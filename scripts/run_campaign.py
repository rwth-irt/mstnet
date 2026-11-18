"""The finetuning campaign: every supervised run the paper needs, on two GPUs.

One queue, two workers, ordered cheapest first so the low-budget cells, which carry the
label-efficiency claim, land within hours rather than at the end. A run that fails is
recorded and the queue continues; rerunning the script skips whatever already has a
result, so it can be stopped and resumed at will.

Batch sizes come from the configs, not from this file. The hyperparameter table in the
appendix is generated from the same configs, so overriding the batch here would silently
make that table describe something we did not run.

Three frameworks, three invocations. OpenPCDet runs in the pcdet env. TED runs on the
same interpreter with its own package ahead on PYTHONPATH, because the dedicated ted env
has corrupted spconv binaries that produce garbage voxel coordinates. LION has its own
env with mamba built from its tree.

The inference-cost sweep is NOT part of this queue and must not be. Both workers run at
once, so any latency measured here would measure the contention with the other card's
job. run_speed_sweep.py waits for both cards to fall silent and then measures on its own;
launch it alongside this script and it will start when this one is finished.

Usage:
  python run_campaign.py --list          the plan and the time estimate, runs nothing
  setsid nohup python run_campaign.py --go > runs/campaign.log 2>&1 &
"""
import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPD = ROOT / "OpenPCDet"
TED = ROOT / "TED"
LION = ROOT / "LION"
CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "runs" / "campaign_results.jsonl"
LOGD = ROOT / "runs" / "logs"
PY_PCDET = sys.executable
PY_LION = os.environ.get("LION_PYTHON", "python")

FW = {
    "openpcdet": dict(repo=OPD, py=PY_PCDET, pp=None),
    "ted": dict(repo=TED, py=PY_PCDET, pp=str(TED)),
    "lion": dict(repo=LION, py=PY_LION, pp=str(LION)),
}

# Labelled frames per split, read once at import so the estimate cannot drift from the
# splits the runs actually consume.
FRAMES = {}
for _ds in ("seepersea", "london"):
    _b = OPD / "data" / f"custom_{_ds}" / "ImageSets"
    FRAMES[_ds] = {100: len((_b / "train.txt").read_text().split())}
    for _p in (5, 10, 20, 50, 80):
        FRAMES[_ds][_p] = len((_b / f"train_{_p}pct.txt").read_text().split())

# Seconds per iteration, measured on one RTX 4090 where we have numbers and estimated
# from architecture class where we do not. Used ONLY to order the queue and to print an
# estimate; every run records its true duration, so the estimate is replaced by
# measurement as soon as the first jobs finish.
# Recalibrated 2026-08-10 from the first ten completed runs. The four marked "measured"
# come from their own recorded durations divided by their iteration counts; the rest are
# still priors and the benchmark family in particular is unverified.
SEC_ITER = {
    "pointpillar": 0.078,   # measured, n=4
    "centerpoint": 0.244,   # measured, n=3
    "voxelnext": 0.084,     # measured, n=1
    "mstnet": 0.447,          # measured, n=2
    "second": 0.13, "transfusion": 0.20, "voxel_rcnn": 0.30, "dsvt": 0.35,
    "pv_rcnn": 0.45, "ted_s": 0.55, "lion_mamba": 0.50, "dsvt_voxel": 0.55,
}
EPOCHS = 80


def base_model(cfg_stem):
    """Strip the budget and variant suffixes to get the timing class."""
    s = re.sub(r"_\d+pct(_ba)?$", "", cfg_stem)
    s = re.sub(r"_(canon|noaug|cap128|iscale|wl|zonly|perclass)$", "", s)
    return s if s in SEC_ITER else re.sub(r"_(b|c|d)$", "", s)


def budget_of(cfg_stem):
    """The label budget of a configuration, in percent, 100 when it carries no suffix."""
    m = re.search(r"_(\d+)pct", cfg_stem)
    return int(m.group(1)) if m else 100


def lr_suffix(lr):
    """Name suffix of a run whose learning rate differs from its config, e.g. _lr3e-3."""
    return f"_lr{lr:.0e}".replace("e-0", "e-") if lr else ""


def job(ds, cfg_stem, fw="openpcdet", ckpt=None, fam="", tag=None, prestage=None, seed=None,
        lr=None):
    """One queue entry, with its name, its tag and an estimate of its duration.

    The name identifies the run everywhere afterwards, in the output directory, in the
    record the queue appends and in the results CSV, so it has to carry the dataset,
    the configuration, the pretraining and the seed.
    """
    b = budget_of(cfg_stem)
    bm = base_model(cfg_stem)
    batch = BATCH.get((ds, cfg_stem), 4)
    iters = FRAMES[ds][b] * EPOCHS / max(batch, 1)
    h = iters * SEC_ITER.get(bm, 0.25) / 3600
    if prestage:
        # Stage one runs 30 epochs over the same frames, so it adds three eighths again.
        h *= 1 + 30 / EPOCHS
    base_tag = tag or ("pseudo2" if prestage else ("ssl" if ckpt else "rand"))
    return dict(
        name=f"{ds}_{cfg_stem}" + (f"_{Path(ckpt).stem}" if ckpt else "_rand")
             + ("_pseudo2" if prestage else "") + lr_suffix(lr) + (f"_s{seed}" if seed else ""),
        ds=ds, cfg_stem=cfg_stem, fw=fw, ckpt=ckpt, family=fam,
        tag=base_tag + lr_suffix(lr) + (f"_s{seed}" if seed else ""),
        prestage=prestage, lr=lr,
        hours=round(h, 2),
    )


def read_epochs(fw, ds, cfg_stem):
    """NUM_EPOCHS out of the config, so the guard below knows when a run is finished."""
    q = FW[fw]["repo"] / "tools" / "cfgs" / f"custom_{ds}_models" / f"{cfg_stem}.yaml"
    if not q.exists():
        return None
    m = re.search(r"NUM_EPOCHS:\s*(\d+)", q.read_text())
    return int(m.group(1)) if m else None


def already_trained(fw, ds, cfg_stem, tag):
    """The final checkpoint of a run that has already finished, or None.

    A campaign killed between a run's last epoch and the write of its record leaves the
    run looking outstanding. train.py then resumes at the final epoch, trains nothing and
    falls into repeat_eval_ckpt, whose exit condition is `total_time > max_waiting_mins *
    60 and first_eval is False` (tools/test.py). With max_waiting_mins zero and nothing
    left to evaluate, first_eval never turns false and the process waits forever, holding
    a card. Observed twice on 2026-08-28. The queue therefore looks for the final
    checkpoint itself and skips straight to the evaluation.
    """
    n = read_epochs(fw, ds, cfg_stem)
    if not n:
        return None
    q = (FW[fw]["repo"] / "output" / f"custom_{ds}_models" / cfg_stem / tag /
         "ckpt" / f"checkpoint_epoch_{n}.pth")
    return q if q.exists() else None


def read_batch(fw, ds, cfg_stem):
    """The batch size out of a configuration, so the estimate matches what will run."""
    p = FW[fw]["repo"] / "tools" / "cfgs" / f"custom_{ds}_models" / f"{cfg_stem}.yaml"
    if not p.exists():
        return None
    m = re.search(r"BATCH_SIZE_PER_GPU:\s*(\d+)", p.read_text())
    return int(m.group(1)) if m else 4


BATCH = {}


def runs(ds, stem, n=1, **kw):
    """n runs of one cell. Run 1 carries no seed suffix, runs 2 and 3 carry _s2 and _s3."""
    return [job(ds, stem, seed=s, **kw) for s in (None, 2, 3)[:n]]


def experiments():
    """Every supervised run of the paper and its appendix, 165 in total.

    E.I    MSTNet without and with in-domain pretraining and the CenterPoint baseline at the
           six label budgets on both datasets
    E.II   MSTNet with the cross-domain maritime, the automotive and the combined
           pretraining at the six budgets on both datasets
    E.III  the eleven published detectors at full supervision on both datasets, and
           PointPillars and VoxelNeXt at the five smaller budgets
    App. C MSTNet pretrained without the SeePerSea validation frames at 5, 20 and 100 %
    lr     MSTNet without pretraining on SeePerSea at 5 and 100 % at the CenterPoint rate of
           0.003, the comparison behind the rate of 0.001 (Section 2.4.2 of the paper)

    Three runs exist for MSTNet and VoxelNeXt on SeePerSea at every budget, for MSTNet on
    Thames at 5 and 100 %, and for CenterPoint on Thames at 50 and 80 %; every other cell
    is a single run. The supervised runs are deliberately unseeded, so a repeat differs from
    its original in nothing but the tag.
    """
    E = []
    budgets = ("_5pct", "_10pct", "_20pct", "_50pct", "_80pct", "")
    indomain = {"seepersea": "mstnet_ssl_seepersea.pth", "london": "mstnet_ssl_london.pth"}
    sources = {"seepersea": ("pohang", "kitti", "pohang_seepersea"),
               "london": ("pohang", "kitti", "pohang_london")}

    for ds in ("seepersea", "london"):
        for b in budgets:
            n = 3 if ds == "seepersea" or b in ("_5pct", "") else 1
            E += runs(ds, f"mstnet{b}", n, fam="E.I")
            E += runs(ds, f"mstnet{b}", n, ckpt=indomain[ds], fam="E.I")
            n = 3 if ds == "london" and b in ("_50pct", "_80pct") else 1
            E += runs(ds, f"centerpoint{b}", n, fam="E.I")

        for s in sources[ds]:
            for b in budgets:
                E += runs(ds, f"mstnet{b}", ckpt=f"mstnet_ssl_{s}.pth", fam="E.II", tag=f"ssl_{s}")

        for m in ("pointpillar", "second", "voxel_rcnn", "pv_rcnn", "dsvt", "dsvt_voxel",
                  "transfusion"):
            E += runs(ds, m, fam="E.III")
        E += runs(ds, "voxelnext", 3 if ds == "seepersea" else 1, fam="E.III")
        # On Thames, TED-S runs with one RoI assignment threshold per class; the upstream
        # single-class recipe trains every medium and large proposal as background.
        E += runs(ds, "ted_s_perclass" if ds == "london" else "ted_s", fw="ted", fam="E.III")
        E += runs(ds, "lion_mamba", fw="lion", fam="E.III")
        for b in budgets[:-1]:
            E += runs(ds, f"pointpillar{b}", fam="E.III")
            E += runs(ds, f"voxelnext{b}", 3 if ds == "seepersea" else 1, fam="E.III")

    for b in ("_5pct", "_20pct", ""):
        E += runs("seepersea", f"mstnet{b}", ckpt="mstnet_ssl_seepersea_noval.pth",
                  fam="appendix-C", tag="ssl_noval")
    for b in ("_5pct", ""):
        E += runs("seepersea", f"mstnet{b}", lr=0.003, fam="lr")
    return E


def log(m):
    """Print one timestamped line of the campaign log."""
    print(f"[{datetime.now():%m-%d %H:%M:%S}] {m}", flush=True)


_lock = threading.Lock()


def env(gpu, fw):
    """The environment of one job: its card, its interpreter and its package path."""
    e = os.environ.copy()
    e["PYTHONNOUSERSITE"] = "1"
    e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    pp = FW[fw]["pp"]
    if pp:
        e["PYTHONPATH"] = pp + (":" + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    return e


def parse_seepersea(logfile):
    """Single-class R40 out of the test.py log. The bbox line has to reset the section,
    otherwise its numbers are read as the 3d and bev ones."""
    res, cur = {}, None
    for line in Path(logfile).read_text(errors="ignore").splitlines():
        m = re.search(r"Car AP_R40@[\d.]+, [\d.]+, ([\d.]+):", line)
        if m:
            cur = "0.7" if m.group(1) == "0.70" else "0.5"
            continue
        if re.search(r"Car AP@", line):
            cur = None
            continue
        if cur:
            mm = re.search(r"^(3d|bev)\s+AP:([\d.]+)", line.strip())
            if mm:
                res[f"{mm.group(1)}@{cur}"] = float(mm.group(2))
    return res


def do_job(e, gpu):
    """Train one cell, evaluate it and append its record.

    A run whose final checkpoint already exists skips straight to the evaluation. The
    per-class evaluation of Thames runs in a subprocess: its numba kernel has aborted
    in the past and, called in this process, took both worker threads with it.
    """
    fw = FW[e["fw"]]
    tools = fw["repo"] / "tools"
    cfg = f"cfgs/custom_{e['ds']}_models/{e['cfg_stem']}.yaml"
    stem = f"custom_{e['ds']}_models/{e['cfg_stem']}"
    tag = e["tag"]
    # A learning rate other than the config's is set on the command line. The override has
    # to be the last argument: --set is parsed with argparse.REMAINDER and swallows whatever
    # follows it. test.py gets it too, so that the logged config is the one that trained.
    lr_set = ["--set", "OPTIMIZATION.LR", str(e["lr"])] if e.get("lr") else []
    rec = dict(e)
    rec["gpu"] = gpu

    pre = e.get("prestage")
    start = str(CKPT / e["ckpt"]) if e["ckpt"] else None
    if pre:
        # Stage one of the pseudo-label arm. The masked pretraining goes into the sparse
        # transformer and the transformer is frozen, so the boxes that DBSCAN produces can
        # train the convolutional backbone and the center head without degrading the
        # encoder. Only its checkpoint then starts the supervised fine-tuning below, which
        # is what the pseudo-label section of the paper describes and what
        # run_pseudo_pretrain.py did before the campaign absorbed it.
        pcfg = f"cfgs/custom_{e['ds']}_models/{pre}.yaml"
        pstem = f"custom_{e['ds']}_models/{pre}"
        phave = already_trained(e["fw"], e["ds"], pre, tag)
        if phave:
            log(f"GPU{gpu} SKIP  {e['name']} stage 1, {phave.name} already on disk")
            rc1 = 0
            rec["stage1_rc"] = 0
            rec["stage1_min"] = 0.0
        else:
            pcmd = [fw["py"], "train.py", "--cfg_file", pcfg, "--extra_tag", tag,
                    "--freeze_modules", "map_to_bev_module"]
            if start:
                pcmd += ["--pretrained_model", start]
            pcmd += lr_set
            log(f"GPU{gpu} STAGE1 {e['name']}")
            t1 = time.time()
            with open(LOGD / f"CAM_{e['name']}_stage1.log", "w") as lf:
                rc1 = subprocess.run(pcmd, cwd=str(tools), env=env(gpu, e["fw"]),
                                     stdout=lf, stderr=subprocess.STDOUT).returncode
            rec["stage1_rc"] = rc1
            rec["stage1_min"] = round((time.time() - t1) / 60, 1)
        pcks = sorted((fw["repo"] / "output" / pstem / tag / "ckpt").glob("checkpoint_epoch_*.pth"),
                      key=lambda q: int(q.stem.split("_")[-1]))
        if rc1 != 0 or not pcks:
            rec["error"] = f"stage1 rc={rc1}" if rc1 else "stage1 no-ckpt"
            log(f"GPU{gpu} FAILED {e['name']} in stage 1")
            with _lock:
                RESULTS.open("a").write(json.dumps(rec) + "\n")
            return
        start = str(pcks[-1])
        rec["stage1_ckpt"] = pcks[-1].name

    have = already_trained(e["fw"], e["ds"], e["cfg_stem"], tag)
    if have:
        log(f"GPU{gpu} SKIP  {e['name']}, {have.name} already on disk")
        rc = 0
        rec["train_rc"] = 0
        rec["train_min"] = 0.0
        rec["resumed_from_disk"] = True
    else:
        cmd = [fw["py"], "train.py", "--cfg_file", cfg, "--extra_tag", tag]
        if start:
            cmd += ["--pretrained_model", start]
        cmd += lr_set
        log(f"GPU{gpu} TRAIN {e['name']}  (est {e['hours']} h)")
        t0 = time.time()
        with open(LOGD / f"CAM_{e['name']}_train.log", "w") as lf:
            rc = subprocess.run(cmd, cwd=str(tools), env=env(gpu, e["fw"]),
                                stdout=lf, stderr=subprocess.STDOUT).returncode
        rec["train_rc"] = rc
        rec["train_min"] = round((time.time() - t0) / 60, 1)
    if rc != 0:
        log(f"GPU{gpu} FAILED {e['name']} rc={rc} after {rec['train_min']} min")
        with _lock:
            RESULTS.open("a").write(json.dumps(rec) + "\n")
        return

    cks = sorted((fw["repo"] / "output" / stem / tag / "ckpt").glob("checkpoint_epoch_*.pth"),
                 key=lambda p: int(p.stem.split("_")[-1]))
    if not cks:
        rec["error"] = "no-ckpt"
        with _lock:
            RESULTS.open("a").write(json.dumps(rec) + "\n")
        return
    rec["ckpt_file"] = cks[-1].name

    log(f"GPU{gpu} EVAL  {e['name']}")
    elog = LOGD / f"CAM_{e['name']}_eval.log"
    with open(elog, "w") as lf:
        subprocess.run([fw["py"], "test.py", "--cfg_file", cfg, "--ckpt", str(cks[-1]),
                        "--extra_tag", tag] + lr_set, cwd=str(tools), env=env(gpu, e["fw"]),
                       stdout=lf, stderr=subprocess.STDOUT)
    if e["ds"] == "seepersea":
        rec["ap"] = parse_seepersea(elog)
    else:
        cands = sorted((fw["repo"] / "output" / stem / tag / "eval").rglob("result.pkl"),
                       key=lambda p: ("eval_with_train" in str(p), str(p)))
        if cands:
            # Out of process, and pinned to this worker's card. The per-class evaluator
            # runs KITTI's rotate_iou as a numba.cuda kernel, and calling it inside the
            # campaign aborted the whole process, both worker threads with it, on three
            # occasions between 2026-08-26 and 2026-08-31. Each time the campaign log
            # ends on a "GPU0 EVAL london_..." line and the run's record is never
            # written, while its checkpoints and its test.py evaluation sit complete on
            # disk. As a subprocess the abort is one failed job instead of a dead queue.
            r = subprocess.run([PY_PCDET, str(ROOT / "scripts" / "eval_london_perclass.py"),
                                "--pkl", str(cands[0])],
                               env=env(gpu, e["fw"]), capture_output=True, text=True)
            line = next((l for l in reversed(r.stdout.splitlines())
                         if l.startswith("{")), None)
            if line:
                rec["ap"] = json.loads(line)
            else:
                rec["error"] = f"perclass-eval rc={r.returncode}"
                log(f"GPU{gpu} per-class evaluation failed for {e['name']}, rc={r.returncode}")
        else:
            rec["error"] = "no-result-pkl"
    log(f"GPU{gpu} DONE  {e['name']}  {rec['train_min']} min  {rec.get('ap', {})}")
    with _lock:
        RESULTS.open("a").write(json.dumps(rec) + "\n")


def worker(gpu, q):
    """Take jobs off the queue and run them on one card until the queue is empty."""
    while True:
        try:
            e = q.get_nowait()
        except queue.Empty:
            return
        try:
            do_job(e, gpu)
        except Exception as ex:
            log(f"GPU{gpu} EXC {e['name']}: {type(ex).__name__} {ex}")
        finally:
            q.task_done()


def done_names():
    """The names of the runs that already have a record, so a restart skips them."""
    d = set()
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("train_rc") == 0 and r.get("ap"):
                d.add(r["name"])
    return d


def main():
    """Print the plan with --list, or run the campaign on two cards with --go."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--go", action="store_true")
    # Which cards to use. Both by default; a subset frees a card for something else, as
    # when a pretraining that the campaign itself depends on has to be run alongside it.
    ap.add_argument("--gpus", default="0,1")
    a = ap.parse_args()
    gpus = [int(x) for x in a.gpus.split(",") if x != ""]

    # Batch sizes have to be read before the jobs are costed.
    for ds in ("seepersea", "london"):
        for fwname in FW:
            d = FW[fwname]["repo"] / "tools" / "cfgs" / f"custom_{ds}_models"
            if d.exists():
                for p in d.glob("*.yaml"):
                    b = read_batch(fwname, ds, p.stem)
                    if b:
                        BATCH[(ds, p.stem)] = b

    E = experiments()
    missing = [e for e in E
               if not (FW[e["fw"]]["repo"] / "tools" / "cfgs" /
                       f"custom_{e['ds']}_models" / f"{e['cfg_stem']}.yaml").exists()]
    nockpt = [e for e in E if e["ckpt"] and not (CKPT / e["ckpt"]).exists()]
    for e in missing:
        log(f"SKIP, config missing: {e['name']}")
    for e in nockpt:
        log(f"SKIP, checkpoint missing: {e['name']}  ({e['ckpt']})")
    bad = {e["name"] for e in missing + nockpt}
    E = [e for e in E if e["name"] not in bad]

    done = done_names()
    todo = [e for e in E if e["name"] not in done]
    # Families in the order of the paper, cheapest first inside each, so that a family
    # finishes as a whole while the next one runs.
    order = {"lr": 0, "E.I": 1, "E.II": 2, "E.III": 3, "appendix-C": 4}
    todo.sort(key=lambda x: (order.get(x["family"], 9), x["hours"]))

    if a.list or not a.go:
        fams = {}
        for e in E:
            fams.setdefault(e["family"], []).append(e)
        print(f"\n{len(E)} runs, {len(done)} already done, {len(todo)} to go\n")
        for f, items in fams.items():
            print(f"  {f:<16} {len(items):3d} runs   {sum(i['hours'] for i in items):7.1f} GPU-h")
        tot = sum(e["hours"] for e in todo)
        print(f"\n  total {tot:.0f} GPU-hours, about {tot / 2 / 24:.1f} days on two cards")
        print(f"  longest single run: " +
              ", ".join(f"{x['name']} {x['hours']:.1f} h"
                        for x in sorted(todo, key=lambda x: -x["hours"])[:5]))
        print("\n  estimates only; every run records its true duration.\n")
        if not a.go:
            return

    LOGD.mkdir(parents=True, exist_ok=True)
    q = queue.Queue()
    for e in todo:
        q.put(e)
    log(f"campaign start: {len(todo)} runs, estimated "
        f"{sum(e['hours'] for e in todo):.0f} GPU-hours, on GPU {gpus}")
    ts = [threading.Thread(target=worker, args=(g, q)) for g in gpus]
    for t in ts:
        t.start()
        time.sleep(20)
    for t in ts:
        t.join()
    log("CAMPAIGN COMPLETE")


if __name__ == "__main__":
    main()
