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
    # The MSTNet variants are carried from the measured MSTNet by their extra work: the same
    # trunk for B, a lighter feature net for C and D, a second stage for the rest.
    "mstnet_b": 0.447, "mstnet_c": 0.402, "mstnet_d": 0.402, "mstnet_zaware": 0.469,
    "mstnet_pseudo": 0.447, "mstnet_mv": 0.626, "mstnet_stage2": 0.715,
    "mstnet_zaware_stage2": 0.760,
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


def job(ds, cfg_stem, fw="openpcdet", ckpt=None, fam="", tag=None, prestage=None, seed=None):
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
             + ("_pseudo2" if prestage else "") + (f"_s{seed}" if seed else ""),
        ds=ds, cfg_stem=cfg_stem, fw=fw, ckpt=ckpt, family=fam,
        tag=base_tag + (f"_s{seed}" if seed else ""),
        prestage=prestage,
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


def experiments():
    """The full plan: every supervised run the paper needs, in one list.

    The comments in the body are the record of which experiment each block serves and
    why the blocks that were dropped were dropped. Nothing here is executed; the queue
    sorts this list and skips whatever already has a result.
    """
    E = []
    S = "mstnet_ssl_seepersea.pth"
    L = "mstnet_ssl_london.pth"

    # O5, the published detectors at full supervision. Three of them also close the
    # budget curves of O1 at 100 percent, so they are filed under budget and run early;
    # the benchmark table uses the same runs. The rest are benchmark cells only and are
    # deferred to the end, since they are two thirds of the campaign's cost and nothing
    # in the argument depends on them until Section 4.
    for ds in ("seepersea", "london"):
        for m in ("pointpillar", "centerpoint", "voxelnext"):
            E.append(job(ds, m, fam="budget"))
        # DSVT enters twice, as the pillar configuration its authors publish for
        # bird's-eye detection and as the voxel one with 32 height bins. They are two of
        # the eleven rows C4 promises, not a variant and its diagnostic, so both run on
        # both datasets.
        for m in ("second", "voxel_rcnn", "pv_rcnn", "dsvt", "dsvt_voxel", "transfusion"):
            E.append(job(ds, m, fam="benchmark"))
        E.append(job(ds, "ted_s", fw="ted", fam="benchmark"))
        E.append(job(ds, "lion_mamba", fw="lion", fam="benchmark"))

    # TED-S on Thames, repaired. The stored cell (mAP 30.49) is not a usable benchmark
    # number. Its RoI target assigner reads CLS_FG_THRESH, CLS_BG_THRESH and
    # REG_FG_THRESH as one entry per class under CLS_SCORE_TYPE roi_iou_x, and the
    # config carried the length-one lists of the upstream single-class KITTI recipe, so
    # every proposal matched to medium_boat or large_boat was trained as background
    # whatever its overlap. Measured consequence on the test split: 271 detections for
    # 268 small vessels at a median score of 0.994, 77 for 1984 medium ones with no
    # score above 0.244 and none at all for the 325 large ones, while the refined boxes
    # before filtering still cover 89.9 percent of the annotations at an overlap of 0.5.
    # ted_s_perclass.yaml is that config with the five lists extended to three entries.
    # SeePerSea is single class and correct as it stands, so only Thames is repeated.
    # It runs ahead of the seeds because it is the longest single job left and Section 4
    # cannot be written without it.
    E.append(job("london", "ted_s_perclass", fw="ted", fam="benchmark-repair"))

    # O2, experiment E.II: the two voxel encoders crossed with the two initializations,
    # at every budget on both datasets. The unlearned mean encoder is MSTNet itself and
    # its two curves are already built above under budget; here the learned feature net
    # gets the same four curves. The SSTOutConv variants that used to sit here, mstnet_b
    # and mstnet_d, were dropped on 2026-08-11: they cross the encoder with the post-SST
    # backbone, which is not what E.II asks and which no other experiment needs.
    C_SSL = {"seepersea": "mstnet_c_ssl_seepersea.pth", "london": "mstnet_c_ssl_london.pth"}
    for ds in ("seepersea", "london"):
        for b in ("", "_5pct", "_10pct", "_20pct", "_50pct", "_80pct"):
            E.append(job(ds, f"mstnet_c{b}", fam="chain"))
            E.append(job(ds, f"mstnet_c{b}", ckpt=C_SSL[ds], fam="chain"))
        E.append(job(ds, "mstnet", fam="chain"))
        E.append(job(ds, "mstnet", ckpt=S if ds == "seepersea" else L, fam="chain"))

    # O1, the label budget sweep. Cheapest budgets first is handled by the sort below.
    for b in (5, 10, 20, 50, 80):
        for ds in ("seepersea", "london"):
            for m in ("pointpillar", "centerpoint", "voxelnext", "mstnet"):
                E.append(job(ds, f"{m}_{b}pct", fam="budget"))
            E.append(job(ds, f"mstnet_{b}pct", ckpt=S if ds == "seepersea" else L,
                         fam="budget"))

    # O3, experiment E.IV. Each pretraining set is always the whole unlabeled source; what
    # varies over the budget is the supervised finetuning on top of it. The in-domain arm
    # is E.I, so only the foreign sources appear here, and they run at every budget because
    # the question is whether a foreign source still buys anything where labels are scarce,
    # which full supervision cannot answer.
    #
    # Dropped on 2026-08-17 by author decision: pretraining one annotated dataset on the
    # other, so Thames on the SeePerSea frames and SeePerSea on the Thames frames. It is
    # not one of the four schemes the paper defines and it will not be reported, so the
    # remaining cells of that arm are not worth the GPU time. The cells that already ran
    # stay in campaign_results.jsonl as a record of what was executed and are excluded from
    # the paper tables by make_paper_tables.py.
    for ds, srcs in {"seepersea": ("pohang", "kitti", "pohang_seepersea"),
                     "london": ("pohang", "kitti", "pohang_london")}.items():
        for s in srcs:
            for b in ("", "_5pct", "_10pct", "_20pct", "_50pct", "_80pct"):
                E.append(job(ds, f"mstnet{b}", ckpt=f"mstnet_ssl_{s}.pth",
                             fam="pretrain-source", tag=f"ssl_{s}"))

    # O4, the interventions against the height deficit. SeePerSea only, since that is
    # where the deficit is.
    # Experiments E.IX to E.XI, the three measures against the height deficit, each
    # across the label budgets rather than at full supervision alone, because a measure
    # that helps only where labels are plentiful would not serve this work. The
    # multi-view stream is not one of the three and its budget sweep was dropped on
    # 2026-08-11; it stays at full supervision as an appendix entry, where its effect of
    # +2.84 in 3D against a seed spread of 1.75 is reported as indistinguishable.
    Z = "mstnet_ssl_seepersea_zaware.pth"
    E.append(job("seepersea", "mstnet_zaware", fam="height"))
    E.append(job("seepersea", "mstnet_zaware_stage2", ckpt=Z, fam="height"))
    E.append(job("seepersea", "mstnet_mv", ckpt=S, fam="height"))
    for b in ("", "_5pct", "_10pct", "_20pct", "_50pct", "_80pct"):
        E.append(job("seepersea", f"mstnet_zaware{b}", ckpt=Z, fam="height-budget"))
        E.append(job("seepersea", f"mstnet_stage2{b}", ckpt=S, fam="height-budget"))
        # E.V. The supervised configuration is the ordinary one; what differs is that its
        # backbone and head start from the pseudo-label stage rather than from random.
        E.append(job("seepersea", f"mstnet{b}", ckpt=S, fam="pseudo",
                     prestage=f"mstnet_pseudo{b}", tag="pseudo2"))

    # The two measures of O4 in combination, over the budgets. Until 2026-08-25 that
    # configuration existed at full supervision alone, which is the one budget where the
    # question O4 asks, whether the vertical deficit can be reduced without more
    # annotation, matters least. The five configs derive from mstnet_zaware_stage2.yaml and
    # differ from it only in the dataset config they inherit.
    for b in ("_5pct", "_10pct", "_20pct", "_50pct", "_80pct"):
        E.append(job("seepersea", f"mstnet_zaware_stage2{b}", ckpt=Z, fam="combo"))

    # The triple combination, added 2026-08-26. The pseudo-label stage of E.V pays at the
    # two smallest budgets and is neutral above them, and the vertical-only refinement of
    # E.VI is measured at full supervision alone, so the question the two leave open is
    # whether the strongest configuration also carries into the low-label regime. Each
    # budget runs twice, once as the vertical-only refinement on its own, which is the
    # control that did not exist below full supervision, and once with the pseudo-label
    # stage in front of it. Stage one uses the z-aware token so the backbone and the head
    # are initialised on the same representation the fine-tuning will use.
    # No repeats: the seed family is what will tell us how much of any difference here is
    # readable, and running repeats of a new arm before that lands would repeat the
    # mistake the E.IV outlier cells already made.
    for b in ("_5pct", "_10pct", "_20pct"):
        E.append(job("seepersea", f"mstnet_zaware_stage2_zonly{b}", ckpt=Z, fam="triple"))
        E.append(job("seepersea", f"mstnet_zaware_stage2_zonly{b}", ckpt=Z, fam="triple",
                     prestage=f"mstnet_pseudo_zaware{b}"))
    # The two budgets the control arm above did not need. Added 2026-08-27 because that
    # arm turned out to hold the highest 3D value of the family at three budgets and to
    # supply the strongest-variant row of the benchmark, so its curve carries the section
    # and may not have holes in the middle. No pseudo-label counterpart: the stage is
    # neutral on MSTNet from 20 percent labels upward.
    for b in ("_50pct", "_80pct"):
        E.append(job("seepersea", f"mstnet_zaware_stage2_zonly{b}", ckpt=Z, fam="triple"))
    # The pseudo-label counterpart at the three larger budgets, added 2026-08-27. The
    # scoping to the three smallest budgets rested on the stage turning neutral on
    # MSTNet from 20 percent labels upward, and that crossing point does not transfer:
    # on this arm the stage still adds 6.03 points in 3D at 20 percent, more than it adds
    # to MSTNet at 10. Full supervision matters most of the three, since that cell is
    # the strongest-variant row of the benchmark table.
    for b in ("_50pct", "_80pct", ""):
        E.append(job("seepersea", f"mstnet_zaware_stage2_zonly{b}", ckpt=Z, fam="triple",
                     prestage=f"mstnet_pseudo_zaware{b}"))

    # Repeated runs for the cells a claim rests on and for the cells whose value falls as
    # their budget grows, which is the only run-to-run spread we can observe. The
    # supervised runs are deliberately unseeded, so a repeat differs from its original in
    # nothing but the tag. Two repeats give three samples per cell.
    SEEDCELLS = [
        ("seepersea", "mstnet", None, ""), ("seepersea", "mstnet", None, "_80pct"),
        ("london", "mstnet", None, "_50pct"), ("london", "mstnet", None, "_80pct"),
        ("london", "centerpoint", None, "_50pct"), ("london", "centerpoint", None, "_80pct"),
        ("seepersea", "mstnet", "mstnet_ssl_kitti.pth", "_20pct"),
        ("seepersea", "mstnet", "mstnet_ssl_pohang_seepersea.pth", "_10pct"),
        ("seepersea", "mstnet", S, ""),
        ("seepersea", "mstnet_zaware", Z, ""),
        ("seepersea", "mstnet_zaware_stage2", Z, ""),
        ("seepersea", "voxelnext", None, ""),
    ]
    for ds, stem, ck, b in SEEDCELLS:
        for s in (2, 3):
            tg = None
            if ck and ck not in (S, Z):
                tg = "ssl_" + Path(ck).stem.replace("mstnet_ssl_", "")
            E.append(job(ds, f"{stem}{b}", ckpt=ck, fam="seeds", tag=tg, seed=s))

    # Diagnostics. These support statements in the discussion and the negative-results
    # appendix; they are not benchmark cells.
    # The cap128 and iscale variants were dropped on 2026-08-11. Both were diagnostics
    # for defects that have since been repaired for every run: the per-cell cap is 128 in
    # the main dataset config, which makes custom_seepersea_cap128.yaml byte-identical to
    # it, and intensity is normalized by the training split's 99th percentile everywhere,
    # which leaves the iscale variant measuring a mismatch it would create itself.
    # Keeping them would put duplicate rows in the appendix table.
    # The vertical line of work that ran through the canonicalised copy is closed and its
    # runs are dropped on 2026-08-17. The data-side finding stands on the annotations, which
    # the rebuild did not touch: the per-frame offset carries almost all of the height
    # scatter because the sensor sits at a different height in every session. The oracle
    # canonicalisation was only ever the gate in front of the learned water-level head, and
    # that head failed for a reason we can name, gt_sampling pastes database boxes at their
    # original absolute height and destroys the single-plane assumption the head needs. Its
    # only defensible test therefore runs without gt_sampling, which no other cell of the
    # study does. Nothing left to measure, so centerpoint_canon, pointpillar_canon,
    # voxelnext_canon, centerpoint_noaug and centerpoint_wl are gone, 14.7 GPU-h. Restore
    # this block if the ceiling is ever wanted as a number again.

    # dsvt_voxel used to sit here as a SeePerSea-only diagnostic. It is a benchmark row
    # on both datasets as of 2026-08-10 and is built above.
    # mstnet_canon went with the block above. The z-only variant of the combined measure stays,
    # it asks whether one configuration can hold the best bird's-eye and the best
    # three-dimensional accuracy at once, which is about our own two measures.
    E.append(job("seepersea", "mstnet_zaware_stage2_zonly", ckpt=Z, fam="diagnostic"))
    for b in (5, 10, 20, 50, 80):
        E.append(job("seepersea", f"pointpillar_{b}pct_ba", fam="diagnostic"))
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
                        "--extra_tag", tag], cwd=str(tools), env=env(gpu, e["fw"]),
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
            r = subprocess.run([PY_PCDET, str(ROOT / "eval_london_perclass.py"),
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
    # Families in the order the paper argues them, cheapest first inside each, so that a
    # family finishes as a whole and its section can be written while the next one runs.
    # A global cheapest-first order would instead leave every family half measured for
    # days, which is the one thing that helps nobody.
    # The pseudo-label arm sits behind the height family, whose runs it shares a dataset
    # with, and ahead of the diagnostics and the benchmark, because it closes O1 while
    # those two only support the discussion and Section 4.7.
    order = {"budget": 0, "chain": 1, "pretrain-source": 2, "height": 3,
             "height-budget": 4, "pseudo": 5, "combo": 6, "benchmark-repair": 7,
             "triple": 8, "seeds": 9, "diagnostic": 10, "benchmark": 11}
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
