"""Inference cost of every architecture the benchmark reports, measured after the
campaign has finished.

The manuscript reports accuracy and nothing about what a configuration costs to run.
This closes that, with three numbers per architecture and dataset: parameter count,
per-frame latency at batch size one, and peak allocated GPU memory.

Three decisions shape what the numbers mean.

TRAINED WEIGHTS. An older script measured untrained models, on the argument that
latency is a property of the architecture. That holds for the forward pass and not for
what follows it. Every detector keeps detections scored above 0.01 and suppresses
overlaps above 0.01, so the cost of post-processing depends on how many boxes survive,
and an untrained center head emits a different number of them than a trained one. We
load the epoch-80 checkpoint of the run the paper reports and skip any cell whose
checkpoint is missing rather than silently substituting an untrained model.

ONE CARD, NOTHING ELSE ON IT. A latency measured next to a training run measures the
contention, not the model. The sweep therefore waits until no compute process is left on
either card and then runs one cell at a time on one of them.

WHAT THE TIMER COVERS. The network forward and the post-processing, from the moment the
batch is on the GPU. Voxelization and disk reads happen in the dataloader and are
excluded, so these are not end-to-end pipeline numbers. The frames come from the same
evaluation split the accuracy is measured on, so the point counts are the ones a
deployment would see.

Usage:
  python run_speed_sweep.py --list     what it would measure, runs nothing
  setsid nohup python run_speed_sweep.py --go > runs/speed_sweep.log 2>&1 &
"""
import argparse
import json
import sys
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPD = ROOT / "OpenPCDet"
TED = ROOT / "TED"
LION = ROOT / "LION"
OUT = ROOT / "results" / "speed.jsonl"
SPEED = ROOT / "scripts" / "benchmark_speed.py"
PY_PCDET = sys.executable
PY_LION = os.environ.get("LION_PYTHON", "python")

# Same three invocations the campaign uses, so a cell is built the way the run that
# produced its accuracy was built. TED runs on the pcdet interpreter with its own
# package ahead on the path, because the dedicated ted env has corrupted spconv binaries.
FW = {
    "openpcdet": dict(repo=OPD, py=PY_PCDET, pp=None),
    "ted": dict(repo=TED, py=PY_PCDET, pp=str(TED)),
    "lion": dict(repo=LION, py=PY_LION, pp=str(LION)),
}

FRAMES = 200
WARMUP = 20
GPU = 0

# (cfg_stem, framework, tag). The tag picks the run whose checkpoint the paper reports.
BENCHMARK = [
    ("pointpillar", "openpcdet", "rand"),
    ("second", "openpcdet", "rand"),
    ("centerpoint", "openpcdet", "rand"),
    ("voxel_rcnn", "openpcdet", "rand"),
    ("pv_rcnn", "openpcdet", "rand"),
    ("voxelnext", "openpcdet", "rand"),
    ("transfusion", "openpcdet", "rand"),
    ("dsvt", "openpcdet", "rand"),
    ("dsvt_voxel", "openpcdet", "rand"),
    ("lion_mamba", "lion", "rand"),
]

# MSTNet with in-domain pretraining; both initializations share the architecture.
MSTNet_BOTH = [
    ("mstnet", "openpcdet", "ssl"),
]


def jobs():
    """The cells to measure, one per architecture and dataset."""
    J = []
    for ds in ("london", "seepersea"):
        for stem, fw, tag in BENCHMARK + MSTNet_BOTH:
            J.append((ds, stem, fw, tag))
        # TED-S on Thames is the repaired per-class config; SeePerSea is single class and
        # runs the original one.
        J.append((ds, "ted_s_perclass" if ds == "london" else "ted_s", "ted", "rand"))
    return J


def ckpt_of(ds, stem, fw, tag):
    """The final checkpoint of one cell, or None when the run is missing."""
    d = (FW[fw]["repo"] / "output" / f"custom_{ds}_models" / stem / tag / "ckpt")
    cks = sorted(d.glob("checkpoint_epoch_*.pth"),
                 key=lambda p: int(p.stem.split("_")[-1]))
    return cks[-1] if cks else None


def log(m):
    """Print one timestamped line of the sweep log."""
    print(f"[{datetime.now():%m-%d %H:%M:%S}] {m}", flush=True)


def busy():
    """PIDs holding a compute context on any card.

    Read from query-compute-apps rather than from used_memory, because the desktop
    session keeps about 3.6 GB resident on card 0 at all times and a memory threshold
    would call that card busy forever.
    """
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                        "--format=csv,noheader"], capture_output=True, text=True)
    return [x for x in r.stdout.split() if x.strip()]


def wait_for_idle():
    """Both cards clear for five consecutive minutes.

    The campaign hands one job to the next within seconds, so a single idle reading
    proves nothing. Five in a row a minute apart do.
    """
    clear = 0
    waited = 0
    log("waiting for both cards to fall silent")
    while clear < 5:
        b = busy()
        if b:
            if clear:
                log(f"  card busy again ({len(b)} process(es)), restarting the count")
            clear = 0
            time.sleep(120)
            waited += 120
            # A heartbeat, so a log that sits untouched for a day still shows the waiter
            # is alive rather than dead.
            if waited % 3600 < 120:
                log(f"  still waiting, {waited // 3600} h so far, "
                    f"{len(b)} compute process(es) on the cards")
        else:
            clear += 1
            time.sleep(60)
            waited += 60
    log("both cards clear for five minutes, starting the sweep")


def measure(ds, stem, fw, tag):
    """Measure one cell on the idle card and return its record."""
    f = FW[fw]
    cfg = f"cfgs/custom_{ds}_models/{stem}.yaml"
    tools = f["repo"] / "tools"
    rec = {"ds": ds, "model": stem, "fw": fw, "tag": tag}

    ck = ckpt_of(ds, stem, fw, tag)
    if ck is None:
        rec["error"] = "no checkpoint"
        return rec

    e = os.environ.copy()
    e["PYTHONNOUSERSITE"] = "1"
    e["CUDA_VISIBLE_DEVICES"] = str(GPU)
    if f["pp"]:
        e["PYTHONPATH"] = f["pp"] + (":" + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    r = subprocess.run([f["py"], str(SPEED), cfg, "--frames", str(FRAMES),
                        "--warmup", str(WARMUP), "--ckpt", str(ck)],
                       cwd=str(tools), env=e, capture_output=True, text=True)
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("{")), None)
    if line:
        rec.update(json.loads(line))
    else:
        rec["error"] = (r.stderr.strip().splitlines() or ["no output"])[-1][:200]
    return rec


def main():
    """List the cells with --list, or measure them once both cards are idle with --go."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--now", action="store_true",
                    help="skip the idle wait, for a dry run only; the numbers a busy "
                         "card produces are contention and must not be reported")
    a = ap.parse_args()

    J = jobs()
    if a.list or not a.go:
        print(f"\n{len(J)} cells, {FRAMES} timed frames each\n")
        for ds, stem, fw, tag in J:
            ck = ckpt_of(ds, stem, fw, tag)
            print(f"  {ds:<10} {stem:<26} {fw:<10} {tag:<6} "
                  f"{ck.name if ck else 'NO CHECKPOINT'}")
        print()
        return

    if not a.now:
        wait_for_idle()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "error" not in r:
                done.add((r["ds"], r["model"]))

    for ds, stem, fw, tag in J:
        if (ds, stem) in done:
            log(f"  {ds}/{stem} already measured, skipping")
            continue
        t0 = time.time()
        rec = measure(ds, stem, fw, tag)
        rec["wall_s"] = round(time.time() - t0, 1)
        OUT.open("a").write(json.dumps(rec) + "\n")
        if "error" in rec:
            log(f"  {ds:<10} {stem:<26} FAILED  {rec['error'][:100]}")
        else:
            log(f"  {ds:<10} {stem:<26} {rec['params_M']:>7} M  "
                f"{rec['median_ms']:>7} ms median  {rec['p90_ms']:>7} ms p90  "
                f"{rec['peak_mem_MB']:>7} MB")
    log(f"sweep complete, {OUT}")


if __name__ == "__main__":
    main()
