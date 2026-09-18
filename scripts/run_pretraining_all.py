"""The nine self-supervised pretraining runs, on the renormalised data.

These are the critical path: every finetuning run waits for one of them. Two GPUs are
used, one run each, and a run starts as soon as a GPU frees up.

Epoch counts differ per configuration on purpose. They equalise how many samples each
encoder sees, since the pretraining sources differ in size by more than a factor of
five. Do not unify them.

Converted checkpoints are written to checkpoints, which is where the
finetuning configs look for them. That directory was emptied along with everything else
before this campaign, so a finetuning run cannot silently pick up a checkpoint trained
on the old intensity scale.
"""
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEO = ROOT / "GeoMAE"
PY = os.environ.get("GEOMAE_PYTHON", "python")
WORK = GEO / "work_dirs"
LOGD = ROOT / "runs" / "logs"
OUT = ROOT / "runs" / "pretraining_results.jsonl"
CKPT = ROOT / "checkpoints"

# (config stem, checkpoint name the finetuning configs expect)
RUNS = [
    ("ssl_seepersea", "mstnet_ssl_seepersea.pth"),
    ("ssl_london", "mstnet_ssl_london.pth"),
    ("ssl_pohang", "mstnet_ssl_pohang.pth"),
    ("ssl_kitti", "mstnet_ssl_kitti.pth"),
    ("ssl_pohang_seepersea", "mstnet_ssl_pohang_seepersea.pth"),
    ("ssl_pohang_london", "mstnet_ssl_pohang_london.pth"),
    ("ssl_seepersea_zaware", "mstnet_ssl_seepersea_zaware.pth"),
    # The learned-encoder arm of E.II. These two configs carry a pillar feature net that
    # is shape-identical to OpenPCDet's DynPillarVFE, ten per-point inputs and two layers
    # of 32 and 64, so its weights transfer into mstnet_c. Their predecessors
    # ssl_seepersea_pfn and ssl_london_pfn had nine inputs and one 64-wide layer, which
    # fits nothing on the detector side; every checkpoint built from them carried the
    # transformer alone and left the detector's feature net randomly initialised.
    ("ssl_seepersea_pfn10", "mstnet_c_ssl_seepersea.pth"),
    ("ssl_london_pfn10", "mstnet_c_ssl_london.pth"),
]

# Configs whose learned voxel encoder has to travel with the transformer. Everything else
# tokenizes with a parameter-free mean, so the SST-only converter is complete for it.
PFN_RUNS = {"ssl_seepersea_pfn10", "ssl_london_pfn10"}

lock = threading.Lock()
queue = list(RUNS)
failures = []  # stems that produced no checkpoint, or raised before reaching run_one's own bookkeeping


def log(m):
    """Print one timestamped line, serialized across the two workers."""
    with lock:
        print(f"[{datetime.now():%m-%d %H:%M:%S}] {m}", flush=True)


OOM = re.compile(r"CUDA out of memory|torch\.cuda\.OutOfMemoryError")


def batch_of(stem):
    """The batch size out of a pretraining configuration."""
    m = re.search(r"samples_per_gpu=(\d+)",
                  (GEO / "configs" / "mae_sst" / f"{stem}.py").read_text())
    return int(m.group(1))


def run_one(gpu, stem, ckpt_name, port):
    """Run one pretraining, halving the batch and starting over if it runs out of memory.

    The batch sizes are set per source from the measured occupied-voxel count of a
    frame, which is what the transformer actually carries, and the densest sources sit
    closer to the card than the sparse ones do. An unlucky draw of dense frames can
    still exceed the card, and it would do so in the first minutes rather than late,
    because it takes only one such batch out of the thousands in an epoch. Rather than
    lose the run and with it every finetuning that waits on its checkpoint, we halve
    and start again. Two halvings are allowed, which covers a factor of four.
    """
    wd = WORK / stem
    LOGD.mkdir(parents=True, exist_ok=True)
    e = os.environ.copy()
    e["PYTHONNOUSERSITE"] = "1"
    e["PYTHONPATH"] = str(GEO)
    e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t0 = time.time()
    batch = None
    for attempt in range(3):
        lf = LOGD / f"PRETRAIN_{stem}.log"
        cmd = [PY, "-m", "torch.distributed.run", "--nproc_per_node=1",
               f"--master_port={port + attempt}", str(GEO / "tools" / "train.py"),
               str(GEO / "configs" / "mae_sst" / f"{stem}.py"),
               "--launcher", "pytorch", "--no-validate",
               "--work-dir", str(wd), "--seed", "0", "--deterministic"]
        if batch is not None:
            cmd += ["--cfg-options", f"data.samples_per_gpu={batch}"]
        log(f"GPU{gpu} START {stem}" + (f"  batch {batch}" if batch else ""))
        with open(lf, "w") as f:
            rc = subprocess.run(cmd, cwd=str(GEO), env=e, stdout=f,
                                stderr=subprocess.STDOUT).returncode
        if rc == 0 or attempt == 2 or not OOM.search(
                lf.read_text(errors="ignore")):
            break
        batch = max(1, (batch or batch_of(stem)) // 2)
        lf.rename(LOGD / f"PRETRAIN_{stem}.oom{attempt + 1}.log")
        log(f"GPU{gpu} OOM   {stem}, retrying at batch {batch}")
    hours = (time.time() - t0) / 3600

    conv = None
    if rc == 0:
        latest = wd / "latest.pth"
        if latest.exists():
            CKPT.mkdir(parents=True, exist_ok=True)
            converter = ("convert_geomae_pfn_ckpt.py" if stem in PFN_RUNS
                         else "convert_geomae_ckpt.py")
            r = subprocess.run([PY, str(ROOT / converter),
                                str(latest), str(CKPT / ckpt_name)],
                               capture_output=True, text=True)
            conv = (CKPT / ckpt_name).exists()
            if not conv:
                log(f"   conversion failed: {r.stderr.strip()[:160]}")
    with lock:
        OUT.open("a").write(json.dumps(
            {"name": stem, "gpu": gpu, "rc": rc, "hours": round(hours, 2),
             "batch": batch or batch_of(stem),
             "checkpoint": ckpt_name if conv else None}) + "\n")
    log(f"GPU{gpu} DONE  {stem}  rc={rc}  {hours:.1f} h  "
        f"{'checkpoint written' if conv else 'NO CHECKPOINT'}")
    if not conv:
        with lock:
            failures.append(stem)


def worker(gpu, port_base):
    """Take pretraining runs off the list and run them on one card.

    A worker thread that raises propagates nowhere on its own: Python prints the
    traceback to stderr and the thread just ends, so a stale GEO path or any other
    exception before run_one's own bookkeeping used to leave main() with two silently
    dead threads and nothing to say so. Every run is therefore wrapped here, and a
    thread that hits an exception records the remainder of its queue as failed instead
    of vanishing.
    """
    n = 0
    while True:
        with lock:
            if not queue:
                return
            stem, ckpt = queue.pop(0)
        try:
            run_one(gpu, stem, ckpt, port_base + 4 * n)
        except Exception as exc:
            log(f"GPU{gpu} CRASHED  {stem}  {type(exc).__name__}: {exc}")
            with lock:
                failures.append(stem)
        n += 1


def main():
    """Run the self-supervised pretraining of every configuration on two cards."""
    log(f"{len(RUNS)} pretraining runs on two GPUs")
    ts = [threading.Thread(target=worker, args=(g, 29600 + 100 * g)) for g in (0, 1)]
    for t in ts:
        t.start()
        time.sleep(30)
    for t in ts:
        t.join()
    if failures:
        log(f"pretraining finished WITH FAILURES: {len(failures)} of {len(RUNS)} produced "
            f"no checkpoint ({', '.join(failures)})")
        return 1
    log("all pretraining finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
