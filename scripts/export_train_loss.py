"""Per-epoch training loss of the supervised runs, out of the TensorBoard events.

Section 4.1 and 4.2 of the paper argue from how the training loss behaves, so those
curves have to be in the results and they have to be reproducible. The framework writes
one scalar per iteration into `output/<group>/<cfg>/<tag>/tensorboard/`; this script
averages them per epoch and writes one tidy CSV that make_paper_tables.py turns into the
figure. A run that was resumed leaves several event files, so all of them are read and
the (step, value) pairs deduplicated before averaging.

It also writes results/train_loss_extra.csv, the endpoints Sections 3.1 and 3.5 quote
for cells that carry no curve of their own.

Usage:
  python export_train_loss.py            # writes both CSVs under runs/
"""
import csv
import glob

from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "train_loss.csv"
EPOCHS = 80

# The eight cells the two encoder arguments rest on, both datasets at full supervision.
RUNS = [(ds, enc, init, f"custom_{ds}_models/{stem}/{tag}")
        for ds in ("london", "seepersea")
        for enc, stem in (("mean", "mstnet"), ("learned", "mstnet_c"))
        for init, tag in (("random", "rand"), ("pretrained", "ssl"))]


def curve(rel):
    """Per-epoch mean training loss of one run, or None if it wrote no events.

    A resumed run leaves several event files, so all of them are read and the
    (step, value) pairs deduplicated before averaging.
    """
    pairs = set()
    for f in sorted(glob.glob(str(ROOT / "OpenPCDet" / "output" / rel / "tensorboard" / "*"))):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        if "train/loss" in ea.Tags()["scalars"]:
            pairs |= {(s.step, s.value) for s in ea.Scalars("train/loss")}
    if not pairs:
        return None
    pairs = sorted(pairs)
    steps = np.array([p[0] for p in pairs], dtype=float)
    vals = np.array([p[1] for p in pairs], dtype=float)
    per = steps.max() / EPOCHS
    out = []
    for e in range(1, EPOCHS + 1):
        m = (steps > (e - 1) * per) & (steps <= e * per)
        if m.any():
            out.append((e, float(vals[m].mean())))
    return out


# The further cells the results section quotes a loss endpoint for. Sections 3.1 and 3.5
# read these numbers off the same per-epoch average as the figure, so they are exported
# here rather than queried by hand. (label, output directory).
EXTRA = [
    ("SeePerSea, random, 80%", "custom_seepersea_models/mstnet_80pct/rand"),
    ("SeePerSea, random, 100%", "custom_seepersea_models/mstnet/rand"),
    ("SeePerSea, in-domain, 10%", "custom_seepersea_models/mstnet_10pct/ssl"),
    ("SeePerSea, in-domain, 20%", "custom_seepersea_models/mstnet_20pct/ssl"),
    ("SeePerSea, in-domain, 50%", "custom_seepersea_models/mstnet_50pct/ssl"),
    ("SeePerSea, automotive, 20%", "custom_seepersea_models/mstnet_20pct/ssl_kitti"),
    ("SeePerSea, automotive, 50%", "custom_seepersea_models/mstnet_50pct/ssl_kitti"),
    ("SeePerSea, combined, 10%", "custom_seepersea_models/mstnet_10pct/ssl_pohang_seepersea"),
    # The two-stage family of E.VI. Section 3.7 quotes these endpoints because the
    # refinement stage at the smallest budget scores far below its own neighbours and the
    # loss is what separates a failed optimization from a property of the measure. Read
    # inside this family only, since the objective carries the RoI terms the one-stage
    # runs do not have.
    ("SeePerSea, refinement, 5%", "custom_seepersea_models/mstnet_stage2_5pct/ssl"),
    ("SeePerSea, refinement, 10%", "custom_seepersea_models/mstnet_stage2_10pct/ssl"),
    ("SeePerSea, refinement, 100%", "custom_seepersea_models/mstnet_stage2/ssl"),
    ("SeePerSea, both, 5%", "custom_seepersea_models/mstnet_zaware_stage2_5pct/ssl"),
    ("SeePerSea, both, 10%", "custom_seepersea_models/mstnet_zaware_stage2_10pct/ssl"),
    ("SeePerSea, both, 100%", "custom_seepersea_models/mstnet_zaware_stage2/ssl"),
]
EXTRA_OUT = ROOT / "results" / "train_loss_extra.csv"


def extras():
    """Return the loss endpoints of the cells that carry no curve of their own."""
    rows = []
    for label, rel in EXTRA:
        c = curve(rel)
        if c is None:
            print(f"  no scalars for {rel}")
            continue
        rows += [dict(run=label, epoch=e, loss=round(v, 4)) for e, v in c]
        print(f"  {label:28s} last {c[-1][1]:6.2f}, lowest {min(v for _, v in c):6.2f}")
    # results/ is not shipped; the reporting scripts create it on first write.
    EXTRA_OUT.parent.mkdir(parents=True, exist_ok=True)
    with EXTRA_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "epoch", "loss"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {EXTRA_OUT}")


def main():
    """Write the two loss CSVs the figure and the results section are built from."""
    rows = []
    for ds, enc, init, rel in RUNS:
        c = curve(rel)
        if c is None:
            print(f"  no scalars for {rel}")
            continue
        rows += [dict(dataset=ds, encoder=enc, init=init, epoch=e, loss=round(v, 4))
                 for e, v in c]
        print(f"  {ds:10s} {enc:8s} {init:11s} {len(c):3d} epochs, "
              f"first {c[0][1]:6.2f}, last {c[-1][1]:6.2f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "encoder", "init", "epoch", "loss"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {OUT}")
    extras()


if __name__ == "__main__":
    main()
