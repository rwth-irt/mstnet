"""Per-epoch training loss of the runs behind the learning rate of MSTNet.

Section 2.4.2 of the paper sets the peak learning rate of MSTNet to 0.001 instead of the
0.003 of CenterPoint, because the lower rate yields a lower final training loss on
SeePerSea at 5 % and 100 % of the labels. This script reads those runs, MSTNet without
pretraining at both rates, out of the TensorBoard events. The framework writes one scalar
per iteration into `output/<group>/<cfg>/<tag>/tensorboard/`; the values are averaged per
epoch and written to one CSV. A resumed run leaves several event files, so all of them are
read and the (step, value) pairs deduplicated before averaging.

Usage:
  python export_train_loss.py            # writes results/train_loss.csv
"""
import csv
import glob

from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "train_loss.csv"
EPOCHS = 80

# (learning rate, label budget in percent, run, output directory below OpenPCDet/output).
RUNS = [(0.001, b, r, f"custom_seepersea_models/{cfg}/rand{sfx}")
        for b, cfg in ((5, "mstnet_5pct"), (100, "mstnet"))
        for r, sfx in ((1, ""), (2, "_s2"), (3, "_s3"))]
RUNS += [(0.003, b, 1, f"custom_seepersea_models/{cfg}/rand_lr3e-3")
         for b, cfg in ((5, "mstnet_5pct"), (100, "mstnet"))]


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


def main():
    """Write the per-epoch loss of every run and print its final value."""
    rows = []
    for lr, budget, run, rel in RUNS:
        c = curve(rel)
        if c is None:
            print(f"  no scalars for {rel}")
            continue
        rows += [dict(lr=lr, budget_pct=budget, run=run, epoch=e, loss=round(v, 4)) for e, v in c]
        print(f"  lr {lr:<6} {budget:3d} %  run {run}  final loss {c[-1][1]:6.2f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lr", "budget_pct", "run", "epoch", "loss"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
