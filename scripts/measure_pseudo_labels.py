"""What the pseudo-label sets of Experiment E.V contain, against the real annotations.

Section 2 describes how the boxes are built, one per DBSCAN cluster with the median size
of the labeled boxes of the budget. This script measures what that produces. It counts
the boxes, matches them to the real training annotations of SeePerSea and reports how
many of them cover a vessel and how many vessels they cover, once under the overlap
criterion of the evaluator and once under a plain distance between centers.

The clustering runs on every training frame regardless of budget, so the six sets differ
only in the size prior. The counts below are therefore identical across budgets and the
box dimensions are not.

Writes nothing; prints the figures the results section quotes.
"""
import pickle
from pathlib import Path

import numpy as np

from measure_localization_error import ious

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "custom_seepersea"
BUDGETS = ["5pct", "10pct", "20pct", "50pct", "80pct", None]


def load(p):
    """Read one info pickle of the SeePerSea dataset."""
    return pickle.load((D / p).open("rb"))


def main():
    """Print the counts and the coverage of the pseudo-label sets against the annotations."""
    print("labeled boxes per budget")
    for b in BUDGETS:
        d = load("custom_infos_train.pkl" if b is None else f"custom_infos_train_{b}.pkl")
        n = sum(len(x["annos"]["name"]) for x in d)
        print(f"  {b or '100pct':7s} {len(d):5d} frames {n:5d} boxes")

    real = {x["point_cloud"]["lidar_idx"]:
            np.asarray(x["annos"]["gt_boxes_lidar"], dtype=float).reshape(-1, 7)
            for x in load("custom_infos_train.pkl")}

    print("\npseudo-label sets")
    for b in BUDGETS:
        ps = load("custom_infos_train_pseudo.pkl" if b is None
                  else f"custom_infos_train_pseudo_{b}.pkl")
        size = None
        npseudo = nreal = tp = near = covn = 0
        for x in ps:
            p = np.asarray(x["annos"]["gt_boxes_lidar"], dtype=float).reshape(-1, 7)
            g = real.get(x["point_cloud"]["lidar_idx"], np.zeros((0, 7)))
            npseudo += len(p)
            nreal += len(g)
            if size is None and len(p):
                size = p[0, 3:6]
            if not len(p) or not len(g):
                continue
            used = set()
            for i in range(len(p)):
                for j in range(len(g)):
                    if j not in used and ious(p[i], g[j])[0] >= 0.5:
                        used.add(j)
                        tp += 1
                        break
            d = np.linalg.norm(p[:, None, :2] - g[None, :, :2], axis=-1)
            near += int((d.min(axis=1) <= 2.0).sum())
            covn += int((d.min(axis=0) <= 2.0).sum())
        print(f"  {b or '100pct':7s} {npseudo:6d} boxes, size prior "
              f"{size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m, "
              f"overlap 0.5: {100*tp/npseudo:4.1f}% of them cover a vessel, "
              f"{100*tp/nreal:4.1f}% of vessels covered; "
              f"within 2 m: {100*near/npseudo:4.1f}% / {100*covn/nreal:4.1f}%")


if __name__ == "__main__":
    main()
