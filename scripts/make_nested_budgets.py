"""Rebuild the SeePerSea annotation budgets as nested subsets.

The Thames budgets are nested, every smaller budget is contained in the next larger one,
and the SeePerSea budgets were drawn independently. Measured on the current files, only
14 of the 148 frames of the five percent budget appear in the ten percent budget. Two
datasets with different budget constructions cannot be plotted on the same label
efficiency curve without a caveat, and the caveat is avoidable.

Construction. Frames are split into those that carry at least one annotated vessel and
those that do not, each group is shuffled once with a fixed seed, and every budget takes
a prefix of both groups in the proportion the full training split has. Taking prefixes
of one fixed permutation is what makes the budgets nested by construction, and keeping
the two groups separate preserves the share of frames that contain an object, which for
SeePerSea is 89.1 percent.

Nothing else about the split changes. The full training split, the validation split and
the frame identities stay as they are.
"""
import argparse
import pickle
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "custom_seepersea"
BUDGETS = [5, 10, 20, 50, 80]
SEED = 20260807


def main():
    """Rebuild the SeePerSea budget lists so each is a subset of the next larger one."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    ids = [l.strip() for l in open(BASE / "ImageSets/train.txt") if l.strip()]
    infos = pickle.load(open(BASE / "custom_infos_train.pkl", "rb"))
    has = {}
    for i in infos:
        a = i.get("annos", {})
        b = np.atleast_2d(a.get("gt_boxes_lidar", np.zeros((0, 7))))
        has[i["point_cloud"]["lidar_idx"]] = b.size > 0

    pos = [i for i in ids if has.get(i, False)]
    neg = [i for i in ids if not has.get(i, False)]
    rng = random.Random(SEED)
    rng.shuffle(pos)
    rng.shuffle(neg)
    print(f"training frames {len(ids)}, with a vessel {len(pos)} "
          f"({100 * len(pos) / len(ids):.1f} %), without {len(neg)}")

    sets = {}
    for p in BUDGETS:
        np_, nn = round(len(pos) * p / 100), round(len(neg) * p / 100)
        sel = pos[:np_] + neg[:nn]
        sets[p] = sel
        print(f"  {p:3d} %  {len(sel):5d} frames, of which {np_} with a vessel "
              f"({100 * np_ / max(len(sel), 1):.1f} %)")

    order = BUDGETS + ["full"]
    sets["full"] = ids
    ok = True
    for a, b in zip(order, order[1:]):
        nested = set(sets[a]) <= set(sets[b])
        ok &= nested
        print(f"  {a} in {b}: {'nested' if nested else 'NOT NESTED'}")
    if not ok:
        raise SystemExit("construction failed, nothing written")

    if not args.apply:
        print("dry run, nothing written")
        return
    for p in BUDGETS:
        out = BASE / f"ImageSets/train_{p}pct.txt"
        out.write_text("\n".join(sorted(sets[p])) + "\n")
        print(f"wrote {out.name}")
    print("done. The info and ground-truth-database files of each budget must now be "
          "regenerated, otherwise they still describe the old draw.")


if __name__ == "__main__":
    main()
