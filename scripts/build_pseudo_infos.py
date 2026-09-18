"""Build custom_infos_train_pseudo_{b}pct.pkl from pseudo_labels_{b}pct/ dirs.

Matches the structure of the existing 5/10pct pseudo infos exactly:
  {'point_cloud': {'num_features': 4, 'lidar_idx': stem},
   'annos': {'name': (N,) <U..>, 'gt_boxes_lidar': (N,7) float32}}
Label line format (from make_pseudo_labels.py): cx cy cz dx dy dz heading Ship
All train.txt frames are included (frames with no cluster get an empty (0,7) array).
"""
import os, pickle, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = f"{ROOT}/data/custom_seepersea"


def build(b):
    """Write the info pickle of the pseudo-label set of one budget.

    The frame list is the full training split at every budget, since the clustering
    does not need annotations. What the budget changes is the size prior the boxes
    were given, so the six sets differ in their box dimensions and not in their
    coverage.
    """
    lab_dir = f"{DS}/pseudo_labels_{b}pct"
    stems = [l.strip() for l in open(f"{DS}/ImageSets/train.txt") if l.strip()]
    infos = []
    nbox = 0
    for st in stems:
        f = f"{lab_dir}/{st}.txt"
        boxes, names = [], []
        if os.path.exists(f):
            for ln in open(f):
                p = ln.split()
                if len(p) >= 8:
                    boxes.append([float(x) for x in p[:7]])
                    names.append(p[7])
        gt = np.array(boxes, dtype=np.float32).reshape(-1, 7)
        nbox += len(gt)
        infos.append({
            'point_cloud': {'num_features': 4, 'lidar_idx': st},
            'annos': {'name': np.array(names), 'gt_boxes_lidar': gt},
        })
    out = f"{DS}/custom_infos_train_pseudo_{b}pct.pkl"
    with open(out, 'wb') as fp:
        pickle.dump(infos, fp)
    print(f"{b}%: {len(infos)} frames, {nbox} pseudo-boxes -> {os.path.basename(out)}")


if __name__ == "__main__":
    for b in (sys.argv[1:] or [20, 50, 80]):
        build(int(b))
