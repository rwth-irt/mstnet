"""Generate pseudo-object labels by clustering, for label-free pretraining.

Motivation: GeoMAE SSL pretrains only the SST encoder (255 tensors). BACKBONE_2D
(~4.8M) and the detection head (~0.4M) start from random, and at a 5% budget they
must be learned from 148 frames. This stage pretrains them too, using labels
obtained without annotation.

What clustering can and cannot give (measured, see report below):
  centres  -> reliable: 100% of GT boxes contain a cluster centre on SeePerSea
  extent   -> unusable: cluster bbox reaches only median IoU 0.21 vs GT, because
              LiDAR sees just the near side of a sparsely sampled hull
So the pseudo-box takes the CENTRE from the cluster and the SIZE from the class
median of the training labels (a dataset-level prior, not per-frame annotation),
which doubles box quality to ~0.42 IoU. Heading comes from PCA of the cluster.

Scope: SeePerSea only. On London 88% of points lie outside any GT box (shoreline
and quay structures are unlabelled); boats merge with that clutter and recall
never exceeds 63% at any eps. That limitation is diagnosed, not worked around --
Ponzini et al. (2025) hit the same issue and needed an explicit 'Quay' class.

Usage: python make_pseudo_labels.py [--eps 5.0] [--min-samples 3] [--report-only]
"""
import argparse, glob, os
from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "data", "custom_seepersea")
OUT = os.path.join(DS, "pseudo_labels")


def class_median_size(label_dir, split_file):
    """Size prior from the training split's labels (dataset-level statistic)."""
    stems = {l.strip() for l in open(split_file) if l.strip()}
    dims = defaultdict(list)
    for stem in stems:
        f = os.path.join(label_dir, f"{stem}.txt")
        if not os.path.exists(f):
            continue
        for ln in open(f):
            p = ln.split()
            if len(p) >= 8:
                dims[p[7]].append((float(p[3]), float(p[4]), float(p[5])))
    return {c: np.median(np.array(v), axis=0) for c, v in dims.items()}


def cluster_boxes(pts, med_size, cls, eps, min_samples, min_pts):
    """One pseudo-box per cluster: cluster centre + class-median size + PCA heading."""
    if len(pts) < min_samples:
        return []
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts[:, :3])
    boxes = []
    for c in set(labels):
        if c == -1:
            continue
        q = pts[labels == c][:, :3]
        if len(q) < min_pts:
            continue
        centre = q.mean(0)
        # heading from the dominant horizontal direction of the cluster
        xy = q[:, :2] - q[:, :2].mean(0)
        if len(q) >= 3:
            v = np.linalg.svd(xy, full_matrices=False)[2][0]
            heading = float(np.arctan2(v[1], v[0]))
        else:
            heading = 0.0
        dx, dy, dz = med_size[cls]
        boxes.append((centre[0], centre[1], centre[2], dx, dy, dz, heading))
    return boxes


def bev_iou(a, b):
    """Overlap of two axis-aligned boxes on the ground plane."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    """Cluster the training frames and write one pseudo-label set per budget."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=5.0)
    ap.add_argument("--min-samples", type=int, default=3)
    ap.add_argument("--min-pts", type=int, default=10, help="drop clusters smaller than this")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only N frames (parameter search)")
    ap.add_argument("--size-split", default="train.txt",
                    help="split whose LABELS may be used for the size prior. At a label "
                         "budget this must be that budget's split (e.g. train_5pct.txt): "
                         "using train.txt would leak sizes from held-out labels "
                         "(median dz 2.05 full vs 1.64 at 5%).")
    ap.add_argument("--out-suffix", default="", help="suffix for the output dir")
    a = ap.parse_args()

    split = os.path.join(DS, "ImageSets", "train.txt")        # frames to cluster (no labels used)
    size_split = os.path.join(DS, "ImageSets", a.size_split)   # labels allowed for the size prior
    med = class_median_size(os.path.join(DS, "labels"), size_split)
    out_dir = OUT + (f"_{a.out_suffix}" if a.out_suffix else "")
    cls = list(med)[0]
    print(f"size prior ({cls}) from {a.size_split}: dx={med[cls][0]:.2f} dy={med[cls][1]:.2f} dz={med[cls][2]:.2f}")

    stems = [l.strip() for l in open(split) if l.strip()]
    if a.limit:
        stems = stems[:a.limit]
    if not a.report_only:
        os.makedirs(out_dir, exist_ok=True)

    n_box = 0; n_frames = 0
    ious, recall_hit, n_gt = [], 0, 0
    for stem in stems:
        pf = os.path.join(DS, "points", f"{stem}.npy")
        if not os.path.exists(pf):
            continue
        pts = np.load(pf)
        boxes = cluster_boxes(pts, med, cls, a.eps, a.min_samples, a.min_pts)
        n_frames += 1; n_box += len(boxes)
        if not a.report_only:
            with open(os.path.join(out_dir, f"{stem}.txt"), "w") as f:
                for b in boxes:
                    f.write(" ".join(f"{v:.4f}" for v in b) + f" {cls}\n")
        # quality against the real labels (diagnostic only -- never used for training)
        lf = os.path.join(DS, "labels", f"{stem}.txt")
        if os.path.exists(lf):
            for ln in open(lf):
                p = ln.split()
                if len(p) < 8:
                    continue
                cx, cy, _, dx, dy, _ = [float(x) for x in p[:6]]
                gt = (cx-dx/2, cy-dy/2, cx+dx/2, cy+dy/2)
                n_gt += 1
                best = 0.0; hit = False
                for b in boxes:
                    pb = (b[0]-b[3]/2, b[1]-b[4]/2, b[0]+b[3]/2, b[1]+b[4]/2)
                    best = max(best, bev_iou(gt, pb))
                    if abs(b[0]-cx) < dx/2 and abs(b[1]-cy) < dy/2:
                        hit = True
                ious.append(best); recall_hit += hit

    ious = np.array(ious)
    print(f"\nframes {n_frames} | pseudo-boxes {n_box} ({n_box/max(n_frames,1):.1f}/frame) "
          f"| real GT boxes {n_gt} ({n_gt/max(n_frames,1):.1f}/frame)")
    print(f"centre recall (GT containing a pseudo-centre): {recall_hit/max(n_gt,1)*100:.0f}%")
    print(f"box BEV-IoU vs GT: median {np.median(ious):.2f} | >0.3 {(ious>0.3).mean()*100:.0f}% "
          f"| >0.5 {(ious>0.5).mean()*100:.0f}%")
    if not a.report_only:
        print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
