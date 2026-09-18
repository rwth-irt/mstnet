"""Where a detection sits relative to the vessel it found, for the runs of E.VI.

The average precision of Section 3 says how many vessels a configuration finds at an
overlap of 0.5, once on the ground plane and once in three dimensions. It does not say
which box parameter costs the difference between the two. This script measures that
directly. It matches every detection to a ground truth box in the bird's-eye view,
exactly as the KITTI evaluator does, and then reports the error of the matched pair per
parameter and how many of those pairs still reach an overlap of 0.5 in 3D.

Matching follows the evaluator. Detections are taken in order of decreasing score and
each one claims the unclaimed ground truth box it overlaps most in the bird's-eye view,
provided that overlap reaches 0.5. A pair formed this way is a true positive of the
bird's-eye-view metric by construction, so the share of pairs that also reach 0.5 in 3D
is the fraction of correct horizontal detections the 3D criterion discards.

Rotated-rectangle intersection is computed with polygon clipping in numpy rather than
with the CUDA kernels of OpenPCDet, so the script runs while the GPUs are training.

Writes results/localization_error.csv.
"""
import csv
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "OpenPCDet" / "output" / "custom_seepersea_models"
GT = ROOT / "data" / "custom_seepersea" / "custom_infos_val.pkl"
DST = ROOT / "results" / "localization_error.csv"

# (label, config stem, run tag). The label is what the manuscript prints.
RUNS = [
    ("MSTNet, random", "mstnet", "rand"),
    ("MSTNet", "mstnet", "ssl"),
    ("Height-aware encoding", "mstnet_zaware", "ssl"),
    ("Refinement stage", "mstnet_stage2", "ssl"),
    ("Both", "mstnet_zaware_stage2", "ssl"),
    ("Both, vertical only", "mstnet_zaware_stage2_zonly", "ssl"),
]


def corners(box):
    """The four ground-plane corners of one box, counterclockwise."""
    x, y, _, dx, dy, _, ry = box[:7]
    c, s = np.cos(ry), np.sin(ry)
    u = np.array([[dx / 2, dy / 2], [-dx / 2, dy / 2], [-dx / 2, -dy / 2], [dx / 2, -dy / 2]])
    R = np.array([[c, -s], [s, c]])
    return u @ R.T + np.array([x, y])


def clip(subject, clipper):
    """Sutherland-Hodgman. Both polygons are convex and counterclockwise."""
    out = list(subject)
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        e = b - a
        inside = lambda p: e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0]) >= -1e-9
        src, out = out, []
        for j in range(len(src)):
            p, q = src[j - 1], src[j]
            if inside(q):
                if not inside(p):
                    out.append(cut(p, q, a, b))
                out.append(q)
            elif inside(p):
                out.append(cut(p, q, a, b))
        if not out:
            return []
    return out


def cut(p, q, a, b):
    """Intersection of the line through p and q with the line through a and b."""
    d1, d2 = q - p, b - a
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12:
        return q
    t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / den
    return p + t * d1


def area(poly):
    """Area of a polygon given as a list of vertices."""
    if len(poly) < 3:
        return 0.0
    p = np.asarray(poly)
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def ccw(poly):
    """The polygon with its vertices in counter-clockwise order."""
    p = np.asarray(poly)
    s = np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))
    return poly if s >= 0 else poly[::-1]


def ious(a, b):
    """Bird's-eye-view and 3D overlap of two boxes in (x, y, z, dx, dy, dz, ry).
    The z of OpenPCDet is the box center height."""
    pa, pb = ccw(corners(a)), ccw(corners(b))
    inter = area(clip(pa, pb))
    aa, ab = a[3] * a[4], b[3] * b[4]
    bev = inter / max(aa + ab - inter, 1e-9)
    lo = max(a[2] - a[5] / 2, b[2] - b[5] / 2)
    hi = min(a[2] + a[5] / 2, b[2] + b[5] / 2)
    oz = max(hi - lo, 0.0)
    vi = inter * oz
    v3 = a[3] * a[4] * a[5] + b[3] * b[4] * b[5] - vi
    return bev, vi / max(v3, 1e-9)


def ranks(v):
    """Ordinal ranks with ties averaged, which is what Spearman's coefficient uses.

    A plain double argsort breaks ties arbitrarily. That is invisible on the scores,
    which are floats and never tie, but the 3D overlap ties whenever a pair has no
    vertical overlap at all and lands at exactly zero. The randomly initialised run has
    101 such pairs, and breaking those ties by index rather than averaging them shifts
    its coefficient by 0.009.
    """
    v = np.asarray(v, dtype=float)
    order = np.argsort(v)
    out = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        out[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return out


def spearman(x, y):
    """Pearson correlation of the two rank vectors."""
    return np.corrcoef(ranks(x), ranks(y))[0, 1]


def measure(stem, tag, gt):
    """Per-parameter error of one run's matched detections, or None if it has no result.

    Detections are taken in order of decreasing score and each claims the unclaimed
    annotation it overlaps most on the ground plane, provided that overlap reaches
    0.5. Every pair formed this way is a true positive of the bird's-eye metric, so
    the share that also reaches 0.5 in three dimensions is what the stricter criterion
    discards.
    """
    p = OUT / stem / tag / "eval" / "epoch_80" / "val" / "default" / "result.pkl"
    if not p.exists():
        return None
    preds = {d["frame_id"]: d for d in pickle.load(p.open("rb"))}
    dz, dh, dxy, i3, sc, hit = [], [], [], [], [], 0
    ngt = 0
    for info in gt:
        fid = info["point_cloud"]["lidar_idx"]
        g = np.asarray(info["annos"]["gt_boxes_lidar"], dtype=float).reshape(-1, 7)
        ngt += len(g)
        d = preds.get(fid)
        if d is None or len(g) == 0 or len(d["boxes_lidar"]) == 0:
            continue
        boxes = np.asarray(d["boxes_lidar"], dtype=float)[:, :7]
        score = np.asarray(d["score"], dtype=float)
        order = np.argsort(-score)
        taken = set()
        for k in order:
            best, bj = 0.5, -1
            for j in range(len(g)):
                if j in taken:
                    continue
                b, _ = ious(boxes[k], g[j])
                if b >= best:
                    best, bj = b, j
            if bj < 0:
                continue
            taken.add(bj)
            _, i = ious(boxes[k], g[bj])
            dz.append(abs(boxes[k][2] - g[bj][2]))
            dh.append(abs(boxes[k][5] - g[bj][5]))
            dxy.append(float(np.hypot(boxes[k][0] - g[bj][0], boxes[k][1] - g[bj][1])))
            i3.append(i)
            sc.append(float(score[k]))
            hit += i >= 0.5
    n = len(dz)
    if n == 0:
        return None
    # Spearman between the score a detection carries and the overlap it actually
    # reaches in 3D, over the matched pairs. The refinement stage is trained to predict
    # that overlap, so this is the quantity its second output is supposed to raise.
    rho = float(spearman(sc, i3)) if n > 2 else float("nan")
    return dict(matched=n, gt=ngt, recall=100.0 * n / ngt,
                dz=float(np.median(dz)), dz90=float(np.percentile(dz, 90)),
                dh=float(np.median(dh)), dxy=float(np.median(dxy)),
                iou3d=float(np.median(i3)), kept=100.0 * hit / n, rho=rho)


def main():
    """Measure every run of E.VI and write the per-parameter errors to the CSV."""
    gt = pickle.load(GT.open("rb"))
    rows = []
    for label, stem, tag in RUNS:
        m = measure(stem, tag, gt)
        if m is None:
            print(f"  missing {stem}/{tag}")
            continue
        m["configuration"] = label
        rows.append(m)
        print(f"{label:24s} matched {m['matched']:4d}/{m['gt']}  "
              f"dz {m['dz']:.2f} (p90 {m['dz90']:.2f})  dh {m['dh']:.2f}  "
              f"dxy {m['dxy']:.2f}  iou3d {m['iou3d']:.3f}  kept {m['kept']:.1f}%  "
              f"rho {m['rho']:.3f}")
    # results/ is not shipped; the reporting scripts create it on first write.
    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("w", newline="") as f:
        w = csv.DictWriter(f, ["configuration", "matched", "gt", "recall",
                               "dxy", "dz", "dz90", "dh", "iou3d", "kept", "rho"])
        w.writeheader()
        w.writerows(rows)
    print(f"-> {DST}")


if __name__ == "__main__":
    main()
