"""Point and annotation statistics of Appendices A.1 and B.3 and Section 2.3.3 of the paper.

A.1  occupied 0.5 m cells of the training splits, the share above the limit of 128 points
     per cell, the expected share of the points inside an annotation that the random
     subsample removes, and the SeePerSea cells beyond y = 100 m (outside the pretraining
     range)
B.3  points per training box, boxes per label budget, the match of the PointPillars box
     priors, median box sizes, vessels per length interval on the SeePerSea validation
     split, and occupied cells and SST windows (12 x 12 cells) per evaluation frame
2.3.3 the best single box prior of a grid search over length and width on SeePerSea

A box prior matches a box if an anchor of PointPillars reaches the matching threshold of
its class. The anchors and the overlap follow OpenPCDet: AnchorGenerator without center
alignment on the feature map of stride 2, rotations 0 and 1.57, and the axis-aligned
bird's-eye-view overlap of boxes3d_nearest_bev_iou, in float32. Only the nine anchor
positions around a box can reach the threshold, so only these are evaluated.

Usage:
  python dataset_statistics.py            # prints all values, writes results/dataset_statistics.json
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFGS = ROOT / "OpenPCDet" / "tools" / "cfgs"
BUDGETS = (5, 10, 20, 50, 80)
CELL, STRIDE, WINDOW, CAP = 0.5, 2, 12, 128


def load(p):
    return pickle.load(open(p, "rb"))


def boxes_of(infos):
    """All annotated boxes of a split and their class names."""
    B = [np.asarray(i["annos"]["gt_boxes_lidar"], dtype=float).reshape(-1, 7) for i in infos]
    N = [np.asarray(i["annos"]["name"]) for i in infos]
    return np.concatenate(B), np.concatenate(N)


def anchors_of(ds):
    """Point cloud range and (class -> anchor size, matching threshold) of PointPillars."""
    c = yaml.safe_load(open(CFGS / f"custom_{ds}_models" / "pointpillar.yaml"))
    gen = c["MODEL"]["DENSE_HEAD"]["ANCHOR_GENERATOR_CONFIG"]
    rng = yaml.safe_load(open(CFGS / "dataset_configs" / f"custom_{ds}.yaml"))["POINT_CLOUD_RANGE"]
    return rng, {a["class_name"]: (a["anchor_sizes"][0][:2], a["matched_threshold"]) for a in gen}


def aligned(x, y, dx, dy, heading):
    """boxes3d_lidar_to_aligned_bev_boxes: dimensions swapped when the heading is nearer 90 degrees."""
    rot = (heading - torch.floor(heading / np.pi + 0.5) * np.pi).abs()
    keep = rot < np.pi / 4
    w, h = torch.where(keep, dx, dy), torch.where(keep, dy, dx)
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def prior_match(boxes, rng, sizes, thr):
    """Share of boxes matched by an anchor, for every prior (length, width) in sizes."""
    nx = round((rng[3] - rng[0]) / CELL / STRIDE)
    ny = round((rng[4] - rng[1]) / CELL / STRIDE)
    sx, sy = (rng[3] - rng[0]) / (nx - 1), (rng[4] - rng[1]) / (ny - 1)
    xs = torch.arange(rng[0], rng[3] + 1e-5, step=sx, dtype=torch.float32)
    ys = torch.arange(rng[1], rng[4] + 1e-5, step=sy, dtype=torch.float32)
    b = torch.tensor(boxes, dtype=torch.float32)
    bx1, by1, bx2, by2 = aligned(b[:, 0], b[:, 1], b[:, 3], b[:, 4], b[:, 6])
    ix = torch.round((b[:, 0] - rng[0]) / sx).long()
    iy = torch.round((b[:, 1] - rng[1]) / sy).long()
    cx = torch.stack([xs[(ix + k).clamp(0, len(xs) - 1)] for k in (-1, 0, 1)], 1)[:, :, None]
    cy = torch.stack([ys[(iy + k).clamp(0, len(ys) - 1)] for k in (-1, 0, 1)], 1)[:, None, :]
    area_b = ((bx2 - bx1) * (by2 - by1))[:, None, None]
    out = []
    for l, w in sizes:
        best = torch.zeros(len(b))
        for rot in (0.0, 1.57):
            a = torch.tensor([l, w, rot], dtype=torch.float32)
            ax1, ay1, ax2, ay2 = aligned(cx, cy, a[0], a[1], a[2])
            xl = (torch.min(ax2, bx2[:, None, None]) - torch.max(ax1, bx1[:, None, None])).clamp_min(0)
            yl = (torch.min(ay2, by2[:, None, None]) - torch.max(ay1, by1[:, None, None])).clamp_min(0)
            inter = xl * yl
            iou = inter / (area_b + (ax2 - ax1) * (ay2 - ay1) - inter).clamp_min(1e-6)
            best = torch.maximum(best, iou.flatten(1).max(1).values)
        out.append(100.0 * float((best >= thr).float().mean()))
    return out


def cells(points, rng):
    """Occupied cells (unique ids) of a point cloud inside the range."""
    p = points[np.all((points[:, :3] >= rng[:3]) & (points[:, :3] < rng[3:]), axis=1)]
    ix = np.floor((p[:, 0] - rng[0]) / CELL).astype(np.int64)
    iy = np.floor((p[:, 1] - rng[1]) / CELL).astype(np.int64)
    return p, ix, iy


def in_box(p, b):
    """Points of p inside the rotated box b."""
    c, s = np.cos(-b[6]), np.sin(-b[6])
    q = p[:, :3] - b[:3]
    x, y = q[:, 0] * c - q[:, 1] * s, q[:, 0] * s + q[:, 1] * c
    return (np.abs(x) <= b[3] / 2) & (np.abs(y) <= b[4] / 2) & (np.abs(q[:, 2]) <= b[5] / 2)


def point_limit(d, infos, rng):
    """Appendix A.1: cells, cells above the point limit, removed in-box points, cells beyond y = 100 m."""
    n = capped = beyond = 0
    inbox = removed = 0.0
    for info in infos:
        pts = np.load(d / "points" / f"{info['point_cloud']['lidar_idx']}.npy")
        p, ix, iy = cells(pts, rng)
        u, inv, cnt = np.unique(ix * 100000 + iy, return_inverse=True, return_counts=True)
        n += len(u)
        capped += int((cnt > CAP).sum())
        beyond += int(((u % 100000) * CELL + rng[1] >= 100).sum())
        frac = np.where(cnt > CAP, (cnt - CAP) / cnt, 0.0)[inv]
        m = np.zeros(len(p), bool)
        for b in info["annos"]["gt_boxes_lidar"]:
            m |= in_box(p, np.asarray(b, dtype=float))
        inbox += m.sum()
        removed += frac[m].sum()
    return dict(cells=n, capped_pct=100 * capped / n, beyond_100m=beyond,
                beyond_100m_pct=100 * beyond / n, removed_inbox_pct=100 * removed / inbox)


def per_frame(d, infos, rng):
    """Median and 90th percentile of the occupied cells and SST windows per evaluation frame."""
    c, w = [], []
    for info in infos:
        _, ix, iy = cells(np.load(d / "points" / f"{info['point_cloud']['lidar_idx']}.npy"), rng)
        u = np.unique(np.stack([ix, iy], 1), axis=0)
        c.append(len(u))
        w.append(len(np.unique(u // WINDOW, axis=0)) if len(u) else 0)
    return dict(cells_median=float(np.median(c)), cells_p90=float(np.percentile(c, 90)),
                windows_median=float(np.median(w)), windows_p90=float(np.percentile(w, 90)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--grid-search", action="store_true", help="best single prior on SeePerSea (slow)")
    a = ap.parse_args()
    out = {}
    for ds in ("london", "seepersea"):
        d = a.data / f"custom_{ds}"
        rng, priors = anchors_of(ds)
        tr, ev = load(d / "custom_infos_train.pkl"), load(d / "custom_infos_val.pkl")
        db = load(d / "custom_dbinfos_train.pkl")
        npts = np.concatenate([[x["num_points_in_gt"] for x in v] for v in db.values()])
        r = dict(points_per_box_median=float(np.median(npts)),
                 boxes_le_10_points_pct=100 * float(np.mean(npts <= 10)),
                 boxes_per_budget={b: sum(len(i["annos"]["name"]) for i in load(d / f"custom_infos_train_{b}pct.pkl"))
                                   for b in BUDGETS} | {100: sum(len(i["annos"]["name"]) for i in tr)})
        B, N = boxes_of(tr)
        r["prior_match_pct"] = {c: prior_match(B[N == c], rng, [s], t)[0] for c, (s, t) in priors.items()}
        E, NE = boxes_of(ev)
        r["eval_median_height_m"] = float(np.median(E[:, 5]))
        r["eval_median_size_m"] = {c: np.median(E[NE == c][:, 3:6], 0).round(2).tolist() for c in sorted(set(NE))}
        if ds == "seepersea":
            L = np.maximum(E[:, 3], E[:, 4])
            r["eval_vessels_by_length"] = [int(((L >= lo) & (L < hi)).sum())
                                           for lo, hi in ((0, 5), (5, 10), (10, 25), (25, np.inf))]
        r["eval_per_frame"] = per_frame(d, ev, rng)
        r["train_point_limit"] = point_limit(d, tr, rng)
        if a.grid_search and ds == "seepersea":
            (s, t), = priors.values()
            grid = [(round(l, 1), round(w, 1)) for l in np.arange(0.5, 20.01, 0.1) for w in np.arange(0.5, l + 1e-9, 0.1)]
            m = prior_match(B, rng, grid, t)
            k = int(np.argmax(m))
            r["grid_search_best_prior"] = dict(length=grid[k][0], width=grid[k][1], match_pct=m[k])
        out[ds] = r
        print(json.dumps({ds: r}, indent=1))
    dst = ROOT / "results" / "dataset_statistics.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dst, "w"), indent=1)
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
