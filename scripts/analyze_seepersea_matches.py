"""Recall by vessel length and by neighbourhood on the SeePerSea validation split.

Sections 2.4.1 and 3.2.1 of the paper resolve the share of matched annotations of every
benchmark entry by the length of the annotated box (below 5 m, 5 to 10 m, 10 to 25 m,
from 25 m), at overlaps of 0.5 and 0.7 in BEV and in 3D. For the vessels below 5 m they
further separate annotations with another annotated vessel within 8 m of the box center
from isolated ones, and report the median distance from an annotation to the nearest
detection center within 2 m. The length is the longer horizontal side of the box.

Matching follows measure_localization_error.py: detections in descending score order claim
the unclaimed annotation with the highest BEV overlap at or above the threshold; a matched
pair counts in 3D if its 3D overlap also reaches the threshold. Repeated cells are averaged
over their runs.

Writes results/recall_by_length.csv and results/small_vessels.csv.
"""
import csv
import pickle
from pathlib import Path

import numpy as np

from measure_localization_error import ious

ROOT = Path(__file__).resolve().parents[1]
GROUP = "custom_seepersea_models"
GT = ROOT / "data" / "custom_seepersea" / "custom_infos_val.pkl"
DST = ROOT / "results"
BINS = ((0, 5), (5, 10), (10, 25), (25, float("inf")))
REPO = {"openpcdet": ROOT / "OpenPCDet", "ted": ROOT / "TED", "lion": ROOT / "LION"}

# (entry, framework, config stem, run tags)
ENTRIES = [(m, "openpcdet", m, ("rand",)) for m in
           ("pointpillar", "second", "centerpoint", "voxel_rcnn", "pv_rcnn", "transfusion",
            "dsvt", "dsvt_voxel")]
ENTRIES += [("voxelnext", "openpcdet", "voxelnext", ("rand", "rand_s2", "rand_s3")),
            ("ted_s", "ted", "ted_s", ("rand",)),
            ("lion_mamba", "lion", "lion_mamba", ("rand",)),
            ("mstnet_random", "openpcdet", "mstnet", ("rand", "rand_s2", "rand_s3")),
            ("mstnet_pretrained", "openpcdet", "mstnet", ("ssl", "ssl_s2", "ssl_s3"))]
# The neighbourhood analysis of Section 3.2.1 compares these four entries.
SMALL = ("transfusion", "centerpoint", "voxelnext", "mstnet_pretrained")


def result(fw, cfg, tag):
    """Predictions of one run on the validation split, keyed by frame."""
    p = REPO[fw] / "output" / GROUP / cfg / tag / "eval" / "epoch_80" / "val" / "default" / "result.pkl"
    return {d["frame_id"]: d for d in pickle.load(p.open("rb"))}


def matches(preds, g, thr, info):
    """Annotation index -> 3D overlap of its match at the BEV threshold thr."""
    d = preds.get(info["point_cloud"]["lidar_idx"])
    if d is None or len(g) == 0 or len(d["boxes_lidar"]) == 0:
        return {}, np.zeros((0, 7))
    boxes = np.asarray(d["boxes_lidar"], dtype=float)[:, :7]
    out = {}
    for k in np.argsort(-np.asarray(d["score"], dtype=float)):
        best, bj, b3 = thr, -1, 0.0
        for j in range(len(g)):
            if j in out:
                continue
            b, i3 = ious(boxes[k], g[j])
            if b >= best:
                best, bj, b3 = b, j, i3
        if bj >= 0:
            out[bj] = b3
    return out, boxes


def length_bins(g):
    """Length interval of every annotation."""
    L = np.maximum(g[:, 3], g[:, 4])
    return np.array([next(k for k, (a, b) in enumerate(BINS) if a <= l < b) for l in L], int)


def by_length(preds, gt, thr):
    """Matched share per length interval in BEV and in 3D, and the annotation counts."""
    n, bev, d3 = np.zeros(len(BINS)), np.zeros(len(BINS)), np.zeros(len(BINS))
    for info in gt:
        g = np.asarray(info["annos"]["gt_boxes_lidar"], dtype=float).reshape(-1, 7)
        bi = length_bins(g)
        np.add.at(n, bi, 1)
        m, _ = matches(preds, g, thr, info)
        for j, i3 in m.items():
            bev[bi[j]] += 1
            d3[bi[j]] += i3 >= thr
    return 100 * bev / n, 100 * d3 / n, n


def small_vessels(preds, gt):
    """Matched share at BEV 0.5 of crowded and isolated vessels below 5 m, and the median
    distance to the nearest detection center within 2 m."""
    hit, tot, off = {True: 0, False: 0}, {True: 0, False: 0}, []
    for info in gt:
        g = np.asarray(info["annos"]["gt_boxes_lidar"], dtype=float).reshape(-1, 7)
        m, boxes = matches(preds, g, 0.5, info)
        for j in range(len(g)):
            if max(g[j, 3], g[j, 4]) >= 5:
                continue
            others = np.delete(g[:, :2], j, 0)
            crowded = bool(len(others)) and np.hypot(*(others - g[j, :2]).T).min() < 8
            tot[crowded] += 1
            hit[crowded] += j in m
            if len(boxes):
                d = np.hypot(*(boxes[:, :2] - g[j, :2]).T).min()
                if d < 2:
                    off.append(d)
    return 100 * hit[True] / tot[True], 100 * hit[False] / tot[False], float(np.median(off)), tot


def main():
    """Measure every entry and write the two CSVs."""
    gt = pickle.load(GT.open("rb"))
    rows, small = [], []
    for name, fw, cfg, tags in ENTRIES:
        preds = [result(fw, cfg, t) for t in tags]
        for thr in (0.5, 0.7):
            r = [by_length(p, gt, thr) for p in preds]
            bev, d3, n = np.mean([x[0] for x in r], 0), np.mean([x[1] for x in r], 0), r[0][2]
            for k, (a, b) in enumerate(BINS):
                rows.append(dict(entry=name, runs=len(tags), overlap=thr, length=f"{a}-{b}",
                                 annotations=int(n[k]), bev=round(bev[k], 1), d3=round(d3[k], 1)))
            print(f"{name:18s} {thr}  BEV {np.round(bev, 1).tolist()}  3D {np.round(d3, 1).tolist()}")
        if name in SMALL:
            s = [small_vessels(p, gt) for p in preds]
            c, i, o = (float(np.mean([x[k] for x in s])) for k in range(3))
            small.append(dict(entry=name, runs=len(tags), crowded=round(c, 1), isolated=round(i, 1),
                              offset_m=round(o, 2), n_crowded=s[0][3][True], n_isolated=s[0][3][False]))
            print(f"{name:18s} below 5 m: crowded {c:.1f} %, isolated {i:.1f} %, offset {o:.2f} m")
    DST.mkdir(parents=True, exist_ok=True)
    for path, data in ((DST / "recall_by_length.csv", rows), (DST / "small_vessels.csv", small)):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print(f"-> {path}")


if __name__ == "__main__":
    main()
