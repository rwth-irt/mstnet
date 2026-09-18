"""Per-class London evaluation — the protocol every reported Thames number uses.

Why this exists: the London dataset config maps all three boat classes onto the
single KITTI class 'Car', so PCDet's built-in eval scores "boat vs background"
(class confusion is impossible) and reports an inflated AP. This script instead
evaluates each class separately, each mapped to 'Car' so every class gets the same
0.7/0.5 IoU thresholds -- KITTI would otherwise hand Pedestrian/Cyclist 0.5/0.25 --
and then averages the per-class APs into the mAP.

Runs on the saved result.pkl predictions, so no re-inference is needed. It still
needs a GPU because KITTI's rotate_iou is a numba.cuda kernel, but that is a
tiny box-IoU workload: the AP values are deterministic and unaffected by other
jobs sharing the GPU. Inference speed and memory are NOT measured here -- those
need an idle GPU and are measured separately by benchmark_speed.py.

Usage: python eval_london_perclass.py [model ...]      (default: all found)
"""
import argparse, copy, pickle, re, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "OpenPCDet"))
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
from pcdet.datasets.kitti import kitti_utils

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "OpenPCDet" / "output" / "custom_london_models"
INFOS = ROOT / "data" / "custom_london" / "custom_infos_val.pkl"
CLASSES = ["small_boat", "medium_boat", "large_boat"]


def filter_annos(annos, cls):
    """Keep only boxes of `cls`, renamed to 'Car' so KITTI applies 0.7/0.5."""
    out = []
    for a in annos:
        a = copy.deepcopy(a)
        # Frames with no detections carry an empty array whose dtype is float64,
        # so a plain `== cls` would collapse to a scalar instead of a mask.
        names = np.asarray(a["name"], dtype=object).reshape(-1)
        mask = (names == cls) if names.size else np.zeros(0, dtype=bool)
        for k, v in list(a.items()):
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == mask.shape[0]:
                a[k] = v[mask]
        a["name"] = np.array(["Car"] * int(mask.sum()))
        out.append(a)
    return out


def parse_result_str(s):
    """Pull the R40 APs out of the printed report.

    ret_dict is unusable: it writes 'Car_3d/moderate_R40' once per IoU setting,
    so the 0.5 pass silently overwrites the 0.7 one. The report keeps both.
    Difficulties are all equal here (no difficulty stratification), so the first
    number on each line is representative.
    """
    res, cur = {}, None
    for line in s.splitlines():
        line = line.strip()
        m40 = re.match(r"Car AP_R40@[\d.]+, [\d.]+, ([\d.]+):", line)
        if m40:
            cur = "0.7" if m40.group(1) == "0.70" else "0.5"
            continue
        # An R11 header ("Car AP@...") follows each R40 block — clear `cur` so its
        # rows are not misfiled under the preceding R40 threshold.
        if re.match(r"Car AP@[\d.]+, [\d.]+, [\d.]+:", line):
            cur = None
            continue
        if cur:
            m = re.match(r"(3d|bev)\s+AP:([\d.]+)", line)
            if m:
                res[f"{m.group(1)}@{cur}"] = float(m.group(2))
    return res


def ap_for_class(det_annos, gt_annos, cls):
    """Average precision of one Thames class, or None if it has no annotations.

    The evaluator is the KITTI one at forty recall positions, applied to the
    detections and annotations of a single class.
    """
    d, g = filter_annos(det_annos, cls), filter_annos(gt_annos, cls)
    n_gt = sum(len(a["name"]) for a in g)
    if n_gt == 0:
        return None, 0
    kitti_utils.transform_annotations_to_kitti_format(d, map_name_to_kitti={"Car": "Car"})
    kitti_utils.transform_annotations_to_kitti_format(g, map_name_to_kitti={"Car": "Car"})
    result_str, _ = kitti_eval.get_official_eval_result(gt_annos=g, dt_annos=d, current_classes=["Car"])
    return parse_result_str(result_str), n_gt


def find_result_pkl(model, tag="full_london"):
    """Locate the saved detections of one run, preferring the test-time evaluation.

    A run directory can hold more than one result, since training also evaluates on
    the training frames. The sort puts those last.
    """
    cands = list((OUT / model / tag / "eval").rglob("result.pkl"))
    cands.sort(key=lambda p: ("eval_with_train" in str(p), str(p)))  # prefer test.py output
    return cands[0] if cands else None


_GT_CACHE = {}


def evaluate_result_pkl(result_pkl):
    """Per-class AP + macro mAP (R40) for one saved London result.pkl.

    Importable entry point for the MSTNet-matrix runner. Returns
    {'per_class': {cls: {'3d@0.7':..}}, 'mAP': {'3d@0.7':..}}.
    """
    import pickle as _pk
    if "gt" not in _GT_CACHE:
        _GT_CACHE["gt"] = [i["annos"] for i in _pk.load(open(INFOS, "rb"))]
    gt = _GT_CACHE["gt"]
    det = _pk.load(open(result_pkl, "rb"))
    if len(det) != len(gt):
        return {"error": f"{len(det)} preds vs {len(gt)} gt"}
    per_class, rows = {}, []
    for c in CLASSES:
        ap, n = ap_for_class(det, gt, c)
        if ap is None:
            continue
        per_class[c] = ap
        rows.append(ap)
    mAP = {k: float(np.mean([r.get(k, np.nan) for r in rows]))
           for k in ("3d@0.7", "3d@0.5", "bev@0.7", "bev@0.5")} if rows else {}
    return {"per_class": per_class, "mAP": mAP}


def main():
    """Evaluate the Thames runs per class, or one saved result file with --pkl."""
    p = argparse.ArgumentParser()
    # --pkl evaluates one saved result.pkl and prints the metrics as one JSON line. The
    # campaign runner used to import evaluate_result_pkl and call it in its own process,
    # which killed the whole run whenever the numba.cuda kernel behind KITTI's rotate_iou
    # aborted; that took down both worker threads and cost three restarts between the
    # 26th and the 31st of August 2026. Run as a subprocess the abort is contained and
    # shows up as a non-zero return code on one job.
    p.add_argument("--pkl", help="evaluate this result.pkl and print JSON on stdout")
    p.add_argument("models", nargs="*")
    args = p.parse_args()
    if args.pkl:
        import json as _json
        print(_json.dumps(evaluate_result_pkl(args.pkl)))
        return

    models = args.models or sorted(d.name for d in OUT.iterdir() if d.is_dir())
    gt_annos_all = [i["annos"] for i in pickle.load(open(INFOS, "rb"))]

    hdr = f"{'model':<16} {'class':<14} {'3D@0.7':>8} {'3D@0.5':>8} {'BEV@0.7':>8} {'BEV@0.5':>8} {'#gt':>7}"
    print(hdr); print("-" * len(hdr))
    for m in models:
        rp = find_result_pkl(m)
        if rp is None:
            print(f"{m:<16} (no result.pkl — not evaluated yet)"); print("-" * len(hdr)); continue
        det = pickle.load(open(rp, "rb"))
        if len(det) != len(gt_annos_all):
            print(f"{m:<16} SKIP: {len(det)} preds vs {len(gt_annos_all)} gt frames"); continue
        rows = []
        for c in CLASSES:
            ap, n_gt = ap_for_class(det, gt_annos_all, c)
            if ap is None:
                print(f"{m:<16} {c:<14} (no GT boxes)"); continue
            rows.append(ap)
            print(f"{m:<16} {c:<14} {ap.get('3d@0.7',float('nan')):>8.2f} {ap.get('3d@0.5',float('nan')):>8.2f} "
                  f"{ap.get('bev@0.7',float('nan')):>8.2f} {ap.get('bev@0.5',float('nan')):>8.2f} {n_gt:>7}")
        if rows:
            mean = lambda k: float(np.mean([r.get(k, np.nan) for r in rows]))
            print(f"{m:<16} {'** mAP **':<14} {mean('3d@0.7'):>8.2f} {mean('3d@0.5'):>8.2f} "
                  f"{mean('bev@0.7'):>8.2f} {mean('bev@0.5'):>8.2f}")
        print("-" * len(hdr))


if __name__ == "__main__":
    main()
