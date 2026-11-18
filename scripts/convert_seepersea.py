"""Convert SeePerSea -> OpenPCDet custom-dataset format.

points/{stem}.npy : float32 [N,4] (x,y,z,intensity), from the aggregated release
labels/{stem}.txt : 'x y z dx dy dz heading Ship' per box (LiDAR frame, box center)
ImageSets/{train,val}.txt

Labels (default --labels 360): the 360-degree point-cloud annotations of the separated
release (<sequence>/annotations/*.json), category 'ship', written as
[x, y, z, dx=width, dy=length, dz=height, heading=rotationYaw]. The KITTI-format labels
of the aggregated release hold an object only inside the camera field of view, while
every detector processes the full scan. aggregated/sequence_file_counts.csv maps each
aggregated frame to its sequence and frame index; every mapping is verified on an
identical point file, or on an identical non-empty KITTI label for the sequences whose
separated clouds are stored organised. A sequence downloaded twice uses its first copy.

--labels camera writes the camera-field-of-view labels of the aggregated release instead
(transformed from the camera to the LiDAR frame with the calibration of each frame).
"""
import argparse, csv, filecmp, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from seepersea_utils import read_pcd, read_kitti_calib, transform_boxes_cam_to_lidar


def camera_boxes(base, stem):
    """Annotations of the aggregated release, transformed from the camera to the LiDAR frame."""
    cam = []
    for line in open(base / "label" / f"{stem}.txt"):
        v = line.split()
        if len(v) < 15 or v[0] != "Car":
            continue
        h, w, l, x, y, z, ry = map(float, v[8:15])
        cam.append([x, y, z, l, w, h, ry])
    if not cam:
        return []
    calib = read_kitti_calib(str(base / "calib" / f"{stem}.txt"))
    return [(x, y, z + h / 2, l, w, h, th) for x, y, z, l, w, h, th in transform_boxes_cam_to_lidar(cam, calib)]


def frame_map(raw):
    """Aggregated frame id -> (separated sequence folder or None, frame index, sequence name)."""
    copies = defaultdict(list)
    for d in sorted((raw / "separated").glob("*/*/")):
        if (d / "annotations").is_dir():
            copies[d.name.replace("_", "-")].append(d)
    amap = {}
    for r in csv.DictReader(open(raw / "aggregated" / "sequence_file_counts.csv")):
        s = r["Sequence"].replace("_", "-")
        for i in range(int(r["Start Count"]), int(r["End Count"]) + 1):
            amap[f"{i:06d}"] = (copies[s][0] if s in copies else None, i - int(r["Start Count"]), s)
    return amap


def verified(base, stem, folder, k):
    """Same frame: identical point file, or identical non-empty KITTI label (organised clouds)."""
    if filecmp.cmp(base / "points" / f"{stem}.pcd", folder / "point_clouds" / f"{k:06d}.pcd", shallow=False):
        return True
    a = (base / "label" / f"{stem}.txt").read_text().strip()
    b = (folder / "kitti_label" / f"{k:06d}.txt").read_text().strip()
    return bool(a) and a == b


def lidar_boxes_360(amap, base, stem):
    """360-degree annotations of the separated release for one aggregated frame."""
    folder, k, seq = amap[stem]
    if folder is None:
        sys.exit(f"frame {stem}: sequence {seq} missing under separated/")
    assert verified(base, stem, folder, k), stem
    jf = folder / "annotations" / f"{k:06d}.json"
    labels = json.load(open(jf))["labels"] if jf.exists() and jf.stat().st_size > 10 else []
    boxes = []
    for lab in labels:
        if lab["category"] != "ship":
            continue
        b = lab["box3d"]
        loc, dim = b["location"], b["dimension"]
        boxes.append((loc["x"], loc["y"], loc["z"], dim["width"], dim["length"], dim["height"],
                      b["orientation"]["rotationYaw"]))
    return boxes


def convert_split(split, base, out, boxes_of, write_points):
    """Convert one SeePerSea split and return the number of frames and boxes kept."""
    stems = [s.strip() for s in open(base / "ImageSets" / f"{split}.txt") if s.strip()]
    kept, nbox = [], 0
    for stem in stems:
        pcd = base / "points" / f"{stem}.pcd"
        if not pcd.exists():
            continue
        if write_points:
            np.save(out / "points" / f"{stem}.npy", read_pcd(str(pcd)).astype(np.float32))
        with open(out / "labels" / f"{stem}.txt", "w") as f:
            for x, y, z, dx, dy, dz, th in boxes_of(stem):
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {dx:.4f} {dy:.4f} {dz:.4f} {th:.6f} Ship\n")
                nbox += 1
        kept.append(stem)
    (out / "ImageSets" / f"{split}.txt").write_text("\n".join(kept) + "\n")
    return len(kept), nbox


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", choices=["360", "camera"], default="360")
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "SeePerSea")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "custom_seepersea")
    ap.add_argument("--labels-only", action="store_true", help="keep existing points/*.npy")
    a = ap.parse_args()
    base = a.raw / "aggregated"
    for sub in ("points", "labels", "ImageSets"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)
    if a.labels == "360":
        amap = frame_map(a.raw)
        boxes_of = lambda stem: lidar_boxes_360(amap, base, stem)
    else:
        boxes_of = lambda stem: camera_boxes(base, stem)
    for split in ("train", "val"):
        n, nb = convert_split(split, base, a.out, boxes_of, not a.labels_only)
        print(f"{split}: {n} frames, {nb} ship boxes ({a.labels} labels)")
    print(f"SeePerSea -> {a.out} DONE")
