"""Convert SeePerSea -> OpenPCDet custom-dataset format.

points/{stem}.npy : float32 [N,4] (x,y,z,intensity)
labels/{stem}.txt : 'x y z dx dy dz heading Ship' per box (LiDAR frame, center z)
ImageSets/{train,val}.txt

Boxes: src cam->lidar (transform_boxes_cam_to_lidar) gives [x,y,z_bottom,l,w,h,theta];
mapped to OpenPCDet [x, y, z_bottom+h/2, dx=l, dy=w, dz=h, heading=theta]
(z-center verified via points-in-box; SeePerSea boxes near-square so orientation low-impact).
Uses the exact src loaders so GT matches MSTNet.
"""
import os, sys
import numpy as np

REPO = str(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))
from seepersea_utils import read_pcd, read_kitti_calib, transform_boxes_cam_to_lidar

BASE = str(ROOT / "data" / "raw" / "SeePerSea" / "aggregated")
OUT = str(ROOT / "data" / "custom_seepersea")
os.makedirs(f"{OUT}/points", exist_ok=True)
os.makedirs(f"{OUT}/labels", exist_ok=True)
os.makedirs(f"{OUT}/ImageSets", exist_ok=True)


def lidar_boxes(stem):
    """Read the annotations of one frame and return them in the LiDAR frame.

    The release stores boxes in the camera frame, so each one is transformed with the
    calibration of its own frame.
    """
    cam = []
    for line in open(f"{BASE}/label/{stem}.txt"):
        v = line.split()
        if len(v) < 15 or v[0] != "Car":
            continue
        h, w, l, x, y, z, ry = map(float, v[8:15])
        cam.append([x, y, z, l, w, h, ry])
    if not cam:
        return []
    calib = read_kitti_calib(f"{BASE}/calib/{stem}.txt")
    return transform_boxes_cam_to_lidar(cam, calib)  # [x,y,z_bot,l,w,h,theta]


def convert_split(split):
    """Convert one SeePerSea split and return the number of frames kept."""
    stems = [s.strip() for s in open(f"{BASE}/ImageSets/{split}.txt") if s.strip()]
    kept = []
    for stem in stems:
        pcd = f"{BASE}/points/{stem}.pcd"
        if not os.path.exists(pcd):
            continue
        pts = read_pcd(pcd).astype(np.float32)          # [N,4] x,y,z,intensity
        np.save(f"{OUT}/points/{stem}.npy", pts)
        with open(f"{OUT}/labels/{stem}.txt", "w") as f:
            for b in lidar_boxes(stem):
                x, y, z, l, w, h, th = b
                f.write(f"{x:.4f} {y:.4f} {z + h/2:.4f} {l:.4f} {w:.4f} {h:.4f} {th:.6f} Ship\n")
        kept.append(stem)
    with open(f"{OUT}/ImageSets/{split}.txt", "w") as f:
        f.write("\n".join(kept) + "\n")
    return len(kept)


if __name__ == "__main__":
    for split in ("train", "val"):
        n = convert_split(split)
        print(f"{split}: {n} frames converted")
    print("SeePerSea -> custom_seepersea DONE")
