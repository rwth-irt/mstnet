#!/usr/bin/env python3
"""
Prepare datasets for GeoMAE SSL pre-training.

Creates NuScenes-style directory structures and SSL info PKL files
for each dataset, bypassing the NuScenes SDK scene-name matching.

Supported datasets:
  - london:    3-dim .bin (x,y,z) -> 4-dim .bin (x,y,z,intensity=1.0)
  - seepersea: ASCII PCD (x,y,z,intensity) -> 4-dim .bin
  - kitti:     4-dim .bin (x,y,z,reflectance), needs NuScenes-SSL PKL

The Pohang corpus is not built here; rebuild_pohang_corpus.py writes it from the
raw archives.

Usage:
  python prepare_ssl_datasets.py london
  python prepare_ssl_datasets.py seepersea
  python prepare_ssl_datasets.py kitti --kitti-root /path/to/kitti
  python prepare_ssl_datasets.py all
"""
import argparse
import os
import pickle
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SSL = ROOT / "data" / "ssl_nuscenes"

import numpy as np
from tqdm import tqdm


def new_token():
    """A fresh identifier, in the format the nuScenes-style reader expects."""
    return uuid.uuid4().hex


def read_pcd_ascii(path):
    """Read ASCII PCD file with FIELDS x y z intensity."""
    header_lines = 0
    num_points = 0
    with open(path, "r") as f:
        for line in f:
            header_lines += 1
            if line.startswith("POINTS"):
                num_points = int(line.strip().split()[-1])
            if line.startswith("DATA"):
                break

    points = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i < header_lines:
                continue
            vals = line.strip().split()
            if len(vals) >= 4:
                points.append([float(v) for v in vals[:4]])
            elif len(vals) == 3:
                points.append([float(v) for v in vals[:3]] + [1.0])

    if not points:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(points, dtype=np.float32)


def create_ssl_pkl(lidar_paths, timestamps, out_pkl_path):
    """Create a NuScenes-style SSL info PKL file.

    Only populates the fields that NuScenesDatasetSSL.get_data_info() uses:
      - lidar_path, token, sweeps, timestamp
    Plus dummy fields for compatibility.
    """
    infos = []
    for lidar_path, ts in zip(lidar_paths, timestamps):
        info = {
            "lidar_path": str(lidar_path),
            "token": new_token(),
            "sweeps": [],
            "cams": dict(),
            "lidar2ego_translation": [0.0, 0.0, 0.0],
            "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
            "ego2global_translation": [0.0, 0.0, 0.0],
            "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
            "timestamp": int(ts),
            "gt_boxes": np.zeros((0, 7), dtype=np.float32),
            "gt_names": np.array([], dtype="<U1"),
            "gt_velocity": np.zeros((0, 2), dtype=np.float32),
            "num_lidar_pts": np.array([], dtype=np.int64),
            "num_radar_pts": np.array([], dtype=np.int64),
            "valid_flag": np.array([], dtype=np.bool_),
        }
        infos.append(info)

    data = {"infos": infos, "metadata": {"version": "v1.0-trainval"}}
    os.makedirs(os.path.dirname(out_pkl_path), exist_ok=True)
    with open(out_pkl_path, "wb") as f:
        pickle.dump(data, f)
    print(f"  Wrote {len(infos)} entries -> {out_pkl_path}")


# ─── London ───────────────────────────────────────────────────────────────────


def prepare_london(
    src_base=str(ROOT / "data" / "raw" / "london_dataset" / "Datasets" / "real_world_temporal_stratified"),
    out_root=str(SSL / "london" / "data" / "nuscenes"),
):
    """Link all London frames (already 4-dim .bin) into NuScenes-SSL format."""
    print("\n=== Preparing London (all frames) ===")

    src_base = Path(src_base)
    out_root = Path(out_root)
    lidar_dir = out_root / "samples" / "LIDAR_TOP"
    lidar_dir.mkdir(parents=True, exist_ok=True)

    # Collect all .bin files from train/val/test
    all_bins = []
    for split in ["training", "validation", "testing"]:
        split_dir = src_base / split / "velodyne"
        if not split_dir.exists():
            print(f"  Warning: {split_dir} not found, skipping")
            continue
        bins = sorted(split_dir.glob("*.bin"))
        print(f"  {split}: {len(bins)} frames")
        all_bins.extend([(split, b) for b in bins])

    print(f"  Total: {len(all_bins)} frames")

    lidar_paths = []
    timestamps = []

    for idx, (split, src_bin) in enumerate(tqdm(all_bins, desc="  Linking London")):
        # London .bin is already 4-dim (x,y,z,intensity)
        out_name = f"{idx:06d}.bin"
        out_path = lidar_dir / out_name
        if not out_path.exists():
            try:
                os.link(str(src_bin), str(out_path))
            except OSError:
                import shutil
                shutil.copy2(str(src_bin), str(out_path))

        lidar_paths.append(str(out_path))
        timestamps.append(idx * 100_000_000)  # synthetic timestamps

    # Create train PKL (all frames for SSL)
    train_pkl = out_root / "nuscenes_ssl_infos_train.pkl"
    create_ssl_pkl(lidar_paths, timestamps, str(train_pkl))

    # Create val PKL (same as train for SSL — no real val needed)
    val_pkl = out_root / "nuscenes_ssl_infos_val.pkl"
    create_ssl_pkl(lidar_paths, timestamps, str(val_pkl))

    print(f"  London SSL data ready at: {out_root}")
    return str(out_root)


# ─── SeePerSea ────────────────────────────────────────────────────────────────


def prepare_seepersea(
    src_base=str(ROOT / "data" / "raw" / "SeePerSea" / "aggregated"),
    out_root=str(SSL / "seepersea" / "data" / "nuscenes"),
):
    """Convert all SeePerSea PCD frames to 4-dim NuScenes-SSL format."""
    print("\n=== Preparing SeePerSea (all frames) ===")

    src_base = Path(src_base)
    out_root = Path(out_root)
    lidar_dir = out_root / "samples" / "LIDAR_TOP"
    lidar_dir.mkdir(parents=True, exist_ok=True)

    points_dir = src_base / "points"
    pcd_files = sorted(points_dir.glob("*.pcd"))
    print(f"  Found {len(pcd_files)} PCD files")

    lidar_paths = []
    timestamps = []

    for i, pcd_file in enumerate(tqdm(pcd_files, desc="  Converting SeePerSea")):
        pts = read_pcd_ascii(str(pcd_file))

        # Write 4-dim .bin
        out_name = f"{i:06d}.bin"
        out_path = lidar_dir / out_name
        pts.astype(np.float32).tofile(str(out_path))

        lidar_paths.append(str(out_path))
        timestamps.append(i * 100_000_000)

    # Create PKLs
    train_pkl = out_root / "nuscenes_ssl_infos_train.pkl"
    create_ssl_pkl(lidar_paths, timestamps, str(train_pkl))

    val_pkl = out_root / "nuscenes_ssl_infos_val.pkl"
    create_ssl_pkl(lidar_paths, timestamps, str(val_pkl))

    print(f"  SeePerSea SSL data ready at: {out_root}")
    return str(out_root)


# ─── Pohang Canal ─────────────────────────────────────────────────────────────


# ─── KITTI ────────────────────────────────────────────────────────────────────


def prepare_kitti(
    kitti_root=None,
    out_root=None,
):
    """Convert KITTI velodyne .bin files to NuScenes-SSL PKL format.

    Both roots default into the repository data tree when not given.

    KITTI .bin files are already 4-dim (x,y,z,reflectance).
    We just need to create the PKL and optionally copy/link files.
    """
    kitti_root = Path(kitti_root) if kitti_root else ROOT / "data" / "raw" / "kitti"
    out_root = Path(out_root) if out_root else SSL / "kitti" / "data" / "nuscenes"
    print("\n=== Preparing KITTI ===")

    kitti_root = Path(kitti_root)
    out_root = Path(out_root)

    velodyne_dir = kitti_root / "training" / "velodyne"
    if not velodyne_dir.exists():
        print(f"  ERROR: KITTI velodyne not found at {velodyne_dir}")
        print("  Please download KITTI 3D Object Detection data from:")
        print("  https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d")
        return None

    lidar_dir = out_root / "samples" / "LIDAR_TOP"
    lidar_dir.mkdir(parents=True, exist_ok=True)

    bin_files = sorted(velodyne_dir.glob("*.bin"))
    print(f"  Found {len(bin_files)} KITTI velodyne frames")

    lidar_paths = []
    timestamps = []

    for i, src_bin in enumerate(tqdm(bin_files, desc="  Linking KITTI")):
        out_path = lidar_dir / src_bin.name
        if not out_path.exists():
            try:
                os.link(str(src_bin), str(out_path))
            except OSError:
                import shutil
                shutil.copy2(str(src_bin), str(out_path))

        lidar_paths.append(str(out_path))
        timestamps.append(i * 100_000_000)

    # Create PKLs
    train_pkl = out_root / "nuscenes_ssl_infos_train.pkl"
    create_ssl_pkl(lidar_paths, timestamps, str(train_pkl))

    val_pkl = out_root / "nuscenes_ssl_infos_val.pkl"
    create_ssl_pkl(lidar_paths, timestamps, str(val_pkl))

    print(f"  KITTI SSL data ready at: {out_root}")
    return str(out_root)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    """Build one pretraining set, or all of them."""
    parser = argparse.ArgumentParser(description="Prepare datasets for GeoMAE SSL pre-training")
    parser.add_argument(
        "dataset",
        choices=["london", "seepersea", "kitti", "all"],
        help="Which dataset to prepare",
    )
    parser.add_argument("--kitti-root", type=str,
                        default=str(ROOT / "data" / "raw" / "kitti"))
    args = parser.parse_args()

    results = {}

    if args.dataset in ("london", "all"):
        results["london"] = prepare_london()

    if args.dataset in ("seepersea", "all"):
        results["seepersea"] = prepare_seepersea()

    if args.dataset in ("kitti", "all"):
        results["kitti"] = prepare_kitti(kitti_root=args.kitti_root)

    print("\n=== Summary ===")
    for name, path in results.items():
        status = "OK" if path else "FAILED/SKIPPED"
        print(f"  {name}: {status} -> {path}")


if __name__ == "__main__":
    main()
