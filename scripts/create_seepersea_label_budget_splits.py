#!/usr/bin/env python3
"""
Create label-budget subsets of the SeePerSea training split.

The original dataset structure is kept intact.  Only additional ImageSets
files are written next to the existing train.txt / val.txt / test.txt:

    ImageSets/
        train.txt          ← original, never modified
        val.txt            ← original, never modified
        test.txt           ← original, never modified
        train_5pct.txt     ← 5 % of training frames (stratified)
        train_10pct.txt
        train_20pct.txt
        train_50pct.txt
        train_80pct.txt

Stratification is done at the *frame* level based on the set of classes
present in each frame (after applying label_map).  This keeps the class
distribution consistent across all budget levels.

Usage:
    python create_seepersea_label_budget_splits.py [--dry-run]
"""

import argparse
import os
from collections import Counter
from pathlib import Path

import numpy as np

# ──────────────────────────── Configuration ────────────────────────────
DATASET_ROOT = str(Path(__file__).resolve().parents[1] / "data" / "raw" / "SeePerSea" / "aggregated")
IMAGESETS_DIR = os.path.join(DATASET_ROOT, "ImageSets")
LABEL_DIR = os.path.join(DATASET_ROOT, "label")
POINTS_DIR = os.path.join(DATASET_ROOT, "points")

# Maps raw annotation class names to canonical class names.
# Matches the label_map in pipeline_config.yaml.
LABEL_MAP = {
    "Car": "Ship",
    "Pedestrian": "Buoy",
}

LABEL_BUDGETS = [0.05, 0.10, 0.20, 0.50, 0.80]
SEED = 42
# ───────────────────────────────────────────────────────────────────────


def read_image_set(path: str) -> list[str]:
    """Return list of frame IDs from an ImageSets text file."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def get_classes_in_frame(frame_id: str) -> frozenset[str]:
    """Return the set of canonical class names present in one label file."""
    label_path = os.path.join(LABEL_DIR, frame_id + ".txt")
    classes: set[str] = set()
    if not os.path.exists(label_path):
        return frozenset()
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            raw_class = parts[0]
            canonical = LABEL_MAP.get(raw_class, raw_class)
            classes.add(canonical)
    return frozenset(classes)


def verify_points_exist(frame_ids: list[str]) -> list[str]:
    """Warn about frame IDs whose point cloud file is missing; return valid ones."""
    valid = []
    for fid in frame_ids:
        pcd_path = os.path.join(POINTS_DIR, fid + ".pcd")
        if not os.path.exists(pcd_path):
            print(f"  WARNING: missing point cloud for frame {fid}, skipping")
        else:
            valid.append(fid)
    return valid


def stratified_subset(
    frames: list[dict],
    fraction: float,
    rng: np.random.Generator,
) -> list[str]:
    """
    Select *fraction* of frames via stratified sampling.

    Stratification key: the frozenset of canonical class names per frame.
    Returns a list of frame IDs.
    """
    # Group by class-set
    strata: dict[frozenset, list[str]] = {}
    for frame in frames:
        strata.setdefault(frame["classes"], []).append(frame["id"])

    selected: list[str] = []
    for key in sorted(strata, key=lambda k: sorted(k)):
        group = list(strata[key])
        rng.shuffle(group)
        n_select = max(1, round(fraction * len(group)))
        selected.extend(group[:n_select])

    rng.shuffle(selected)
    return selected


def print_statistics(
    label: str,
    frame_ids: list[str],
    all_classes: list[str],
) -> None:
    """Print the class counts and the annotated share of one budget."""
    counts: Counter = Counter()
    n_frames = len(frame_ids)
    for fid in frame_ids:
        label_path = os.path.join(LABEL_DIR, fid + ".txt")
        if not os.path.exists(label_path):
            continue
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    raw = parts[0]
                    canonical = LABEL_MAP.get(raw, raw)
                    if canonical in all_classes:
                        counts[canonical] += 1

    parts_str = "  ".join(f"{c}: {counts[c]}" for c in all_classes)
    total = sum(counts.values())
    print(f"  {label:<20} {n_frames:>5} frames  |  {parts_str}  |  total objects: {total}")


def main():
    """Build the nested label-budget lists of the SeePerSea training split."""
    parser = argparse.ArgumentParser(
        description="Create label-budget ImageSets for SeePerSea"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without writing any files",
    )
    parser.add_argument("--root", default=DATASET_ROOT, help="SeePerSea aggregated root")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    imagesets_dir = os.path.join(args.root, "ImageSets")
    label_dir = os.path.join(args.root, "label")
    points_dir = os.path.join(args.root, "points")

    # Override globals if --root was changed
    global LABEL_DIR, POINTS_DIR
    LABEL_DIR = label_dir
    POINTS_DIR = points_dir

    train_txt = os.path.join(imagesets_dir, "train.txt")
    print(f"Reading training split from: {train_txt}")
    train_ids = read_image_set(train_txt)
    print(f"  {len(train_ids)} frame IDs found in train.txt")

    # Verify point clouds exist
    print("Verifying point clouds...")
    train_ids = verify_points_exist(train_ids)
    print(f"  {len(train_ids)} valid frames after verification")

    # Build frame metadata
    print("Reading label files for stratification...")
    frames = [{"id": fid, "classes": get_classes_in_frame(fid)} for fid in train_ids]

    # Report stratification groups
    strata_counts: Counter = Counter()
    for f in frames:
        strata_counts[f["classes"]] += 1
    print("\nStratification groups (canonical class combinations):")
    for key, cnt in sorted(strata_counts.items(), key=lambda x: -x[1]):
        label_str = sorted(key) if key else ["(empty)"]
        print(f"  {label_str}: {cnt} frames")

    all_classes = sorted({LABEL_MAP.get(c, c) for frame in frames for c in frame["classes"]})
    if not all_classes:
        all_classes = list(set(LABEL_MAP.values()))

    print("\nClass distribution in full training set:")
    print_statistics("train (100%)", train_ids, all_classes)

    # Create subsets
    rng = np.random.default_rng(args.seed)
    results: dict[str, list[str]] = {}
    for budget in LABEL_BUDGETS:
        subset = stratified_subset(frames, budget, rng)
        key = f"{int(budget * 100)}pct"
        results[key] = subset

    print("\nLabel budget subsets:")
    for key, subset_ids in results.items():
        print_statistics(f"train_{key}", subset_ids, all_classes)

    # Write output files
    if args.dry_run:
        print("\nDRY RUN — no files written.")
        print("Would write:")
        for key in results:
            out_path = os.path.join(imagesets_dir, f"train_{key}.txt")
            print(f"  {out_path}  ({len(results[key])} frames)")
    else:
        print(f"\nWriting ImageSets files to {imagesets_dir} ...")
        for key, subset_ids in results.items():
            out_path = os.path.join(imagesets_dir, f"train_{key}.txt")
            with open(out_path, "w") as f:
                for fid in subset_ids:
                    f.write(fid + "\n")
            print(f"  Wrote {out_path}  ({len(subset_ids)} frames)")

        print("\nDone. Original train.txt / val.txt / test.txt are unchanged.")
        print("To use a budget, set training_image_sets_file in pipeline_config.yaml, e.g.:")
        print('  training_image_sets_file: ".../ImageSets/train_10pct.txt"')


if __name__ == "__main__":
    main()
