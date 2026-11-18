#!/usr/bin/env python3
"""
Create a temporally-clean, class-balanced split of the London maritime dataset.

Improvement over the original chunked split (create_temporal_stratified_split.py):
- Uses 100-frame chunks (about 6.7 s at the 15 Hz of the recording) instead of 500
  for finer granularity
- Optimizes chunk assignment for class proportion, distance, AND volume balance
- Achieves near-proportional class distribution in all splits

Ten frames are discarded on each side of every boundary between splits, so that
adjacent, nearly identical frames do not fall into different splits. Moored vessels
that remain visible over long periods still appear in several splits (Section 4.3 of
the paper).

Usage:
    python create_balanced_temporal_split.py --dry-run
    python create_balanced_temporal_split.py
"""

import argparse
import os
from pathlib import Path
import shutil
from collections import Counter, defaultdict

import numpy as np

# ──────────────────────────── Configuration ────────────────────────────
_LB = Path(__file__).resolve().parents[1] / "data" / "raw" / "london_dataset" / "Datasets"
SRC_ROOTS = [
    str(_LB / "real_world" / "training"),
    str(_LB / "real_world" / "testing"),
]
DST_ROOT = str(_LB / "real_world_balanced_temporal")

CHUNK_SIZE = 100   # frames per chunk (about 6.7 s at 15 Hz)
GAP_SIZE = 10      # frames excluded at each side of cross-split boundary (about 0.7 s at 15 Hz)
SEED = 42
N_SEARCH_ITER = 2_000_000

CLASSES = ["small_boat", "medium_boat", "large_boat"]
SPLIT_NAMES = ["training", "validation", "testing"]
# ───────────────────────────────────────────────────────────────────────


def collect_frames(src_roots: list[str]) -> list[dict]:
    """Collect all frames with full annotation details, sorted by stem."""
    frames = []
    for src_root in src_roots:
        label_dir = os.path.join(src_root, "label_2")
        velo_dir = os.path.join(src_root, "velodyne")
        if not os.path.isdir(label_dir):
            print(f"  WARNING: {label_dir} not found, skipping")
            continue
        for f in sorted(os.listdir(label_dir)):
            if not f.endswith(".txt"):
                continue
            stem = int(f.replace(".txt", ""))
            velo_path = os.path.join(velo_dir, f.replace(".txt", ".bin"))
            if not os.path.exists(velo_path):
                print(f"  WARNING: missing velodyne for {f}, skipping")
                continue
            objects = []
            with open(os.path.join(label_dir, f)) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) >= 15:
                        cls = parts[0]
                        l, w, h = float(parts[8]), float(parts[9]), float(parts[10])
                        x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
                        dist = np.sqrt(x**2 + y**2 + z**2)
                        objects.append({"cls": cls, "dist": dist, "vol": l * w * h})
            frames.append({
                "stem": stem,
                "label": os.path.join(label_dir, f),
                "velodyne": velo_path,
                "objects": objects,
            })
    frames.sort(key=lambda f: f["stem"])
    return frames


def build_chunks(frames: list[dict], chunk_size: int) -> list[dict]:
    """Divide frames into temporal chunks with per-class statistics."""
    chunks = []
    for i in range(0, len(frames), chunk_size):
        cf = frames[i : i + len(frames[i : i + chunk_size])]
        stats = {}
        for cls in CLASSES:
            objs = [o for f in cf for o in f["objects"] if o["cls"] == cls]
            if objs:
                stats[cls] = {
                    "count": len(objs),
                    "mean_dist": np.mean([o["dist"] for o in objs]),
                    "mean_vol": np.mean([o["vol"] for o in objs]),
                }
            else:
                stats[cls] = {"count": 0, "mean_dist": 0, "mean_vol": 0}
        chunks.append({
            "idx": len(chunks),
            "start": i,
            "end": i + len(cf),
            "n": len(cf),
            "stats": stats,
            "frames": cf,
        })
    return chunks


def weighted_avg(vals, weights):
    """Weighted mean, zero when every weight is zero."""
    mask = weights > 0
    if not mask.any():
        return 0
    return np.average(vals[mask], weights=weights[mask])


def optimize_split(chunks: list[dict], n_iter: int, seed: int) -> tuple:
    """Find optimal val/test chunk assignment via random search."""
    n_chunks = len(chunks)
    total_frames = sum(c["n"] for c in chunks)

    # Pre-compute arrays
    c_f = np.array([c["n"] for c in chunks])
    c_counts = {}
    c_dists = {}
    c_vols = {}
    for cls in CLASSES:
        c_counts[cls] = np.array([c["stats"][cls]["count"] for c in chunks])
        c_dists[cls] = np.array([c["stats"][cls]["mean_dist"] for c in chunks])
        c_vols[cls] = np.array([c["stats"][cls]["mean_vol"] for c in chunks])

    totals = {cls: c_counts[cls].sum() for cls in CLASSES}
    overall_d = {cls: weighted_avg(c_dists[cls], c_counts[cls]) for cls in CLASSES}
    overall_v = {cls: weighted_avg(c_vols[cls], c_counts[cls]) for cls in CLASSES}

    def score(val_idx, test_idx):
        """Penalty of one candidate assignment of chunks to validation and test.

        Lower is better. The penalty grows with the distance of each split's class
        distribution and frame count from the target.
        """
        val_mask = np.zeros(n_chunks, dtype=bool)
        test_mask = np.zeros(n_chunks, dtype=bool)
        val_mask[list(val_idx)] = True
        test_mask[list(test_idx)] = True

        val_f = c_f[val_mask].sum()
        test_f = c_f[test_mask].sum()

        # Must have all 3 classes in val and test
        for mask in [val_mask, test_mask]:
            for cls in CLASSES:
                if c_counts[cls][mask].sum() == 0:
                    return float("inf")

        # 1. Frame ratio penalty
        frame_pen = abs(val_f / total_frames - 0.1) + abs(test_f / total_frames - 0.1)

        # 2. Class proportion penalty (squared for outlier sensitivity)
        cls_pen = 0
        for cls in CLASSES:
            if totals[cls] == 0:
                continue
            for mask, f in [(val_mask, val_f), (test_mask, test_f)]:
                ideal = totals[cls] * f / total_frames
                actual = c_counts[cls][mask].sum()
                cls_pen += (abs(actual - ideal) / totals[cls]) ** 2

        # 3. Distance balance penalty
        dist_pen = 0
        for cls in CLASSES:
            if overall_d[cls] == 0:
                continue
            for mask in [val_mask, test_mask]:
                sel = c_counts[cls][mask]
                if sel.sum() > 0:
                    wd = weighted_avg(c_dists[cls][mask], sel)
                    dist_pen += (abs(wd - overall_d[cls]) / overall_d[cls]) ** 2

        # 4. Volume balance penalty
        vol_pen = 0
        for cls in CLASSES:
            if overall_v[cls] == 0:
                continue
            for mask in [val_mask, test_mask]:
                sel = c_counts[cls][mask]
                if sel.sum() > 0:
                    wv = weighted_avg(c_vols[cls][mask], sel)
                    vol_pen += (abs(wv - overall_v[cls]) / overall_v[cls]) ** 2

        # 5. Gap penalty
        combined = np.zeros(n_chunks, dtype=int)
        combined[val_mask] = 1
        combined[test_mask] = 2
        boundaries = sum(1 for i in range(1, n_chunks) if combined[i] != combined[i - 1])
        gap_pen = boundaries * GAP_SIZE * 2 / total_frames

        return frame_pen * 15 + cls_pen * 10 + dist_pen * 5 + vol_pen * 3 + gap_pen * 2

    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_val = best_test = None

    # Adaptive chunk count range: target ~10% of chunks, ±3
    target_n = max(1, round(n_chunks * 0.1))
    chunk_range = list(range(max(1, target_n - 3), target_n + 4))

    for _ in range(n_iter):
        perm = rng.permutation(n_chunks)
        val_n = rng.choice(chunk_range)
        test_n = rng.choice(chunk_range)

        val_idx = tuple(sorted(perm[:val_n]))
        test_idx = tuple(sorted(perm[val_n : val_n + test_n]))

        # Quick reject
        val_f = c_f[list(val_idx)].sum()
        test_f = c_f[list(test_idx)].sum()
        if val_f / total_frames < 0.08 or val_f / total_frames > 0.13:
            continue
        if test_f / total_frames < 0.08 or test_f / total_frames > 0.13:
            continue

        s = score(val_idx, test_idx)
        if s < best_score:
            best_score = s
            best_val = val_idx
            best_test = test_idx

    return best_val, best_test, best_score


def build_frame_assignment(
    frames: list[dict],
    chunks: list[dict],
    val_chunks: tuple,
    test_chunks: tuple,
    gap_size: int,
) -> list[str | None]:
    """Build per-frame split assignment with gap buffers at boundaries."""
    n = len(frames)
    chunk_split = {}
    for c in chunks:
        if c["idx"] in val_chunks:
            split = "validation"
        elif c["idx"] in test_chunks:
            split = "testing"
        else:
            split = "training"
        chunk_split[c["idx"]] = split

    # Phase 1: assign every frame
    raw = [None] * n
    for c in chunks:
        s = chunk_split[c["idx"]]
        for idx in range(c["start"], c["end"]):
            raw[idx] = s

    # Phase 2: apply gap buffer at cross-split boundaries
    assignment = list(raw)
    for idx in range(1, n):
        if raw[idx] != raw[idx - 1]:
            lo = max(0, idx - gap_size)
            hi = min(n - 1, idx + gap_size - 1)
            for g in range(lo, hi + 1):
                assignment[g] = None

    return assignment


def print_statistics(frames, assignment, total_n):
    """Print detailed split statistics."""
    split_data = {s: [] for s in SPLIT_NAMES}
    n_excluded = 0
    for idx, split in enumerate(assignment):
        if split is None:
            n_excluded += 1
        else:
            split_data[split].append(frames[idx])

    print(f"\n{'=' * 100}")
    print(f"{'Split':<14} {'Frames':>7} {'%':>6} | {'small cnt':>9} {'%':>6} {'dist':>6} {'vol':>7} | "
          f"{'med cnt':>9} {'%':>6} {'dist':>6} {'vol':>7} | {'large cnt':>9} {'%':>6} {'dist':>6} {'vol':>7}")
    print(f"{'=' * 100}")

    totals = {cls: 0 for cls in CLASSES}
    for f in frames:
        for o in f["objects"]:
            totals[o["cls"]] = totals.get(o["cls"], 0) + 1

    for name in SPLIT_NAMES:
        sf = split_data[name]
        n_f = len(sf)

        cls_stats = {}
        for cls in CLASSES:
            objs = [o for f in sf for o in f["objects"] if o["cls"] == cls]
            if objs:
                cls_stats[cls] = {
                    "count": len(objs),
                    "pct": len(objs) / totals[cls] * 100 if totals[cls] > 0 else 0,
                    "dist": np.mean([o["dist"] for o in objs]),
                    "vol": np.mean([o["vol"] for o in objs]),
                }
            else:
                cls_stats[cls] = {"count": 0, "pct": 0, "dist": 0, "vol": 0}

        sm, md, lg = cls_stats["small_boat"], cls_stats["medium_boat"], cls_stats["large_boat"]
        print(f"{name:<14} {n_f:>7} {n_f/total_n*100:>5.1f}% | "
              f"{sm['count']:>9} {sm['pct']:>5.1f}% {sm['dist']:>5.1f}m {sm['vol']:>6.0f} | "
              f"{md['count']:>9} {md['pct']:>5.1f}% {md['dist']:>5.1f}m {md['vol']:>6.0f} | "
              f"{lg['count']:>9} {lg['pct']:>5.1f}% {lg['dist']:>5.1f}m {lg['vol']:>6.0f}")

    print(f"{'─' * 100}")
    print(f"{'excluded':<14} {n_excluded:>7} {n_excluded/total_n*100:>5.1f}%")
    print(f"{'=' * 100}")


def write_split(frames, assignment, dst_root, dry_run):
    """Write the split to disk with re-indexed filenames."""
    split_frames = {s: [] for s in SPLIT_NAMES}
    for idx, split in enumerate(assignment):
        if split is not None:
            split_frames[split].append(frames[idx])

    counts = {}
    for name, sf in split_frames.items():
        label_dst = os.path.join(dst_root, name, "label_2")
        velo_dst = os.path.join(dst_root, name, "velodyne")

        if not dry_run:
            os.makedirs(label_dst, exist_ok=True)
            os.makedirs(velo_dst, exist_ok=True)

        for new_idx, frame in enumerate(sf):
            new_stem = f"{new_idx:05d}"
            if dry_run:
                if new_idx < 3 or new_idx == len(sf) - 1:
                    print(f"  [{name}] stem {frame['stem']:05d} -> {new_stem}")
                elif new_idx == 3:
                    print(f"  ... ({len(sf)} total in {name})")
            else:
                shutil.copy2(frame["label"], os.path.join(label_dst, new_stem + ".txt"))
                shutil.copy2(frame["velodyne"], os.path.join(velo_dst, new_stem + ".bin"))
        counts[name] = len(sf)

    return counts


def main():
    """Build the balanced temporal split of the Thames dataset."""
    parser = argparse.ArgumentParser(
        description="Balanced temporal split of London LiDAR dataset"
    )
    parser.add_argument("--dst", default=DST_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-iter", type=int, default=N_SEARCH_ITER)
    args = parser.parse_args()

    print(f"Chunk size   : {CHUNK_SIZE} frames (~{CHUNK_SIZE / 10:.0f}s at 10Hz)")
    print(f"Gap size     : {GAP_SIZE} frames (~{GAP_SIZE / 10:.1f}s)")
    print(f"Search iters : {args.n_iter:,}")
    print(f"Seed         : {args.seed}")
    print(f"Destination  : {args.dst}")
    print()

    # 1. Collect frames
    print("Collecting frames...")
    frames = collect_frames(SRC_ROOTS)
    n = len(frames)
    print(f"  {n} frames (stems {frames[0]['stem']:05d} - {frames[-1]['stem']:05d})")

    # 2. Build chunks
    chunks = build_chunks(frames, CHUNK_SIZE)
    print(f"  {len(chunks)} temporal chunks of <={CHUNK_SIZE} frames")

    # 3. Optimize split
    print(f"\nSearching for optimal split ({args.n_iter:,} iterations)...")
    val_chunks, test_chunks, score = optimize_split(chunks, args.n_iter, args.seed)
    print(f"  Best score: {score:.6f}")
    print(f"  Val chunks:  {val_chunks}")
    print(f"  Test chunks: {test_chunks}")

    # 4. Build frame assignment
    assignment = build_frame_assignment(frames, chunks, val_chunks, test_chunks, GAP_SIZE)

    # 5. Statistics
    print_statistics(frames, assignment, n)

    # 6. Write
    if args.dry_run:
        print("\nDRY RUN — sample file mappings:")
        write_split(frames, assignment, args.dst, dry_run=True)
        print("\nRe-run without --dry-run to write files.")
    else:
        print(f"\nWriting to {args.dst}...")
        counts = write_split(frames, assignment, args.dst, dry_run=False)
        print("Frames written:")
        for name, c in counts.items():
            print(f"  {name:<14}: {c}")

        # Write metadata
        meta = os.path.join(args.dst, "split_info.txt")
        with open(meta, "w") as f:
            f.write("London maritime LiDAR — balanced temporal split\n")
            f.write(f"Chunk size   : {CHUNK_SIZE} frames\n")
            f.write(f"Gap size     : {GAP_SIZE} frames\n")
            f.write(f"Search iters : {args.n_iter}\n")
            f.write(f"Seed         : {args.seed}\n")
            f.write(f"Score        : {score:.6f}\n")
            f.write(f"Val chunks   : {val_chunks}\n")
            f.write(f"Test chunks  : {test_chunks}\n")
            f.write(f"Total frames : {n}\n\n")
            for name, c in counts.items():
                f.write(f"{name}: {c} frames\n")
            f.write(f"excluded (gap): {n - sum(counts.values())} frames\n")
        print(f"\nMetadata: {meta}")
        print(f"Done. Update pipeline_config.yaml to point to: {args.dst}")


if __name__ == "__main__":
    main()
