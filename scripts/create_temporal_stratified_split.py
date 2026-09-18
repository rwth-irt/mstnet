#!/usr/bin/env python3
"""
Create a temporally-clean, class-stratified split of the London maritime dataset.

PROBLEM WITH NAIVE FRAME-LEVEL STRATIFICATION
──────────────────────────────────────────────
The London dataset is a single continuous LiDAR sequence (~9286 frames at
10 Hz). A random frame-level stratified split produces val frame 33 adjacent
to train frame 32 (0.1 s apart), causing temporal leakage and mAP ≈ 1.0.

TWO AVAILABLE MODES
───────────────────

  --mode sequential  (default)
      The sequence is split into three contiguous temporal blocks
      (train | val | test) with a GAP_SIZE-frame exclusion zone at each
      block boundary.  Guarantees zero temporal leakage.
      Drawback: rare classes (small_boat, large_boat) that cluster in
      the early portion of the recording may be absent from val/test.

  --mode chunked
      The sequence is divided into large temporal chunks of CHUNK_SIZE
      frames (~50 s at 10 Hz, covering >250 m of boat movement).  Chunks
      are stratified by their class composition and randomly assigned to
      train/val/test.  At each boundary where adjacent chunks belong to
      different splits a small GAP_SIZE-frame exclusion zone is applied.
      Better class balance; leakage-free within one chunk period.

RECOMMENDATION
──────────────
Use `--mode chunked --chunk-size 500 --gap-size 20` for the thesis.
Boats travel 250–375 m within one chunk (10 knots over 50 s), so adjacent
cross-split chunks contain no repeated spatial context.

Usage:
    python create_temporal_stratified_split.py --mode chunked --dry-run
    python create_temporal_stratified_split.py --mode sequential --dry-run
    python create_temporal_stratified_split.py --mode chunked --chunk-size 500 --gap-size 20
"""

import argparse
import glob
import os
from pathlib import Path
import shutil
from collections import Counter

import numpy as np

# ──────────────────────────── Configuration ────────────────────────────
_LB = Path(__file__).resolve().parents[1] / "data" / "raw" / "london_dataset" / "Datasets"
SRC_ROOTS = [
    str(_LB / "real_world" / "training"),
    str(_LB / "real_world" / "testing"),
]
DST_ROOT = str(_LB / "real_world_temporal_stratified")

SPLIT_RATIOS = {"training": 0.80, "validation": 0.10, "testing": 0.10}
CHUNK_SIZE   = 500   # frames per atomic chunk for chunked mode (~50 s at 10 Hz)
GAP_SIZE     = 20    # frames excluded at each side of a cross-split boundary
SEED         = 42

CLASSES = ["small_boat", "medium_boat", "large_boat"]
# ───────────────────────────────────────────────────────────────────────


# ─────────────────────────── Frame collection ───────────────────────────

def parse_classes(label_path: str) -> frozenset[str]:
    """Return the set of class names annotated in one label file."""
    classes: set[str] = set()
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                classes.add(parts[0])
    return frozenset(classes)


def collect_all_frames(src_roots: list[str]) -> list[dict]:
    """Collect all frames from src_roots, sorted by integer stem (= temporal order)."""
    frames: list[dict] = []
    for src_root in src_roots:
        label_dir = os.path.join(src_root, "label_2")
        velo_dir  = os.path.join(src_root, "velodyne")
        if not os.path.isdir(label_dir):
            print(f"  WARNING: {label_dir} not found, skipping")
            continue
        for label_path in sorted(glob.glob(os.path.join(label_dir, "*.txt"))):
            stem = os.path.splitext(os.path.basename(label_path))[0]
            velo_path = os.path.join(velo_dir, stem + ".bin")
            if not os.path.exists(velo_path):
                print(f"  WARNING: missing velodyne for {label_path}, skipping")
                continue
            frames.append({
                "stem":     int(stem),
                "stem_str": stem,
                "label":    label_path,
                "velodyne": velo_path,
                "classes":  parse_classes(label_path),
            })
    frames.sort(key=lambda f: f["stem"])
    return frames


# ──────────────────────────── Mode: sequential ──────────────────────────

def build_sequential_assignment(
    frames: list[dict],
    ratios: dict[str, float],
    gap_size: int,
) -> list[str | None]:
    """
    Assign frames to three contiguous temporal blocks with gap buffers.

    Layout (80/10/10, gap=50):
        [0 … 7428] | gap | [7478 … 8306] | gap | [8356 … 9285]
    """
    n = len(frames)
    split_names = list(ratios.keys())
    cumulative  = np.cumsum([ratios[s] for s in split_names])

    boundaries = [0]
    for frac in cumulative[:-1]:
        boundaries.append(int(round(frac * n)))
    boundaries.append(n)

    assignment: list[str | None] = [None] * n
    for i, name in enumerate(split_names):
        lo = boundaries[i]     + (gap_size if i > 0                       else 0)
        hi = boundaries[i + 1] - (gap_size if i < len(split_names) - 1   else 0)
        if lo >= hi:
            print(f"  WARNING: {name} block empty after gap exclusion — reduce gap_size.")
            continue
        for idx in range(lo, hi):
            assignment[idx] = name

    return assignment


# ────────────────────────── Mode: chunked ───────────────────────────────

def build_chunks(frames: list[dict], chunk_size: int) -> list[dict]:
    """Divide the sorted frame list into non-overlapping temporal chunks."""
    chunks = []
    for i in range(0, len(frames), chunk_size):
        chunk_frames = frames[i: i + chunk_size]
        all_classes  = frozenset().union(*(f["classes"] for f in chunk_frames))
        chunks.append({
            "start":   i,
            "end":     i + len(chunk_frames),   # exclusive
            "frames":  chunk_frames,
            "classes": all_classes,
        })
    return chunks


def stratify_chunks(
    chunks: list[dict],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, list[dict]]:
    """
    Assign chunks to splits using stratified sampling on class composition.
    Within each stratum, chunks are sorted by temporal position before
    splitting, so each stratum contributes one contiguous block per split
    (minimising the total number of cross-split boundaries).

    After proportional allocation, a repair pass ensures every split
    contains every class present in the dataset.  If a split is missing a
    class, one chunk containing that class is swapped in from training
    (the largest split).
    """
    rng = np.random.default_rng(seed)
    split_names = list(ratios.keys())
    cumulative  = np.cumsum([ratios[s] for s in split_names])

    strata: dict[frozenset, list[dict]] = {}
    for chunk in chunks:
        strata.setdefault(chunk["classes"], []).append(chunk)

    result: dict[str, list[dict]] = {s: [] for s in split_names}

    for key in sorted(strata, key=lambda k: sorted(k)):
        group = sorted(strata[key], key=lambda c: c["start"])   # temporal order
        rng.shuffle(group)   # shuffle within stratum for balanced assignment
        n = len(group)
        boundaries = [0]
        for c in cumulative[:-1]:
            boundaries.append(int(round(c * n)))
        boundaries.append(n)
        for i, name in enumerate(split_names):
            result[name].extend(group[boundaries[i]: boundaries[i + 1]])

    # ── Repair pass: ensure every class appears in every split ──
    all_classes: set = set()
    for chunk in chunks:
        all_classes |= chunk["classes"]

    donor = split_names[0]  # training (largest split) donates chunks

    for target in split_names[1:]:
        target_classes: set = set()
        for chunk in result[target]:
            target_classes |= chunk["classes"]

        missing = all_classes - target_classes
        if not missing:
            continue

        for cls in sorted(missing):
            # Find a donor chunk that contains the missing class
            for i, chunk in enumerate(result[donor]):
                if cls in chunk["classes"]:
                    moved = result[donor].pop(i)
                    result[target].append(moved)
                    print(f"  Repair: moved chunk (frames {moved['start']}-{moved['end']-1}) "
                          f"from {donor} to {target} for class '{cls}'")
                    break

    return result


def build_chunked_assignment(
    frames: list[dict],
    chunk_splits: dict[str, list[dict]],
    gap_size: int,
) -> list[str | None]:
    """
    Build frame-level assignment from chunk assignments.
    At each boundary where adjacent chunks belong to different splits,
    exclude gap_size frames from both sides.
    """
    n = len(frames)
    # Phase 1: assign every frame the split of its chunk
    raw: list[str | None] = [None] * n
    for name, chunks in chunk_splits.items():
        for chunk in chunks:
            for idx in range(chunk["start"], chunk["end"]):
                raw[idx] = name

    # Phase 2: apply gap buffer at cross-split boundaries
    assignment: list[str | None] = list(raw)
    for idx in range(1, n):
        if raw[idx] != raw[idx - 1]:          # boundary found
            lo = max(0,     idx - gap_size)
            hi = min(n - 1, idx + gap_size - 1)
            for g in range(lo, hi + 1):
                assignment[g] = None

    return assignment


# ──────────────────────────── Shared utilities ──────────────────────────

def verify_temporal_gap(
    frames: list[dict],
    assignment: list[str | None],
    gap_size: int,
) -> None:
    """Check that no evaluation frame lies within the gap around a training frame.

    This is what makes the split temporally clean: neighbouring frames of one
    recording are near-duplicates, so an evaluation frame next to a training frame
    would be leakage.
    """
    train_idx = {i for i, s in enumerate(assignment) if s == "training"}
    others    = [(i, s) for i, s in enumerate(assignment)
                 if s is not None and s != "training"]

    min_gap = gap_size + 1
    worst   = None
    for idx, split in others:
        for offset in range(1, gap_size + 5):
            if (idx - offset) in train_idx:
                if offset < min_gap:
                    min_gap, worst = offset, (idx - offset, idx, split)
                break
            if (idx + offset) in train_idx:
                if offset < min_gap:
                    min_gap, worst = offset, (idx + offset, idx, split)
                break

    print("\nTemporal gap verification:")
    if worst is None:
        print(f"  OK — no training frame within {gap_size} frames of val/test.")
    else:
        ti, oi, sp = worst
        print(f"  Closest train–{sp} pair: frame indices {ti} and {oi}")
        print(f"  Gap: {min_gap} frames  (~{min_gap/10:.1f} s at 10 Hz)")
        print(f"  {'OK' if min_gap >= gap_size else 'WARNING — below gap_size'}")


def print_statistics(
    frames: list[dict],
    assignment: list[str | None],
    all_classes: list[str],
    n_total: int,
) -> None:
    """Print the frame and class counts of every split."""
    split_frames: dict[str, list[dict]] = {s: [] for s in SPLIT_RATIOS}
    n_excluded = 0
    for idx, split in enumerate(assignment):
        if split is None:
            n_excluded += 1
        else:
            split_frames[split].append(frames[idx])

    col_w = 15
    header = f"{'Split':<14}  {'Frames':>7}"
    for c in all_classes:
        header += f"  {c:>{col_w}}"
    header += f"  {'total_obj':>10}"
    sep = "=" * (len(header) + 20)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for name in SPLIT_RATIOS:
        sf = split_frames[name]
        obj: Counter  = Counter()
        frm: Counter  = Counter()
        for frame in sf:
            with open(frame["label"]) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and parts[0] in all_classes:
                        obj[parts[0]] += 1
            for c in frame["classes"]:
                if c in all_classes:
                    frm[c] += 1

        row = f"{name:<14}  {len(sf):>7}"
        for c in all_classes:
            pct = f"({frm[c]/len(sf)*100:.0f}%)" if sf else ""
            row += f"  {obj[c]:>7} {pct:>6}"
        row += f"  {sum(obj.values()):>10}"
        print(row)

    print(f"{'─' * len(sep)}")
    print(f"Excluded (gap):       {n_excluded:>7}  ({n_excluded/n_total*100:.1f}% of {n_total} total)")
    print(f"{sep}\n")


def create_output_dirs(dst_root: str, split_names: list[str]) -> None:
    """Create the label and velodyne directories of every split."""
    for name in split_names:
        os.makedirs(os.path.join(dst_root, name, "label_2"),  exist_ok=True)
        os.makedirs(os.path.join(dst_root, name, "velodyne"), exist_ok=True)


def write_split(
    frames: list[dict],
    assignment: list[str | None],
    split_names: list[str],
    dst_root: str,
    dry_run: bool,
) -> dict[str, int]:
    """Copy the frames of every split into the output tree."""
    split_frames: dict[str, list[dict]] = {s: [] for s in split_names}
    for idx, split in enumerate(assignment):
        if split is not None:
            split_frames[split].append(frames[idx])

    counts: dict[str, int] = {}
    for name, sf in split_frames.items():
        label_dst = os.path.join(dst_root, name, "label_2")
        velo_dst  = os.path.join(dst_root, name, "velodyne")
        for new_idx, frame in enumerate(sf):
            new_stem = f"{new_idx:05d}"
            if dry_run:
                if new_idx < 3 or new_idx == len(sf) - 1:
                    print(f"  [{name}] stem {frame['stem']:05d} → {new_stem}")
                elif new_idx == 3:
                    print(f"  ... ({len(sf)} total in {name})")
            else:
                shutil.copy2(frame["label"],
                             os.path.join(label_dst, new_stem + ".txt"))
                shutil.copy2(frame["velodyne"],
                             os.path.join(velo_dst,  new_stem + ".bin"))
        counts[name] = len(sf)
    return counts


# ────────────────────────────── Main ────────────────────────────────────

def main() -> None:
    """Build the temporally stratified split of the Thames dataset."""
    parser = argparse.ArgumentParser(
        description="Temporally-clean split of London LiDAR dataset"
    )
    parser.add_argument(
        "--mode",
        choices=["sequential", "chunked"],
        default="chunked",
        help="Split mode: 'sequential' (contiguous blocks) or "
             "'chunked' (stratified temporal chunks).  Default: chunked",
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"Frames per chunk for chunked mode (default: {CHUNK_SIZE})")
    parser.add_argument("--gap-size",   type=int, default=GAP_SIZE,
                        help=f"Exclusion frames at cross-split boundaries (default: {GAP_SIZE})")
    parser.add_argument("--seed",       type=int, default=SEED)
    parser.add_argument("--dst",        default=DST_ROOT)
    parser.add_argument("--dry-run",    action="store_true")
    args = parser.parse_args()

    print(f"Mode         : {args.mode}")
    if args.mode == "chunked":
        print(f"Chunk size   : {args.chunk_size} frames  "
              f"(~{args.chunk_size/10:.0f} s · boats drift "
              f"~{args.chunk_size/10 * 5:.0f}–{args.chunk_size/10 * 7:.0f} m at 10–14 kn)")
    print(f"Gap size     : {args.gap_size} frames  (~{args.gap_size/10:.1f} s at 10 Hz)")
    print(f"Split ratios : {SPLIT_RATIOS}")
    print(f"Seed         : {args.seed}")
    print(f"Destination  : {args.dst}")
    print()

    # 1. Collect frames
    print("Collecting frames...")
    frames = collect_all_frames(SRC_ROOTS)
    n = len(frames)
    print(f"  {n} frames  (stems {frames[0]['stem']:05d} – {frames[-1]['stem']:05d})")

    # 2. Build assignment
    if args.mode == "sequential":
        assignment = build_sequential_assignment(frames, SPLIT_RATIOS, args.gap_size)

    else:  # chunked
        chunks = build_chunks(frames, args.chunk_size)
        print(f"  {len(chunks)} temporal chunks of ≤{args.chunk_size} frames")

        strata_counts: Counter = Counter()
        for c in chunks:
            strata_counts[c["classes"]] += 1
        print("\nChunk class-composition groups:")
        for key, cnt in sorted(strata_counts.items(), key=lambda x: -x[1]):
            print(f"  {sorted(key)}: {cnt} chunks")

        chunk_splits = stratify_chunks(chunks, SPLIT_RATIOS, args.seed)
        print("\nChunks per split (before gap exclusion):")
        for name, cs in chunk_splits.items():
            total = sum(len(c["frames"]) for c in cs)
            print(f"  {name:<12}: {len(cs):>3} chunks  ({total} raw frames)")

        assignment = build_chunked_assignment(frames, chunk_splits, args.gap_size)

    # 3. Summary counts
    counts_preview: Counter = Counter(s if s else "excluded" for s in assignment)
    print(f"\nFrame counts after gap exclusion:")
    for s in [*SPLIT_RATIOS.keys(), "excluded"]:
        c = counts_preview[s]
        print(f"  {s:<12}: {c:>5}  ({c/n*100:.1f}%)")

    # 4. Verify temporal gap
    verify_temporal_gap(frames, assignment, args.gap_size)

    # 5. Class statistics
    print_statistics(frames, assignment, CLASSES, n)

    # 6. Write or dry-run
    if args.dry_run:
        print("DRY RUN — sample file mappings:")
        write_split(frames, assignment, list(SPLIT_RATIOS.keys()), args.dst, dry_run=True)
        print("\nRe-run without --dry-run to write files.")
    else:
        print(f"Creating directories under {args.dst}...")
        create_output_dirs(args.dst, list(SPLIT_RATIOS.keys()))
        written = write_split(
            frames, assignment, list(SPLIT_RATIOS.keys()), args.dst, dry_run=False
        )
        print("Frames written:")
        for name, c in written.items():
            print(f"  {name:<12}: {c}")

        meta = os.path.join(args.dst, "split_info.txt")
        with open(meta, "w") as f:
            f.write(f"London maritime LiDAR — temporally-clean {args.mode} split\n")
            f.write(f"Mode         : {args.mode}\n")
            if args.mode == "chunked":
                f.write(f"Chunk size   : {args.chunk_size} frames\n")
            f.write(f"Gap size     : {args.gap_size} frames\n")
            f.write(f"Split ratios : {SPLIT_RATIOS}\n")
            f.write(f"Seed         : {args.seed}\n")
            f.write(f"Total frames : {n}\n\n")
            for name, c in written.items():
                f.write(f"{name}: {c} frames\n")
            f.write(f"excluded (gap): {n - sum(written.values())} frames\n")
        print(f"\nMetadata: {meta}")
        print(f"Done.  Update pipeline_config.yaml to point to: {args.dst}")


if __name__ == "__main__":
    main()
