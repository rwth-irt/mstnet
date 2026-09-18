"""Build the two combined pretraining sets out of the three single-source ones.

Experiment E.IV pretrains on Pohang Canal together with each target domain, so two
further corpora are needed beyond the single-source ones. Neither copies any point
cloud. The frame list of a combined set is the concatenation of the two info lists it
joins, and its LIDAR_TOP directory is a tree of symlinks into the two sources, each
name prefixed with the source it came from so frames of the two cannot collide.

The three single-source sets have to exist first, written by prepare_ssl_datasets.py
(london, seepersea) and rebuild_pohang_corpus.py (pohang).

Usage:
  python scripts/build_combined_ssl_sets.py
"""
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SSL = ROOT / "data" / "ssl_nuscenes"
NUSCENES_SUBDIR = "data/nuscenes"

SRC_POHANG = SSL / "pohang"
SRC_LONDON = SSL / "london"
SRC_SEEPERSEA = SSL / "seepersea"
DST_POHANG_LONDON = SSL / "pohang_london"
DST_POHANG_SEEPERSEA = SSL / "pohang_seepersea"


def log(msg):
    """Print one timestamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_pkl(pkl_path: Path):
    """Read one pickle."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def save_pkl(data, pkl_path: Path):
    """Write one pickle, creating the parent directory."""
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    log(f"  wrote {pkl_path} ({len(data.get('infos', []))} infos)")


def load_source(src: Path, label: str):
    """Read the info list of one pretraining set, or exit if it has not been built.

    `label` names the set in the error message, so a missing input says which of the
    build steps of data/README.md was skipped.
    """
    pkl_path = src / NUSCENES_SUBDIR / "nuscenes_ssl_infos_train.pkl"
    if not pkl_path.exists():
        sys.exit(f"{pkl_path} not found, build the {label} set first (see data/README.md)")
    data = load_pkl(pkl_path)
    infos = data.get("infos", data if isinstance(data, list) else [])
    log(f"  {label}: {len(infos)} frames")
    return infos


def link_frames(sources, dst: Path, label: str):
    """Symlink the point clouds of several sets into one LIDAR_TOP directory.

    `sources` is a list of (path, prefix) pairs. The prefix is prepended to every link
    name, so two sets that number their frames the same way stay distinguishable.
    """
    dst_lidar = dst / NUSCENES_SUBDIR / "samples" / "LIDAR_TOP"
    dst_lidar.mkdir(parents=True, exist_ok=True)
    existing = set(os.listdir(dst_lidar))
    total = 0
    for src_path, prefix in sources:
        src_lidar = src_path / NUSCENES_SUBDIR / "samples" / "LIDAR_TOP"
        if not src_lidar.exists():
            sys.exit(f"{src_lidar} not found")
        created = 0
        for fname in sorted(os.listdir(src_lidar)):
            link_name = f"{prefix}_{fname}" if prefix else fname
            if link_name in existing:
                continue
            try:
                os.symlink(src_lidar / fname, dst_lidar / link_name)
                created += 1
            except FileExistsError:
                pass
        total += created
        log(f"    {prefix}: {created} new links")
    log(f"  {label}: {total} links in {dst_lidar}")


def combine(infos_a, label_a, infos_b, label_b, dst: Path, label: str):
    """Write the info list of one combined set as the concatenation of two."""
    infos = list(infos_a) + list(infos_b)
    log(f"  {label}: {label_a} ({len(infos_a)}) + {label_b} ({len(infos_b)}) = {len(infos)}")
    save_pkl({"infos": infos, "metadata": {"version": "v1.0-trainval"}},
             dst / NUSCENES_SUBDIR / "nuscenes_ssl_infos_train.pkl")


def main():
    """Build pohang_london and pohang_seepersea from the three single-source sets."""
    log("reading the single-source sets")
    pohang = load_source(SRC_POHANG, "pohang")
    london = load_source(SRC_LONDON, "london")
    seepersea = load_source(SRC_SEEPERSEA, "seepersea")

    log("linking the point clouds")
    link_frames([(SRC_POHANG, "pohang"), (SRC_LONDON, "london")],
                DST_POHANG_LONDON, "pohang_london")
    link_frames([(SRC_POHANG, "pohang"), (SRC_SEEPERSEA, "seepersea")],
                DST_POHANG_SEEPERSEA, "pohang_seepersea")

    log("writing the combined frame lists")
    combine(pohang, "pohang", london, "london", DST_POHANG_LONDON, "pohang_london")
    combine(pohang, "pohang", seepersea, "seepersea", DST_POHANG_SEEPERSEA, "pohang_seepersea")
    log("done")


if __name__ == "__main__":
    main()
