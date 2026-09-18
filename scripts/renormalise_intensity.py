"""Put the intensity channel of every dataset on one scale, in both the pretraining and
the finetuning path.

Why. The two conversion scripts normalised differently. The detection converter divided
SeePerSea by 255 and clipped, which is not a property of that sensor and set 12.8
percent of its points to exactly 1.0. The self-supervised converter wrote the same
recordings raw. The encoder was therefore pretrained on a median intensity near 53 and
finetuned on 0.2, a factor of roughly 120, and a control run showed that mismatch alone
costs about 5 average precision in three dimensions.

The rule. Divide by the 99th percentile of that dataset's own training split, then clip
to the unit interval. Per-dataset because each sensor encodes reflectance differently,
from the training split because no test statistic may enter the pipeline, and the 99th
percentile rather than the maximum because these distributions have long tails: for
SeePerSea the 99th percentile is 821 while the maximum is 7256.

Divisors, measured on the respective training splits:

    SeePerSea   821      p50 53,  p90 291,  max 7256
    Thames      100      p50 17,            max  255
    Pohang      433      p50 86,            max 2300
    KITTI         0.62   p50 0.29,          max    1.0

Afterwards every dataset has its 99th percentile at 1.0 and the two paths of each
dataset agree.

The combined pretraining sets are symlink farms over the individual ones, so they are
skipped here and follow automatically. Skipping them is not cosmetic; scaling through a
symlink would apply the divisor twice.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# The self-supervised sets. Values are the divisors above.
SSL = {"seepersea": 821.0, "london": 100.0, "pohang": 433.0, "kitti": 0.62}
# The detection sets. Thames only needs scaling; SeePerSea has to be converted again
# from the source recordings because the old clipping destroyed information that
# rescaling cannot restore.
DETECT_SCALE = {"custom_london": 100.0}


def rescale_bin(path, div):
    """Divide the intensity column of one binary frame, returning whether it was written."""
    a = np.fromfile(path, dtype=np.float32)
    if a.size % 4:
        return False
    b = a.reshape(-1, 4)
    b[:, 3] = np.clip(b[:, 3] / div, 0.0, 1.0)
    b.astype(np.float32).tofile(path)
    return True


def rescale_npy(path, div):
    """Divide the intensity column of one numpy frame and clip it to the unit range."""
    a = np.load(path)
    a[:, 3] = np.clip(a[:, 3] / div, 0.0, 1.0)
    np.save(path, a)
    return True


def do_ssl(apply):
    """Bring the pretraining corpora onto one intensity scale."""
    for name, div in SSL.items():
        allf = sorted(glob.glob(str(ROOT / f"data/ssl_nuscenes/{name}/**/*.bin"),
                                recursive=True))
        real = [f for f in allf if not os.path.islink(f)]
        links = [f for f in allf if os.path.islink(f)]

        # A link that leaves this set points at an original dataset, which must not be
        # written to. KITTI is stored that way. Those entries are materialised as real
        # scaled files instead. A link that stays inside data/ssl_nuscenes belongs to a
        # combined set and is left alone, since its target is scaled in its own pass.
        outside = [f for f in links
                   if "ssl_nuscenes" not in os.path.realpath(f)]
        print(f"SSL {name:10s} divisor {div:8.2f}   {len(real):6d} own files, "
              f"{len(outside):6d} to materialise, {len(links) - len(outside):6d} internal links",
              flush=True)
        if not apply:
            continue
        for k, f in enumerate(real):
            rescale_bin(f, div)
            if (k + 1) % 10000 == 0:
                print(f"   {k + 1}", flush=True)
        for k, f in enumerate(outside):
            src = os.path.realpath(f)
            a = np.fromfile(src, dtype=np.float32)
            if a.size % 4:
                continue
            b = a.reshape(-1, 4)
            b[:, 3] = np.clip(b[:, 3] / div, 0.0, 1.0)
            os.remove(f)
            b.astype(np.float32).tofile(f)
            if (k + 1) % 5000 == 0:
                print(f"   materialised {k + 1}", flush=True)


def do_detect_scale(apply):
    """Bring the detection datasets onto the same intensity scale."""
    for name, div in DETECT_SCALE.items():
        files = sorted(glob.glob(str(ROOT / f"data/{name}/points/*.npy")))
        print(f"detect {name:18s} {len(files):6d} files, divisor {div}", flush=True)
        if not apply:
            continue
        for k, f in enumerate(files):
            rescale_npy(f, div)
            if (k + 1) % 5000 == 0:
                print(f"   {k + 1}", flush=True)


def do_seepersea_reconvert(apply):
    """Rewrite the point files from the source recordings. read_pcd now applies the new
    divisor, so this also removes the clipping of the old conversion."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from seepersea_utils import read_pcd, SEEPERSEA_INTENSITY_P99
    src = ROOT / "data" / "raw" / "SeePerSea" / "aggregated" / "points"
    out = ROOT / "data/custom_seepersea/points"
    stems = [p.stem for p in sorted(out.glob("*.npy"))]
    print(f"detect custom_seepersea    {len(stems):6d} files, divisor "
          f"{SEEPERSEA_INTENSITY_P99}, reconverted from source", flush=True)
    if not apply:
        return
    missing = 0
    for k, s in enumerate(stems):
        p = src / f"{s}.pcd"
        if not p.exists():
            missing += 1
            continue
        np.save(out / f"{s}.npy", read_pcd(str(p)).astype(np.float32))
        if (k + 1) % 500 == 0:
            print(f"   {k + 1}", flush=True)
    if missing:
        print(f"   WARNING {missing} source files missing", flush=True)


def main():
    """Report the intensity scales, and rewrite them with --apply."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="without this only reports")
    args = ap.parse_args()
    do_ssl(args.apply)
    do_detect_scale(args.apply)
    do_seepersea_reconvert(args.apply)
    print("done" if args.apply else "dry run, nothing written")


if __name__ == "__main__":
    main()
