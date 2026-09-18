"""Rebuild the Pohang pretraining set from the raw release, with the record layout the
files actually have.

The defect this repairs. The corpus in use was written with 114688 points of 8 float32
per frame. A raw file holds 3670016 bytes, which divides by both 32 and 28, so the wrong
record size produces a file of exactly the expected length and nothing complains. The
correct size is 28 bytes, i.e. 7 float32, giving 131072 points, which is 64 beams times
the 2048 azimuth steps the dataset paper states. Three independent checks confirm it:

  * 131072 = 64 x 2048 exactly, and 114688 = 64 x 1792 matches nothing in the sensor.
  * Read as (64, 2048, 7), the median elevation per row falls monotonically from
    +17.5 to -12.1 degrees, which is the beam pattern of a 64-beam Ouster. The
    8-column reading gives no such structure.
  * Along a row the azimuth sweeps continuously from +151 to -150 degrees.

Under the 8-column reading every point mixes fields of neighbouring points, so the
geometry of the previous corpus was scrambled rather than merely carrying a wrong
intensity channel.

What this script writes. Columns 0 to 2 are x, y, z and column 3 is intensity; columns
4 to 6 are identically zero in the release and are dropped. Points with no return are
stored by the sensor at the origin and are removed, because they would otherwise pile
tens of thousands of identical points into the single cell containing the origin in
every frame. The output keeps the four-column layout and the directory structure of the
previous set, so no path or config outside this script changes.

Reads straight from the archives, so no intermediate extraction of 80 GB is needed.
"""
import argparse
import pickle
import zipfile
from pathlib import Path

import numpy as np

RAW = ROOT / "data" / "raw" / "PohangCanal_raw"
ARCHIVES = ["pohang00_lidar_lidar_front.zip", "pohang01_lidar_lidar_front.zip"]
RECORD = 7          # float32 per point in the raw release
KEEP = 4            # x, y, z, intensity


def convert(out_root, limit=None):
    """Convert the Pohang archives into the pretraining layout and return the frame list."""
    samples = out_root / "samples" / "LIDAR_TOP"
    samples.mkdir(parents=True, exist_ok=True)
    written, dropped_frames = [], 0
    kept_pts = raw_pts = 0

    for arc in ARCHIVES:
        with zipfile.ZipFile(RAW / arc) as z:
            names = sorted(n for n in z.namelist() if n.endswith(".bin"))
            if limit:
                names = names[:limit]
            print(f"{arc}: {len(names)} frames", flush=True)
            for k, n in enumerate(names):
                buf = z.read(n)
                a = np.frombuffer(buf, dtype=np.float32)
                if a.size % RECORD:
                    dropped_frames += 1
                    continue
                b = a.reshape(-1, RECORD)[:, :KEEP]
                r = np.linalg.norm(b[:, :3], axis=1)
                b = np.ascontiguousarray(b[r > 0.01])
                raw_pts += len(r)
                kept_pts += len(b)
                if len(b) == 0:
                    dropped_frames += 1
                    continue
                stem = Path(n).stem
                b.astype(np.float32).tofile(samples / f"{stem}.bin")
                written.append(str((samples / f"{stem}.bin").resolve()))
                if (k + 1) % 5000 == 0:
                    print(f"   {k + 1} done", flush=True)

    infos = [{"lidar_path": p, "token": Path(p).stem, "sweeps": [], "cams": {},
              "lidar2ego_translation": [0.0, 0.0, 0.0],
              "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
              "ego2global_translation": [0.0, 0.0, 0.0],
              "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
              "timestamp": int(Path(p).stem) if Path(p).stem.isdigit() else 0,
              "gt_boxes": np.zeros((0, 7), dtype=np.float32),
              "gt_names": np.array([], dtype="<U1"),
              "gt_velocity": np.zeros((0, 2), dtype=np.float32)}
             for p in written]
    for name in ("nuscenes_ssl_infos_train.pkl", "nuscenes_ssl_infos_val.pkl"):
        with (out_root / name).open("wb") as f:
            pickle.dump({"infos": infos, "metadata": {"version": "pohang_front_00_01"}}, f)

    print(f"\nframes written {len(written)}, skipped {dropped_frames}")
    print(f"points kept {100 * kept_pts / max(raw_pts, 1):.1f} % "
          f"({kept_pts} of {raw_pts}); the remainder are beams without a return")


def main():
    """Build the Pohang pretraining set from the raw archives."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ssl_nuscenes/pohang/data/nuscenes")
    ap.add_argument("--limit", type=int, default=None, help="frames per archive, for a smoke test")
    args = ap.parse_args()
    convert(Path(args.out).resolve(), args.limit)


if __name__ == "__main__":
    main()
