"""Convert London/Thames -> OpenPCDet custom-dataset format.

London KITTI label (LiDAR frame): fields ... h w l x y z ry.
Verified mapping (points-in-box on elongated boats):
  OpenPCDet [x, y, z + h/2, dx=w, dy=l, dz=h, heading=-ry]
Classes: small_boat, medium_boat, large_boat.
Splits: train = training/ ; val = testing/ (= MSTNet's London eval split).
Stems are prefixed by split to avoid collisions in the shared points/labels dirs.
"""
import os, glob
import numpy as np

LB = str(ROOT / "data" / "raw" / "london_dataset" / "Datasets" / "real_world_balanced_temporal")
OUT = str(ROOT / "data" / "custom_london")
os.makedirs(f"{OUT}/points", exist_ok=True)
os.makedirs(f"{OUT}/labels", exist_ok=True)
os.makedirs(f"{OUT}/ImageSets", exist_ok=True)
CLASSES = {"small_boat", "medium_boat", "large_boat"}


def convert(split_dir, prefix, out_split):
    """Convert one Thames split into the detector's layout and return the frame count.

    Writes the points and the labels under the output root and the frame list under
    ImageSets, with `prefix` prepended to every stem so the two splits cannot collide.
    """
    kept = []
    for bin_f in sorted(glob.glob(f"{LB}/{split_dir}/velodyne/*.bin")):
        stem = os.path.basename(bin_f)[:-4]
        lf = f"{LB}/{split_dir}/label_2/{stem}.txt"
        if not os.path.exists(lf):
            continue
        out_stem = f"{prefix}{stem}"
        pts = np.fromfile(bin_f, dtype=np.float32).reshape(-1, 4)
        np.save(f"{OUT}/points/{out_stem}.npy", pts)
        with open(f"{OUT}/labels/{out_stem}.txt", "w") as f:
            for line in open(lf):
                v = line.split()
                if not v or v[0] not in CLASSES:
                    continue
                h, w, l, x, y, z, ry = map(float, v[8:15])
                # OpenPCDet: [x, y, z+h/2, dx=w, dy=l, dz=h, heading=-ry]
                f.write(f"{x:.4f} {y:.4f} {z + h/2:.4f} {w:.4f} {l:.4f} {h:.4f} {-ry:.6f} {v[0]}\n")
        kept.append(out_stem)
    with open(f"{OUT}/ImageSets/{out_split}.txt", "w") as f:
        f.write("\n".join(kept) + "\n")
    return len(kept)


if __name__ == "__main__":
    n_tr = convert("training", "train_", "train")
    n_va = convert("testing", "test_", "val")   # MSTNet's London eval split
    print(f"train (training/): {n_tr} frames | val (testing/): {n_va} frames")
    print("London -> custom_london DONE")
