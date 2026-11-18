"""Write the SeePerSea pretraining infos without the validation frames.

Reads data/ssl_nuscenes/seepersea/data/nuscenes/nuscenes_ssl_infos_train.pkl, removes
every frame of data/custom_seepersea/ImageSets/val.txt, and writes
nuscenes_ssl_infos_train_noval.pkl beside it for GeoMAE/configs/mae_sst/ssl_seepersea_noval.py
(control of Appendix C of the paper: 540 of the 8111 frames are validation frames, 7571 remain).
The frame id is the file name of each lidar_path.
"""
import argparse
import os
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ssl", type=Path, default=ROOT / "data" / "ssl_nuscenes" / "seepersea" / "data" / "nuscenes")
    ap.add_argument("--val", type=Path, default=ROOT / "data" / "custom_seepersea" / "ImageSets" / "val.txt")
    a = ap.parse_args()
    val = {s.strip() for s in open(a.val) if s.strip()}
    src = pickle.load(open(a.ssl / "nuscenes_ssl_infos_train.pkl", "rb"))
    keep = [i for i in src["infos"] if os.path.splitext(os.path.basename(i["lidar_path"]))[0] not in val]
    out = a.ssl / "nuscenes_ssl_infos_train_noval.pkl"
    pickle.dump({"infos": keep, "metadata": src["metadata"]}, open(out, "wb"))
    print(f"{len(src['infos'])} frames, {len(src['infos']) - len(keep)} validation frames removed, "
          f"{len(keep)} written to {out}")
