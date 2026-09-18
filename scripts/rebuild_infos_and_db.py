"""Regenerate the info files and the ground-truth databases after the renormalisation
and the new nested budgets.

Two things went stale and would otherwise be mixed with the new data.

The ground-truth databases hold the points cut out of each annotated object, including
their intensity, and were written before the renormalisation. Leaving them in place
would make the sampling augmentation paste objects on the old scale into frames on the
new one, which is exactly the kind of silent inconsistency this whole exercise is
meant to remove. Both datasets are affected.

The SeePerSea budget info files still describe the old independent draws. The frame
lists were rewritten as nested subsets, so the info files no longer match them.

The full train and val infos are also rewritten, because the boxes are unchanged but
the point files behind them were replaced.
"""
import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPD = ROOT / "OpenPCDet"
sys.path.insert(0, str(OPD))

import yaml
from easydict import EasyDict

from pcdet.datasets.custom.custom_dataset import CustomDataset
from pcdet.utils import common_utils

DATASETS = {
    "seepersea": {"cfg": "tools/cfgs/dataset_configs/custom_seepersea.yaml",
                  "classes": ["Ship"],
                  "splits": ["train", "val", "train_5pct", "train_10pct",
                             "train_20pct", "train_50pct", "train_80pct"]},
    "london": {"cfg": "tools/cfgs/dataset_configs/custom_london.yaml",
               "classes": ["small_boat", "medium_boat", "large_boat"],
               # Thames budgets were already nested and are not being redrawn, but
               # their databases still carry the old intensity scale.
               "splits": ["train", "val", "train_5pct", "train_10pct",
                          "train_20pct", "train_50pct", "train_80pct"]},
}


def build(name, spec, apply, workers=8):
    """Rebuild the info pickles and the sampling database of one dataset.

    The ground-truth sampling database has to be rebuilt whenever the budget lists
    change, because a database built on the full training split would put annotated
    objects of frames outside the budget into a run that is not supposed to see
    them.
    """
    cfg = EasyDict(yaml.safe_load(open(OPD / spec["cfg"])))
    root = OPD / "data" / f"custom_{name}"
    logger = common_utils.create_logger()
    n_feat = len(cfg.POINT_FEATURE_ENCODING.src_feature_list)

    for split in spec["splits"]:
        ds = CustomDataset(dataset_cfg=cfg, class_names=spec["classes"],
                           root_path=root, training=False, logger=logger)
        ds.set_split(split)
        ids = len(ds.sample_id_list)
        info_path = root / f"custom_infos_{split}.pkl"
        print(f"{name}/{split}: {ids} frames -> {info_path.name}", flush=True)
        if not apply:
            continue
        infos = ds.get_infos(spec["classes"], num_workers=workers, has_label=True,
                             num_features=n_feat)
        with open(info_path, "wb") as f:
            pickle.dump(infos, f)

        if split == "val":
            continue        # the sampling database is only ever built from training data
        ds.set_split(split)
        ds.create_groundtruth_database(info_path, split=split)
        print(f"   database custom_dbinfos_{split}.pkl written", flush=True)


def main():
    """Rebuild the info pickles and sampling databases of both datasets."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--only", default=None, help="seepersea or london")
    args = ap.parse_args()
    for name, spec in DATASETS.items():
        if args.only and name != args.only:
            continue
        build(name, spec, args.apply)
    print("done" if args.apply else "dry run, nothing written")


if __name__ == "__main__":
    main()
