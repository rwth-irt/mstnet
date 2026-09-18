"""Build the info pickles of one custom dataset.

The dataset readers do not scan the disk at training time; they read one pickle per
split that lists the frames and their annotations. This is the entry point that
writes those pickles, kept separate from the dataset module so it can run in a plain
interpreter without importing the detector.

Usage:
  python make_infos.py <dataset_config.yaml> <data_dir> <comma-separated class names>
"""
import sys, yaml
from pathlib import Path
from easydict import EasyDict
from pcdet.datasets.custom.custom_dataset import create_custom_infos
cfg_path, data_dir, classes = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
dataset_cfg = EasyDict(yaml.safe_load(open(cfg_path)))
root = Path(data_dir).resolve()
create_custom_infos(dataset_cfg=dataset_cfg, class_names=classes, data_path=root, save_path=root)
print("INFOS DONE")
