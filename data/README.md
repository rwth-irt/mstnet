# Data tree

Nothing in this directory ships with the repository except the split index files
under `custom_*/ImageSets/`. Everything else is built locally from the public
dataset releases. The full tree needs roughly 80 GB. If the repository disk is
small, put the tree elsewhere and link it:

```bash
mv data /path/to/large/disk/mstnet_data
ln -s /path/to/large/disk/mstnet_data data
```

The four forks reach this tree through relative symlinks, `OpenPCDet/data`,
`TED/data`, `LION/data` and `GeoMAE/data`, so a clone works wherever it is placed.

## Layout

```
data/
  raw/                      the untouched public releases, the inputs
    london_dataset/         Thames, in the layout Datasets/real_world*/...
    SeePerSea/aggregated/   the aggregated SeePerSea release (points, label, calib, ImageSets)
    PohangCanal_raw/        the archives, fetched by scripts/download_pohang_raw.py
    kitti/training/velodyne the KITTI 3D object detection scans
  custom_london/            detection dataset, written by scripts/convert_london.py
  custom_seepersea/         detection dataset, written by scripts/convert_seepersea.py
  ssl_nuscenes/             the six pretraining corpora
    london/  seepersea/  pohang/  kitti/  pohang_london/  pohang_seepersea/
```

The two combined corpora are trees of symlinks over the single-source ones, so they
cost no additional storage.

## Where to get the raw datasets

| dataset | used for | reference |
|---|---|---|
| Thames | supervised training and evaluation | Xie et al., *Reliable LiDAR-based ship detection and tracking for Autonomous Surface Vehicles*, Ocean Engineering, 2024, doi 10.1016/j.oceaneng.2024.119288 |
| SeePerSea | supervised training and evaluation | Jeong et al., *SeePerSea: Multimodal Perception Dataset of In-Water Objects for Autonomous Surface Vehicles*, IEEE T-FR, 2025, doi 10.1109/TFR.2025.3602937 |
| Pohang Canal | unlabeled pretraining | Chung et al., *Pohang canal dataset*, IJRR, 2023, doi 10.1177/02783649231191145 |
| KITTI | unlabeled pretraining, automotive control | Geiger et al., *Are we ready for autonomous driving?*, CVPR, 2012, doi 10.1109/CVPR.2012.6248074 |

Each dataset is distributed by its own authors under their own terms, and this
repository redistributes none of it. Obtain each release from its authors and
observe the terms they attach to it. Of SeePerSea we use the aggregated release;
of Pohang Canal the forward-facing Ouster OS1-64 of sequences 00 and 01; of KITTI
the point clouds of the 3D object detection release and none of its annotations.

## What the conversion produces

`convert_london.py` and `convert_seepersea.py` write the detection datasets: one
point cloud and one label file per frame, plus the frame lists. The Thames
conversion reads the balanced temporal layout that
`create_balanced_temporal_split.py` builds, not the released partition; the reason
is in the paper and in `custom_london/ImageSets/README.md`.

`prepare_ssl_datasets.py`, `rebuild_pohang_corpus.py` and
`build_combined_ssl_sets.py` write the pretraining corpora in the nuScenes-style
layout the GeoMAE code reads. These carry point clouds only, never annotations.

`renormalise_intensity.py` brings every corpus onto one intensity scale. Run it
without `--apply` first; it then reports the scales it finds and changes nothing.
