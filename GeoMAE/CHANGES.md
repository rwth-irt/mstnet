# Changes against upstream GeoMAE

Fork of https://github.com/Tsinghua-MARS-Lab/GeoMAE at commit
`962d6db2fcbd2738342242d3cbc36743995a322a`, which is the repository's only public
commit and the one the 2025-09-16 subtree import records. This copy is stripped of
`build/`, `work_dirs/`, the data tree and every compiled artifact. The list below
was derived by comparing this tree against a fresh clone of that commit, file by
file, not written by hand.

## Modified upstream files

| File | Why |
|---|---|
| `mmdet3d/__init__.py`, `mmdet3d/apis/__init__.py`, `mmdet3d/apis/train.py` | version-check relaxation and deterministic training setup |
| `mmdet3d/core/evaluation/__init__.py` | import surface trimmed to what the pretraining needs |
| `mmdet3d/datasets/__init__.py` | registers the maritime pretraining datasets |
| `mmdet3d/datasets/nuscenes_dataset.py`, `nuscenes_mono_dataset.py`, `nuscenes_ssl_dataset.py`, `nuscenes_ssl_dataset_oppsite.py` | the nuScenes-format reader adapted to the converted maritime corpora |
| `mmdet3d/datasets/pipelines/data_augment_utils.py` | numba and NumPy compatibility |
| `mmdet3d/models/detectors/multi_sub_voxel_dynamic_voxelnet_ssl.py` | the masked pretraining detector, its targets and its loss accounting |
| `mmdet3d/models/voxel_encoders/pillar_encoder.py` | the mean-encoder variant the pretraining uses, so a pretrained encoder transfers into the detector unchanged |
| `mmdet3d/models/__init__.py`, `models/backbones/__init__.py`, `models/builder.py`, `models/detectors/__init__.py`, `models/voxel_encoders/__init__.py` | registration of the added modules |
| `mmdet3d/ops/__init__.py`, `mmdet3d/ops/sst/sst_ops.py`, `mmdet3d/ops/voxel/__init__.py` | torch >= 2 compatibility of the ops surface |
| `requirements/runtime.txt`, `setup.py` | dependency pins matching the `geomae` environment |
| `tools/train.py` | imports `compat_shim` when running under mmcv 2.x |
| `tools/dist_train.sh` | single-node launch without a hardcoded GPU count |

## Added files

- `compat_shim.py` — provides mmcv 1.x-style imports under mmcv 2.x; imported by `tools/train.py`
- `mmdet3d/models/detectors/multi_sub_voxel_dynamic_voxelnet_ssl_zaware.py` — the height-aware pretraining detector
- `mmdet3d/models/voxel_encoders/dynamic_zbinned_vfe.py` — the height-binned dynamic encoder
- `configs/mae_sst/ssl_*.py` — the nine pretraining configurations the paper uses

The nine are one per pretraining corpus (`ssl_london`, `ssl_seepersea`,
`ssl_pohang`, `ssl_kitti`, `ssl_pohang_london`, `ssl_pohang_seepersea`), the
height-aware variant `ssl_seepersea_zaware`, and the two learned-encoder variants
`ssl_seepersea_pfn10` and `ssl_london_pfn10`. The `pfn10` suffix matters: those two
carry ten input features and the layer widths the detector's feature net expects,
and `run_pretraining_all.py` uses them. Earlier `_pfn` variants with nine inputs
existed during development, produced a checkpoint the converter could not map, and
are not part of the study.
