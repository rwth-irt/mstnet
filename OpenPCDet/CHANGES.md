# Changes against upstream OpenPCDet

Fork of https://github.com/open-mmlab/OpenPCDet at commit `233f849`.
This copy is stripped of `.git`, `build/`, `output/`, the data tree and every
compiled artifact. Everything else is upstream code or one of the changes below.
The list was derived from the fork's own git status against that commit, not
written by hand.

## Modified upstream files

| File | Why |
|---|---|
| `pcdet/datasets/__init__.py` | registers the two maritime datasets |
| `pcdet/datasets/custom/custom_dataset.py` | the maritime dataset reader, its evaluation protocol and the per-budget splits |
| `pcdet/datasets/dataset.py` | collate support for the front-view stream |
| `pcdet/datasets/augmentor/database_sampler.py` | budget-aware ground-truth sampling, so a run never samples an object out of a frame outside its budget |
| `pcdet/datasets/processor/data_processor.py` | front-view voxelization, per-cell point cap, deterministic shuffle |
| `pcdet/models/backbones_2d/__init__.py` | registers the height-profile fusion module |
| `pcdet/models/backbones_2d/base_bev_backbone.py` | `np.int` removal for NumPy >= 1.24 |
| `pcdet/models/backbones_2d/map_to_bev/__init__.py` | registers the sparse transformer |
| `pcdet/models/backbones_3d/__init__.py` | registers the front-view branch |
| `pcdet/models/backbones_3d/dsvt.py` | `key_padding_mask` as bool for torch >= 2 on the voxel path |
| `pcdet/models/backbones_3d/vfe/__init__.py` | registers the height-binned mean encoder |
| `pcdet/models/dense_heads/center_head.py` | GroupNorm in `shared_conv` when the water-level head is enabled, because BatchNorm train and eval statistics leak session identity |
| `pcdet/models/dense_heads/transfusion_head.py` | code size 8 without velocity targets instead of a hardcoded 10 |
| `pcdet/models/detectors/detector3d_template.py` | the `map_to_bev_fusion` slot in the build order |
| `pcdet/models/model_utils/centernet_utils.py` | top-k class recovery uses the actual per-class stride `min(K, N)`; upstream divides by `K`, which mislabels detections on frames with fewer than `K` candidates |
| `pcdet/models/roi_heads/__init__.py` | registers the second-stage refinement head |
| `tools/cfgs/dataset_configs/custom_dataset.yaml` | evaluation on the test split, 128 points per voxel |
| `tools/train.py` | `--freeze_modules`, applied before the optimizer is built |

## Added files

- `make_infos.py` — builds the info pickles of the custom datasets
- `pcdet/models/backbones_2d/height_profile_fusion.py`
- `pcdet/models/backbones_2d/map_to_bev/sst/` — the sparse transformer implementation
- `pcdet/models/backbones_2d/map_to_bev/sst_bev.py` — MSTNet's bird's-eye module
- `pcdet/models/backbones_2d/sst_out_conv.py`
- `pcdet/models/backbones_3d/frontview_sst.py`
- `pcdet/models/backbones_3d/vfe/zbinned_mean_vfe.py` — the height-aware encoder
- `pcdet/models/model_utils/roi_point_crop_utils.py`
- `pcdet/models/roi_heads/points_crop_conv3d_head.py` — the second-stage refinement
- `tools/cfgs/custom_london_models/` — 36 experiment configs (Thames)
- `tools/cfgs/custom_seepersea_models/` — 78 experiment configs (SeePerSea)
- `tools/cfgs/dataset_configs/custom_london*.yaml`, `custom_seepersea*.yaml` — 41 dataset
  configs, one per label budget and per dataset variant

The two model directories hold exactly the configurations the campaign resolves;
every supervised run of the paper is one file there.
