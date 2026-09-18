# Changes against upstream TED

Fork of https://github.com/hailanyi/TED at commit `c2e4847`, stripped of `.git`,
`build/`, `output/`, the data tree and every compiled artifact. The list was
derived from the fork's own git status against that commit.

## Modified upstream files

**Build compatibility with current PyTorch**, no behavioral change: the deprecated
`THC/THC.h` include and the `THCState` extern were removed from

- `pcdet/ops/dcn/src/{deform_conv,deform_pool}_cuda_kernel.cu`
- `pcdet/ops/pointnet2/pointnet2_batch/src/{ball_query,group_points,interpolate,sampling}.cpp`
- `pcdet/ops/pointnet2/pointnet2_stack/src/{ball_query,ball_query_deform,group_points,interpolate,sampling,vector_pool,voxel_query}.cpp`
- `pcdet/ops/votr_ops/src/{build_attention_indices,build_mapping,group_features}.cpp`

plus `tools/train_utils/optimization/fastai_optim.py` (`collections.abc.Iterable`
for Python >= 3.10) and `pcdet/version.py`, which records the upstream commit.

**Behavioral changes:**

| File | Why |
|---|---|
| `pcdet/datasets/__init__.py` | registers the maritime dataset |
| `pcdet/models/roi_heads/target_assigner/proposal_target_layer.py` | a frame with no qualifying foreground or background RoI, common on sparse single-object maritime frames, falls back to sampling all RoIs as background instead of raising `NotImplementedError` |

## Added files

- `pcdet/datasets/custom/` — the maritime dataset reader
- `pcdet/datasets/kitti/__init__.py`, `kitti_object_eval_python/__init__.py`,
  `kitti_utils.py` — import shims that make the KITTI evaluation reachable as a package
- `tools/cfgs/dataset_configs/custom_london.yaml`, `custom_seepersea.yaml`
- `tools/cfgs/custom_london_models/` — `ted_s.yaml` and `ted_s_perclass.yaml`
- `tools/cfgs/custom_seepersea_models/` — `ted_s.yaml`

`ted_s_perclass.yaml` is `ted_s.yaml` with its five threshold lists extended to one
entry per class. Under `CLS_SCORE_TYPE: roi_iou_x` the RoI target assigner indexes
`CLS_FG_THRESH`, `CLS_BG_THRESH` and `REG_FG_THRESH` per class, and the length-one
lists of the upstream single-class recipe trained every proposal matched to the
second or third Thames class as background whatever its overlap. The Thames row of
the benchmark comes from the repaired config. SeePerSea is single class and its
configuration is correct as it stands.
