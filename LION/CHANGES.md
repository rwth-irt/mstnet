# Changes against upstream LION

Fork of https://github.com/happinesslz/LION at commit `f7cbdf9`, stripped of
`.git`, `build/`, `output/`, the data tree, the prebuilt `mamba_ssm` egg under
`pcdet/ops/mamba/dist/` and every compiled artifact; build mamba from the shipped
sources. The list was derived from the fork's own git status against that commit.

## Modified upstream files

| File | Why |
|---|---|
| `pcdet/datasets/custom/custom_dataset.py` | rewritten for the maritime datasets, their info pickles, evaluation protocol and per-budget splits |
| `pcdet/datasets/kitti/kitti_object_eval_python/rotate_iou.py` | correctness fix in upstream code: a two-argument `min(a, b)` inside a CUDA kernel resolved to the one-argument form under numba and crashed the evaluation. Replaced by an explicit conditional and verified bit-identical on the cases that did work |
| `tools/cfgs/dataset_configs/custom_dataset.yaml` | evaluation split and voxel settings for the maritime data |

## Added files

- `tools/cfgs/dataset_configs/custom_london.yaml`, `custom_seepersea.yaml`
- `tools/cfgs/custom_london_models/lion_mamba.yaml`
- `tools/cfgs/custom_seepersea_models/lion_mamba.yaml`

One remaining absolute path, `/home/xywang/...` in
`pcdet/models/model_utils/rwkv_cls.py`, is upstream content citing the author's own
machine in a comment. It is left untouched.
