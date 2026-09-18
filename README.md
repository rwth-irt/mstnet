# MSTNet: Label-Efficient Maritime 3D LiDAR Object Detection by Self-Supervised Pretraining and Vertical Refinement

Code, configurations and results for the paper of the same name.

Tim Rehbronn, Laurens Keuker, Heike Vallery
Institute of Automatic Control, RWTH Aachen University

[Reproduce](docs/REPRODUCE.md) · [Environments](docs/ENVIRONMENTS.md) · [Data](data/README.md) · [Third-party code](THIRD_PARTY.md)

The pipeline is called **MSTNet** in the paper and throughout this repository.

The code was written by Tim Rehbronn and Laurens Keuker, with coding assistance and refactoring by Anthropic's Claude.

## Abstract

> Maritime autonomous vessels need 3D object detection for safe navigation. This
> requires supervised training with annotated data, which is scarce because
> annotation is resource-intensive. Unlabeled vessel recordings are increasingly
> available. This paper presents a Maritime Sparse Transformer Network (MSTNet), a
> detection pipeline that enables label-free pretraining with supervised
> fine-tuning. We replace the pillar feature net and CenterPoint's scatter
> operation with a parameter-free mean voxel encoder and a single-stride sparse
> transformer, and we pretrain the transformer with a masked geometric objective. A
> second stage on clustered pseudo-labels initializes the convolutional backbone
> and the center head. We evaluate on a re-split Thames and the SeePerSea datasets
> across six label budgets and four pretraining data sources. On SeePerSea,
> pretraining raises 3D mean average precision from 12.14 to 58.04 mAP at full
> supervision, and MSTNet matches CenterPoint's full-supervision 3D accuracy using
> 20 % of the annotations. Every detector we test localizes less accurately in 3D
> than in the bird's-eye view. A height-aware token, together with a refinement
> stage restricted to the vertical box parameters, cuts the median center height
> error from 0.14 m to 0.08 m without further annotation. Against eleven
> state-of-the-art detectors, MSTNet in its strongest configuration, the
> height-aware token with vertical refinement, leads on SeePerSea with 77.52 mAP in
> the bird's-eye view and 66.37 in 3D at 36 ms per frame, while PointPillars stays
> ahead on the denser Thames recording. The code, the re-split dataset, and the
> checkpoints are publicly available.

![Graphical abstract](assets/graphical_abstract.svg)

## What this is

Annotating maritime LiDAR point clouds is expensive; recording them is not. MSTNet
puts the unlabeled recordings to work, in a pipeline built from published modules.
The detector is CenterPoint with two modules exchanged: a mean voxel encoder
replaces the learned pillar features, and a single-stride sparse transformer builds
the bird's-eye feature map. That transformer is pretrained on unlabeled maritime
point clouds with masked geometric pretraining (GeoMAE): most occupied pillars are
masked and the encoder learns to reconstruct their centroids, surface normals and
sub-voxel occupancy. The pretrained weights then initialize supervised training,
which we run at label budgets from 5 % to 100 % to measure how much annotation the
pretraining replaces, and with pretraining data from inside and outside the target
domain (Thames, SeePerSea, Pohang Canal, KITTI) to measure how far from the target
domain the unlabeled data may come.

Masked pretraining updates only the transformer, so at the smallest label budgets
the convolutional backbone and the center head still start from a random
initialization with few labels to fit them. A second, equally label-free stage
addresses that: boxes fitted to clusters of unlabeled points pretrain the backbone
and the head before fine-tuning on the real labels.

Every detector we benchmark localizes vessels well in the bird's-eye view and loses
most of its accuracy in the vertical axis, an error the annotation budget does not
close. Two measures address it without spending a further label: a height-aware
token that carries a coarse vertical profile of the pillar, and a local refinement
stage restricted to the vertical box parameters.

The paper closes with a benchmark that did not exist for this domain. Eleven
published detector configurations (PointPillars, SECOND, CenterPoint, Voxel R-CNN,
PV-RCNN, DSVT in its pillar and voxel form, VoxelNeXt, TransFusion, TED-S,
LION-Mamba) and MSTNet are trained under one shared protocol on both datasets, with
every deviation from that protocol documented and machine-checked.

## Repository structure

```
OpenPCDet/  TED/  LION/  GeoMAE/   the four vendored forks; each carries a CHANGES.md
                                   naming its upstream commit and every file we changed
scripts/                           data preparation, checks, runners, evaluation and
                                   reporting; every entry point lives here
data/                              built locally from the public releases; only the split
                                   index files under custom_*/ImageSets ship
envs/                              conda exports and pip freezes of the three environments
docs/                              REPRODUCE.md and ENVIRONMENTS.md
results/                           not shipped; where the reporting scripts write the
                                   collected measurements and the generated tables
assets/                            the graphical abstract, SVG and TikZ source
```

OpenPCDet hosts nine of the eleven published detectors and MSTNet itself. TED and
LION are separate forks because each patches into its own copy of the `pcdet`
package. GeoMAE hosts the self-supervised pretraining and runs on an older torch
generation. The forks reach the data tree through relative symlinks, so a clone
works wherever it is placed.

## Installation

Three conda environments, because the detection frameworks run on torch 2.3 while
the pretraining is mmdetection3d-generation code on torch 2.0.
[docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md) explains the split and gives the build
commands for the compiled ops of each fork.

```bash
conda env create -n pcdet  -f envs/env_pcdet.yml
conda env create -n lion   -f envs/env_lion.yml
conda env create -n geomae -f envs/env_geomae.yml
```

Everything under `scripts/` runs on the interpreter it is started with, which should
be `pcdet`. The two others are announced through environment variables; no script
contains an absolute interpreter path.

```bash
export LION_PYTHON=~/miniconda3/envs/lion/bin/python
export GEOMAE_PYTHON=~/miniconda3/envs/geomae/bin/python
```

## Data

Obtain the four public releases from their authors as described in
[data/README.md](data/README.md), place them under `data/raw/`, then build
everything derived from them. The ordered command list is in
[docs/REPRODUCE.md](docs/REPRODUCE.md). This repository redistributes no dataset
content; what it publishes are the split index files and the scripts that build
them, so the partition can be reproduced without the recordings changing hands.

## Usage

**Check** before any GPU hour is spent. `check_configs.py` resolves every training
configuration and reports deviations from the shared protocol, and should print
`no deviations`. `check_training.py` runs static checks over splits, intensity scales
and leftovers.

```bash
python scripts/check_configs.py
python scripts/check_training.py
```

**Train.** The nine self-supervised runs first, then the supervised campaign. The
campaign plans 204 runs in total, about 983 GPU-hours; `run_campaign.py --list`
reports only the runs it can start immediately, which is 97 of them at 538
GPU-hours before any pretraining checkpoint exists, and grows toward the full 204
as `checkpoints/` fills in. Both runners drive two GPUs, record every result, and
skip finished work on restart, so they can be stopped and resumed at will.

```bash
python scripts/run_pretraining_all.py                        # writes checkpoints/
python scripts/run_campaign.py --list                        # the plan, trains nothing
setsid nohup python scripts/run_campaign.py --go > runs/campaign.log 2>&1 &
```

**Report.** Collect the evaluation logs into one CSV, take the measurements that are
not average precision, and generate every table and figure from them.

```bash
python scripts/collect_results.py             # -> results/all_results.csv
python scripts/eval_london_perclass.py        # per-class Thames evaluation
python scripts/export_train_loss.py           # per-epoch training loss
python scripts/measure_localization_error.py  # per-parameter error of matched detections
python scripts/run_speed_sweep.py --go        # parameters, latency, peak memory
python scripts/make_paper_tables.py           # -> results/tables/
```

## Results

This repository publishes the code, not the numbers. The measurements themselves
belong to the paper, and the checkpoints behind them are on Zenodo.

The reporting path builds them locally. `collect_results.py` reads the evaluation
logs of a campaign into `results/all_results.csv`, one row per evaluation with the
configuration it came from and all four metric combinations. The other reporting
scripts write the measurements that are not average precision beside it:
per-epoch training loss, the error of matched detections per box parameter, and
parameters, latency and peak memory per architecture. `make_paper_tables.py` then
generates every table and figure of the paper from those files into
`results/tables/`; none is written by hand, so a rerun of a cell cannot leave a
table disagreeing with the logs.

## Checkpoints

The pretrained encoders and the trained detector weights are published separately
on Zenodo, one checkpoint per reported run, with a table that maps each file to its
experiment and configuration.

> **DOI:** [10.5281/zenodo.22825184](https://doi.org/10.5281/zenodo.22825184)
> (reserved; the record is not yet public, so the link may not resolve until it is).

`scripts/make_checkpoint_release.py` assembles that folder from
`results/all_results.csv`, so the path from a published checkpoint back to the row
whose number it produced is itself reproducible.

### Using the published checkpoints

Downloading a checkpoint from Zenodo replaces the GPU-hours it was trained for,
not the data: the datasets still have to be obtained and converted as described
in [Data](#data) and [docs/REPRODUCE.md](docs/REPRODUCE.md), because every config
reads its input from `data/custom_london/` or `data/custom_seepersea/`, not from
the checkpoint. Two things become possible once that data tree exists and a
checkpoint is downloaded.

**Skip pretraining.** The Zenodo `pretraining/` files are exactly the contents of
`checkpoints/`, one file per name (`mstnet_ssl_seepersea.pth`, and so on --
`python scripts/run_campaign.py --list` prints which name each run needs). Place
the ones you want under `checkpoints/` and `run_campaign.py` finds them; the runs
that depend on them stop printing `SKIP, checkpoint missing` and enter the plan,
same as if `run_pretraining_all.py` had just produced them.

**Evaluate one reported result without training it.** A `finetuning/` file is a
detector checkpoint, and `--ckpt` takes any path, so it does not need to sit
inside an `output/` tree first. From the framework's `tools/` directory (`cd
OpenPCDet/tools` for MSTNet and nine of the eleven benchmarked detectors, `TED/`
or `LION/tools` for TED-S or LION-Mamba):

```bash
python test.py --cfg_file cfgs/custom_seepersea_models/mstnet.yaml \
                --ckpt /path/to/seepersea_mstnet_mstnet_ssl_seepersea.pth \
                --extra_tag from_zenodo
```

The config is the one named in the checkpoint's row of the table on Zenodo
(`custom_<dataset>_models/<config>.yaml`, relative to that framework's `tools/`).
Results land under `output/<config-path>/from_zenodo/eval/`, including a
`result.pkl` of the raw predictions. On Thames, that mAP is not the one the paper
reports: the dataset config maps all three vessel classes to the single KITTI
class `Car`, so the built-in evaluator cannot separate them and reports an
inflated figure. Run `python scripts/eval_london_perclass.py` afterward, from the
repository root, to get the per-class number the paper actually uses; it reads
the `result.pkl` `test.py` just wrote, so no re-inference is needed.

## Citation

> *Placeholder. The reference of the accepted version is inserted here once the
> paper is published, with the DOI, volume and page numbers it is assigned.*

```bibtex
@article{rehbronn2026mstnet,
  title   = {{MSTNet}: Label-Efficient Maritime {3D} {LiDAR} Object Detection by
             Self-Supervised Pretraining and Vertical Refinement},
  author  = {Rehbronn, Tim and Keuker, Laurens and Vallery, Heike},
  journal = {Journal of Intelligent \& Robotic Systems},
  year    = {2026},
  note    = {Submitted},
}
```

## License

Apache 2.0, see [LICENSE](LICENSE). The four vendored forks keep their upstream
Apache 2.0 licenses, listed in [THIRD_PARTY.md](THIRD_PARTY.md). The datasets are
distributed by their own authors under their own terms, named in
[data/README.md](data/README.md).

## Acknowledgements

This work has partially been funded by the German Federal Ministry of Research,
Technology and Space (BMFTR) under the Robotics Institute Germany (RIG). The four
vendored code bases and the four datasets are the work of their authors, cited in
the paper and named in [THIRD_PARTY.md](THIRD_PARTY.md) and
[data/README.md](data/README.md).
