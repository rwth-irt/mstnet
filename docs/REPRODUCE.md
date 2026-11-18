# Reproducing the experiments

The ordered command list, from a fresh clone to the tables of the paper. Every
command runs from the repository root in the `pcdet` environment unless stated
otherwise. Steps 1 and 2 are prose, everything after is copy and paste.

## 1. Environments

Create the three environments and build the compiled ops as described in
`docs/ENVIRONMENTS.md`, then export the two other interpreters:

```bash
export LION_PYTHON=~/miniconda3/envs/lion/bin/python
export GEOMAE_PYTHON=~/miniconda3/envs/geomae/bin/python
```

## 2. Raw data

Store the public dataset releases under `data/raw/` in the layout described in
`data/README.md`; SeePerSea requires both the aggregated and the separated release.
Thames, SeePerSea and KITTI are obtained from their authors, and the Pohang Canal
archives are downloaded by a script in step 3.

## 3. Build the datasets

```bash
# Pohang raw archives, about 54 GB
python scripts/download_pohang_raw.py

# Thames derived layouts, deterministic under fixed seeds
python scripts/create_temporal_stratified_split.py   # pretraining source layout
python scripts/create_balanced_temporal_split.py     # detection layout

# Detection datasets -> data/custom_{london,seepersea}
python scripts/convert_london.py
python scripts/convert_seepersea.py                  # 360-degree annotations of the separated release

# Pretraining corpora -> data/ssl_nuscenes/
python scripts/prepare_ssl_datasets.py all           # london, seepersea, kitti
python scripts/rebuild_pohang_corpus.py              # pohang
python scripts/build_combined_ssl_sets.py            # pohang_london, pohang_seepersea
python scripts/make_ssl_noval_infos.py               # seepersea without the validation frames (Appendix C)

# One intensity scale everywhere; run without --apply first to see the report
python scripts/renormalise_intensity.py --apply

# Info pickles and ground-truth sampling databases
python scripts/rebuild_infos_and_db.py --apply
```

The label budget lists under `data/custom_*/ImageSets/` ship with the repository and
are the ones the paper used. Do not regenerate them; results on redrawn splits are
not comparable. For SeePerSea the generators are included so the procedure is
inspectable, not so the lists get replaced, and none of the steps above calls them:
`create_seepersea_label_budget_splits.py` draws the budgets and
`make_nested_budgets.py` makes them strictly nested. The Thames budget lists ship
without their generator, so for those the lists themselves are the record; how they
were drawn is described in the paper.

## 4. Check before training

```bash
python scripts/check_configs.py     # every config resolves; protocol deviations listed
python scripts/check_training.py         # static checks over splits, scales and leftovers
```

`check_configs.py` should print `no deviations`. The four exceptions it lists after
that are intrinsic to the architectures and are documented in the paper.

## 5. Pretraining, then the campaign

```bash
# The seven self-supervised runs on two GPUs; writes checkpoints/
python scripts/run_pretraining_all.py

# The supervised campaign, 165 runs, about 545 GPU-hours on two RTX 4090
python scripts/run_campaign.py --list
setsid nohup python scripts/run_campaign.py --go > runs/campaign.log 2>&1 &
```

Both runners are resumable: rerunning skips whatever already has a result. Campaign
runs that need a pretrained encoder are skipped while `checkpoints/` is empty, so
the two stages can run on different machines. `--list` therefore reports only the
runs it can start right away, 98 of the 165 before any pretraining checkpoint exists;
the rest appear in the plan as `checkpoints/` fills in and disappear from the SKIP
lines above it.

## 6. Results

The inference cost is measured separately, after the campaign, because a latency
measured next to a training run measures the contention and not the model.
`run_speed_sweep.py` waits until no compute process is left on either card and then
measures one cell at a time:

```bash
python scripts/run_speed_sweep.py --list
setsid nohup python scripts/run_speed_sweep.py --go > runs/speed_sweep.log 2>&1 &
```

The tables, figures and further measurements of the paper are then generated from the
run records, the checkpoints and the data:

```bash
python scripts/make_paper_tables.py            # Figures 6 and 7, Tables 3 to 5 and C1, per-run mAP
python scripts/export_train_loss.py            # training loss behind the learning rate, Section 2.4.2
python scripts/measure_localization_error.py   # localization of the matched detections
python scripts/analyze_seepersea_matches.py    # recall by vessel length and neighbourhood
python scripts/dataset_statistics.py --grid-search   # Appendices A.1 and B.3, Section 2.3.3
```

The checkpoints behind the reported numbers are assembled for publication by

```bash
python scripts/make_checkpoint_release.py --dry-run
python scripts/make_checkpoint_release.py --go
```

`results/` is not part of the repository; the scripts above write it. The mAP of every
run the paper reports is published in `release_data/results/map_per_run.csv`, and
`make_paper_tables.py` writes the same file, `results/map_per_run.csv`, from a new
campaign.
