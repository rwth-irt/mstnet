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

Place the public dataset releases under `data/raw/` as described in
`data/README.md`. Thames, SeePerSea and KITTI are downloaded by hand from their
authors; Pohang is fetched by script in step 3.

## 3. Build the datasets

```bash
# Pohang raw archives, about 54 GB
python scripts/download_pohang_raw.py

# Thames derived layouts, deterministic under fixed seeds
python scripts/create_temporal_stratified_split.py   # pretraining source layout
python scripts/create_balanced_temporal_split.py     # detection layout

# Detection datasets -> data/custom_{london,seepersea}
python scripts/convert_london.py
python scripts/convert_seepersea.py

# Pretraining corpora -> data/ssl_nuscenes/
python scripts/prepare_ssl_datasets.py all           # london, seepersea, kitti
python scripts/rebuild_pohang_corpus.py              # pohang
python scripts/build_combined_ssl_sets.py            # pohang_london, pohang_seepersea

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

Only the `mstnet_pseudo*` runs additionally need clustering-based pseudo labels,
SeePerSea only. Each budget gets its own pass so the size prior sees that budget's
labels alone, then the info pickles are built in one go:

```bash
python scripts/make_pseudo_labels.py                                          # full labels
for b in 5 10 20 50 80; do
  python scripts/make_pseudo_labels.py --size-split train_${b}pct.txt --out-suffix ${b}pct
done
python scripts/build_pseudo_infos.py
```

## 4. Check before training

```bash
python scripts/check_configs.py     # every config resolves; protocol deviations listed
python scripts/check_training.py         # static checks over splits, scales and leftovers
```

`check_configs.py` should print `no deviations`. The four exceptions it lists after
that are intrinsic to the architectures and are documented in the paper.

## 5. Pretraining, then the campaign

```bash
# The nine self-supervised runs on two GPUs; writes checkpoints/
python scripts/run_pretraining_all.py

# The supervised campaign, 204 runs, about 983 GPU-hours in total
python scripts/run_campaign.py --list
setsid nohup python scripts/run_campaign.py --go > runs/campaign.log 2>&1 &
```

Both runners are resumable: rerunning skips whatever already has a result. Campaign
runs that need a pretrained encoder are skipped while `checkpoints/` is empty, so
the two stages can run on different machines. `--list` therefore reports only the
runs it can start right away: 97 of the 204, at 538 of the 983 GPU-hours, before any
pretraining checkpoint exists; the rest appear in the plan as `checkpoints/` fills
in and disappear from the SKIP lines above it.

## 6. Results

```bash
python scripts/collect_results.py             # evaluation logs -> results/all_results.csv
python scripts/eval_london_perclass.py        # per-class Thames evaluation
python scripts/export_train_loss.py           # per-epoch training loss
python scripts/measure_localization_error.py  # per-parameter error of matched detections
python scripts/measure_pseudo_labels.py       # what the pseudo-label sets contain
python scripts/make_paper_tables.py           # -> results/tables/
```

The inference cost is measured separately, after the campaign, because a latency
measured next to a training run measures the contention and not the model.
`run_speed_sweep.py` waits until no compute process is left on either card and then
measures one cell at a time:

```bash
python scripts/run_speed_sweep.py --list
setsid nohup python scripts/run_speed_sweep.py --go > runs/speed_sweep.log 2>&1 &
```

The checkpoints behind the reported numbers are assembled for publication by

```bash
python scripts/make_checkpoint_release.py --dry-run
python scripts/make_checkpoint_release.py --go
```

`results/` is not part of the repository. The reporting scripts above write it, and
what they write are the paper's numbers, which belong to the paper. A fresh clone
therefore starts without it and fills it from the campaign it runs itself.
