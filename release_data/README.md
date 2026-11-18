# MSTNet release data

Frame lists and per-run results of the paper *MSTNet: Label-Efficient Maritime 3D LiDAR Object Detection by Self-Supervised Pretraining*.

## Thames (`thames/`)

Rebuilt partition of the real-world Thames recording (Xie et al., 2024). Each line names a frame of the public release as `<folder>/<stem>`, where `<folder>` is `training` or `testing` of the release and `<stem>` is its file name without extension (`00000` to `09285`).

| File | Frames | Content |
|---|---:|---|
| `train.txt` | 7130 | training split (100 % label budget) |
| `val.txt` | 740 | validation split |
| `test.txt` | 796 | test split; all Thames results refer to this split |
| `train_{5,10,20,50,80}pct.txt` | 356 / 713 / 1427 / 3565 / 5703 | label budgets |

## SeePerSea (`seepersea/`)

Each line is a frame identifier of the aggregated SeePerSea release (Jeong et al., 2025), i.e. the file name of `aggregated/points/<id>.pcd`. Training and evaluation use the 360° point-cloud annotations of the separated release; `frame_map.csv` gives, for every training and validation frame, the sequence and the frame index (`annotations/<frame>.json`, `point_clouds/<frame>.pcd`) in the separated release. Only the category `ship` is used.

| File | Frames | Content |
|---|---:|---|
| `train.txt` | 2951 | official training split (100 % label budget) |
| `val.txt` | 629 | official validation split; all SeePerSea results refer to this split |
| `train_{5,10,20,50,80}pct.txt` | 147 / 295 / 591 / 1476 / 2360 | label budgets |
| `frame_map.csv` | 3580 | aggregated id → separated sequence and frame |

The label budgets of both datasets are nested: every budget is contained in the next larger one.

## Results (`results/map_per_run.csv`)

One row per trained model and run, 163 rows. Values are average precision in percent at 40 recall positions (KITTI protocol) on the evaluation split, at 100 % or the given label budget.

| Column | Meaning |
|---|---|
| `dataset` | `Thames` or `SeePerSea` |
| `experiment` | `E.I` label efficiency, `E.II` pretraining data, `E.III` benchmark and label-budget sweeps, `Appendix C` pretraining without validation frames; `E.I, E.III` marks full-supervision runs used in both |
| `detector` | detector name as in the paper |
| `pretraining` | `none` (random initialization), `in-domain`, `cross-domain maritime` (Pohang Canal), `out-of-domain automotive` (KITTI), `combined` (Pohang Canal and in-domain), `in-domain without validation frames` |
| `label_budget_pct` | 5, 10, 20, 50, 80 or 100 |
| `run` | 1, 2 or 3 for repeated cells |
| `mAP_{bev,3d}@{0.5,0.7}` | mean average precision in BEV and in 3D at overlap 0.5 and 0.7; on SeePerSea the AP of the single class |
| `AP_<class>_{bev,3d}@{0.5,0.7}` | Thames only: AP of `small_boat`, `medium_boat` and `large_boat` |

The in-domain cells of E.II are the in-domain MSTNet runs of E.I and are listed once, under `E.I`. MSTNet is trained at a peak learning rate of 0.001, every other detector at the rate of its reference configuration (Appendix A of the paper). The means over the runs of a cell reproduce the values in the tables and figures of the paper.
