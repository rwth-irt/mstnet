# SeePerSea split index files

The frame lists of the SeePerSea dataset of Jeong et al. (2025), doi
10.1109/TFR.2025.3602937. Stems refer to the **aggregated** release, which ships
`points/`, `label/`, `calib/` and `ImageSets/` with globally unique frame names, so
a stem alone identifies a frame. Only the lists are published here.

| file | frames | content |
|---|---|---|
| `train.txt` | 2951 | supervised training pool, the official train split, the 100 % budget |
| `val.txt` | 629 | the official validation split, which every reported number is measured on |
| `train_5pct.txt` | 147 | 5 % label budget, nested |
| `train_10pct.txt` | 295 | 10 % |
| `train_20pct.txt` | 591 | 20 % |
| `train_50pct.txt` | 1476 | 50 % |
| `train_80pct.txt` | 2360 | 80 % |

`train.txt` and `val.txt` are the official splits of the release, reproduced
unchanged. The official test split of 1141 frames carries no public labels and
cannot be evaluated locally, which is why the validation split serves as the
held-out set. It never enters supervised training.

The unlabeled point clouds of all frames, the validation frames included, do enter
the self-supervised pretraining corpora by design. The paper discloses this.

The budget lists are ours and strictly nested. They are built by splitting the
training frames into those with at least one annotated vessel and those without,
shuffling each group once under a fixed seed, and taking a prefix of both in the
proportion the full split has. The annotated share is thereby held between 89.0 %
and 89.2 % at every budget, against 89.1 % in the full split, and each budget is a
subset of the next larger one. Do not regenerate them; results measured on redrawn lists are not
comparable with the paper.
