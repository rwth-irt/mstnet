# Thames split index files

The frame lists of the Thames dataset of Xie et al. (2024), doi
10.1016/j.oceaneng.2024.119288. Only the lists are published here; the recordings
belong to their authors.

| file | frames | content |
|---|---|---|
| `train.txt` | 7130 | supervised training pool, the 100 % budget |
| `val.txt` | 796 | the split every reported Thames number is measured on |
| `train_5pct.txt` | 356 | 5 % label budget, nested |
| `train_10pct.txt` | 713 | 10 % |
| `train_20pct.txt` | 1427 | 20 % |
| `train_50pct.txt` | 3565 | 50 % |
| `train_80pct.txt` | 5703 | 80 % |

`val.txt` holds the test split of the partition, not its validation split. The
detector framework evaluates whatever it calls `val`, and pointing that name at the
test frames is how the reported numbers come from the held-out set. The 740
validation frames of the partition are not part of this layout.

## Why the released partition is not used

The release assigns 8600 real-world frames to training and 686 to evaluation, uses
the same frames for validation and for testing, and is not stratified: small vessels
appear in 24.1 % of its training frames and in 4.1 % of its evaluation frames. The
recording is also continuous, so neighbouring frames are near duplicates and a
frame-level partition lets a detector recognize a frame it has already seen.

We therefore rebuild it. The frames are cut into consecutive chunks of 100, about
6.7 seconds at the recorded rate, and whole chunks are assigned by stratified
sampling on their class composition. At every boundary between chunks of different
splits, 10 frames on each side are discarded, which removes 620 frames or 6.7 % of
the release. Small vessels then appear in 20.6 % of the test frames.
`create_balanced_temporal_split.py` performs this, deterministically under a fixed
seed.

The budget lists are ours and strictly nested, 5 % a subset of 10 % and so on up to
`train.txt`. Do not regenerate them; results measured on redrawn lists are not
comparable with the paper.
