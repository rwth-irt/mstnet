# Third-party code

This repository vendors four research code bases. Each keeps its upstream license,
which is Apache 2.0 in all four cases, and each carries a `CHANGES.md` naming the
upstream commit and every file we modified or added. The change lists are derived
mechanically, three of them from the fork's own git history against the upstream
commit and GeoMAE's from a file-by-file comparison against a fresh clone.

| Directory | Project | Upstream | Commit | License |
|---|---|---|---|---|
| `OpenPCDet/` | OpenPCDet | https://github.com/open-mmlab/OpenPCDet | `233f849` | Apache 2.0 |
| `TED/` | TED | https://github.com/hailanyi/TED | `c2e4847` | Apache 2.0 |
| `LION/` | LION | https://github.com/happinesslz/LION | `f7cbdf9` | Apache 2.0 |
| `GeoMAE/` | GeoMAE | https://github.com/Tsinghua-MARS-Lab/GeoMAE | `962d6db` | Apache 2.0 |

OpenPCDet hosts nine of the eleven published detectors of the benchmark and MSTNet
itself. TED and LION are separate forks because each patches into its own copy of
the `pcdet` package and the two cannot share one installation. GeoMAE hosts the
self-supervised pretraining and runs on an older torch generation, which is why it
needs its own environment.

The papers behind the eleven benchmarked detectors are cited in the manuscript.
