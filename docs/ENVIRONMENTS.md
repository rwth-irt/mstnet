# The three conda environments

The pipeline spans two software generations. The detection frameworks run on torch
2.3, while the GeoMAE pretraining is mmdetection3d-generation code that needs torch
2.0 with a compatibility shim for mmcv 1.x. One environment cannot serve both, and
LION additionally needs its mamba kernels compiled against its own tree. Hence
three environments. The exact package lists are exported under `envs/`.

| env | python | torch | used for |
|---|---|---|---|
| `pcdet` | 3.10 | 2.3.1+cu121 | OpenPCDet and TED training and evaluation, every entry point under `scripts/` |
| `lion` | 3.10 | 2.3.1+cu121 | LION training and evaluation, with `mamba_ssm` built from `LION/pcdet/ops/mamba/` |
| `geomae` | 3.10 | 2.0.1+cu118 | the nine self-supervised pretraining runs |

There is deliberately no TED environment. TED runs on the `pcdet` interpreter with
`TED/` first on `PYTHONPATH`, so its own `pcdet` package shadows the installed one.
The torch versions match, so TED's compiled ops load. A separate TED environment
produced corrupted spconv binaries in our setup.

## Creating the environments

```bash
conda env create -n pcdet  -f envs/env_pcdet.yml
conda env create -n lion   -f envs/env_lion.yml
conda env create -n geomae -f envs/env_geomae.yml
```

The `env_*.yml` files are exports of what we ran with; `pip_*.txt` are the full pip
freezes of the same environments, for reference. The CUDA-specific wheels, torch,
spconv and mmcv, may need their index URLs adjusted to your driver.

Then build the compiled ops, each inside its environment:

```bash
conda run -n pcdet  bash -c "cd OpenPCDet && python setup.py develop"
conda run -n pcdet  bash -c "cd TED && python setup.py develop"
conda run -n lion   bash -c "cd LION && python setup.py develop && cd pcdet/ops/mamba && python setup.py install"
conda run -n geomae bash -c "cd GeoMAE && pip install -e ."
```

## How the scripts find the interpreters

Everything under `scripts/` runs on the interpreter it is started with, which should
be `pcdet`. The two other interpreters are read from environment variables and
default to `python`:

```bash
export LION_PYTHON=~/miniconda3/envs/lion/bin/python
export GEOMAE_PYTHON=~/miniconda3/envs/geomae/bin/python
```

`run_campaign.py` and `run_speed_sweep.py` use `LION_PYTHON`,
`run_pretraining_all.py` uses `GEOMAE_PYTHON`. No script contains an absolute
interpreter path.
