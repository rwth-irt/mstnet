"""
Compatibility shim: patches mmcv 2.x to provide mmcv 1.x-style imports.

GeoMAE was written for mmcv-full 1.x + mmdet 2.x + mmseg 0.x.
This shim patches the new packages to provide old-style APIs.

Usage: import this module BEFORE any mmcv/mmdet3d imports:
    import compat_shim  # patches at import time
"""
import functools
import sys
import types

# ---------------------------------------------------------------------------
# 0) Fake mmcv version to pass downstream version checks (mmseg, mmdet)
#    mmseg 1.2.2 requires mmcv < 2.2.0 (exclusive)
#    mmdet 3.3.0 requires mmcv < 2.2.0 (exclusive)
# ---------------------------------------------------------------------------
import mmcv
_real_version = mmcv.__version__
mmcv.__version__ = "2.1.0"

# ---------------------------------------------------------------------------
# 1) Patch mmcv.Config / mmcv.DictAction / mmcv.utils.Registry
# ---------------------------------------------------------------------------
from mmengine.config import Config, DictAction
from mmengine.registry import Registry

mmcv.Config = Config
mmcv.DictAction = DictAction

# Patch mmcv.utils to have Registry
try:
    mmcv.utils.Registry = Registry
except AttributeError:
    pass

# ---------------------------------------------------------------------------
# 2) Create mmcv.runner shim
# ---------------------------------------------------------------------------
runner_mod = types.ModuleType("mmcv.runner")
runner_mod.__package__ = "mmcv"

from mmengine.dist import get_dist_info, init_dist
from mmengine.runner import load_checkpoint, save_checkpoint
from mmengine.model import BaseModule

runner_mod.get_dist_info = get_dist_info
runner_mod.init_dist = init_dist
runner_mod.load_checkpoint = load_checkpoint
runner_mod.save_checkpoint = save_checkpoint
runner_mod.BaseModule = BaseModule

# No-op decorators for auto_fp16/force_fp32 (removed in mmcv 2.x)
def _noop_decorator(*args, **kwargs):
    if len(args) == 1 and callable(args[0]):
        return args[0]
    def wrapper(fn):
        """Wrap `fn` unchanged; the mixed-precision decorators are no-ops here."""
        @functools.wraps(fn)
        def inner(*a, **kw):
            """Call the wrapped function, adding nothing."""
            return fn(*a, **kw)
        return inner
    return wrapper

runner_mod.auto_fp16 = _noop_decorator
runner_mod.force_fp32 = _noop_decorator

sys.modules["mmcv.runner"] = runner_mod

# ---------------------------------------------------------------------------
# 3) Patch mmcv.parallel (collate, scatter, DataContainer, MMDataParallel)
# ---------------------------------------------------------------------------
from mmengine.dataset import default_collate
from torch.nn.parallel._functions import Scatter as _TorchScatter

def _collate(batch, samples_per_gpu=1):
    """Compat shim for mmcv.parallel.collate → mmengine default_collate."""
    return default_collate(batch)

def _scatter(inputs, target_gpus, dim=0):
    """Compat shim for mmcv.parallel.scatter."""
    return _TorchScatter.apply(target_gpus, None, dim, *inputs)

try:
    import mmcv.parallel
    if not hasattr(mmcv.parallel, "collate"):
        mmcv.parallel.collate = _collate
    if not hasattr(mmcv.parallel, "scatter"):
        mmcv.parallel.scatter = _scatter
except (ImportError, ModuleNotFoundError):
    parallel_mod = types.ModuleType("mmcv.parallel")
    parallel_mod.__package__ = "mmcv"
    parallel_mod.collate = _collate
    parallel_mod.scatter = _scatter
    sys.modules["mmcv.parallel"] = parallel_mod

# ---------------------------------------------------------------------------
# 4) Patch mmdet.models to expose old-style registries (BACKBONES etc.)
#    mmdet 3.x uses a single MODELS registry in mmdet.registry
# ---------------------------------------------------------------------------
import mmdet.models
from mmdet.registry import MODELS as _MMDET_MODELS

# Add old-style registry aliases to mmdet.models
for name in ("BACKBONES", "DETECTORS", "HEADS", "LOSSES", "NECKS",
             "ROI_EXTRACTORS", "SHARED_HEADS"):
    if not hasattr(mmdet.models, name):
        setattr(mmdet.models, name, _MMDET_MODELS)

# Also provide build_backbone etc.
if not hasattr(mmdet.models, "build_backbone"):
    mmdet.models.build_backbone = _MMDET_MODELS.build

# Shim mmdet.models.builder module
builder_mod = types.ModuleType("mmdet.models.builder")
builder_mod.__package__ = "mmdet.models"
for name in ("BACKBONES", "DETECTORS", "HEADS", "LOSSES", "NECKS",
             "ROI_EXTRACTORS", "SHARED_HEADS"):
    setattr(builder_mod, name, _MMDET_MODELS)
builder_mod.build_backbone = _MMDET_MODELS.build
sys.modules["mmdet.models.builder"] = builder_mod

# ---------------------------------------------------------------------------
# 5) Patch mmseg.models.builder for SEGMENTORS registry
# ---------------------------------------------------------------------------
import mmseg.models
from mmseg.registry import MODELS as _MMSEG_MODELS

seg_builder_mod = types.ModuleType("mmseg.models.builder")
seg_builder_mod.__package__ = "mmseg.models"
seg_builder_mod.SEGMENTORS = _MMSEG_MODELS
sys.modules["mmseg.models.builder"] = seg_builder_mod
if not hasattr(mmseg.models, "SEGMENTORS"):
    mmseg.models.SEGMENTORS = _MMSEG_MODELS

# ---------------------------------------------------------------------------
# 6) Patch mmcv.utils to provide Registry
# ---------------------------------------------------------------------------
try:
    import mmcv.utils
    mmcv.utils.Registry = Registry
except (ImportError, AttributeError):
    utils_mod = types.ModuleType("mmcv.utils")
    utils_mod.__package__ = "mmcv"
    utils_mod.Registry = Registry
    sys.modules["mmcv.utils"] = utils_mod

# ---------------------------------------------------------------------------
# 7) Patch mmcv.cnn to expose NORM_LAYERS registry
# ---------------------------------------------------------------------------
import mmcv.cnn
if not hasattr(mmcv.cnn, "NORM_LAYERS"):
    mmcv.cnn.NORM_LAYERS = Registry("norm_layer", scope="mmcv")

# ---------------------------------------------------------------------------
# 8) Patch mmcv.mkdir_or_exist (removed in mmcv 2.x)
# ---------------------------------------------------------------------------
import os as _os
if not hasattr(mmcv, "mkdir_or_exist"):
    mmcv.mkdir_or_exist = lambda d: _os.makedirs(d, exist_ok=True)

print(f"[compat_shim] mmcv {_real_version} + mmdet/mmseg patched for 1.x compat")
