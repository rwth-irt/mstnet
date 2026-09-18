import mmcv

# Fix double registration: mmcv-full already registers SparseConv* in
# CONV_LAYERS; GeoMAE's spconv re-registers the same names.
try:
    from mmcv.cnn import CONV_LAYERS
    _orig_conv_register = CONV_LAYERS.register_module
    def _force_conv_register(*args, **kwargs):
        kwargs.setdefault('force', True)
        return _orig_conv_register(*args, **kwargs)
    CONV_LAYERS.register_module = _force_conv_register
except Exception:
    pass

import mmdet
import mmseg
from .version import __version__, short_version

__all__ = ['__version__', 'short_version']
