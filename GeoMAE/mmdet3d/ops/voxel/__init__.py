# Redirect to mmcv 2.x implementations (local C extensions failed to compile)
try:
    from .scatter_points import DynamicScatter, dynamic_scatter
except (ImportError, ModuleNotFoundError):
    from mmcv.ops import DynamicScatter, dynamic_scatter

try:
    from .voxelize import Voxelization, voxelization, Voxelization_with_flag
except (ImportError, ModuleNotFoundError):
    from mmcv.ops import Voxelization, voxelization
    # Voxelization_with_flag is GeoMAE-specific; provide a stub if not available
    Voxelization_with_flag = Voxelization

__all__ = ['Voxelization', 'voxelization', 'dynamic_scatter', 'DynamicScatter', 'Voxelization_with_flag']
