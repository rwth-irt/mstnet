from .builder import (FUSION_LAYERS, SPARSE_ENCODERS, MIDDLE_ENCODERS, VOXEL_ENCODERS,
                      build_sparse_encoder,
                      build_backbone, build_detector, build_fusion_layer,
                      build_head, build_loss, build_middle_encoder,
                      build_model, build_neck, build_roi_extractor,
                      build_shared_head, build_voxel_encoder)

# Import each submodule separately so a failure in one (e.g. missing
# PointNet2 CUDA extensions) doesn't block registration of others.
_submodules = [
    'backbones', 'decode_heads', 'dense_heads', 'detectors',
    'fusion_layers', 'losses', 'middle_encoders', 'model_utils',
    'necks', 'roi_heads', 'segmentors', 'voxel_encoders',
    'sparse_encoder', 'sst',
]
for _sub in _submodules:
    try:
        __import__(f'{__name__}.{_sub}')
    except ImportError:
        pass

__all__ = [
    'VOXEL_ENCODERS', 'MIDDLE_ENCODERS', 'FUSION_LAYERS', 'build_backbone',
    'build_neck', 'build_roi_extractor', 'build_shared_head', 'build_head',
    'build_loss', 'build_detector', 'build_fusion_layer', 'build_model',
    'build_middle_encoder', 'build_voxel_encoder', 'SPARSE_ENCODERS', 'build_sparse_encoder'
]
