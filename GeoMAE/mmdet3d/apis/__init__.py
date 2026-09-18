from .train import train_model

# Lazy-import inference/test to avoid pulling in mmdet.core (removed in mmdet 3.x).
# SSL pre-training only needs train_model.
def __getattr__(name):
    if name in ('convert_SyncBN', 'inference_detector',
                'inference_mono_3d_detector', 'inference_multi_modality_detector',
                'inference_segmentor', 'init_model', 'show_result_meshlab'):
        from . import inference
        return getattr(inference, name)
    if name == 'single_gpu_test':
        from .test import single_gpu_test
        return single_gpu_test
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'inference_detector', 'init_model', 'single_gpu_test',
    'inference_mono_3d_detector', 'show_result_meshlab', 'convert_SyncBN',
    'train_model', 'inference_multi_modality_detector', 'inference_segmentor'
]
