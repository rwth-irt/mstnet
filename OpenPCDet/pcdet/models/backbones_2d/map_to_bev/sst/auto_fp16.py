# This code was created with the help of generative AI to mimic mmcv/runner/auto_fp16, a decorator to handle mixed-precision training.
# Original can be found here: https://github.com/open-mmlab/mmcv/blob/v1.7.1/mmcv/runner/epoch_based_runner.py
# This is necessary to recreate the GeoMAE SST for use as a PointPillars feature extractor.

import functools
import inspect
from typing import Callable, Iterable, Mapping, Sequence, Optional, Any, Tuple, Union
import torch
import torch.nn as nn

try:
    from torch.cuda.amp import autocast
except Exception:  # pragma: no cover
    # Fallback no-op if AMP isn't available (CPU-only env, etc.)
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled: bool = True):
        yield


TensorLike = torch.Tensor
Container = Union[Mapping, Sequence]


def _is_container(x: Any) -> bool:
    return isinstance(x, Mapping) or (
        isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray))
    )


def cast_tensor_type(obj: Any, src_dtype: torch.dtype, dst_dtype: torch.dtype) -> Any:
    """
    Recursively cast floating point tensors of dtype==src_dtype to dst_dtype.
    Leaves non-tensors and non-float tensors untouched. Handles (nested) dict/list/tuple.
    """
    if isinstance(obj, TensorLike):
        # Only cast floats and only when it matches src_dtype exactly
        if obj.is_floating_point() and obj.dtype == src_dtype:
            return obj.to(dst_dtype)
        return obj
    if _is_container(obj):
        if isinstance(obj, Mapping):
            return type(obj)(
                (k, cast_tensor_type(v, src_dtype, dst_dtype)) for k, v in obj.items()
            )
        # For tuples, keep tuple type; for other sequences, reconstruct same type if possible
        casted = [cast_tensor_type(v, src_dtype, dst_dtype) for v in obj]
        return type(obj)(casted) if not isinstance(obj, tuple) else tuple(casted)
    return obj


def auto_fp16(
    apply_to: Optional[Iterable[str]] = None,
    out_fp32: bool = False,
    supported_types: Tuple[type, ...] = (nn.Module,),
) -> Callable:
    """
    Decorator to enable fp16 autocasting of specified arguments when the owning
    module has `self.fp16_enabled == True`.

    Args:
        apply_to: Iterable of argument names to cast (by name). If None, casts all args/kwargs (except `self`).
        out_fp32: If True, casts the function's output back to fp32.
        supported_types: Classes this decorator is allowed to wrap (default: nn.Module).
    """

    def auto_fp16_wrapper(old_func: Callable) -> Callable:
        sig = inspect.signature(old_func)

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # Ensure it's used to decorate a method of supported type
            if not args or not isinstance(args[0], supported_types):
                raise TypeError(
                    "@auto_fp16 can only decorate methods of {}".format(supported_types)
                )

            module_self = args[0]
            # If module doesn't have fp16_enabled or it's False, just run original
            if not (hasattr(module_self, "fp16_enabled") and module_self.fp16_enabled):
                return old_func(*args, **kwargs)

            # Bind args so we can modify by parameter names
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            # Determine which names to cast
            if apply_to is None:
                names_to_cast = [n for n in sig.parameters.keys() if n != "self"]
            else:
                names_to_cast = list(apply_to)

            # Cast chosen args/kwargs from fp32 -> fp16
            for name in names_to_cast:
                if name in bound.arguments:
                    bound.arguments[name] = cast_tensor_type(
                        bound.arguments[name], torch.float32, torch.float16
                    )

            # Run under autocast so down-stream ops can use Tensor Cores
            with autocast(enabled=True):
                out = old_func(*bound.args, **bound.kwargs)

            # Optionally cast output back to fp32 for numerical stability
            if out_fp32:
                out = cast_tensor_type(out, torch.float16, torch.float32)
            return out

        return new_func

    return auto_fp16_wrapper
