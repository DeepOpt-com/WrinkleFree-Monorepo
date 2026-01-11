"""GPU utilities for automatic batch size detection and OOM handling.

Two approaches supported:
1. Estimation-based: estimate_starting_batch_size() for initial guess
2. Dynamic: find_executable_batch_size from Accelerate for automatic retry

The estimation approach is faster (no retries) but may need tuning.
The dynamic approach always works but may waste time on failed attempts.
"""

import gc
import logging
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Generator, TypeVar

import torch

logger = logging.getLogger(__name__)

# Re-export from accelerate for convenience
try:
    from accelerate.utils import find_executable_batch_size
except ImportError:
    # Fallback if accelerate not installed
    def find_executable_batch_size(function: Callable = None, starting_batch_size: int = 128):
        """Fallback decorator that just runs with starting batch size."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(starting_batch_size, *args, **kwargs)
            return wrapper

        if function is not None:
            return decorator(function)
        return decorator


def get_gpu_memory_gb() -> float:
    """Get GPU VRAM in GB."""
    if not torch.cuda.is_available():
        return 0.0
    total_memory = torch.cuda.get_device_properties(0).total_memory
    return total_memory / (1024**3)


def get_gpu_name() -> str:
    """Get GPU name for logging."""
    if not torch.cuda.is_available():
        return "CPU"
    return torch.cuda.get_device_name(0)


def estimate_starting_batch_size(model_name: str, stage: str) -> int:
    """Estimate a good starting batch size for find_executable_batch_size.

    This is the MAXIMUM batch size to try first. The decorator will
    automatically reduce it on OOM.

    Args:
        model_name: Model config name (e.g., "qwen3_4b", "smollm2_135m")
        stage: Training stage (e.g., "stage1_9", "stage2")

    Returns:
        Starting batch size (will be reduced on OOM)
    """
    vram_gb = get_gpu_memory_gb()

    # Estimate based on VRAM and model size
    # These are aggressive starting points - decorator will reduce on OOM

    # Detect if this is a distillation stage (loads teacher + student = 2x memory)
    stage_lower = stage.lower()
    is_distillation = any(kw in stage_lower for kw in [
        "stage1_9", "1.9", "layerwise", "distillation"
    ])

    if "smollm" in model_name.lower() or "135m" in model_name.lower():
        # Small model - can use large batches
        if is_distillation:
            # Still need room for teacher model
            if vram_gb >= 70:
                return 64
            elif vram_gb >= 35:
                return 32
            elif vram_gb >= 20:
                return 16
            else:
                return 8
        else:
            if vram_gb >= 70:
                return 128
            elif vram_gb >= 35:
                return 64
            elif vram_gb >= 20:
                return 32
            else:
                return 16
    elif "4b" in model_name.lower() or "qwen3" in model_name.lower():
        # 4B model - conservative for single GPU training
        # Qwen3-4B is ~16GB in BF16, optimizer states add ~32-48GB
        # Total: ~64-80GB, leaving minimal headroom on H100
        if is_distillation:
            # Stage 1.9/distillation loads teacher + student = 2x memory
            # Qwen3-4B: ~16GB per model × 2 = 32GB base
            # Plus optimizer states, activations, gradients - very memory heavy
            if vram_gb >= 70:
                return 2  # Very conservative for 80GB
            elif vram_gb >= 35:
                return 1
            else:
                return 1
        else:
            # Stage 2/3 - only student model
            # batch=32 OOMs on single H100, batch=8 is safe with influence updates
            if vram_gb >= 70:
                return 8  # Reduced from 32 - single GPU needs headroom for influence
            elif vram_gb >= 35:
                return 4
            else:
                return 2
    else:
        # Unknown model - be conservative
        return 8


def log_gpu_info() -> None:
    """Log GPU information at startup."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available!")
        return

    gpu_name = get_gpu_name()
    vram_gb = get_gpu_memory_gb()
    logger.info(f"GPU: {gpu_name} ({vram_gb:.1f}GB VRAM)")


def clear_cuda_cache() -> None:
    """Clear CUDA cache to free memory before batch size search."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@contextmanager
def oom_retry_context(
    max_retries: int = 3,
    batch_size_ref: list[int] | None = None,
) -> Generator[None, None, None]:
    """Context manager for graceful OOM handling.

    On OOM, clears CUDA cache and optionally halves batch size.

    Args:
        max_retries: Maximum retry attempts
        batch_size_ref: Mutable list [batch_size] to update on retry.
                       If provided, batch size is halved on each retry.

    Usage:
        batch_size = [16]  # Mutable reference
        for attempt in range(max_retries):
            with oom_retry_context(batch_size_ref=batch_size):
                train_step(batch_size[0])
                break  # Success

    Or simpler (just clears cache on OOM):
        with oom_retry_context():
            train_step()
    """
    try:
        yield
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(f"CUDA OOM detected, clearing cache...")

            # Clear memory
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Optionally reduce batch size
            if batch_size_ref is not None and len(batch_size_ref) > 0:
                old_size = batch_size_ref[0]
                batch_size_ref[0] = max(1, old_size // 2)
                logger.warning(f"Reducing batch size: {old_size} -> {batch_size_ref[0]}")

            raise  # Re-raise for caller to handle retry
        else:
            raise  # Non-OOM error


def probe_batch_size(
    probe_fn: Callable[[int], None],
    starting_batch_size: int,
    min_batch_size: int = 1,
    reduction_factor: float = 0.5,
) -> int:
    """Find the maximum batch size that fits in memory.

    Uses halving approach: multiply by reduction_factor (0.5) on OOM.
    Faster than fine-grained search - finds optimal in 1-3 tries.

    Run this BEFORE training to find the optimal batch size.

    Args:
        probe_fn: Function that takes batch_size and runs a few steps.
                  Should raise RuntimeError on OOM.
        starting_batch_size: Initial batch size to try
        min_batch_size: Minimum batch size before giving up
        reduction_factor: Multiply batch by this on OOM (default 0.9)

    Returns:
        The largest batch size that works

    Example:
        def probe(batch_size):
            loader = DataLoader(dataset, batch_size=batch_size)
            batch = next(iter(loader))
            loss = model(batch)
            loss.backward()
            torch.cuda.synchronize()

        optimal_batch = probe_batch_size(probe, starting_batch_size=32)
    """
    batch_size = starting_batch_size

    while batch_size >= min_batch_size:
        try:
            logger.info(f"Probing batch_size={batch_size}...")

            # Clear cache before probe
            gc.collect()
            torch.cuda.empty_cache()

            probe_fn(batch_size)

            # Success! Leave some headroom (95% of max)
            safe_batch = max(min_batch_size, int(batch_size * 0.95))
            logger.info(f"✓ Batch size {batch_size} works, using {safe_batch} with headroom")
            return safe_batch

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                new_batch = max(min_batch_size, int(batch_size * reduction_factor))
                if new_batch == batch_size:
                    new_batch = batch_size - 1  # Ensure progress
                logger.warning(f"OOM at batch_size={batch_size}, trying {new_batch}")
                batch_size = new_batch

                # Clear memory for retry
                gc.collect()
                torch.cuda.empty_cache()
            else:
                raise

    raise RuntimeError(f"Even batch_size={min_batch_size} causes OOM!")


__all__ = [
    # Re-exported from Accelerate
    "find_executable_batch_size",
    # GPU info
    "get_gpu_memory_gb",
    "get_gpu_name",
    "log_gpu_info",
    # Batch size detection
    "estimate_starting_batch_size",  # Fast heuristic
    "probe_batch_size",  # Dynamic probing (Accelerate-style)
    # OOM handling
    "oom_retry_context",
    "clear_cuda_cache",
]
