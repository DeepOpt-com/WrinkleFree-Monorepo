"""GPU memory tracking utilities."""

import torch
from typing import Optional
from contextlib import contextmanager


class MemoryTracker:
    """GPU memory tracking utilities for benchmarking.

    Provides methods to track, reset, and query GPU memory usage
    during benchmark trials.
    """

    def __init__(self, device: Optional[torch.device] = None):
        """Initialize memory tracker.

        Args:
            device: CUDA device to track. Defaults to current device.
        """
        self.device = device or torch.device("cuda")
        self._snapshot_allocated = 0.0
        self._snapshot_reserved = 0.0

    @staticmethod
    def is_cuda_available() -> bool:
        """Check if CUDA is available."""
        return torch.cuda.is_available()

    def reset(self) -> None:
        """Reset memory statistics and clear cache.

        Call this before starting a benchmark trial to ensure
        clean memory measurements.
        """
        if not self.is_cuda_available():
            return

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)

    def get_allocated_gb(self) -> float:
        """Get currently allocated memory in GB.

        Returns:
            Currently allocated GPU memory in gigabytes.
        """
        if not self.is_cuda_available():
            return 0.0

        return torch.cuda.memory_allocated(self.device) / (1024 ** 3)

    def get_peak_gb(self) -> float:
        """Get peak allocated memory in GB.

        Returns:
            Peak GPU memory allocation since last reset, in gigabytes.
        """
        if not self.is_cuda_available():
            return 0.0

        return torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

    def get_reserved_gb(self) -> float:
        """Get reserved (cached) memory in GB.

        This is the total memory held by the caching allocator,
        which may be larger than currently allocated memory.

        Returns:
            Reserved GPU memory in gigabytes.
        """
        if not self.is_cuda_available():
            return 0.0

        return torch.cuda.memory_reserved(self.device) / (1024 ** 3)

    def get_free_gb(self) -> float:
        """Get free GPU memory in GB.

        Returns:
            Free GPU memory in gigabytes.
        """
        if not self.is_cuda_available():
            return 0.0

        free, total = torch.cuda.mem_get_info(self.device)
        return free / (1024 ** 3)

    def get_total_gb(self) -> float:
        """Get total GPU memory in GB.

        Returns:
            Total GPU memory in gigabytes.
        """
        if not self.is_cuda_available():
            return 0.0

        free, total = torch.cuda.mem_get_info(self.device)
        return total / (1024 ** 3)

    def snapshot(self) -> None:
        """Take a snapshot of current memory usage.

        Can be used to track memory changes between operations.
        """
        self._snapshot_allocated = self.get_allocated_gb()
        self._snapshot_reserved = self.get_reserved_gb()

    def get_delta_allocated_gb(self) -> float:
        """Get change in allocated memory since last snapshot.

        Returns:
            Change in allocated memory in GB since snapshot().
        """
        return self.get_allocated_gb() - self._snapshot_allocated

    def get_delta_reserved_gb(self) -> float:
        """Get change in reserved memory since last snapshot.

        Returns:
            Change in reserved memory in GB since snapshot().
        """
        return self.get_reserved_gb() - self._snapshot_reserved

    def get_memory_summary(self) -> dict:
        """Get comprehensive memory summary.

        Returns:
            Dictionary with all memory metrics.
        """
        return {
            "allocated_gb": self.get_allocated_gb(),
            "peak_gb": self.get_peak_gb(),
            "reserved_gb": self.get_reserved_gb(),
            "free_gb": self.get_free_gb(),
            "total_gb": self.get_total_gb(),
        }

    @contextmanager
    def track(self):
        """Context manager to track memory during a block.

        Yields a dictionary that will be populated with memory metrics
        after the block completes.

        Example:
            tracker = MemoryTracker()
            with tracker.track() as mem:
                model = create_model()
                # ... training ...
            print(f"Peak memory: {mem['peak_gb']:.2f} GB")
        """
        self.reset()
        result = {}
        try:
            yield result
        finally:
            result["allocated_gb"] = self.get_allocated_gb()
            result["peak_gb"] = self.get_peak_gb()
            result["reserved_gb"] = self.get_reserved_gb()

    def __str__(self) -> str:
        """Human-readable memory summary."""
        summary = self.get_memory_summary()
        return (
            f"GPU Memory: {summary['allocated_gb']:.2f} GB allocated, "
            f"{summary['peak_gb']:.2f} GB peak, "
            f"{summary['free_gb']:.2f} GB free / {summary['total_gb']:.2f} GB total"
        )


def get_memory_stats() -> dict:
    """Get current GPU memory statistics.

    Convenience function for quick memory checks.

    Returns:
        Dictionary with memory statistics in GB.
    """
    tracker = MemoryTracker()
    return tracker.get_memory_summary()


def clear_memory() -> None:
    """Clear GPU memory cache and reset statistics.

    Convenience function for memory cleanup between trials.
    """
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        except RuntimeError as e:
            # CUDA may be in bad state from killed process, try to recover
            if "CUDA" in str(e):
                import gc
                gc.collect()
                try:
                    # Reinitialize CUDA context
                    torch.cuda.init()
                    torch.cuda.empty_cache()
                except Exception:
                    pass  # Best effort cleanup
            else:
                raise
