"""KV Cache management for sgl-kernel.

Provides Python bindings for CPU-optimized KV cache operations:
- Page-based allocation for vLLM-style paged attention
- AVX-512 optimized gather/scatter kernels
- Thread-safe allocation with OpenMP parallelization

Uses torch extension for native kernel calls.
"""

import torch
from typing import Optional, Tuple


# Kernel availability flag
_kernel_available = False


def _check_kernel_available() -> bool:
    """Check if KV cache kernels are available via torch ops."""
    global _kernel_available
    try:
        import sgl_kernel
        if hasattr(torch.ops.sgl_kernel, 'kv_cache_create'):
            _kernel_available = True
            return True
    except (ImportError, AttributeError):
        pass
    return False


# Check on import
_check_kernel_available()


def check_kernel_available() -> bool:
    """Check if KV cache kernels are available.

    Returns:
        True if native kernels are available, False otherwise.
    """
    return _kernel_available


class KVCacheManager:
    """Manages KV cache for paged attention.

    This class wraps the C++ KVCacheManager and provides a Python interface
    for page allocation and KV gather/scatter operations.

    Attributes:
        handle: C++ manager handle (opaque int)
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension per head
        page_size: Tokens per page
        max_pages: Maximum number of pages
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int,
        max_pages: int,
        use_fp16: bool = False,
    ):
        """Initialize KV cache manager.

        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension per head
            page_size: Tokens per page (typically 256)
            max_pages: Maximum number of pages to allocate
            use_fp16: Use FP16 storage (default: FP32)
        """
        if not _kernel_available:
            raise RuntimeError(
                "KV cache kernels not available. "
                "Ensure sgl_kernel is properly installed."
            )

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages
        self.use_fp16 = use_fp16

        self.handle = torch.ops.sgl_kernel.kv_cache_create(
            num_layers, num_heads, head_dim, page_size, max_pages, use_fp16
        )

    def __del__(self):
        """Destroy the cache manager."""
        if hasattr(self, 'handle') and self.handle is not None:
            try:
                torch.ops.sgl_kernel.kv_cache_destroy(self.handle)
            except Exception:
                pass  # Ignore errors during cleanup

    def allocate_page(self) -> int:
        """Allocate a single page.

        Returns:
            Page ID, or -1 if no pages available.
        """
        return torch.ops.sgl_kernel.kv_cache_allocate_page(self.handle)

    def allocate_pages(self, num_pages: int) -> torch.Tensor:
        """Allocate multiple pages.

        Args:
            num_pages: Number of pages to allocate

        Returns:
            Tensor of page IDs [num_pages], or empty tensor if not enough pages.
        """
        return torch.ops.sgl_kernel.kv_cache_allocate_pages(self.handle, num_pages)

    def free_page(self, page_id: int) -> None:
        """Free a single page.

        Args:
            page_id: Page ID to free
        """
        torch.ops.sgl_kernel.kv_cache_free_page(self.handle, page_id)

    def free_pages(self, page_ids: torch.Tensor) -> None:
        """Free multiple pages.

        Args:
            page_ids: Tensor of page IDs to free
        """
        torch.ops.sgl_kernel.kv_cache_free_pages(self.handle, page_ids)

    def num_free_pages(self) -> int:
        """Get number of free pages.

        Returns:
            Number of available pages.
        """
        return torch.ops.sgl_kernel.kv_cache_num_free_pages(self.handle)

    def gather_kv(
        self,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
        page_indices: torch.Tensor,
        slot_indices: torch.Tensor,
        layer_id: int,
    ) -> None:
        """Gather K and V from cache.

        Args:
            k_out: Output K tensor [num_tokens, num_heads * head_dim]
            v_out: Output V tensor [num_tokens, num_heads * head_dim]
            page_indices: Page indices [num_tokens]
            slot_indices: Slot indices within pages [num_tokens]
            layer_id: Layer index
        """
        torch.ops.sgl_kernel.kv_cache_gather(
            self.handle, k_out, v_out, page_indices, slot_indices, layer_id
        )

    def scatter_kv(
        self,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        page_indices: torch.Tensor,
        slot_indices: torch.Tensor,
        layer_id: int,
    ) -> None:
        """Scatter K and V to cache.

        Args:
            k_in: Input K tensor [num_tokens, num_heads * head_dim]
            v_in: Input V tensor [num_tokens, num_heads * head_dim]
            page_indices: Page indices [num_tokens]
            slot_indices: Slot indices within pages [num_tokens]
            layer_id: Layer index
        """
        torch.ops.sgl_kernel.kv_cache_scatter(
            self.handle, k_in, v_in, page_indices, slot_indices, layer_id
        )


# Export public API
__all__ = [
    "check_kernel_available",
    "KVCacheManager",
]
