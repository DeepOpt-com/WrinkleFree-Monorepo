"""Test KV Cache C++ implementation vs Python reference.

This test file validates that C++ KV cache operations match Python reference
implementations. Tests are added incrementally as C++ features are implemented.

Iteration 1: Basic allocation tests (Python reference only)
Iteration 2-3: Gather/scatter tests (after C++ implementation)
Iteration 4+: Full integration tests (after Python bindings)
"""

import pytest
import torch
import numpy as np


# =============================================================================
# Python Reference Implementations
# =============================================================================

class PythonKVCacheReference:
    """Pure Python KV cache reference implementation for correctness testing."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int,
        max_pages: int,
        dtype: torch.dtype = torch.float32,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages
        self.dtype = dtype

        # Storage: [max_pages, num_layers, 2, num_heads, head_dim]
        # 2 = K and V
        self.cache_storage = torch.zeros(
            max_pages, num_layers, 2, num_heads, head_dim, dtype=dtype
        )

        # Free page list
        self._free_pages = list(range(max_pages - 1, -1, -1))

    def allocate_page(self) -> int:
        """Allocate a page. Returns -1 if no pages available."""
        if not self._free_pages:
            return -1
        return self._free_pages.pop()

    def allocate_pages(self, num_pages: int) -> torch.Tensor:
        """Allocate multiple pages. Returns empty tensor if not enough."""
        if len(self._free_pages) < num_pages:
            return torch.empty(0, dtype=torch.int32)
        pages = [self._free_pages.pop() for _ in range(num_pages)]
        return torch.tensor(pages, dtype=torch.int32)

    def free_page(self, page_id: int):
        """Return a page to the free list."""
        if 0 <= page_id < self.max_pages:
            self._free_pages.append(page_id)

    def free_pages_batch(self, page_ids: torch.Tensor):
        """Return multiple pages to the free list."""
        for page_id in page_ids.tolist():
            self.free_page(page_id)

    def num_free_pages(self) -> int:
        """Get number of free pages."""
        return len(self._free_pages)

    def gather_kv(
        self,
        page_indices: torch.Tensor,
        slot_indices: torch.Tensor,
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K and V from cache.

        Args:
            page_indices: [num_tokens] page indices
            slot_indices: [num_tokens] slot within page (unused in Iter 1)
            layer_id: Layer index

        Returns:
            k_out: [num_tokens, num_heads, head_dim]
            v_out: [num_tokens, num_heads, head_dim]
        """
        num_tokens = page_indices.shape[0]
        k_out = torch.zeros(num_tokens, self.num_heads, self.head_dim, dtype=self.dtype)
        v_out = torch.zeros(num_tokens, self.num_heads, self.head_dim, dtype=self.dtype)

        for t in range(num_tokens):
            page_id = page_indices[t].item()
            # K = cache[page, layer, 0, :, :]
            # V = cache[page, layer, 1, :, :]
            k_out[t] = self.cache_storage[page_id, layer_id, 0]
            v_out[t] = self.cache_storage[page_id, layer_id, 1]

        return k_out, v_out

    def scatter_kv(
        self,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        page_indices: torch.Tensor,
        slot_indices: torch.Tensor,
        layer_id: int,
    ):
        """Scatter K and V to cache.

        Args:
            k_in: [num_tokens, num_heads, head_dim]
            v_in: [num_tokens, num_heads, head_dim]
            page_indices: [num_tokens] page indices
            slot_indices: [num_tokens] slot within page (unused in Iter 1)
            layer_id: Layer index
        """
        num_tokens = page_indices.shape[0]

        for t in range(num_tokens):
            page_id = page_indices[t].item()
            self.cache_storage[page_id, layer_id, 0] = k_in[t]
            self.cache_storage[page_id, layer_id, 1] = v_in[t]


# =============================================================================
# Test Configuration
# =============================================================================

# BitNet-b1.58-2B-4T dimensions
BITNET_CONFIG = {
    "num_layers": 32,
    "num_heads": 20,
    "head_dim": 128,
    "page_size": 256,
    "max_pages": 100,
}


# =============================================================================
# Iteration 1 Tests: Basic Allocation
# =============================================================================

class TestPythonReference:
    """Test Python reference implementation (used as ground truth)."""

    def test_init(self):
        """Test cache initialization."""
        cache = PythonKVCacheReference(**BITNET_CONFIG)
        assert cache.num_free_pages() == BITNET_CONFIG["max_pages"]
        assert cache.cache_storage.shape == (
            BITNET_CONFIG["max_pages"],
            BITNET_CONFIG["num_layers"],
            2,  # K, V
            BITNET_CONFIG["num_heads"],
            BITNET_CONFIG["head_dim"],
        )

    def test_allocate_single(self):
        """Test single page allocation."""
        cache = PythonKVCacheReference(**BITNET_CONFIG)
        page1 = cache.allocate_page()
        assert 0 <= page1 < BITNET_CONFIG["max_pages"]
        assert cache.num_free_pages() == BITNET_CONFIG["max_pages"] - 1

        page2 = cache.allocate_page()
        assert page2 != page1
        assert cache.num_free_pages() == BITNET_CONFIG["max_pages"] - 2

    def test_allocate_batch(self):
        """Test batch page allocation."""
        cache = PythonKVCacheReference(**BITNET_CONFIG)
        pages = cache.allocate_pages(10)
        assert pages.shape == (10,)
        assert cache.num_free_pages() == BITNET_CONFIG["max_pages"] - 10

        # All pages should be unique
        assert len(set(pages.tolist())) == 10

    def test_allocate_too_many(self):
        """Test allocation failure when not enough pages."""
        cache = PythonKVCacheReference(**BITNET_CONFIG)
        # Allocate all pages
        all_pages = cache.allocate_pages(BITNET_CONFIG["max_pages"])
        assert all_pages.shape == (BITNET_CONFIG["max_pages"],)

        # Try to allocate more
        more_pages = cache.allocate_pages(1)
        assert more_pages.shape == (0,)

        single = cache.allocate_page()
        assert single == -1

    def test_free_pages(self):
        """Test page deallocation."""
        cache = PythonKVCacheReference(**BITNET_CONFIG)
        pages = cache.allocate_pages(5)
        assert cache.num_free_pages() == BITNET_CONFIG["max_pages"] - 5

        cache.free_pages_batch(pages)
        assert cache.num_free_pages() == BITNET_CONFIG["max_pages"]


class TestGatherScatterReference:
    """Test gather/scatter in Python reference."""

    def test_scatter_then_gather(self):
        """Test roundtrip: scatter then gather."""
        cache = PythonKVCacheReference(**BITNET_CONFIG)

        # Allocate some pages
        num_tokens = 8
        pages = cache.allocate_pages(num_tokens)
        slots = torch.zeros(num_tokens, dtype=torch.int32)  # Not used in Iter 1
        layer_id = 5

        # Create random K/V
        k_in = torch.randn(num_tokens, BITNET_CONFIG["num_heads"], BITNET_CONFIG["head_dim"])
        v_in = torch.randn(num_tokens, BITNET_CONFIG["num_heads"], BITNET_CONFIG["head_dim"])

        # Scatter to cache
        cache.scatter_kv(k_in, v_in, pages, slots, layer_id)

        # Gather back
        k_out, v_out = cache.gather_kv(pages, slots, layer_id)

        # Should match
        assert torch.allclose(k_out, k_in, atol=1e-5)
        assert torch.allclose(v_out, v_in, atol=1e-5)

    def test_gather_different_layers(self):
        """Test that different layers are independent."""
        cache = PythonKVCacheReference(**BITNET_CONFIG)

        num_tokens = 4
        pages = cache.allocate_pages(num_tokens)
        slots = torch.zeros(num_tokens, dtype=torch.int32)

        # Scatter to layer 0
        k0 = torch.randn(num_tokens, BITNET_CONFIG["num_heads"], BITNET_CONFIG["head_dim"])
        v0 = torch.randn(num_tokens, BITNET_CONFIG["num_heads"], BITNET_CONFIG["head_dim"])
        cache.scatter_kv(k0, v0, pages, slots, layer_id=0)

        # Scatter different values to layer 1
        k1 = torch.randn(num_tokens, BITNET_CONFIG["num_heads"], BITNET_CONFIG["head_dim"])
        v1 = torch.randn(num_tokens, BITNET_CONFIG["num_heads"], BITNET_CONFIG["head_dim"])
        cache.scatter_kv(k1, v1, pages, slots, layer_id=1)

        # Gather from both layers
        k0_out, v0_out = cache.gather_kv(pages, slots, layer_id=0)
        k1_out, v1_out = cache.gather_kv(pages, slots, layer_id=1)

        # Layer 0 should match original layer 0 data
        assert torch.allclose(k0_out, k0, atol=1e-5)
        assert torch.allclose(v0_out, v0, atol=1e-5)

        # Layer 1 should match original layer 1 data
        assert torch.allclose(k1_out, k1, atol=1e-5)
        assert torch.allclose(v1_out, v1, atol=1e-5)


# =============================================================================
# Iteration 4+ Tests: C++ vs Python Comparison
# =============================================================================

def _cpp_available():
    """Check if C++ KV cache bindings are available."""
    try:
        from sgl_kernel.kvcache import KVCacheManager, check_kernel_available
        return check_kernel_available()
    except ImportError:
        return False


@pytest.mark.skipif(not _cpp_available(), reason="C++ KV cache bindings not available")
class TestCppVsPython:
    """Compare C++ implementation against Python reference."""

    def test_allocation_cpp_vs_python(self):
        """Test C++ allocation matches Python behavior."""
        from sgl_kernel.kvcache import KVCacheManager

        # Create both implementations
        py_cache = PythonKVCacheReference(**BITNET_CONFIG)
        cpp_cache = KVCacheManager(**BITNET_CONFIG)

        # Test initial state
        assert py_cache.num_free_pages() == cpp_cache.num_free_pages()

        # Allocate same number of pages
        py_pages = py_cache.allocate_pages(10)
        cpp_pages = cpp_cache.allocate_pages(10)

        assert py_pages.shape == cpp_pages.shape
        assert py_cache.num_free_pages() == cpp_cache.num_free_pages()

        # Free pages and check count
        py_cache.free_pages_batch(py_pages)
        cpp_cache.free_pages(cpp_pages)
        assert py_cache.num_free_pages() == cpp_cache.num_free_pages()

    def test_gather_scatter_roundtrip_cpp_vs_python(self):
        """Test C++ scatter→gather roundtrip matches Python reference."""
        from sgl_kernel.kvcache import KVCacheManager

        num_tokens = 16
        py_cache = PythonKVCacheReference(**BITNET_CONFIG)
        cpp_cache = KVCacheManager(**BITNET_CONFIG)

        # Allocate pages
        py_pages = py_cache.allocate_pages(num_tokens)
        cpp_pages = cpp_cache.allocate_pages(num_tokens)
        slots = torch.zeros(num_tokens, dtype=torch.int32)
        layer_id = 5

        # Create random K/V tensors
        k_in = torch.randn(num_tokens, BITNET_CONFIG["num_heads"], BITNET_CONFIG["head_dim"])
        v_in = torch.randn(num_tokens, BITNET_CONFIG["num_heads"], BITNET_CONFIG["head_dim"])

        # Python scatter & gather
        py_cache.scatter_kv(k_in, v_in, py_pages, slots, layer_id)
        py_k_out, py_v_out = py_cache.gather_kv(py_pages, slots, layer_id)

        # C++ scatter & gather (reshape for C++ interface)
        k_in_flat = k_in.reshape(num_tokens, -1)
        v_in_flat = v_in.reshape(num_tokens, -1)
        cpp_k_out = torch.zeros_like(k_in_flat)
        cpp_v_out = torch.zeros_like(v_in_flat)

        cpp_cache.scatter_kv(k_in_flat, v_in_flat, cpp_pages, slots, layer_id)
        cpp_cache.gather_kv(cpp_k_out, cpp_v_out, cpp_pages, slots, layer_id)

        # Reshape C++ output for comparison
        cpp_k_out = cpp_k_out.reshape(num_tokens, BITNET_CONFIG["num_heads"], BITNET_CONFIG["head_dim"])
        cpp_v_out = cpp_v_out.reshape(num_tokens, BITNET_CONFIG["num_heads"], BITNET_CONFIG["head_dim"])

        # Compare outputs
        assert torch.allclose(py_k_out, k_in, atol=1e-5), "Python K roundtrip failed"
        assert torch.allclose(py_v_out, v_in, atol=1e-5), "Python V roundtrip failed"
        assert torch.allclose(cpp_k_out, k_in, atol=1e-5), "C++ K roundtrip failed"
        assert torch.allclose(cpp_v_out, v_in, atol=1e-5), "C++ V roundtrip failed"

    def test_multiple_layers_cpp_vs_python(self):
        """Test C++ handles multiple layers correctly."""
        from sgl_kernel.kvcache import KVCacheManager

        num_tokens = 8
        cpp_cache = KVCacheManager(**BITNET_CONFIG)
        pages = cpp_cache.allocate_pages(num_tokens)
        slots = torch.zeros(num_tokens, dtype=torch.int32)

        # Scatter different data to different layers
        k_layers = []
        v_layers = []
        for layer_id in range(BITNET_CONFIG["num_layers"]):
            k = torch.randn(num_tokens, BITNET_CONFIG["num_heads"] * BITNET_CONFIG["head_dim"])
            v = torch.randn(num_tokens, BITNET_CONFIG["num_heads"] * BITNET_CONFIG["head_dim"])
            k_layers.append(k)
            v_layers.append(v)
            cpp_cache.scatter_kv(k, v, pages, slots, layer_id)

        # Gather from each layer and verify
        for layer_id in range(BITNET_CONFIG["num_layers"]):
            k_out = torch.zeros_like(k_layers[0])
            v_out = torch.zeros_like(v_layers[0])
            cpp_cache.gather_kv(k_out, v_out, pages, slots, layer_id)

            assert torch.allclose(k_out, k_layers[layer_id], atol=1e-5), f"Layer {layer_id} K mismatch"
            assert torch.allclose(v_out, v_layers[layer_id], atol=1e-5), f"Layer {layer_id} V mismatch"
