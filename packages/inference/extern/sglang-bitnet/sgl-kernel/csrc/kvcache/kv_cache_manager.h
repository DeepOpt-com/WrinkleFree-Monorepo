/**
 * KV Cache Manager for CPU inference.
 *
 * Provides high-performance KV cache management with:
 * - Page-based memory allocation
 * - AVX-512 optimized gather/scatter operations
 * - OpenMP parallelization
 *
 * References:
 * - vLLM paged attention: https://arxiv.org/abs/2309.06180
 * - SGLang memory management: sglang/srt/mem_cache/
 */

#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <mutex>

namespace sgl_kernel {
namespace kvcache {

/**
 * Configuration for KV cache.
 */
struct KVCacheConfig {
    int num_layers;       // Number of transformer layers
    int num_heads;        // Number of attention heads
    int head_dim;         // Dimension per head
    int page_size;        // Tokens per page (default: 256)
    int max_pages;        // Maximum number of pages
    bool use_fp16;        // Use FP16 storage (default: false = FP32)
};

/**
 * KV Cache Manager - manages paged KV cache for transformer inference.
 *
 * Thread-safe for concurrent allocation/deallocation.
 * Gather/scatter operations are parallelized with OpenMP.
 */
class KVCacheManager {
public:
    /**
     * Create a new KV cache manager.
     *
     * @param config Cache configuration
     */
    explicit KVCacheManager(const KVCacheConfig& config);

    ~KVCacheManager();

    // Disable copy
    KVCacheManager(const KVCacheManager&) = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;

    /**
     * Allocate a page from the cache.
     *
     * @return Page index, or -1 if no pages available
     */
    int allocate_page();

    /**
     * Allocate multiple pages.
     *
     * @param num_pages Number of pages to allocate
     * @return Tensor of page indices, empty if allocation failed
     */
    at::Tensor allocate_pages(int num_pages);

    /**
     * Free a page back to the cache.
     *
     * @param page_id Page index to free
     */
    void free_page(int page_id);

    /**
     * Free multiple pages.
     *
     * @param page_ids Tensor of page indices to free
     */
    void free_pages(const at::Tensor& page_ids);

    /**
     * Gather KV values from cache for given token indices.
     *
     * @param k_out Output tensor for keys [num_tokens, num_heads, head_dim]
     * @param v_out Output tensor for values [num_tokens, num_heads, head_dim]
     * @param page_indices Page indices for each token [num_tokens]
     * @param slot_indices Slot within page for each token [num_tokens]
     * @param layer_id Layer index
     */
    void gather_kv(
        at::Tensor& k_out,
        at::Tensor& v_out,
        const at::Tensor& page_indices,
        const at::Tensor& slot_indices,
        int layer_id
    );

    /**
     * Scatter KV values to cache for given token indices.
     *
     * @param k_in Input keys [num_tokens, num_heads, head_dim]
     * @param v_in Input values [num_tokens, num_heads, head_dim]
     * @param page_indices Page indices for each token [num_tokens]
     * @param slot_indices Slot within page for each token [num_tokens]
     * @param layer_id Layer index
     */
    void scatter_kv(
        const at::Tensor& k_in,
        const at::Tensor& v_in,
        const at::Tensor& page_indices,
        const at::Tensor& slot_indices,
        int layer_id
    );

    /**
     * Get number of free pages.
     */
    int num_free_pages() const;

    /**
     * Get total number of pages.
     */
    int num_total_pages() const { return config_.max_pages; }

    /**
     * Get configuration.
     */
    const KVCacheConfig& config() const { return config_; }

private:
    KVCacheConfig config_;

    // Page storage: [num_pages, num_layers, 2, num_heads, head_dim]
    // The "2" dimension is for K and V
    at::Tensor cache_storage_;

    // Free page list (thread-safe)
    std::vector<int> free_pages_;
    mutable std::mutex alloc_mutex_;

    // Cached raw pointer for fast access
    float* cache_ptr_;

    // Helper to compute offset into cache
    size_t compute_offset(int page_id, int layer_id, int kv_idx, int head_id, int dim_idx) const;

    // AVX-512 optimized gather/scatter (implemented in .cpp)
    void gather_kv_avx512(
        float* k_out, float* v_out,
        const int* page_indices, const int* slot_indices,
        int num_tokens, int layer_id
    );

    void scatter_kv_avx512(
        const float* k_in, const float* v_in,
        const int* page_indices, const int* slot_indices,
        int num_tokens, int layer_id
    );
};

/**
 * Global manager registry for Python bindings.
 * Returns handle (int) that can be used to reference the manager.
 */
int create_kv_cache_manager(const KVCacheConfig& config);
KVCacheManager* get_kv_cache_manager(int handle);
void destroy_kv_cache_manager(int handle);

}  // namespace kvcache
}  // namespace sgl_kernel
