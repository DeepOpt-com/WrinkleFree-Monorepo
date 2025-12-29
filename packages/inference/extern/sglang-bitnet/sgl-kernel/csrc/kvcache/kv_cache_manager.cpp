/**
 * KV Cache Manager implementation.
 *
 * Iteration 1: Basic skeleton with simple gather/scatter.
 * Later iterations will add AVX-512 optimization.
 */

#include "kv_cache_manager.h"

#include <stdexcept>
#include <algorithm>
#include <unordered_map>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sgl_kernel {
namespace kvcache {

// -----------------------------------------------------------------------------
// Global manager registry
// -----------------------------------------------------------------------------

static std::unordered_map<int, std::unique_ptr<KVCacheManager>> g_managers;
static std::mutex g_registry_mutex;
static int g_next_handle = 1;

int create_kv_cache_manager(const KVCacheConfig& config) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    int handle = g_next_handle++;
    g_managers[handle] = std::make_unique<KVCacheManager>(config);
    return handle;
}

KVCacheManager* get_kv_cache_manager(int handle) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    auto it = g_managers.find(handle);
    if (it == g_managers.end()) {
        return nullptr;
    }
    return it->second.get();
}

void destroy_kv_cache_manager(int handle) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    g_managers.erase(handle);
}

// -----------------------------------------------------------------------------
// KVCacheManager implementation
// -----------------------------------------------------------------------------

KVCacheManager::KVCacheManager(const KVCacheConfig& config)
    : config_(config), cache_ptr_(nullptr) {

    // Validate config
    if (config_.num_layers <= 0 || config_.num_heads <= 0 ||
        config_.head_dim <= 0 || config_.page_size <= 0 ||
        config_.max_pages <= 0) {
        throw std::invalid_argument("Invalid KVCacheConfig");
    }

    // Allocate cache storage: [max_pages, num_layers, 2, num_heads, head_dim]
    // 2 = K and V
    std::vector<int64_t> shape = {
        config_.max_pages,
        config_.num_layers,
        2,  // K, V
        config_.num_heads,
        config_.head_dim
    };

    auto options = at::TensorOptions()
        .dtype(config_.use_fp16 ? torch::kFloat16 : torch::kFloat32)
        .device(torch::kCPU);

    cache_storage_ = at::zeros(shape, options);
    cache_ptr_ = cache_storage_.data_ptr<float>();

    // Initialize free page list with all pages
    free_pages_.reserve(config_.max_pages);
    for (int i = config_.max_pages - 1; i >= 0; --i) {
        free_pages_.push_back(i);
    }
}

KVCacheManager::~KVCacheManager() {
    // Storage is automatically freed by PyTorch
}

int KVCacheManager::allocate_page() {
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    if (free_pages_.empty()) {
        return -1;
    }
    int page_id = free_pages_.back();
    free_pages_.pop_back();
    return page_id;
}

at::Tensor KVCacheManager::allocate_pages(int num_pages) {
    std::lock_guard<std::mutex> lock(alloc_mutex_);

    if (static_cast<int>(free_pages_.size()) < num_pages) {
        // Not enough pages
        return at::empty({0}, torch::kInt32);
    }

    auto result = at::empty({num_pages}, torch::kInt32);
    int* data = result.data_ptr<int>();

    for (int i = 0; i < num_pages; ++i) {
        data[i] = free_pages_.back();
        free_pages_.pop_back();
    }

    return result;
}

void KVCacheManager::free_page(int page_id) {
    if (page_id < 0 || page_id >= config_.max_pages) {
        throw std::out_of_range("Invalid page_id");
    }

    std::lock_guard<std::mutex> lock(alloc_mutex_);
    free_pages_.push_back(page_id);
}

void KVCacheManager::free_pages(const at::Tensor& page_ids) {
    auto accessor = page_ids.accessor<int, 1>();
    std::lock_guard<std::mutex> lock(alloc_mutex_);

    for (int64_t i = 0; i < page_ids.size(0); ++i) {
        int page_id = accessor[i];
        if (page_id >= 0 && page_id < config_.max_pages) {
            free_pages_.push_back(page_id);
        }
    }
}

int KVCacheManager::num_free_pages() const {
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    return static_cast<int>(free_pages_.size());
}

size_t KVCacheManager::compute_offset(
    int page_id, int layer_id, int kv_idx, int head_id, int dim_idx
) const {
    // Layout: [max_pages, num_layers, 2, num_heads, head_dim]
    return (((page_id * config_.num_layers + layer_id) * 2 + kv_idx) *
            config_.num_heads + head_id) * config_.head_dim + dim_idx;
}

void KVCacheManager::gather_kv(
    at::Tensor& k_out,
    at::Tensor& v_out,
    const at::Tensor& page_indices,
    const at::Tensor& slot_indices,
    int layer_id
) {
    int num_tokens = page_indices.size(0);

    // Validate inputs
    TORCH_CHECK(page_indices.device().is_cpu(), "page_indices must be on CPU");
    TORCH_CHECK(slot_indices.device().is_cpu(), "slot_indices must be on CPU");
    TORCH_CHECK(k_out.size(0) == num_tokens, "k_out size mismatch");
    TORCH_CHECK(v_out.size(0) == num_tokens, "v_out size mismatch");

    const int* page_ptr = page_indices.data_ptr<int>();
    const int* slot_ptr = slot_indices.data_ptr<int>();
    float* k_ptr = k_out.data_ptr<float>();
    float* v_ptr = v_out.data_ptr<float>();

    const int num_heads = config_.num_heads;
    const int head_dim = config_.head_dim;
    const int head_size = num_heads * head_dim;

#ifdef __AVX512F__
    // AVX-512 optimized path: process 16 floats per iteration
    const int vec_size = 16;  // AVX-512 = 512 bits = 16 floats

    #pragma omp parallel for
    for (int t = 0; t < num_tokens; ++t) {
        int page_id = page_ptr[t];
        // slot_ptr[t] unused in this version - page_id is the direct index

        // Compute cache offset for this token's K and V
        size_t base_offset = (page_id * config_.num_layers + layer_id) * 2 * head_size;
        const float* k_src = cache_ptr_ + base_offset;
        const float* v_src = cache_ptr_ + base_offset + head_size;
        float* k_dst = k_ptr + t * head_size;
        float* v_dst = v_ptr + t * head_size;

        // Vectorized copy with prefetching
        int i = 0;
        for (; i + vec_size <= head_size; i += vec_size) {
            // Prefetch next cache lines
            _mm_prefetch(reinterpret_cast<const char*>(k_src + i + 64), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(v_src + i + 64), _MM_HINT_T0);

            // Load 16 floats from K cache and store to output
            __m512 k_vec = _mm512_loadu_ps(k_src + i);
            _mm512_storeu_ps(k_dst + i, k_vec);

            // Load 16 floats from V cache and store to output
            __m512 v_vec = _mm512_loadu_ps(v_src + i);
            _mm512_storeu_ps(v_dst + i, v_vec);
        }

        // Handle remainder (if head_size not divisible by 16)
        for (; i < head_size; ++i) {
            k_dst[i] = k_src[i];
            v_dst[i] = v_src[i];
        }
    }
#else
    // Fallback: simple memcpy
    #pragma omp parallel for
    for (int t = 0; t < num_tokens; ++t) {
        int page_id = page_ptr[t];

        size_t base_offset = (page_id * config_.num_layers + layer_id) * 2 * head_size;
        size_t k_offset = base_offset;
        size_t v_offset = base_offset + head_size;

        std::memcpy(k_ptr + t * head_size, cache_ptr_ + k_offset, head_size * sizeof(float));
        std::memcpy(v_ptr + t * head_size, cache_ptr_ + v_offset, head_size * sizeof(float));
    }
#endif
}

void KVCacheManager::scatter_kv(
    const at::Tensor& k_in,
    const at::Tensor& v_in,
    const at::Tensor& page_indices,
    const at::Tensor& slot_indices,
    int layer_id
) {
    int num_tokens = page_indices.size(0);

    // Validate inputs
    TORCH_CHECK(page_indices.device().is_cpu(), "page_indices must be on CPU");
    TORCH_CHECK(slot_indices.device().is_cpu(), "slot_indices must be on CPU");
    TORCH_CHECK(k_in.size(0) == num_tokens, "k_in size mismatch");
    TORCH_CHECK(v_in.size(0) == num_tokens, "v_in size mismatch");

    const int* page_ptr = page_indices.data_ptr<int>();
    const int* slot_ptr = slot_indices.data_ptr<int>();
    const float* k_ptr = k_in.data_ptr<float>();
    const float* v_ptr = v_in.data_ptr<float>();

    const int num_heads = config_.num_heads;
    const int head_dim = config_.head_dim;
    const int head_size = num_heads * head_dim;

#ifdef __AVX512F__
    // AVX-512 optimized scatter: process 16 floats per iteration
    const int vec_size = 16;

    #pragma omp parallel for
    for (int t = 0; t < num_tokens; ++t) {
        int page_id = page_ptr[t];
        // slot_ptr[t] unused in this version

        size_t base_offset = (page_id * config_.num_layers + layer_id) * 2 * head_size;
        float* k_dst = cache_ptr_ + base_offset;
        float* v_dst = cache_ptr_ + base_offset + head_size;
        const float* k_src = k_ptr + t * head_size;
        const float* v_src = v_ptr + t * head_size;

        // Vectorized scatter with non-temporal stores for better cache behavior
        int i = 0;
        for (; i + vec_size <= head_size; i += vec_size) {
            // Load from input tensors
            __m512 k_vec = _mm512_loadu_ps(k_src + i);
            __m512 v_vec = _mm512_loadu_ps(v_src + i);

            // Store to cache (non-temporal to avoid polluting cache)
            _mm512_storeu_ps(k_dst + i, k_vec);
            _mm512_storeu_ps(v_dst + i, v_vec);
        }

        // Handle remainder
        for (; i < head_size; ++i) {
            k_dst[i] = k_src[i];
            v_dst[i] = v_src[i];
        }
    }
#else
    // Fallback: simple memcpy
    #pragma omp parallel for
    for (int t = 0; t < num_tokens; ++t) {
        int page_id = page_ptr[t];

        size_t base_offset = (page_id * config_.num_layers + layer_id) * 2 * head_size;
        size_t k_offset = base_offset;
        size_t v_offset = base_offset + head_size;

        std::memcpy(cache_ptr_ + k_offset, k_ptr + t * head_size, head_size * sizeof(float));
        std::memcpy(cache_ptr_ + v_offset, v_ptr + t * head_size, head_size * sizeof(float));
    }
#endif
}

// Note: AVX-512 is now integrated directly into gather_kv() and scatter_kv()
// via compile-time #ifdef __AVX512F__. These private methods are kept for
// future use if we need raw-pointer interfaces.
void KVCacheManager::gather_kv_avx512(
    float* k_out, float* v_out,
    const int* page_indices, const int* slot_indices,
    int num_tokens, int layer_id
) {
    // Not used - AVX-512 is in gather_kv()
    (void)k_out; (void)v_out; (void)page_indices;
    (void)slot_indices; (void)num_tokens; (void)layer_id;
}

void KVCacheManager::scatter_kv_avx512(
    const float* k_in, const float* v_in,
    const int* page_indices, const int* slot_indices,
    int num_tokens, int layer_id
) {
    // Not used - AVX-512 is in scatter_kv()
    (void)k_in; (void)v_in; (void)page_indices;
    (void)slot_indices; (void)num_tokens; (void)layer_id;
}

}  // namespace kvcache
}  // namespace sgl_kernel
