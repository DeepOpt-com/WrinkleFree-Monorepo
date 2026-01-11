/**
 * KV Cache for BitNet Inference Engine
 *
 * Manages key-value cache for transformer attention layers.
 * Uses contiguous memory layout for cache-friendly access.
 */

#ifndef BITNET_KV_CACHE_H
#define BITNET_KV_CACHE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// KV cache struct (full definition for unique_ptr compatibility)
struct KVCache {
    int32_t num_layers;
    int32_t num_heads;
    int32_t head_dim;
    int32_t max_seq_len;
    int32_t current_seq_len;

    // Contiguous memory for all layers
    // Layout: [num_layers, 2 (k/v), max_seq_len, num_heads, head_dim]
    float* data;
    size_t layer_stride;  // Stride between layers
    size_t kv_stride;     // Stride between K and V within layer
    size_t pos_stride;    // Stride between positions
};

/**
 * Create KV cache for a model.
 *
 * @param num_layers Number of transformer layers
 * @param num_heads Number of attention heads
 * @param head_dim Dimension per head
 * @param max_seq_len Maximum sequence length
 * @return KV cache handle, or NULL on error
 */
KVCache* kv_cache_create(
    int32_t num_layers,
    int32_t num_heads,
    int32_t head_dim,
    int32_t max_seq_len
);

/**
 * Destroy KV cache and free memory.
 */
void kv_cache_destroy(KVCache* cache);

/**
 * Get pointer to key cache for a layer.
 *
 * @param cache KV cache handle
 * @param layer Layer index
 * @return Pointer to key cache [max_seq_len, num_heads, head_dim]
 */
float* kv_cache_get_key(KVCache* cache, int32_t layer);

/**
 * Get pointer to value cache for a layer.
 *
 * @param cache KV cache handle
 * @param layer Layer index
 * @return Pointer to value cache [max_seq_len, num_heads, head_dim]
 */
float* kv_cache_get_value(KVCache* cache, int32_t layer);

/**
 * Update key cache at position.
 *
 * @param cache KV cache handle
 * @param layer Layer index
 * @param position Sequence position
 * @param key Key tensor [num_heads, head_dim]
 */
void kv_cache_update_key(
    KVCache* cache,
    int32_t layer,
    int32_t position,
    const float* key
);

/**
 * Update value cache at position.
 *
 * @param cache KV cache handle
 * @param layer Layer index
 * @param position Sequence position
 * @param value Value tensor [num_heads, head_dim]
 */
void kv_cache_update_value(
    KVCache* cache,
    int32_t layer,
    int32_t position,
    const float* value
);

/**
 * Reset cache to empty state.
 */
void kv_cache_reset(KVCache* cache);

/**
 * Get current sequence length in cache.
 */
int32_t kv_cache_seq_len(KVCache* cache);

/**
 * Set current sequence length.
 */
void kv_cache_set_seq_len(KVCache* cache, int32_t len);

/**
 * Get total memory usage in bytes.
 */
size_t kv_cache_memory_usage(KVCache* cache);

#ifdef __cplusplus
}
#endif

#endif // BITNET_KV_CACHE_H
