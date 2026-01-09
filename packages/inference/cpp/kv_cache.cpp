/**
 * KV Cache Implementation
 */

#include "kv_cache.h"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <new>  // for std::nothrow

extern "C" {

KVCache* kv_cache_create(
    int32_t num_layers,
    int32_t num_heads,
    int32_t head_dim,
    int32_t max_seq_len
) {
    KVCache* cache = new (std::nothrow) KVCache;
    if (!cache) return nullptr;

    cache->num_layers = num_layers;
    cache->num_heads = num_heads;
    cache->head_dim = head_dim;
    cache->max_seq_len = max_seq_len;
    cache->current_seq_len = 0;

    // Calculate strides
    cache->pos_stride = num_heads * head_dim;
    cache->kv_stride = max_seq_len * cache->pos_stride;
    cache->layer_stride = 2 * cache->kv_stride;

    // Allocate contiguous memory
    size_t total_size = num_layers * cache->layer_stride;
    cache->data = static_cast<float*>(aligned_alloc(64, total_size * sizeof(float)));
    if (!cache->data) {
        delete cache;
        return nullptr;
    }

    // Zero initialize
    std::memset(cache->data, 0, total_size * sizeof(float));

    return cache;
}

void kv_cache_destroy(KVCache* cache) {
    if (cache) {
        free(cache->data);
        delete cache;
    }
}

float* kv_cache_get_key(KVCache* cache, int32_t layer) {
    return cache->data + layer * cache->layer_stride;
}

float* kv_cache_get_value(KVCache* cache, int32_t layer) {
    return cache->data + layer * cache->layer_stride + cache->kv_stride;
}

void kv_cache_update_key(
    KVCache* cache,
    int32_t layer,
    int32_t position,
    const float* key
) {
    float* dst = kv_cache_get_key(cache, layer) + position * cache->pos_stride;
    std::memcpy(dst, key, cache->pos_stride * sizeof(float));
}

void kv_cache_update_value(
    KVCache* cache,
    int32_t layer,
    int32_t position,
    const float* value
) {
    float* dst = kv_cache_get_value(cache, layer) + position * cache->pos_stride;
    std::memcpy(dst, value, cache->pos_stride * sizeof(float));
}

void kv_cache_reset(KVCache* cache) {
    cache->current_seq_len = 0;
    // Note: We don't zero the memory for performance - it will be overwritten
}

int32_t kv_cache_seq_len(KVCache* cache) {
    return cache->current_seq_len;
}

void kv_cache_set_seq_len(KVCache* cache, int32_t len) {
    cache->current_seq_len = std::min(len, cache->max_seq_len);
}

size_t kv_cache_memory_usage(KVCache* cache) {
    return cache->num_layers * cache->layer_stride * sizeof(float);
}

} // extern "C"
