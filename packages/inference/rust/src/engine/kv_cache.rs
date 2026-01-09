//! KV cache for transformer inference.
//!
//! Stores key and value projections from attention layers to avoid
//! recomputation during autoregressive decoding.

/// KV cache configuration
#[derive(Debug, Clone)]
pub struct KVCacheConfig {
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of key-value heads
    pub num_kv_heads: usize,
    /// Dimension per head (head_dim = hidden_size / num_heads)
    pub head_dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
}

/// Single layer's KV cache
pub struct LayerKVCache {
    /// Key cache: [max_seq_len, num_kv_heads, head_dim]
    pub keys: Vec<f32>,
    /// Value cache: [max_seq_len, num_kv_heads, head_dim]
    pub values: Vec<f32>,
    /// Current sequence length (number of cached positions)
    pub seq_len: usize,
    /// Configuration
    config: LayerKVConfig,
}

struct LayerKVConfig {
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
}

impl LayerKVCache {
    /// Create a new layer KV cache.
    pub fn new(num_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let cache_size = max_seq_len * num_kv_heads * head_dim;
        Self {
            keys: vec![0.0; cache_size],
            values: vec![0.0; cache_size],
            seq_len: 0,
            config: LayerKVConfig {
                num_kv_heads,
                head_dim,
                max_seq_len,
            },
        }
    }

    /// Update cache with new key/value at the given position.
    ///
    /// # Arguments
    /// * `pos` - Position in the sequence
    /// * `key` - Key projection [num_kv_heads, head_dim]
    /// * `value` - Value projection [num_kv_heads, head_dim]
    pub fn update(&mut self, pos: usize, key: &[f32], value: &[f32]) {
        let kv_size = self.config.num_kv_heads * self.config.head_dim;
        debug_assert!(pos < self.config.max_seq_len, "Position exceeds max_seq_len");
        debug_assert_eq!(key.len(), kv_size, "Key size mismatch");
        debug_assert_eq!(value.len(), kv_size, "Value size mismatch");

        let offset = pos * kv_size;
        self.keys[offset..offset + kv_size].copy_from_slice(key);
        self.values[offset..offset + kv_size].copy_from_slice(value);
        self.seq_len = self.seq_len.max(pos + 1);
    }

    /// Get keys for positions 0..seq_len.
    ///
    /// Returns a slice of shape [seq_len, num_kv_heads, head_dim]
    pub fn get_keys(&self, seq_len: usize) -> &[f32] {
        let size = seq_len * self.config.num_kv_heads * self.config.head_dim;
        &self.keys[..size]
    }

    /// Get values for positions 0..seq_len.
    ///
    /// Returns a slice of shape [seq_len, num_kv_heads, head_dim]
    pub fn get_values(&self, seq_len: usize) -> &[f32] {
        let size = seq_len * self.config.num_kv_heads * self.config.head_dim;
        &self.values[..size]
    }

    /// Get key at a specific position.
    pub fn get_key_at(&self, pos: usize) -> &[f32] {
        let kv_size = self.config.num_kv_heads * self.config.head_dim;
        let offset = pos * kv_size;
        &self.keys[offset..offset + kv_size]
    }

    /// Get value at a specific position.
    pub fn get_value_at(&self, pos: usize) -> &[f32] {
        let kv_size = self.config.num_kv_heads * self.config.head_dim;
        let offset = pos * kv_size;
        &self.values[offset..offset + kv_size]
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.seq_len = 0;
        // Note: We don't zero the memory for performance, seq_len tracks valid data
    }

    /// Get the current sequence length.
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }
}

/// Full KV cache for all layers.
pub struct KVCache {
    /// Per-layer caches
    layers: Vec<LayerKVCache>,
    /// Configuration
    pub config: KVCacheConfig,
}

impl KVCache {
    /// Create a new KV cache for all layers.
    pub fn new(config: KVCacheConfig) -> Self {
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(LayerKVCache::new(
                config.num_kv_heads,
                config.head_dim,
                config.max_seq_len,
            ));
        }
        Self { layers, config }
    }

    /// Get the cache for a specific layer.
    pub fn layer(&self, layer_idx: usize) -> &LayerKVCache {
        &self.layers[layer_idx]
    }

    /// Get mutable cache for a specific layer.
    pub fn layer_mut(&mut self, layer_idx: usize) -> &mut LayerKVCache {
        &mut self.layers[layer_idx]
    }

    /// Update cache for a layer.
    pub fn update(&mut self, layer_idx: usize, pos: usize, key: &[f32], value: &[f32]) {
        self.layers[layer_idx].update(pos, key, value);
    }

    /// Clear all caches.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    /// Get current sequence length (same across all layers).
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.seq_len).unwrap_or(0)
    }

    /// Get number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Calculate memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let per_layer = self.config.max_seq_len * self.config.num_kv_heads * self.config.head_dim * 4 * 2;
        per_layer * self.config.num_layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_kv_cache() {
        let mut cache = LayerKVCache::new(8, 64, 2048);
        assert_eq!(cache.len(), 0);

        // Update position 0
        let key = vec![1.0; 8 * 64];
        let value = vec![2.0; 8 * 64];
        cache.update(0, &key, &value);
        assert_eq!(cache.len(), 1);

        // Check values
        let k0 = cache.get_key_at(0);
        assert_eq!(k0.len(), 8 * 64);
        assert_eq!(k0[0], 1.0);

        let v0 = cache.get_value_at(0);
        assert_eq!(v0[0], 2.0);

        // Update position 1
        let key2 = vec![3.0; 8 * 64];
        let value2 = vec![4.0; 8 * 64];
        cache.update(1, &key2, &value2);
        assert_eq!(cache.len(), 2);

        // Get all keys
        let all_keys = cache.get_keys(2);
        assert_eq!(all_keys.len(), 2 * 8 * 64);
    }

    #[test]
    fn test_kv_cache() {
        let config = KVCacheConfig {
            num_layers: 4,
            num_kv_heads: 8,
            head_dim: 64,
            max_seq_len: 2048,
        };

        let mut cache = KVCache::new(config);
        assert_eq!(cache.num_layers(), 4);
        assert_eq!(cache.seq_len(), 0);

        // Update layer 0
        let key = vec![1.0; 8 * 64];
        let value = vec![2.0; 8 * 64];
        cache.update(0, 0, &key, &value);

        assert_eq!(cache.layer(0).len(), 1);

        // Clear
        cache.clear();
        assert_eq!(cache.seq_len(), 0);
    }
}
