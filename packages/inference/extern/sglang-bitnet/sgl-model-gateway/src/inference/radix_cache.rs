//! RadixCache: Prefix caching for KV cache reuse.
//!
//! This module implements RadixAttention-style prefix caching to enable:
//! - Zero-copy KV cache sharing via llama.cpp's `kv_cache_seq_cp`
//! - Fast prefix matching in O(k) where k = matched prefix length
//! - LRU eviction for memory management
//!
//! # Performance Optimizations
//! - RwLock for read-heavy workloads (prefix matching >> insertions)
//! - Atomics for lock-free counter updates
//! - HashMap for O(1) child lookup
//! - Minimal allocations in match_prefix hot path

use super::radix_tree::{common_prefix_len, current_time_ms, RadixTreeNode, TokenVec};
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tracing::{debug, trace};

/// Eviction policy for cache management.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used - evict oldest accessed first
    LRU,
    /// Least Frequently Used - evict least accessed first (not implemented yet)
    LFU,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self::LRU
    }
}

/// Configuration for RadixCache.
#[derive(Debug, Clone)]
pub struct RadixCacheConfig {
    /// Maximum cached tokens before eviction triggers.
    pub max_cached_tokens: usize,

    /// Eviction policy.
    pub eviction_policy: EvictionPolicy,

    /// Whether the cache is enabled.
    pub enabled: bool,
}

impl Default for RadixCacheConfig {
    fn default() -> Self {
        Self {
            max_cached_tokens: 100_000, // ~100K tokens default
            eviction_policy: EvictionPolicy::LRU,
            enabled: true,
        }
    }
}

/// Result of a prefix match operation.
#[derive(Debug)]
pub struct MatchResult {
    /// Number of tokens matched from the input.
    pub matched_len: usize,

    /// The node where the match ended.
    /// May be the root if no match found.
    pub last_node: Arc<RadixTreeNode>,

    /// The llama.cpp sequence ID to copy KV cache from.
    /// None if no reusable cache found.
    pub reuse_seq_id: Option<i32>,

    /// Position in KV cache where the matched prefix ends.
    /// This is where new tokens should start being written.
    pub kv_end_pos: i32,
}

impl MatchResult {
    /// Create an empty match result (no match found).
    pub fn empty(root: &Arc<RadixTreeNode>) -> Self {
        Self {
            matched_len: 0,
            last_node: Arc::clone(root),
            reuse_seq_id: None,
            kv_end_pos: 0,
        }
    }

    /// Check if any prefix was matched.
    #[inline]
    pub fn has_match(&self) -> bool {
        self.matched_len > 0 && self.reuse_seq_id.is_some()
    }
}

/// Entry for the eviction priority queue.
#[derive(Debug)]
struct EvictionEntry {
    /// Priority (negated for min-heap behavior with BinaryHeap).
    priority: u64,
    /// Node to potentially evict.
    node: Arc<RadixTreeNode>,
}

impl PartialEq for EvictionEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for EvictionEntry {}

impl PartialOrd for EvictionEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EvictionEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse order for min-heap (evict lowest priority first)
        other.priority.cmp(&self.priority)
    }
}

/// RadixCache manages prefix caching for KV cache reuse.
///
/// Thread-safe for concurrent access from multiple request handlers.
pub struct RadixCache {
    /// Root of the radix tree.
    root: Arc<RadixTreeNode>,

    /// Configuration.
    config: RadixCacheConfig,

    /// Total number of cached tokens.
    cached_token_count: AtomicUsize,

    /// Node ID counter for unique identification.
    next_node_id: AtomicU64,

    /// Mutex to serialize insert operations to prevent race conditions during tree modification.
    /// match_prefix is lock-free (read-only), but insert/split requires exclusive access.
    insert_lock: Mutex<()>,
}

impl RadixCache {
    /// Create a new RadixCache with the given configuration.
    pub fn new(config: RadixCacheConfig) -> Self {
        let root = Arc::new(RadixTreeNode::root(0));
        Self {
            root,
            config,
            cached_token_count: AtomicUsize::new(0),
            next_node_id: AtomicU64::new(1),
            insert_lock: Mutex::new(()),
        }
    }

    /// Create a RadixCache with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(RadixCacheConfig::default())
    }

    /// Check if the cache is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the current number of cached tokens.
    #[inline]
    pub fn cached_tokens(&self) -> usize {
        self.cached_token_count.load(Ordering::Relaxed)
    }

    /// Get a reference to the root node.
    #[inline]
    pub fn root(&self) -> &Arc<RadixTreeNode> {
        &self.root
    }

    /// Generate a new unique node ID.
    #[inline]
    fn next_node_id(&self) -> u64 {
        self.next_node_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Find the longest prefix of `tokens` that exists in the cache.
    ///
    /// This is the hot path - optimized for read performance:
    /// - Only acquires read locks
    /// - Minimal allocations
    /// - Early exit on evicted nodes
    ///
    /// # Arguments
    /// * `tokens` - The token sequence to match against the cache
    ///
    /// # Returns
    /// A `MatchResult` containing:
    /// - Length of matched prefix
    /// - Last matched node
    /// - Sequence ID for KV cache copy (if any)
    /// - KV cache end position
    pub fn match_prefix(&self, tokens: &[i32]) -> MatchResult {
        if !self.config.enabled || tokens.is_empty() {
            return MatchResult::empty(&self.root);
        }

        let mut current = Arc::clone(&self.root);
        let mut matched_len: usize = 0;
        let mut token_idx: usize = 0;
        let mut last_valid_node = Arc::clone(&self.root);
        let mut last_valid_seq_id: Option<i32> = None;
        let mut last_valid_pos: i32 = 0;

        while token_idx < tokens.len() {
            let first_token = tokens[token_idx];

            // Try to get child (read lock only)
            let child = match current.get_child(first_token) {
                Some(c) => c,
                None => break,
            };

            // Check if child's KV cache is still valid
            let kv_seq_id = child.kv_seq_id.load(Ordering::Acquire);
            if kv_seq_id < 0 {
                // This node's KV cache has been evicted
                trace!(
                    "Node {} evicted, stopping match at {} tokens",
                    child.node_id,
                    matched_len
                );
                break;
            }

            // Compare tokens
            let remaining = &tokens[token_idx..];
            let common_len = common_prefix_len(&child.tokens, remaining);

            if common_len == 0 {
                break;
            }

            // Update access time for LRU
            child.touch();

            matched_len += common_len;
            token_idx += common_len;

            // Track last valid match point
            if common_len == child.tokens.len() {
                // Full match with this node
                last_valid_node = Arc::clone(&child);
                last_valid_seq_id = Some(kv_seq_id);
                last_valid_pos = child.kv_start_pos.load(Ordering::Acquire)
                    + child.tokens.len() as i32;
                current = child;
            } else {
                // Partial match - can still use up to common_len
                last_valid_node = Arc::clone(&child);
                last_valid_seq_id = Some(kv_seq_id);
                last_valid_pos = child.kv_start_pos.load(Ordering::Acquire) + common_len as i32;
                // Can't continue past partial match
                break;
            }
        }

        trace!(
            "Matched {} tokens, seq_id={:?}, kv_end_pos={}",
            matched_len,
            last_valid_seq_id,
            last_valid_pos
        );

        MatchResult {
            matched_len,
            last_node: last_valid_node,
            reuse_seq_id: if matched_len > 0 {
                last_valid_seq_id
            } else {
                None
            },
            kv_end_pos: last_valid_pos,
        }
    }

    /// Insert a new token sequence into the cache.
    ///
    /// Called after a sequence has completed prefill to make its KV cache
    /// available for future requests.
    ///
    /// # Arguments
    /// * `tokens` - The full token sequence that was processed
    /// * `seq_id` - The llama.cpp sequence ID that owns this KV cache
    ///
    /// # Returns
    /// Number of new tokens added to the cache.
    pub fn insert(&self, tokens: &[i32], seq_id: i32) -> usize {
        if !self.config.enabled || tokens.is_empty() {
            return 0;
        }

        // Acquire insert lock to prevent race conditions during tree modification.
        // This serializes all insert operations but allows concurrent match_prefix reads.
        let _insert_guard = self.insert_lock.lock().unwrap();

        let mut current = Arc::clone(&self.root);
        let mut token_idx: usize = 0;
        let mut new_tokens_added: usize = 0;

        while token_idx < tokens.len() {
            let first_token = tokens[token_idx];

            // Check if child exists
            if let Some(child) = current.get_child(first_token) {
                let remaining = &tokens[token_idx..];
                let common_len = common_prefix_len(&child.tokens, remaining);

                if common_len == 0 {
                    // No match, create new branch
                    break;
                }

                if common_len < child.tokens.len() {
                    // Partial match - need to split the node
                    let (upper, _lower) = self.split_node(&current, &child, common_len);

                    // Update sequence ID on the split point
                    upper.kv_seq_id.store(seq_id, Ordering::Release);

                    token_idx += common_len;
                    current = upper;
                } else {
                    // Full match with child
                    token_idx += child.tokens.len();
                    current = child;
                }
            } else {
                // No existing child - create new node with remaining tokens
                let remaining: TokenVec = tokens[token_idx..].into();
                let remaining_len = remaining.len();

                let new_node = Arc::new(RadixTreeNode::new(
                    remaining,
                    seq_id,
                    token_idx as i32,
                    token_idx + remaining_len,
                    self.next_node_id(),
                ));

                new_node.set_parent(Some(&current));
                current.insert_child(Arc::clone(&new_node));

                new_tokens_added += remaining_len;
                self.cached_token_count
                    .fetch_add(remaining_len, Ordering::Relaxed);

                debug!(
                    "Inserted {} new tokens at node {}, total cached: {}",
                    remaining_len,
                    new_node.node_id,
                    self.cached_tokens()
                );

                break;
            }
        }

        new_tokens_added
    }

    /// Split a node at the given position.
    ///
    /// Before: parent -> child([1,2,3,4,5])
    /// After:  parent -> upper([1,2,3]) -> lower([4,5])
    ///
    /// Returns (upper_node, lower_node).
    fn split_node(
        &self,
        parent: &Arc<RadixTreeNode>,
        child: &Arc<RadixTreeNode>,
        split_pos: usize,
    ) -> (Arc<RadixTreeNode>, Arc<RadixTreeNode>) {
        debug_assert!(split_pos > 0 && split_pos < child.tokens.len());

        // Use SmallVec to avoid heap allocation for short token sequences
        let upper_tokens: TokenVec = child.tokens[..split_pos].into();
        let lower_tokens: TokenVec = child.tokens[split_pos..].into();

        // Capture lengths before moving into constructors
        let upper_len = upper_tokens.len();
        let lower_len = lower_tokens.len();

        let child_start_pos = child.kv_start_pos.load(Ordering::Acquire);
        let child_prefix_len = child.prefix_len.load(Ordering::Acquire);

        // Create upper node (takes the common prefix)
        let upper = Arc::new(RadixTreeNode::new(
            upper_tokens,
            child.kv_seq_id.load(Ordering::Acquire),
            child_start_pos,
            child_start_pos as usize + split_pos,
            self.next_node_id(),
        ));

        // Create lower node (takes the suffix)
        let lower = Arc::new(RadixTreeNode::new(
            lower_tokens,
            child.kv_seq_id.load(Ordering::Acquire),
            child_start_pos + split_pos as i32,
            child_prefix_len,
            self.next_node_id(),
        ));

        // Copy lock_ref from original child to lower (preserves active references)
        let lock_ref = child.lock_ref.load(Ordering::Acquire);
        lower.lock_ref.store(lock_ref, Ordering::Release);

        // Set up parent-child relationships
        upper.set_parent(Some(parent));
        lower.set_parent(Some(&upper));
        upper.insert_child(Arc::clone(&lower));

        // Move original child's children to lower node
        {
            let child_children = child.children.read().unwrap();
            let mut lower_children = lower.children.write().unwrap();
            for (k, v) in child_children.iter() {
                v.set_parent(Some(&lower));
                lower_children.insert(*k, Arc::clone(v));
            }
        }

        // Replace child with upper in parent
        if let Some(first_token) = child.tokens.first() {
            parent.remove_child(*first_token);
        }
        parent.insert_child(Arc::clone(&upper));

        debug!(
            "Split node {}: upper({} tokens) -> lower({} tokens)",
            child.node_id,
            upper_len,
            lower_len
        );

        (upper, lower)
    }

    /// Evict cached prefixes to free up space.
    ///
    /// Uses LRU policy: evicts least recently accessed leaves first.
    /// Nodes with lock_ref > 0 are protected from eviction.
    ///
    /// # Arguments
    /// * `num_tokens_to_free` - Target number of tokens to free
    ///
    /// # Returns
    /// Number of tokens actually freed.
    pub fn evict(&self, num_tokens_to_free: usize) -> usize {
        if !self.config.enabled || num_tokens_to_free == 0 {
            return 0;
        }

        // Acquire insert lock to prevent race conditions with concurrent inserts.
        let _insert_guard = self.insert_lock.lock().unwrap();

        // Collect all evictable leaf nodes
        let mut heap = BinaryHeap::new();
        self.collect_evictable_leaves(&self.root, &mut heap);

        let mut freed: usize = 0;

        while freed < num_tokens_to_free {
            let entry = match heap.pop() {
                Some(e) => e,
                None => break,
            };

            let node = &entry.node;

            // Double-check evictability (might have changed)
            if !node.can_evict() || node.is_evicted() {
                continue;
            }

            let token_count = node.tokens.len();
            let seq_id = node.kv_seq_id.load(Ordering::Acquire);

            // Mark as evicted
            node.kv_seq_id.store(-1, Ordering::Release);

            freed += token_count;
            self.cached_token_count
                .fetch_sub(token_count, Ordering::Relaxed);

            debug!(
                "Evicted node {} (seq_id={}, {} tokens), total freed: {}",
                node.node_id, seq_id, token_count, freed
            );

            // Remove from parent
            if let Some(parent) = node.get_parent() {
                if let Some(first_token) = node.tokens.first() {
                    parent.remove_child(*first_token);
                }

                // If parent becomes a leaf with no active refs, add to heap
                if parent.is_leaf() && parent.can_evict() && !parent.is_root() {
                    heap.push(EvictionEntry {
                        priority: parent.last_access_time.load(Ordering::Relaxed),
                        node: parent,
                    });
                }
            }
        }

        debug!(
            "Eviction complete: freed {} tokens, {} remaining",
            freed,
            self.cached_tokens()
        );

        freed
    }

    /// Collect all evictable leaf nodes into the priority heap.
    fn collect_evictable_leaves(
        &self,
        node: &Arc<RadixTreeNode>,
        heap: &mut BinaryHeap<EvictionEntry>,
    ) {
        let children = node.children.read().unwrap();

        if children.is_empty() {
            // This is a leaf
            if !node.is_root() && node.can_evict() && !node.is_evicted() {
                heap.push(EvictionEntry {
                    priority: node.last_access_time.load(Ordering::Relaxed),
                    node: Arc::clone(node),
                });
            }
        } else {
            // Recurse into children
            for child in children.values() {
                self.collect_evictable_leaves(child, heap);
            }
        }
    }

    /// Increment lock reference count on a node and all ancestors.
    ///
    /// This protects the entire path from root to this node from eviction.
    pub fn inc_lock_ref(&self, node: &Arc<RadixTreeNode>) {
        let mut current = Some(Arc::clone(node));
        while let Some(n) = current {
            n.inc_lock_ref();
            current = n.get_parent();
        }
    }

    /// Decrement lock reference count on a node and all ancestors.
    pub fn dec_lock_ref(&self, node: &Arc<RadixTreeNode>) {
        let mut current = Some(Arc::clone(node));
        while let Some(n) = current {
            n.dec_lock_ref();
            current = n.get_parent();
        }
    }

    /// Clear all cached entries.
    pub fn clear(&self) {
        let mut children = self.root.children.write().unwrap();
        children.clear();
        self.cached_token_count.store(0, Ordering::Relaxed);
        debug!("RadixCache cleared");
    }

    /// Get statistics about the cache.
    pub fn stats(&self) -> RadixCacheStats {
        let (node_count, leaf_count, max_depth) = self.compute_tree_stats();
        RadixCacheStats {
            cached_tokens: self.cached_tokens(),
            max_tokens: self.config.max_cached_tokens,
            node_count,
            leaf_count,
            max_depth,
            enabled: self.config.enabled,
        }
    }

    fn compute_tree_stats(&self) -> (usize, usize, usize) {
        fn traverse(node: &RadixTreeNode, depth: usize) -> (usize, usize, usize) {
            let children = node.children.read().unwrap();
            if children.is_empty() {
                (1, 1, depth)
            } else {
                let mut nodes = 1;
                let mut leaves = 0;
                let mut max_d = depth;
                for child in children.values() {
                    let (n, l, d) = traverse(child, depth + 1);
                    nodes += n;
                    leaves += l;
                    max_d = max_d.max(d);
                }
                (nodes, leaves, max_d)
            }
        }
        traverse(&self.root, 0)
    }
}

/// Statistics about the RadixCache.
#[derive(Debug, Clone)]
pub struct RadixCacheStats {
    /// Current number of cached tokens.
    pub cached_tokens: usize,
    /// Maximum tokens before eviction.
    pub max_tokens: usize,
    /// Total number of nodes in tree.
    pub node_count: usize,
    /// Number of leaf nodes.
    pub leaf_count: usize,
    /// Maximum depth of tree.
    pub max_depth: usize,
    /// Whether cache is enabled.
    pub enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache() -> RadixCache {
        RadixCache::new(RadixCacheConfig {
            max_cached_tokens: 1000,
            eviction_policy: EvictionPolicy::LRU,
            enabled: true,
        })
    }

    #[test]
    fn test_empty_cache() {
        let cache = make_cache();
        let result = cache.match_prefix(&[1, 2, 3]);
        assert_eq!(result.matched_len, 0);
        assert!(result.reuse_seq_id.is_none());
    }

    #[test]
    fn test_insert_and_exact_match() {
        let cache = make_cache();

        // Insert a sequence
        let added = cache.insert(&[1, 2, 3, 4, 5], 0);
        assert_eq!(added, 5);
        assert_eq!(cache.cached_tokens(), 5);

        // Exact match
        let result = cache.match_prefix(&[1, 2, 3, 4, 5]);
        assert_eq!(result.matched_len, 5);
        assert_eq!(result.reuse_seq_id, Some(0));
        assert_eq!(result.kv_end_pos, 5);
    }

    #[test]
    fn test_prefix_match() {
        let cache = make_cache();

        cache.insert(&[1, 2, 3, 4, 5], 0);

        // Partial prefix match
        let result = cache.match_prefix(&[1, 2, 3]);
        assert_eq!(result.matched_len, 3);
        assert_eq!(result.reuse_seq_id, Some(0));
        assert_eq!(result.kv_end_pos, 3);
    }

    #[test]
    fn test_no_match() {
        let cache = make_cache();

        cache.insert(&[1, 2, 3], 0);

        // Different tokens - no match
        let result = cache.match_prefix(&[4, 5, 6]);
        assert_eq!(result.matched_len, 0);
        assert!(result.reuse_seq_id.is_none());
    }

    #[test]
    fn test_partial_token_match() {
        let cache = make_cache();

        cache.insert(&[1, 2, 3, 4, 5], 0);

        // Match diverges partway through
        let result = cache.match_prefix(&[1, 2, 3, 6, 7]);
        assert_eq!(result.matched_len, 3);
        assert_eq!(result.reuse_seq_id, Some(0));
    }

    #[test]
    fn test_node_splitting() {
        let cache = make_cache();

        // First insert
        cache.insert(&[1, 2, 3, 4, 5], 0);

        // Second insert with shared prefix - should cause split
        cache.insert(&[1, 2, 3, 6, 7], 1);

        // Both should match their full length
        let result1 = cache.match_prefix(&[1, 2, 3, 4, 5]);
        assert_eq!(result1.matched_len, 5);

        let result2 = cache.match_prefix(&[1, 2, 3, 6, 7]);
        assert_eq!(result2.matched_len, 5);

        // Common prefix should match both
        let result3 = cache.match_prefix(&[1, 2, 3]);
        assert_eq!(result3.matched_len, 3);
    }

    #[test]
    fn test_multiple_branches() {
        let cache = make_cache();

        cache.insert(&[1, 2, 3], 0);
        cache.insert(&[1, 2, 4], 1);
        cache.insert(&[1, 5, 6], 2);
        cache.insert(&[7, 8, 9], 3);

        let r1 = cache.match_prefix(&[1, 2, 3]);
        assert_eq!(r1.matched_len, 3);

        let r2 = cache.match_prefix(&[1, 2, 4]);
        assert_eq!(r2.matched_len, 3);

        let r3 = cache.match_prefix(&[1, 5, 6]);
        assert_eq!(r3.matched_len, 3);

        let r4 = cache.match_prefix(&[7, 8, 9]);
        assert_eq!(r4.matched_len, 3);
    }

    #[test]
    fn test_eviction_lru() {
        let cache = make_cache();

        // Insert some sequences - use explicit timestamps via sleep
        cache.insert(&[1, 2, 3, 4, 5], 0);
        std::thread::sleep(std::time::Duration::from_millis(50));
        cache.insert(&[6, 7, 8, 9, 10], 1);
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Access first one to make it more recent
        let r = cache.match_prefix(&[1, 2, 3, 4, 5]);
        assert_eq!(r.matched_len, 5, "First sequence should match before eviction");

        // Evict 5 tokens - should evict the second (older) one
        let freed = cache.evict(5);
        assert_eq!(freed, 5, "Should free exactly 5 tokens");

        // Second should be evicted (checked first since it might fail)
        let r2 = cache.match_prefix(&[6, 7, 8, 9, 10]);
        assert_eq!(r2.matched_len, 0, "Second sequence should be evicted");

        // First should still be cached (it was touched more recently)
        let r1 = cache.match_prefix(&[1, 2, 3, 4, 5]);
        assert_eq!(r1.matched_len, 5, "First sequence should still be cached after eviction");
    }

    #[test]
    fn test_lock_ref_prevents_eviction() {
        let cache = make_cache();

        cache.insert(&[1, 2, 3, 4, 5], 0);
        cache.insert(&[6, 7, 8, 9, 10], 1);

        // Lock the first one
        let result = cache.match_prefix(&[1, 2, 3, 4, 5]);
        cache.inc_lock_ref(&result.last_node);

        // Try to evict everything
        cache.evict(100);

        // First should still be cached (locked)
        let r1 = cache.match_prefix(&[1, 2, 3, 4, 5]);
        assert_eq!(r1.matched_len, 5);

        // Unlock
        cache.dec_lock_ref(&result.last_node);

        // Now it can be evicted
        cache.evict(100);
        let r2 = cache.match_prefix(&[1, 2, 3, 4, 5]);
        assert_eq!(r2.matched_len, 0);
    }

    #[test]
    fn test_disabled_cache() {
        let cache = RadixCache::new(RadixCacheConfig {
            enabled: false,
            ..Default::default()
        });

        let added = cache.insert(&[1, 2, 3], 0);
        assert_eq!(added, 0);

        let result = cache.match_prefix(&[1, 2, 3]);
        assert_eq!(result.matched_len, 0);
    }

    #[test]
    fn test_empty_input() {
        let cache = make_cache();
        cache.insert(&[1, 2, 3], 0);

        let result = cache.match_prefix(&[]);
        assert_eq!(result.matched_len, 0);
    }

    #[test]
    fn test_long_sequence() {
        let cache = make_cache();

        // Insert a long sequence (8K tokens)
        let long_seq: Vec<i32> = (0..8000).collect();
        cache.insert(&long_seq, 0);

        // Match various prefixes
        let r1 = cache.match_prefix(&long_seq[..100]);
        assert_eq!(r1.matched_len, 100);

        let r2 = cache.match_prefix(&long_seq[..4000]);
        assert_eq!(r2.matched_len, 4000);

        let r3 = cache.match_prefix(&long_seq);
        assert_eq!(r3.matched_len, 8000);
    }

    #[test]
    fn test_stats() {
        let cache = make_cache();

        cache.insert(&[1, 2, 3], 0);
        cache.insert(&[1, 2, 4], 1);

        let stats = cache.stats();
        assert!(stats.cached_tokens > 0);
        assert!(stats.node_count >= 2);
        assert!(stats.leaf_count >= 2);
    }

    #[test]
    fn test_clear() {
        let cache = make_cache();

        cache.insert(&[1, 2, 3], 0);
        cache.insert(&[4, 5, 6], 1);
        assert!(cache.cached_tokens() > 0);

        cache.clear();
        assert_eq!(cache.cached_tokens(), 0);

        let result = cache.match_prefix(&[1, 2, 3]);
        assert_eq!(result.matched_len, 0);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let cache = Arc::new(make_cache());
        let mut handles = vec![];

        // Spawn multiple threads doing inserts and matches
        for i in 0..10 {
            let cache = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                let base = i * 100;
                let tokens: Vec<i32> = (base..(base + 50)).collect();

                // Insert
                cache.insert(&tokens, i);

                // Multiple matches
                for _ in 0..100 {
                    let _ = cache.match_prefix(&tokens[..25]);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify all sequences are accessible
        for i in 0..10 {
            let base = i * 100;
            let tokens: Vec<i32> = (base..(base + 50)).collect();
            let result = cache.match_prefix(&tokens);
            assert!(result.matched_len > 0, "Sequence {} not found", i);
        }
    }
}
