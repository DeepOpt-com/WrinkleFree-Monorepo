//! Radix tree for efficient prefix matching in KV cache.
//!
//! This module provides a radix tree (compressed trie) optimized for:
//! - Fast O(k) prefix matching where k = prefix length
//! - Lock-free read operations via RwLock (readers don't block each other)
//! - Atomic reference counting for eviction protection
//! - LRU tracking for cache eviction

use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, Weak};
use std::time::{SystemTime, UNIX_EPOCH};
use smallvec::SmallVec;

/// Token storage type - SmallVec inlines up to 8 tokens to avoid heap allocation.
/// Most radix tree nodes have short edge labels (1-5 tokens), so this avoids
/// heap allocation for the majority of nodes.
pub type TokenVec = SmallVec<[i32; 8]>;

/// Get current time in milliseconds for LRU tracking.
#[inline]
pub fn current_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// A node in the radix tree representing a sequence of tokens.
///
/// Design for performance:
/// - `tokens`: Stored inline to avoid pointer chasing
/// - `children`: RwLock<HashMap> for O(1) lookup, concurrent reads
/// - Atomics for counters to avoid locks on hot paths
pub struct RadixTreeNode {
    /// Token IDs stored at this node (the "edge label" in radix tree terms).
    /// For the root node, this is empty.
    /// Uses SmallVec to inline up to 8 tokens, avoiding heap allocation for most nodes.
    pub tokens: TokenVec,

    /// Children indexed by first token of their edge label.
    /// Using HashMap for O(1) lookup instead of linear scan.
    pub children: RwLock<HashMap<i32, Arc<RadixTreeNode>>>,

    /// Parent node (weak ref to avoid reference cycles).
    pub parent: RwLock<Option<Weak<RadixTreeNode>>>,

    /// The llama.cpp sequence ID that "owns" this prefix in KV cache.
    /// Set to -1 if this node's KV cache has been evicted.
    pub kv_seq_id: AtomicI32,

    /// Starting position in KV cache for this node's tokens.
    /// This is the cumulative position from root to the START of this node.
    pub kv_start_pos: AtomicI32,

    /// Number of tokens from root to END of this node (inclusive).
    pub prefix_len: AtomicUsize,

    /// Reference count: how many active requests are using this prefix.
    /// Nodes with lock_ref > 0 cannot be evicted.
    pub lock_ref: AtomicI32,

    /// Last access time in milliseconds (for LRU eviction).
    pub last_access_time: AtomicU64,

    /// Unique node ID for debugging and identification.
    pub node_id: u64,
}

impl RadixTreeNode {
    /// Create a new tree node.
    pub fn new(
        tokens: impl Into<TokenVec>,
        kv_seq_id: i32,
        kv_start_pos: i32,
        prefix_len: usize,
        node_id: u64,
    ) -> Self {
        Self {
            tokens: tokens.into(),
            children: RwLock::new(HashMap::new()),
            parent: RwLock::new(None),
            kv_seq_id: AtomicI32::new(kv_seq_id),
            kv_start_pos: AtomicI32::new(kv_start_pos),
            prefix_len: AtomicUsize::new(prefix_len),
            lock_ref: AtomicI32::new(0),
            last_access_time: AtomicU64::new(current_time_ms()),
            node_id,
        }
    }

    /// Create a root node (empty tokens, no parent).
    pub fn root(node_id: u64) -> Self {
        Self::new(Vec::new(), -1, 0, 0, node_id)
    }

    /// Check if this node is the root.
    #[inline]
    pub fn is_root(&self) -> bool {
        self.tokens.is_empty() && self.parent.read().unwrap().is_none()
    }

    /// Check if this node is a leaf (no children).
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.children.read().unwrap().is_empty()
    }

    /// Check if this node's KV cache has been evicted.
    #[inline]
    pub fn is_evicted(&self) -> bool {
        self.kv_seq_id.load(Ordering::Acquire) < 0
    }

    /// Check if this node can be evicted (no active references).
    #[inline]
    pub fn can_evict(&self) -> bool {
        self.lock_ref.load(Ordering::Acquire) == 0
    }

    /// Get the number of children.
    #[inline]
    pub fn child_count(&self) -> usize {
        self.children.read().unwrap().len()
    }

    /// Update last access time to now.
    #[inline]
    pub fn touch(&self) {
        self.last_access_time
            .store(current_time_ms(), Ordering::Release);
    }

    /// Increment lock reference count.
    /// Returns the new count.
    #[inline]
    pub fn inc_lock_ref(&self) -> i32 {
        self.lock_ref.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Decrement lock reference count.
    /// Returns the new count.
    #[inline]
    pub fn dec_lock_ref(&self) -> i32 {
        let prev = self.lock_ref.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(prev > 0, "lock_ref underflow");
        prev - 1
    }

    /// Get child by first token, if it exists.
    #[inline]
    pub fn get_child(&self, first_token: i32) -> Option<Arc<RadixTreeNode>> {
        self.children.read().unwrap().get(&first_token).cloned()
    }

    /// Insert a child node.
    pub fn insert_child(&self, child: Arc<RadixTreeNode>) {
        if let Some(first_token) = child.tokens.first() {
            self.children
                .write()
                .unwrap()
                .insert(*first_token, child);
        }
    }

    /// Remove a child by first token.
    pub fn remove_child(&self, first_token: i32) -> Option<Arc<RadixTreeNode>> {
        self.children.write().unwrap().remove(&first_token)
    }

    /// Set parent reference.
    pub fn set_parent(&self, parent: Option<&Arc<RadixTreeNode>>) {
        *self.parent.write().unwrap() = parent.map(Arc::downgrade);
    }

    /// Get parent as Arc, if it still exists.
    pub fn get_parent(&self) -> Option<Arc<RadixTreeNode>> {
        self.parent.read().unwrap().as_ref()?.upgrade()
    }
}

impl std::fmt::Debug for RadixTreeNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RadixTreeNode")
            .field("node_id", &self.node_id)
            .field("tokens_len", &self.tokens.len())
            .field("prefix_len", &self.prefix_len.load(Ordering::Relaxed))
            .field("kv_seq_id", &self.kv_seq_id.load(Ordering::Relaxed))
            .field("lock_ref", &self.lock_ref.load(Ordering::Relaxed))
            .field("children", &self.child_count())
            .finish()
    }
}

/// Find the length of the common prefix between two slices.
/// Optimized for performance with early exit.
#[inline]
pub fn common_prefix_len(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = RadixTreeNode::new(vec![1, 2, 3], 0, 0, 3, 1);
        assert_eq!(node.tokens, vec![1, 2, 3]);
        assert_eq!(node.kv_seq_id.load(Ordering::Relaxed), 0);
        assert_eq!(node.prefix_len.load(Ordering::Relaxed), 3);
        assert!(!node.is_root());
        assert!(node.is_leaf());
    }

    #[test]
    fn test_root_node() {
        let root = RadixTreeNode::root(0);
        assert!(root.is_root());
        assert!(root.is_leaf());
        assert!(root.tokens.is_empty());
    }

    #[test]
    fn test_lock_ref() {
        let node = RadixTreeNode::new(vec![1], 0, 0, 1, 1);
        assert!(node.can_evict());

        assert_eq!(node.inc_lock_ref(), 1);
        assert!(!node.can_evict());

        assert_eq!(node.inc_lock_ref(), 2);
        assert!(!node.can_evict());

        assert_eq!(node.dec_lock_ref(), 1);
        assert!(!node.can_evict());

        assert_eq!(node.dec_lock_ref(), 0);
        assert!(node.can_evict());
    }

    #[test]
    fn test_child_operations() {
        let parent = Arc::new(RadixTreeNode::root(0));
        let child = Arc::new(RadixTreeNode::new(vec![1, 2, 3], 1, 0, 3, 1));

        parent.insert_child(Arc::clone(&child));
        assert_eq!(parent.child_count(), 1);

        let retrieved = parent.get_child(1).unwrap();
        assert_eq!(retrieved.node_id, 1);

        parent.remove_child(1);
        assert_eq!(parent.child_count(), 0);
        assert!(parent.get_child(1).is_none());
    }

    #[test]
    fn test_common_prefix_len() {
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 4]), 2);
        assert_eq!(common_prefix_len(&[1, 2, 3], &[4, 5, 6]), 0);
        assert_eq!(common_prefix_len(&[1, 2], &[1, 2, 3, 4]), 2);
        assert_eq!(common_prefix_len(&[], &[1, 2, 3]), 0);
    }

    #[test]
    fn test_parent_child_relationship() {
        let parent = Arc::new(RadixTreeNode::root(0));
        let child = Arc::new(RadixTreeNode::new(vec![1, 2], 1, 0, 2, 1));

        child.set_parent(Some(&parent));
        parent.insert_child(Arc::clone(&child));

        let retrieved_parent = child.get_parent().unwrap();
        assert_eq!(retrieved_parent.node_id, 0);
    }

    #[test]
    fn test_touch_updates_time() {
        let node = RadixTreeNode::new(vec![1], 0, 0, 1, 1);
        let initial_time = node.last_access_time.load(Ordering::Relaxed);

        std::thread::sleep(std::time::Duration::from_millis(10));
        node.touch();

        let new_time = node.last_access_time.load(Ordering::Relaxed);
        assert!(new_time >= initial_time);
    }
}
