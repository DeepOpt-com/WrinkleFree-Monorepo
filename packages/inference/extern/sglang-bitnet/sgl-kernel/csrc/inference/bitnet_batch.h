/**
 * BitNet Batch Inference API - Continuous Batching Support
 *
 * Extends the BitNet engine with multi-sequence batched inference.
 * Enables continuous batching where new requests can join mid-generation.
 *
 * Key concepts:
 * - Sequence ID (seq_id): Unique identifier for each concurrent sequence
 * - Slot: Pre-allocated KV cache region for a sequence
 * - Batch: Collection of tokens from multiple sequences processed together
 */

#ifndef BITNET_BATCH_H
#define BITNET_BATCH_H

#include "bitnet_engine.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Types
// =============================================================================

/// Sequence ID type (matches llama_seq_id for compatibility)
typedef int32_t bitnet_seq_id;

/// Batch engine handle (extends BitNetEngine with multi-sequence support)
typedef struct BitNetBatchEngine BitNetBatchEngine;

/// Batch structure for multiple sequences
typedef struct {
    int32_t n_tokens;           ///< Number of tokens in batch

    int32_t* token;             ///< Token IDs [n_tokens]
    int32_t* pos;               ///< Position in sequence [n_tokens]
    int32_t* n_seq_id;          ///< Number of seq_ids per token [n_tokens]
    bitnet_seq_id** seq_id;     ///< Sequence IDs [n_tokens][n_seq_id[i]]
    int8_t* logits;             ///< Whether to output logits [n_tokens]

    // Internal allocation tracking
    int32_t _capacity;          ///< Allocated capacity
    int32_t _n_seq_max;         ///< Max sequences per token
} BitNetBatch;

/// Batch engine configuration
typedef struct {
    int32_t max_batch_size;     ///< Maximum tokens per batch (e.g., 512)
    int32_t max_sequences;      ///< Maximum concurrent sequences (e.g., 16)
    int32_t n_ctx;              ///< Total context size
    int32_t n_ctx_per_seq;      ///< Context per sequence (n_ctx / max_sequences)
    int32_t num_threads;        ///< Number of threads (0 = auto)
} BitNetBatchConfig;

/// Sequence state
typedef enum {
    SEQ_STATE_IDLE = 0,         ///< Slot available
    SEQ_STATE_PREFILLING,       ///< Processing prompt
    SEQ_STATE_DECODING,         ///< Generating tokens
    SEQ_STATE_FINISHED          ///< Generation complete
} BitNetSeqState;

/// Per-sequence information (returned by query functions)
typedef struct {
    bitnet_seq_id seq_id;
    bool is_active;
    int32_t position;           ///< Current position in sequence
    int32_t prompt_len;         ///< Original prompt length
    int32_t generated_count;    ///< Tokens generated so far
} BitNetSeqInfo;

/// Sampling parameters for batch API
typedef struct {
    float temperature;          ///< Temperature (0 = greedy)
    float top_p;                ///< Top-p sampling
    float top_k;                ///< Top-k sampling (0 = disabled)
    float repetition_penalty;   ///< Repetition penalty
    int32_t max_tokens;         ///< Maximum tokens to generate
} BitNetSamplingParams;

// =============================================================================
// Batch Engine Lifecycle
// =============================================================================

/**
 * Create batch engine with multi-sequence support.
 *
 * @param model_path Path to model (HuggingFace format)
 * @param config Batch configuration (NULL for defaults)
 * @return Engine handle, or NULL on error
 */
BitNetBatchEngine* bitnet_batch_engine_create(
    const char* model_path,
    const BitNetBatchConfig* config
);

/**
 * Destroy batch engine and free resources.
 */
void bitnet_batch_engine_destroy(BitNetBatchEngine* engine);

/**
 * Get default batch configuration.
 */
BitNetBatchConfig bitnet_batch_config_default(void);

// =============================================================================
// Batch Management
// =============================================================================

/**
 * Initialize a batch structure.
 *
 * @param n_tokens Maximum tokens the batch can hold
 * @param n_seq_max Maximum sequences per token (usually 1)
 * @return Batch handle, or NULL on error
 */
BitNetBatch* bitnet_batch_init(int32_t n_tokens, int32_t n_seq_max);

/**
 * Free batch structure.
 */
void bitnet_batch_free(BitNetBatch* batch);

/**
 * Clear batch for reuse (resets n_tokens to 0).
 */
void bitnet_batch_clear(BitNetBatch* batch);

/**
 * Add a token to the batch.
 *
 * @param batch Batch to add to
 * @param token Token ID
 * @param pos Position in sequence
 * @param seq_ids Array of sequence IDs this token belongs to
 * @param n_seq_ids Number of sequence IDs (usually 1)
 * @param output_logits Whether to compute logits for this token
 */
void bitnet_batch_add(
    BitNetBatch* batch,
    int32_t token,
    int32_t pos,
    const bitnet_seq_id* seq_ids,
    int32_t n_seq_ids,
    int8_t output_logits
);

// =============================================================================
// Sequence Management
// =============================================================================

/**
 * Allocate a sequence slot.
 *
 * @param engine Batch engine
 * @return New sequence ID, or -1 if no slots available
 */
bitnet_seq_id bitnet_seq_alloc(BitNetBatchEngine* engine);

/**
 * Free a sequence slot (clears its KV cache).
 *
 * @param engine Batch engine
 * @param seq_id Sequence to free
 */
void bitnet_seq_free(BitNetBatchEngine* engine, bitnet_seq_id seq_id);

/**
 * Get sequence information.
 *
 * @param engine Batch engine
 * @param seq_id Sequence ID
 * @param info Output info structure
 * @return 0 on success, -1 if seq_id invalid
 */
int bitnet_seq_get_info(
    BitNetBatchEngine* engine,
    bitnet_seq_id seq_id,
    BitNetSeqInfo* info
);

/**
 * Get number of active sequences.
 */
int32_t bitnet_seq_active_count(BitNetBatchEngine* engine);

/**
 * Get number of available slots.
 */
int32_t bitnet_seq_available_slots(BitNetBatchEngine* engine);

// =============================================================================
// Batch Inference
// =============================================================================

/**
 * Process batch through model.
 *
 * Handles both prefill (multiple tokens per seq) and decode (1 token per seq).
 * Updates KV cache for all sequences in the batch.
 *
 * @param engine Batch engine
 * @param batch Input batch
 * @return 0 on success, 1 if KV cache full, negative on error
 */
int bitnet_batch_decode(
    BitNetBatchEngine* engine,
    const BitNetBatch* batch
);

/**
 * Get logits for a batch position.
 *
 * Only valid for positions where logits=1 was set in batch_add.
 *
 * @param engine Batch engine
 * @param batch_idx Index in batch (0 to n_tokens-1)
 * @return Pointer to logits [vocab_size], valid until next decode
 */
const float* bitnet_get_logits_ith(
    BitNetBatchEngine* engine,
    int32_t batch_idx
);

/**
 * Sample token from logits at batch position.
 *
 * @param engine Batch engine
 * @param batch_idx Batch index
 * @param params Sampling parameters
 * @return Sampled token ID
 */
int32_t bitnet_batch_sample(
    BitNetBatchEngine* engine,
    int32_t batch_idx,
    const BitNetSamplingParams* params
);

// =============================================================================
// Tokenization
// =============================================================================

/**
 * Tokenize text to token IDs.
 *
 * @param engine Batch engine
 * @param text Input text
 * @param text_len Length of text (-1 for null-terminated)
 * @param tokens Output token array
 * @param n_tokens_max Maximum tokens to output
 * @param add_special Add special tokens (BOS, etc.)
 * @return Number of tokens, or negative on error
 */
int32_t bitnet_tokenize(
    BitNetBatchEngine* engine,
    const char* text,
    int32_t text_len,
    int32_t* tokens,
    int32_t n_tokens_max,
    bool add_special
);

/**
 * Detokenize token IDs to text.
 *
 * @param engine Batch engine
 * @param tokens Input token array
 * @param n_tokens Number of tokens
 * @param text Output text buffer
 * @param text_len_max Maximum text length
 * @return Length of output text, or negative on error
 */
int32_t bitnet_detokenize(
    BitNetBatchEngine* engine,
    const int32_t* tokens,
    int32_t n_tokens,
    char* text,
    int32_t text_len_max
);

// =============================================================================
// KV Cache Management (Per-Sequence)
// =============================================================================

/**
 * Remove KV cache entries for a sequence.
 *
 * @param engine Batch engine
 * @param seq_id Sequence to clear (-1 for all sequences)
 * @param p0 Start position (-1 for beginning)
 * @param p1 End position (-1 for end)
 * @return 0 on success
 */
int bitnet_kv_cache_seq_rm(
    BitNetBatchEngine* engine,
    bitnet_seq_id seq_id,
    int32_t p0,
    int32_t p1
);

/**
 * Copy KV cache from one sequence to another.
 *
 * Useful for sharing system prompts across sequences.
 *
 * @param engine Batch engine
 * @param seq_id_src Source sequence
 * @param seq_id_dst Destination sequence
 * @param p0 Start position
 * @param p1 End position
 */
void bitnet_kv_cache_seq_cp(
    BitNetBatchEngine* engine,
    bitnet_seq_id seq_id_src,
    bitnet_seq_id seq_id_dst,
    int32_t p0,
    int32_t p1
);

/**
 * Get maximum position in KV cache for a sequence.
 *
 * @param engine Batch engine
 * @param seq_id Sequence ID
 * @return Maximum position, or -1 if sequence has no cache
 */
int32_t bitnet_kv_cache_seq_pos_max(
    BitNetBatchEngine* engine,
    bitnet_seq_id seq_id
);

/**
 * Get number of used KV cache cells.
 */
int32_t bitnet_kv_cache_used_cells(BitNetBatchEngine* engine);

/**
 * Get total KV cache capacity.
 */
int32_t bitnet_kv_cache_capacity(BitNetBatchEngine* engine);

/**
 * Clear entire KV cache.
 */
void bitnet_kv_cache_clear(BitNetBatchEngine* engine);

// =============================================================================
// Model Information
// =============================================================================

/**
 * Get EOS token ID.
 */
int32_t bitnet_batch_eos_token(BitNetBatchEngine* engine);

/**
 * Check if token is end-of-generation token.
 */
bool bitnet_batch_is_eos(BitNetBatchEngine* engine, int32_t token);

/**
 * Get vocabulary size.
 */
int32_t bitnet_batch_vocab_size(BitNetBatchEngine* engine);

/**
 * Get context length.
 */
int32_t bitnet_batch_n_ctx(BitNetBatchEngine* engine);

/**
 * Get embedding dimension.
 */
int32_t bitnet_batch_n_embd(BitNetBatchEngine* engine);

/**
 * Get maximum number of concurrent sequences.
 */
int32_t bitnet_batch_max_sequences(BitNetBatchEngine* engine);

/**
 * Get number of currently active sequences.
 */
int32_t bitnet_batch_active_sequences(BitNetBatchEngine* engine);

/**
 * Get maximum context length per sequence.
 */
int32_t bitnet_batch_max_ctx_per_seq(BitNetBatchEngine* engine);

/**
 * Get last error message.
 */
const char* bitnet_batch_get_error(void);

#ifdef __cplusplus
}
#endif

#endif // BITNET_BATCH_H
