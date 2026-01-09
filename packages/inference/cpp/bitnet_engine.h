/**
 * BitNet C++ Inference Engine - C-Compatible API
 *
 * This provides a C-compatible interface for Rust FFI to call BitNet inference
 * without going through Python. Uses the native SIMD kernels from sgl-kernel.
 */

#ifndef BITNET_ENGINE_H
#define BITNET_ENGINE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque engine handle
typedef struct BitNetEngine BitNetEngine;

// Engine configuration
typedef struct {
    int32_t max_seq_len;      // Maximum sequence length (default: 2048)
    int32_t num_threads;      // Number of threads (default: auto)
    int32_t kv_cache_size;    // KV cache size in MB (default: 256)
} BitNetConfig;

// Sampling parameters
typedef struct {
    float temperature;        // Temperature for sampling (0 = greedy)
    float top_p;              // Top-p (nucleus) sampling
    int32_t top_k;            // Top-k sampling (0 = disabled)
    float repetition_penalty; // Repetition penalty (1.0 = disabled)
    int32_t max_tokens;       // Maximum tokens to generate
} SamplingParams;

// Generation result
typedef struct {
    int32_t* output_ids;      // Generated token IDs (caller must free)
    int32_t num_tokens;       // Number of generated tokens
    float* logits;            // Final logits (optional, NULL if not requested)
    int32_t logits_size;      // Size of logits array
} GenerationResult;

// ============================================================================
// Engine Lifecycle
// ============================================================================

/**
 * Create a new BitNet engine from a HuggingFace model path.
 *
 * @param model_path Path to HuggingFace model directory or model ID
 * @param config Engine configuration (NULL for defaults)
 * @return Engine handle, or NULL on error
 */
BitNetEngine* bitnet_engine_create(const char* model_path, const BitNetConfig* config);

/**
 * Destroy engine and free all resources.
 */
void bitnet_engine_destroy(BitNetEngine* engine);

/**
 * Get last error message.
 * @return Error string (valid until next API call)
 */
const char* bitnet_get_error(void);

// ============================================================================
// Inference
// ============================================================================

/**
 * Run full forward pass and generate tokens.
 *
 * @param engine Engine handle
 * @param input_ids Input token IDs
 * @param num_input_tokens Number of input tokens
 * @param params Sampling parameters
 * @param result Output result (caller owns output_ids, must free)
 * @return 0 on success, negative on error
 */
int bitnet_generate(
    BitNetEngine* engine,
    const int32_t* input_ids,
    int32_t num_input_tokens,
    const SamplingParams* params,
    GenerationResult* result
);

/**
 * Prefill phase - process input tokens and populate KV cache.
 *
 * @param engine Engine handle
 * @param input_ids Input token IDs
 * @param num_tokens Number of input tokens
 * @return 0 on success, negative on error
 */
int bitnet_prefill(
    BitNetEngine* engine,
    const int32_t* input_ids,
    int32_t num_tokens
);

/**
 * Single decode step - generate one token.
 *
 * @param engine Engine handle
 * @param position Current position in sequence
 * @param params Sampling parameters
 * @param output_id Output: generated token ID
 * @return 0 on success, negative on error
 */
int bitnet_decode_step(
    BitNetEngine* engine,
    int32_t position,
    const SamplingParams* params,
    int32_t* output_id
);

/**
 * Reset KV cache (start new sequence).
 */
void bitnet_reset_cache(BitNetEngine* engine);

// ============================================================================
// Model Information
// ============================================================================

/**
 * Get model vocabulary size.
 */
int32_t bitnet_vocab_size(BitNetEngine* engine);

/**
 * Get model hidden dimension.
 */
int32_t bitnet_hidden_size(BitNetEngine* engine);

/**
 * Get number of layers.
 */
int32_t bitnet_num_layers(BitNetEngine* engine);

/**
 * Get maximum sequence length.
 */
int32_t bitnet_max_seq_len(BitNetEngine* engine);

// ============================================================================
// Memory Management
// ============================================================================

/**
 * Free result memory allocated by bitnet_generate.
 */
void bitnet_free_result(GenerationResult* result);

#ifdef __cplusplus
}
#endif

#endif // BITNET_ENGINE_H
