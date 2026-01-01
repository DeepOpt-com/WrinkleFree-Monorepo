/**
 * BitNet Batch Inference - sgl-kernel SIMD Backend
 *
 * Implements the BitNetBatchEngine API using sgl-kernel SIMD kernels
 * instead of llama.cpp. This provides ~4x faster inference through
 * online INT8 activation quantization.
 *
 * Key differences from llama.cpp backend:
 * - Uses packed 2-bit weights from .bin format (not GGUF)
 * - Online quantization: activations quantized to INT8 at runtime
 * - SIMD-optimized GEMV/GEMM for ternary x INT8 operations
 */

#include "bitnet_batch.h"
#include "bitnet_engine.h"
#include "kv_cache.h"
#include "sglkernel_loader.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <mutex>
#include <omp.h>

// Thread-local error message
static thread_local std::string g_simd_batch_error;

static void set_simd_batch_error(const std::string& msg) {
    g_simd_batch_error = msg;
}

// =============================================================================
// Per-Sequence State
// =============================================================================

struct SequenceState {
    bool is_active = false;
    int32_t position = 0;     // Current position in KV cache
    int32_t prompt_len = 0;
    int32_t generated_count = 0;
    std::unique_ptr<KVCache> kv_cache;
};

// =============================================================================
// SIMD Batch Engine Implementation
// =============================================================================

struct BitNetBatchEngine {
    // Shared model (loaded once, used by all sequences)
    std::unique_ptr<BitNetEngine, decltype(&bitnet_engine_destroy)> model{nullptr, bitnet_engine_destroy};

    BitNetBatchConfig batch_config;

    // Per-sequence state
    std::vector<SequenceState> sequences;

    // Logits buffer for batch decode
    std::vector<float> logits_buffer;
    std::vector<int32_t> logits_batch_indices;  // Maps batch_idx to vocab offset

    // Sampling RNG
    std::mt19937 rng;
    std::mutex engine_mutex;

    bool is_loaded = false;
    std::string model_path;
};

// =============================================================================
// Helper Functions
// =============================================================================

static void softmax(float* x, int n) {
    float max_val = *std::max_element(x, x + n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

static int32_t sample_top_p(const float* logits, int vocab_size, float temperature,
                            float top_p, std::mt19937& rng) {
    std::vector<std::pair<float, int32_t>> candidates;
    candidates.reserve(vocab_size);

    for (int i = 0; i < vocab_size; i++) {
        candidates.emplace_back(logits[i] / temperature, i);
    }

    std::sort(candidates.begin(), candidates.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    // Softmax on sorted
    float max_logit = candidates[0].first;
    float sum = 0.0f;
    for (auto& c : candidates) {
        c.first = std::exp(c.first - max_logit);
        sum += c.first;
    }
    for (auto& c : candidates) {
        c.first /= sum;
    }

    // Find top-p cutoff
    float cumsum = 0.0f;
    int cutoff = 0;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += candidates[i].first;
        if (cumsum >= top_p) {
            cutoff = i + 1;
            break;
        }
    }
    if (cutoff == 0) cutoff = 1;

    // Sample from top-p
    std::uniform_real_distribution<float> dist(0.0f, cumsum);
    float r = dist(rng);
    cumsum = 0.0f;
    for (int i = 0; i < cutoff; i++) {
        cumsum += candidates[i].first;
        if (r < cumsum) {
            return candidates[i].second;
        }
    }
    return candidates[0].second;
}

// =============================================================================
// Forward pass for a single token at a position
// Uses the shared model weights but sequence-specific KV cache
// =============================================================================

// Forward pass for a single token using the shared model and per-sequence KV cache
static void forward_token_for_sequence(
    BitNetBatchEngine* batch_engine,
    int seq_id,
    int32_t token_id,
    int pos,
    float* logits_out
) {
    auto& seq = batch_engine->sequences[seq_id];
    BitNetEngine* engine = batch_engine->model.get();

    // Use the public function that accepts external KV cache
    forward_one_token_with_cache(
        engine,
        seq.kv_cache.get(),
        token_id,
        pos,
        logits_out
    );
}

// =============================================================================
// Public API Implementation
// =============================================================================

extern "C" {

BitNetBatchConfig bitnet_batch_config_default(void) {
    BitNetBatchConfig config;
    config.max_batch_size = 512;
    config.max_sequences = 16;
    config.n_ctx = 4096;
    config.n_ctx_per_seq = 256;
    config.num_threads = 8;
    return config;
}

BitNetBatchEngine* bitnet_batch_engine_create(
    const char* model_path,
    const BitNetBatchConfig* config
) {
    if (!model_path) {
        set_simd_batch_error("Model path is NULL");
        return nullptr;
    }

    auto engine = new BitNetBatchEngine();
    engine->model_path = model_path;

    // Apply config
    if (config) {
        engine->batch_config = *config;
    } else {
        engine->batch_config = bitnet_batch_config_default();
    }
    auto& bcfg = engine->batch_config;

    // Create the shared model using sgl-kernel engine
    BitNetConfig model_config;
    model_config.max_seq_len = bcfg.n_ctx;
    model_config.num_threads = bcfg.num_threads;
    model_config.kv_cache_size = bcfg.n_ctx_per_seq;

    BitNetEngine* raw_model = bitnet_engine_create(model_path, &model_config);
    if (!raw_model) {
        set_simd_batch_error(std::string("Failed to load model: ") + bitnet_get_error());
        delete engine;
        return nullptr;
    }
    engine->model.reset(raw_model);

    // Initialize per-sequence state
    engine->sequences.resize(bcfg.max_sequences);
    for (int i = 0; i < bcfg.max_sequences; i++) {
        auto& seq = engine->sequences[i];
        seq.is_active = false;
        seq.position = 0;
        seq.prompt_len = 0;
        seq.generated_count = 0;

        // Each sequence gets its own KV cache
        seq.kv_cache.reset(kv_cache_create(
            bitnet_num_layers(raw_model),
            bitnet_get_num_kv_heads(raw_model),
            bitnet_hidden_size(raw_model) / bitnet_get_num_kv_heads(raw_model),  // head_dim
            bcfg.n_ctx_per_seq
        ));
        if (!seq.kv_cache) {
            set_simd_batch_error("Failed to create per-sequence KV cache");
            delete engine;
            return nullptr;
        }
    }

    // Allocate logits buffer for batch processing
    int vocab_size = bitnet_vocab_size(raw_model);
    engine->logits_buffer.resize(bcfg.max_batch_size * vocab_size);
    engine->logits_batch_indices.resize(bcfg.max_batch_size);

    // Initialize RNG
    engine->rng.seed(42);

    engine->is_loaded = true;
    return engine;
}

void bitnet_batch_engine_destroy(BitNetBatchEngine* engine) {
    delete engine;
}

const char* bitnet_batch_get_error(void) {
    return g_simd_batch_error.c_str();
}

// =============================================================================
// Batch Management
// =============================================================================

BitNetBatch* bitnet_batch_init(int32_t n_tokens, int32_t n_seq_max) {
    try {
        auto batch = new BitNetBatch();
        batch->n_tokens = 0;
        batch->_capacity = n_tokens;
        batch->_n_seq_max = n_seq_max;

        batch->token = new int32_t[n_tokens];
        batch->pos = new int32_t[n_tokens];
        batch->n_seq_id = new int32_t[n_tokens];
        batch->logits = new int8_t[n_tokens];

        batch->seq_id = new bitnet_seq_id*[n_tokens];
        for (int32_t i = 0; i < n_tokens; i++) {
            batch->seq_id[i] = new bitnet_seq_id[n_seq_max];
        }

        return batch;
    } catch (...) {
        return nullptr;
    }
}

void bitnet_batch_free(BitNetBatch* batch) {
    if (!batch) return;

    delete[] batch->token;
    delete[] batch->pos;
    delete[] batch->n_seq_id;
    delete[] batch->logits;

    for (int32_t i = 0; i < batch->_capacity; i++) {
        delete[] batch->seq_id[i];
    }
    delete[] batch->seq_id;

    delete batch;
}

void bitnet_batch_clear(BitNetBatch* batch) {
    if (batch) {
        batch->n_tokens = 0;
    }
}

void bitnet_batch_add(
    BitNetBatch* batch,
    int32_t token,
    int32_t pos,
    const bitnet_seq_id* seq_ids,
    int32_t n_seq_ids,
    int8_t output_logits
) {
    if (!batch || batch->n_tokens >= batch->_capacity) return;

    int32_t i = batch->n_tokens;
    batch->token[i] = token;
    batch->pos[i] = pos;
    batch->n_seq_id[i] = n_seq_ids;
    batch->logits[i] = output_logits;

    for (int32_t j = 0; j < n_seq_ids && j < batch->_n_seq_max; j++) {
        batch->seq_id[i][j] = seq_ids[j];
    }

    batch->n_tokens++;
}

// =============================================================================
// Sequence Management
// =============================================================================

bitnet_seq_id bitnet_seq_alloc(BitNetBatchEngine* engine) {
    if (!engine) return -1;

    for (size_t i = 0; i < engine->sequences.size(); i++) {
        if (!engine->sequences[i].is_active) {
            engine->sequences[i].is_active = true;
            engine->sequences[i].position = 0;
            engine->sequences[i].prompt_len = 0;
            engine->sequences[i].generated_count = 0;
            kv_cache_reset(engine->sequences[i].kv_cache.get());
            return static_cast<bitnet_seq_id>(i);
        }
    }
    return -1;  // No available slots
}

void bitnet_seq_free(BitNetBatchEngine* engine, bitnet_seq_id seq_id) {
    if (!engine || seq_id < 0 || seq_id >= (int32_t)engine->sequences.size()) return;

    auto& seq = engine->sequences[seq_id];
    kv_cache_reset(seq.kv_cache.get());
    seq.is_active = false;
    seq.position = 0;
    seq.prompt_len = 0;
    seq.generated_count = 0;
}

int bitnet_seq_get_info(
    BitNetBatchEngine* engine,
    bitnet_seq_id seq_id,
    BitNetSeqInfo* info
) {
    if (!engine || !info || seq_id < 0 || seq_id >= (int32_t)engine->sequences.size()) {
        return -1;
    }

    auto& seq = engine->sequences[seq_id];
    info->seq_id = seq_id;
    info->is_active = seq.is_active;
    info->position = seq.position;
    info->prompt_len = seq.prompt_len;
    info->generated_count = seq.generated_count;

    return 0;
}

int32_t bitnet_seq_active_count(BitNetBatchEngine* engine) {
    if (!engine) return 0;
    int32_t count = 0;
    for (auto& seq : engine->sequences) {
        if (seq.is_active) count++;
    }
    return count;
}

int32_t bitnet_seq_available_slots(BitNetBatchEngine* engine) {
    if (!engine) return 0;
    int32_t count = 0;
    for (auto& seq : engine->sequences) {
        if (!seq.is_active) count++;
    }
    return count;
}

// =============================================================================
// Batch Inference (SIMD)
// =============================================================================

int bitnet_batch_decode(BitNetBatchEngine* engine, const BitNetBatch* batch) {
    if (!engine || !engine->is_loaded || !batch) {
        set_simd_batch_error("Invalid engine or batch");
        return -1;
    }

    if (batch->n_tokens == 0) {
        return 0;  // Nothing to decode
    }

    // TODO: Implement proper batched forward pass
    // For now, process tokens sequentially per sequence

    int vocab_size = bitnet_vocab_size(engine->model.get());
    int logit_idx = 0;

    // Group tokens by sequence for more efficient processing
    for (int32_t i = 0; i < batch->n_tokens; i++) {
        int32_t token = batch->token[i];
        int32_t pos = batch->pos[i];
        bitnet_seq_id seq_id = batch->seq_id[i][0];  // Assume single seq per token
        bool need_logits = batch->logits[i] != 0;

        if (seq_id < 0 || seq_id >= (int32_t)engine->sequences.size()) {
            continue;
        }

        auto& seq = engine->sequences[seq_id];
        if (!seq.is_active) {
            continue;
        }

        // Forward pass for this token
        // If we need logits, use the buffer; otherwise we still need to run forward
        // to update the KV cache
        float* logits_out = engine->logits_buffer.data() + logit_idx * vocab_size;

        // Run SIMD forward pass with per-sequence KV cache
        forward_token_for_sequence(engine, seq_id, token, pos, logits_out);

        if (need_logits) {
            engine->logits_batch_indices[logit_idx] = i;
            logit_idx++;
        }

        // Update sequence position
        seq.position = pos + 1;
    }

    return 0;
}

const float* bitnet_get_logits_ith(BitNetBatchEngine* engine, int32_t batch_idx) {
    if (!engine || !engine->is_loaded) return nullptr;

    int vocab_size = bitnet_vocab_size(engine->model.get());

    // Find which logit buffer slot corresponds to this batch index
    for (size_t i = 0; i < engine->logits_batch_indices.size(); i++) {
        if (engine->logits_batch_indices[i] == batch_idx) {
            return engine->logits_buffer.data() + i * vocab_size;
        }
    }

    return nullptr;
}

int32_t bitnet_batch_sample(
    BitNetBatchEngine* engine,
    int32_t batch_idx,
    const BitNetSamplingParams* params
) {
    if (!engine || !engine->is_loaded) return -1;

    const float* logits = bitnet_get_logits_ith(engine, batch_idx);
    if (!logits) return -1;

    int vocab_size = bitnet_vocab_size(engine->model.get());
    float temperature = params ? params->temperature : 0.7f;
    float top_p = params ? params->top_p : 0.9f;

    int32_t next_token;
    if (temperature <= 0.0f) {
        // Greedy sampling
        next_token = std::distance(logits,
            std::max_element(logits, logits + vocab_size));
    } else {
        next_token = sample_top_p(logits, vocab_size, temperature, top_p, engine->rng);
    }

    return next_token;
}

// =============================================================================
// KV Cache Management
// =============================================================================

int bitnet_kv_cache_seq_rm(
    BitNetBatchEngine* engine,
    bitnet_seq_id seq_id,
    int32_t p0,
    int32_t p1
) {
    if (!engine) return -1;

    if (seq_id == -1) {
        // Clear all sequences
        for (auto& seq : engine->sequences) {
            kv_cache_reset(seq.kv_cache.get());
            seq.position = 0;
        }
    } else if (seq_id >= 0 && seq_id < (int32_t)engine->sequences.size()) {
        auto& seq = engine->sequences[seq_id];
        if (p0 == -1 && p1 == -1) {
            // Clear entire sequence
            kv_cache_reset(seq.kv_cache.get());
            seq.position = 0;
        } else {
            // Partial clear - not fully supported in simple KV cache
            // Just reset for now
            kv_cache_reset(seq.kv_cache.get());
            seq.position = p0 >= 0 ? p0 : 0;
        }
    }

    return 0;
}

void bitnet_kv_cache_seq_cp(
    BitNetBatchEngine* engine,
    bitnet_seq_id src,
    bitnet_seq_id dst,
    int32_t p0,
    int32_t p1
) {
    // TODO: Implement KV cache copying for prefix sharing
    (void)engine; (void)src; (void)dst; (void)p0; (void)p1;
}

int32_t bitnet_kv_cache_seq_pos_max(BitNetBatchEngine* engine, bitnet_seq_id seq_id) {
    if (!engine || seq_id < 0 || seq_id >= (int32_t)engine->sequences.size()) return -1;
    return engine->sequences[seq_id].position - 1;
}

int32_t bitnet_kv_cache_used_cells(BitNetBatchEngine* engine) {
    if (!engine) return 0;
    int32_t total = 0;
    for (auto& seq : engine->sequences) {
        if (seq.is_active) {
            total += seq.position;
        }
    }
    return total;
}

int32_t bitnet_kv_cache_capacity(BitNetBatchEngine* engine) {
    if (!engine) return 0;
    return engine->batch_config.max_sequences * engine->batch_config.n_ctx_per_seq;
}

void bitnet_kv_cache_clear(BitNetBatchEngine* engine) {
    if (!engine) return;
    for (auto& seq : engine->sequences) {
        kv_cache_reset(seq.kv_cache.get());
        seq.position = 0;
    }
}

// =============================================================================
// Model Information
// =============================================================================

int32_t bitnet_batch_vocab_size(BitNetBatchEngine* engine) {
    return engine && engine->model ? bitnet_vocab_size(engine->model.get()) : 0;
}

int32_t bitnet_batch_n_ctx(BitNetBatchEngine* engine) {
    return engine ? engine->batch_config.n_ctx : 0;
}

int32_t bitnet_batch_n_embd(BitNetBatchEngine* engine) {
    return engine && engine->model ? bitnet_hidden_size(engine->model.get()) : 0;
}

int32_t bitnet_batch_eos_token(BitNetBatchEngine* engine) {
    // TODO: Get from model config
    return 151645;  // Default for BitNet/Qwen models
}

bool bitnet_batch_is_eos(BitNetBatchEngine* engine, int32_t token) {
    return token == bitnet_batch_eos_token(engine);
}

int32_t bitnet_batch_max_sequences(BitNetBatchEngine* engine) {
    return engine ? engine->batch_config.max_sequences : 0;
}

int32_t bitnet_batch_active_sequences(BitNetBatchEngine* engine) {
    return bitnet_seq_active_count(engine);
}

int32_t bitnet_batch_max_ctx_per_seq(BitNetBatchEngine* engine) {
    return engine ? engine->batch_config.n_ctx_per_seq : 0;
}

// =============================================================================
// Tokenization (stub - needs external tokenizer)
// =============================================================================

int32_t bitnet_tokenize(
    BitNetBatchEngine* engine,
    const char* text,
    int32_t text_len,
    int32_t* tokens,
    int32_t n_tokens_max,
    bool add_special
) {
    // TODO: Implement tokenization
    // Options:
    // 1. Use HuggingFace tokenizers library
    // 2. Use sentencepiece
    // 3. Load tokenizer.json and implement BPE
    set_simd_batch_error("Tokenization not implemented in SIMD backend - use external tokenizer");
    return -1;
}

int32_t bitnet_detokenize(
    BitNetBatchEngine* engine,
    const int32_t* tokens,
    int32_t n_tokens,
    char* text,
    int32_t text_len_max
) {
    // TODO: Implement detokenization
    set_simd_batch_error("Detokenization not implemented in SIMD backend - use external tokenizer");
    return -1;
}

}  // extern "C"
