/**
 * BitNet Batch Inference Implementation - llama.cpp Backend
 *
 * Uses llama.cpp's native multi-sequence batching for continuous batching.
 * This replaces the custom forward pass implementation with llama.cpp calls.
 */

#include "bitnet_batch.h"

// Include llama.cpp headers
#include "llama.h"
#include "ggml.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <mutex>

// Thread-local error message
static thread_local std::string g_batch_last_error;

static void set_batch_error(const std::string& msg) {
    g_batch_last_error = msg;
}

// =============================================================================
// Batch Engine Implementation (llama.cpp backend)
// =============================================================================

struct BitNetBatchEngine {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;

    BitNetBatchConfig batch_config;

    // Model info (queried from llama.cpp)
    int32_t n_vocab = 0;
    int32_t n_ctx = 0;
    int32_t n_embd = 0;
    int32_t n_layer = 0;

    // Track active sequences
    std::vector<bool> seq_active;

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
        set_batch_error("Model path is NULL");
        return nullptr;
    }

    // Initialize llama backend
    llama_backend_init();

    auto engine = new BitNetBatchEngine();
    engine->model_path = model_path;

    // Apply config
    if (config) {
        engine->batch_config = *config;
    } else {
        engine->batch_config = bitnet_batch_config_default();
    }
    auto& bcfg = engine->batch_config;

    // Model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU only for BitNet
    model_params.use_mmap = true;

    // Load model
    engine->model = llama_load_model_from_file(model_path, model_params);
    if (!engine->model) {
        set_batch_error(std::string("Failed to load model from: ") + model_path);
        delete engine;
        return nullptr;
    }

    // Context parameters with multi-sequence support
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = bcfg.n_ctx;
    ctx_params.n_batch = bcfg.max_batch_size;
    ctx_params.n_seq_max = bcfg.max_sequences;  // Enable multi-sequence batching!
    ctx_params.n_threads = bcfg.num_threads > 0 ? bcfg.num_threads : 8;
    ctx_params.n_threads_batch = ctx_params.n_threads;

    // Create context
    engine->ctx = llama_new_context_with_model(engine->model, ctx_params);
    if (!engine->ctx) {
        set_batch_error("Failed to create llama context");
        llama_free_model(engine->model);
        delete engine;
        return nullptr;
    }

    // Store model info
    engine->n_vocab = llama_n_vocab(engine->model);
    engine->n_ctx = llama_n_ctx(engine->ctx);
    engine->n_embd = llama_n_embd(engine->model);
    engine->n_layer = llama_n_layer(engine->model);

    // Track active sequences
    engine->seq_active.resize(bcfg.max_sequences, false);

    // Initialize RNG
    engine->rng.seed(42);

    engine->is_loaded = true;
    return engine;
}

void bitnet_batch_engine_destroy(BitNetBatchEngine* engine) {
    if (engine) {
        if (engine->ctx) {
            llama_free(engine->ctx);
        }
        if (engine->model) {
            llama_free_model(engine->model);
        }
        delete engine;
    }
}

const char* bitnet_batch_get_error(void) {
    return g_batch_last_error.c_str();
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
// Inference (using llama.cpp)
// =============================================================================

int bitnet_batch_decode(BitNetBatchEngine* engine, const BitNetBatch* batch) {
    if (!engine || !engine->ctx || !batch) {
        set_batch_error("Invalid engine or batch");
        return -1;
    }

    if (batch->n_tokens == 0) {
        return 0;  // Nothing to decode
    }

    // Convert our batch format to llama_batch
    llama_batch lb = llama_batch_init(batch->n_tokens, 0, batch->_n_seq_max);

    for (int32_t i = 0; i < batch->n_tokens; i++) {
        lb.token[i] = batch->token[i];
        lb.pos[i] = batch->pos[i];
        lb.n_seq_id[i] = batch->n_seq_id[i];
        for (int32_t j = 0; j < batch->n_seq_id[i]; j++) {
            lb.seq_id[i][j] = batch->seq_id[i][j];
        }
        lb.logits[i] = batch->logits[i];
    }
    lb.n_tokens = batch->n_tokens;

    // Call llama.cpp decode
    int result = llama_decode(engine->ctx, lb);

    llama_batch_free(lb);

    if (result != 0) {
        set_batch_error("llama_decode failed");
    }

    return result;
}

const float* bitnet_get_logits_ith(BitNetBatchEngine* engine, int32_t batch_idx) {
    if (!engine || !engine->ctx) return nullptr;
    return llama_get_logits_ith(engine->ctx, batch_idx);
}

int32_t bitnet_batch_sample(
    BitNetBatchEngine* engine,
    int32_t batch_idx,
    const BitNetSamplingParams* params
) {
    if (!engine || !engine->ctx) return -1;

    const float* logits = llama_get_logits_ith(engine->ctx, batch_idx);
    if (!logits) return -1;

    float temperature = params ? params->temperature : 0.7f;
    float top_p = params ? params->top_p : 0.9f;

    int32_t next_token;
    if (temperature <= 0.0f) {
        // Greedy sampling
        next_token = std::distance(logits,
            std::max_element(logits, logits + engine->n_vocab));
    } else {
        next_token = sample_top_p(logits, engine->n_vocab, temperature, top_p, engine->rng);
    }

    return next_token;
}

// =============================================================================
// KV Cache Management (delegating to llama.cpp)
// =============================================================================

int bitnet_kv_cache_seq_rm(
    BitNetBatchEngine* engine,
    bitnet_seq_id seq_id,
    int32_t p0,
    int32_t p1
) {
    if (!engine || !engine->ctx) return -1;

    bool success = llama_kv_cache_seq_rm(engine->ctx, seq_id, p0, p1);

    // Mark sequence as inactive if we're clearing it entirely
    if (p0 == 0 && p1 == -1 && seq_id >= 0 && seq_id < (int32_t)engine->seq_active.size()) {
        engine->seq_active[seq_id] = false;
    }

    return success ? 0 : -1;
}

void bitnet_kv_cache_seq_cp(
    BitNetBatchEngine* engine,
    bitnet_seq_id src,
    bitnet_seq_id dst,
    int32_t p0,
    int32_t p1
) {
    if (!engine || !engine->ctx) return;
    llama_kv_cache_seq_cp(engine->ctx, src, dst, p0, p1);
}

int32_t bitnet_kv_cache_seq_pos_max(BitNetBatchEngine* engine, bitnet_seq_id seq_id) {
    if (!engine || !engine->ctx) return -1;
    return llama_kv_cache_seq_pos_max(engine->ctx, seq_id);
}

// =============================================================================
// Sequence Management
// =============================================================================

bitnet_seq_id bitnet_seq_alloc(BitNetBatchEngine* engine) {
    if (!engine) return -1;

    for (size_t i = 0; i < engine->seq_active.size(); i++) {
        if (!engine->seq_active[i]) {
            engine->seq_active[i] = true;
            return static_cast<bitnet_seq_id>(i);
        }
    }
    return -1;  // No available slots
}

void bitnet_seq_free(BitNetBatchEngine* engine, bitnet_seq_id seq_id) {
    if (!engine || seq_id < 0 || seq_id >= (int32_t)engine->seq_active.size()) return;

    // Clear KV cache for this sequence
    llama_kv_cache_seq_rm(engine->ctx, seq_id, 0, -1);
    engine->seq_active[seq_id] = false;
}

int bitnet_seq_get_info(
    BitNetBatchEngine* engine,
    bitnet_seq_id seq_id,
    BitNetSeqInfo* info
) {
    if (!engine || !info || seq_id < 0 || seq_id >= (int32_t)engine->seq_active.size()) {
        return -1;
    }

    info->seq_id = seq_id;
    info->is_active = engine->seq_active[seq_id];
    info->position = llama_kv_cache_seq_pos_max(engine->ctx, seq_id) + 1;
    info->prompt_len = 0;  // Not tracked
    info->generated_count = 0;  // Not tracked

    return 0;
}

// =============================================================================
// Tokenization (using llama.cpp)
// =============================================================================

int32_t bitnet_tokenize(
    BitNetBatchEngine* engine,
    const char* text,
    int32_t text_len,
    int32_t* tokens,
    int32_t n_tokens_max,
    bool add_special
) {
    if (!engine || !engine->model || !text || !tokens) return -1;

    return llama_tokenize(
        engine->model,
        text,
        text_len,
        tokens,
        n_tokens_max,
        add_special,
        true  // parse_special
    );
}

int32_t bitnet_detokenize(
    BitNetBatchEngine* engine,
    const int32_t* tokens,
    int32_t n_tokens,
    char* text,
    int32_t text_len_max
) {
    if (!engine || !engine->model || !tokens || !text || n_tokens <= 0) return -1;

    int32_t total_len = 0;
    for (int32_t i = 0; i < n_tokens && total_len < text_len_max - 1; i++) {
        char piece[256];
        int32_t piece_len = llama_token_to_piece(
            engine->model, tokens[i], piece, sizeof(piece) - 1, 0, true
        );
        if (piece_len < 0) continue;

        int32_t copy_len = std::min(piece_len, text_len_max - total_len - 1);
        memcpy(text + total_len, piece, copy_len);
        total_len += copy_len;
    }
    text[total_len] = '\0';
    return total_len;
}

// =============================================================================
// Model Info
// =============================================================================

int32_t bitnet_batch_vocab_size(BitNetBatchEngine* engine) {
    return engine ? engine->n_vocab : 0;
}

int32_t bitnet_batch_n_ctx(BitNetBatchEngine* engine) {
    return engine ? engine->n_ctx : 0;
}

int32_t bitnet_batch_n_embd(BitNetBatchEngine* engine) {
    return engine ? engine->n_embd : 0;
}

int32_t bitnet_batch_eos_token(BitNetBatchEngine* engine) {
    if (!engine || !engine->model) return -1;
    return llama_token_eos(engine->model);
}

bool bitnet_batch_is_eos(BitNetBatchEngine* engine, int32_t token) {
    if (!engine || !engine->model) return false;
    return llama_token_is_eog(engine->model, token);
}

int32_t bitnet_batch_max_sequences(BitNetBatchEngine* engine) {
    return engine ? engine->batch_config.max_sequences : 0;
}

int32_t bitnet_batch_active_sequences(BitNetBatchEngine* engine) {
    if (!engine) return 0;
    int32_t count = 0;
    for (bool active : engine->seq_active) {
        if (active) count++;
    }
    return count;
}

}  // extern "C"
