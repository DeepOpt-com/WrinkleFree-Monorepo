/**
 * BitNet Inference Engine - llama.cpp Backend
 *
 * This wraps llama.cpp's C API to provide inference for BitNet GGUF models.
 * Uses Microsoft's BitNet.cpp which extends llama.cpp with 1.58-bit kernels.
 *
 * MIT License - includes code from:
 * - llama.cpp (https://github.com/ggerganov/llama.cpp)
 * - BitNet.cpp (https://github.com/microsoft/BitNet)
 */

#include "bitnet_engine.h"

// Include llama.cpp headers
// Note: These paths assume BitNet.cpp is built in the parent project
#include "llama.h"
#include "ggml.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

// Thread-local error message
static thread_local std::string g_last_error;

// Engine implementation using llama.cpp
struct BitNetEngine {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;

    // Model info
    int32_t n_vocab = 0;
    int32_t n_ctx = 0;
    int32_t n_embd = 0;
    int32_t n_layer = 0;

    // Generation state
    std::vector<llama_token> tokens;
    int32_t n_past = 0;

    // Sampling
    std::mt19937 rng;

    std::string model_path;
};

// ============================================================================
// Helper Functions
// ============================================================================

static void set_error(const std::string& msg) {
    g_last_error = msg;
}

// Softmax for sampling
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

// Top-p (nucleus) sampling
static llama_token sample_top_p(const float* logits, int n_vocab, float temperature,
                                 float top_p, std::mt19937& rng) {
    // Apply temperature
    std::vector<std::pair<float, llama_token>> candidates;
    candidates.reserve(n_vocab);

    for (int i = 0; i < n_vocab; i++) {
        candidates.emplace_back(logits[i] / temperature, i);
    }

    // Sort by logit
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
    for (int i = 0; i < n_vocab; i++) {
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

// ============================================================================
// Public C API Implementation
// ============================================================================

extern "C" {

BitNetEngine* bitnet_engine_create(const char* model_path, const BitNetConfig* config) {
    if (!model_path) {
        set_error("Model path is NULL");
        return nullptr;
    }

    // Initialize llama backend
    llama_backend_init();

    auto engine = new BitNetEngine();
    engine->model_path = model_path;

    // Model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU only for BitNet
    model_params.use_mmap = true;

    // Load model
    engine->model = llama_load_model_from_file(model_path, model_params);
    if (!engine->model) {
        set_error(std::string("Failed to load model from: ") + model_path);
        delete engine;
        return nullptr;
    }

    // Context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config && config->max_seq_len > 0 ? config->max_seq_len : 4096;
    ctx_params.n_batch = 512;
    // Default to 8 threads for better CPU utilization
    ctx_params.n_threads = config && config->num_threads > 0 ? config->num_threads : 8;
    ctx_params.n_threads_batch = ctx_params.n_threads;

    // Create context
    engine->ctx = llama_new_context_with_model(engine->model, ctx_params);
    if (!engine->ctx) {
        set_error("Failed to create llama context");
        llama_free_model(engine->model);
        delete engine;
        return nullptr;
    }

    // Store model info
    engine->n_vocab = llama_n_vocab(engine->model);
    engine->n_ctx = llama_n_ctx(engine->ctx);
    engine->n_embd = llama_n_embd(engine->model);
    engine->n_layer = llama_n_layer(engine->model);

    // Initialize RNG
    engine->rng.seed(42);

    return engine;
}

void bitnet_engine_destroy(BitNetEngine* engine) {
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

const char* bitnet_get_error(void) {
    return g_last_error.c_str();
}

int bitnet_generate(
    BitNetEngine* engine,
    const int32_t* input_ids,
    int32_t num_input_tokens,
    const SamplingParams* params,
    GenerationResult* result
) {
    if (!engine || !engine->ctx) {
        set_error("Engine not initialized");
        return -1;
    }

    // Clear KV cache
    llama_kv_cache_clear(engine->ctx);

    // Convert input tokens
    std::vector<llama_token> tokens(input_ids, input_ids + num_input_tokens);

    // Process prompt (prefill)
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size(), 0, 0);
    if (llama_decode(engine->ctx, batch) != 0) {
        set_error("Failed to process prompt");
        return -1;
    }

    int n_past = tokens.size();

    // Generate tokens
    std::vector<llama_token> generated;
    int max_tokens = params ? params->max_tokens : 256;
    float temperature = params ? params->temperature : 0.7f;
    float top_p = params ? params->top_p : 0.9f;

    for (int i = 0; i < max_tokens; i++) {
        // Get logits
        float* logits = llama_get_logits_ith(engine->ctx, -1);
        if (!logits) {
            set_error("Failed to get logits");
            return -1;
        }

        // Sample next token
        llama_token next_token;
        if (temperature <= 0.0f) {
            // Greedy
            next_token = std::distance(logits,
                std::max_element(logits, logits + engine->n_vocab));
        } else {
            next_token = sample_top_p(logits, engine->n_vocab,
                                       temperature, top_p, engine->rng);
        }

        // Check for EOS
        if (llama_token_is_eog(engine->model, next_token)) {
            break;
        }

        generated.push_back(next_token);

        // Decode next token
        batch = llama_batch_get_one(&next_token, 1, n_past, 0);
        if (llama_decode(engine->ctx, batch) != 0) {
            set_error("Failed to decode token");
            return -1;
        }
        n_past++;
    }

    // Fill result
    result->num_tokens = generated.size();
    result->output_ids = static_cast<int32_t*>(
        malloc(generated.size() * sizeof(int32_t))
    );
    for (size_t i = 0; i < generated.size(); i++) {
        result->output_ids[i] = generated[i];
    }
    result->logits = nullptr;
    result->logits_size = 0;

    return 0;
}

int bitnet_prefill(
    BitNetEngine* engine,
    const int32_t* input_ids,
    int32_t num_tokens
) {
    if (!engine || !engine->ctx) {
        set_error("Engine not initialized");
        return -1;
    }

    // Clear KV cache
    llama_kv_cache_clear(engine->ctx);

    // Store tokens
    engine->tokens.assign(input_ids, input_ids + num_tokens);

    // Process all tokens
    llama_batch batch = llama_batch_get_one(
        reinterpret_cast<llama_token*>(engine->tokens.data()),
        engine->tokens.size(), 0, 0
    );

    if (llama_decode(engine->ctx, batch) != 0) {
        set_error("Prefill failed");
        return -1;
    }

    engine->n_past = num_tokens;
    return 0;
}

int bitnet_decode_step(
    BitNetEngine* engine,
    int32_t position,
    const SamplingParams* params,
    int32_t* output_id
) {
    if (!engine || !engine->ctx) {
        set_error("Engine not initialized");
        return -1;
    }

    // Get logits from last decode
    float* logits = llama_get_logits_ith(engine->ctx, -1);
    if (!logits) {
        set_error("Failed to get logits");
        return -1;
    }

    // Sample
    llama_token next_token;
    float temperature = params ? params->temperature : 0.7f;
    float top_p = params ? params->top_p : 0.9f;

    if (temperature <= 0.0f) {
        next_token = std::distance(logits,
            std::max_element(logits, logits + engine->n_vocab));
    } else {
        next_token = sample_top_p(logits, engine->n_vocab,
                                   temperature, top_p, engine->rng);
    }

    *output_id = next_token;

    // Decode the token for next step
    llama_batch batch = llama_batch_get_one(&next_token, 1, engine->n_past, 0);
    if (llama_decode(engine->ctx, batch) != 0) {
        set_error("Decode step failed");
        return -1;
    }
    engine->n_past++;

    return 0;
}

void bitnet_reset_cache(BitNetEngine* engine) {
    if (engine && engine->ctx) {
        llama_kv_cache_clear(engine->ctx);
        engine->n_past = 0;
        engine->tokens.clear();
    }
}

int32_t bitnet_vocab_size(BitNetEngine* engine) {
    return engine ? engine->n_vocab : 0;
}

int32_t bitnet_hidden_size(BitNetEngine* engine) {
    return engine ? engine->n_embd : 0;
}

int32_t bitnet_num_layers(BitNetEngine* engine) {
    return engine ? engine->n_layer : 0;
}

int32_t bitnet_max_seq_len(BitNetEngine* engine) {
    return engine ? engine->n_ctx : 0;
}

void bitnet_free_result(GenerationResult* result) {
    if (result) {
        free(result->output_ids);
        free(result->logits);
        result->output_ids = nullptr;
        result->logits = nullptr;
    }
}

}  // extern "C"
