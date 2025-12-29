/**
 * BitNet Batch Inference Implementation
 *
 * Extends the BitNet engine with multi-sequence continuous batching.
 */

#include "bitnet_batch.h"
#include "kv_cache.h"
#include "../bitnet/bitnet_gemv.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <mutex>

// Thread-local error message
static thread_local std::string g_batch_last_error;

static void set_batch_error(const std::string& msg) {
    g_batch_last_error = msg;
}

// =============================================================================
// Multi-Sequence KV Cache
// =============================================================================

/// KV cache with per-sequence isolation
struct MultiSeqKVCache {
    int32_t num_layers;
    int32_t num_heads;
    int32_t head_dim;
    int32_t max_seq_len;        ///< Per-sequence max length
    int32_t max_sequences;      ///< Number of sequence slots

    // Layout: [max_sequences, num_layers, 2 (k/v), max_seq_len, num_heads, head_dim]
    float* data;
    size_t seq_stride;          ///< Stride between sequences
    size_t layer_stride;        ///< Stride between layers
    size_t kv_stride;           ///< Stride between K and V

    // Per-sequence state
    std::vector<int32_t> seq_positions;  ///< Current position per sequence
    std::vector<bool> seq_active;        ///< Whether slot is in use

    MultiSeqKVCache(int32_t n_layers, int32_t n_heads, int32_t h_dim,
                    int32_t max_len, int32_t max_seqs)
        : num_layers(n_layers), num_heads(n_heads), head_dim(h_dim),
          max_seq_len(max_len), max_sequences(max_seqs),
          seq_positions(max_seqs, 0), seq_active(max_seqs, false)
    {
        kv_stride = static_cast<size_t>(max_seq_len) * num_heads * head_dim;
        layer_stride = 2 * kv_stride;  // K and V
        seq_stride = static_cast<size_t>(num_layers) * layer_stride;

        size_t total_size = static_cast<size_t>(max_sequences) * seq_stride;
        data = static_cast<float*>(aligned_alloc(64, total_size * sizeof(float)));
        if (data) {
            memset(data, 0, total_size * sizeof(float));
        }
    }

    ~MultiSeqKVCache() {
        free(data);
    }

    float* get_key(int32_t seq_id, int32_t layer) {
        return data + seq_id * seq_stride + layer * layer_stride;
    }

    float* get_value(int32_t seq_id, int32_t layer) {
        return data + seq_id * seq_stride + layer * layer_stride + kv_stride;
    }

    void clear_seq(int32_t seq_id) {
        float* start = data + seq_id * seq_stride;
        memset(start, 0, seq_stride * sizeof(float));
        seq_positions[seq_id] = 0;
        seq_active[seq_id] = false;
    }

    void copy_seq(int32_t src, int32_t dst, int32_t p0, int32_t p1) {
        if (p0 < 0) p0 = 0;
        if (p1 < 0) p1 = seq_positions[src];

        int32_t len = p1 - p0;
        if (len <= 0) return;

        size_t pos_stride = static_cast<size_t>(num_heads) * head_dim;

        for (int32_t l = 0; l < num_layers; l++) {
            // Copy keys
            float* k_src = get_key(src, l) + p0 * pos_stride;
            float* k_dst = get_key(dst, l) + p0 * pos_stride;
            memcpy(k_dst, k_src, len * pos_stride * sizeof(float));

            // Copy values
            float* v_src = get_value(src, l) + p0 * pos_stride;
            float* v_dst = get_value(dst, l) + p0 * pos_stride;
            memcpy(v_dst, v_src, len * pos_stride * sizeof(float));
        }

        seq_positions[dst] = std::max(seq_positions[dst], p1);
    }

    int32_t alloc_seq() {
        for (int32_t i = 0; i < max_sequences; i++) {
            if (!seq_active[i]) {
                seq_active[i] = true;
                seq_positions[i] = 0;
                return i;
            }
        }
        return -1;  // No available slots
    }

    void free_seq(int32_t seq_id) {
        if (seq_id >= 0 && seq_id < max_sequences) {
            clear_seq(seq_id);
        }
    }

    int32_t active_count() const {
        int32_t count = 0;
        for (bool active : seq_active) {
            if (active) count++;
        }
        return count;
    }

    int32_t available_slots() const {
        return max_sequences - active_count();
    }
};

// =============================================================================
// Model Configuration (BitNet-b1.58-2B-4T)
// =============================================================================

struct BatchModelConfig {
    int32_t vocab_size = 152064;
    int32_t hidden_size = 2560;
    int32_t intermediate_size = 6912;
    int32_t num_hidden_layers = 26;
    int32_t num_attention_heads = 20;
    int32_t num_key_value_heads = 20;
    int32_t head_dim = 128;
    int32_t max_position_embeddings = 4096;
    float rms_norm_eps = 1e-6f;
    int32_t bos_token_id = 151643;
    int32_t eos_token_id = 151645;
    int32_t pad_token_id = 151643;
};

// Weight tensor (reuse from bitnet_engine.cpp pattern)
struct BatchWeightTensor {
    std::vector<uint8_t> packed_data;
    std::vector<float> fp32_data;
    std::vector<int32_t> shape;
    float scale = 1.0f;
    bool is_packed = false;
};

struct BatchLayerWeights {
    BatchWeightTensor q_proj, k_proj, v_proj, o_proj;
    BatchWeightTensor gate_proj, up_proj, down_proj;
    BatchWeightTensor input_layernorm, post_attention_layernorm;
};

// =============================================================================
// Batch Engine Implementation
// =============================================================================

struct BitNetBatchEngine {
    BatchModelConfig config;
    BitNetBatchConfig batch_config;
    std::vector<BatchLayerWeights> layers;
    BatchWeightTensor embed_tokens;
    BatchWeightTensor lm_head;
    BatchWeightTensor final_norm;

    std::unique_ptr<MultiSeqKVCache> kv_cache;

    // Scratch buffers for batch processing
    std::vector<float> hidden_states;   // [batch_size, hidden_size]
    std::vector<float> residual;
    std::vector<float> attn_output;
    std::vector<float> mlp_output;
    std::vector<float> logits_buffer;   // [batch_size, vocab_size]
    std::vector<int8_t> quant_buffer;

    // Track which batch positions have logits
    std::vector<int32_t> logits_indices;
    int32_t n_logits = 0;

    std::mt19937 rng;
    std::mutex engine_mutex;

    bool is_loaded = false;
};

// =============================================================================
// Helper Functions
// =============================================================================

static void rms_norm(float* output, const float* input, const float* weight,
                     int n, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += input[i] * input[i];
    }
    float rms = std::sqrt(sum_sq / n + eps);
    float scale = 1.0f / rms;
    for (int i = 0; i < n; i++) {
        output[i] = input[i] * scale * weight[i];
    }
}

static void silu(float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

static void elementwise_mul(float* output, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = a[i] * b[i];
    }
}

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

static int32_t sample_top_p(const float* probs, int vocab_size, float top_p,
                            std::mt19937& rng) {
    std::vector<std::pair<float, int32_t>> sorted_probs;
    sorted_probs.reserve(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        sorted_probs.emplace_back(probs[i], i);
    }
    std::sort(sorted_probs.begin(), sorted_probs.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    float cumsum = 0.0f;
    int cutoff = 0;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += sorted_probs[i].first;
        if (cumsum >= top_p) {
            cutoff = i + 1;
            break;
        }
    }
    if (cutoff == 0) cutoff = 1;

    std::uniform_real_distribution<float> dist(0.0f, cumsum);
    float r = dist(rng);
    cumsum = 0.0f;
    for (int i = 0; i < cutoff; i++) {
        cumsum += sorted_probs[i].first;
        if (r < cumsum) {
            return sorted_probs[i].second;
        }
    }
    return sorted_probs[0].second;
}

// =============================================================================
// BitNet Operations (using sgl-kernel)
// =============================================================================

static void bitnet_linear(float* output, const float* input, int8_t* quant_input,
                          const BatchWeightTensor& weight, int M, int K,
                          float* quant_scale) {
    using namespace sgl_kernel::bitnet;

    quantize_activations_i8(K, quant_input, input, quant_scale);

    const int K_packed = K / 4;
    for (int m = 0; m < M; m++) {
        bitnet_vec_dot_i2_i8(K, &output[m],
                             weight.packed_data.data() + m * K_packed,
                             quant_input);
        output[m] *= weight.scale * (*quant_scale);
    }
}

static void fp32_linear(float* output, const float* input, const float* weight,
                        int M, int K) {
    for (int m = 0; m < M; m++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += input[k] * weight[m * K + k];
        }
        output[m] = sum;
    }
}

static void embed_lookup(float* output, const float* embed_table,
                         int32_t token_id, int hidden_size) {
    std::memcpy(output, embed_table + token_id * hidden_size,
                hidden_size * sizeof(float));
}

// =============================================================================
// Attention Forward (Single Token, Specific Sequence)
// =============================================================================

static void attention_forward_batched(
    BitNetBatchEngine* engine,
    int layer_idx,
    int32_t token,
    int32_t pos,
    bitnet_seq_id seq_id,
    float* hidden,
    float* output
) {
    auto& cfg = engine->config;
    auto& layer = engine->layers[layer_idx];

    const int H = cfg.hidden_size;
    const int heads = cfg.num_attention_heads;
    const int head_dim = cfg.head_dim;

    std::vector<float> q(H), k(H), v(H);
    int8_t* quant_buf = engine->quant_buffer.data();
    float quant_scale;

    bitnet_linear(q.data(), hidden, quant_buf, layer.q_proj, H, H, &quant_scale);
    bitnet_linear(k.data(), hidden, quant_buf, layer.k_proj, H, H, &quant_scale);
    bitnet_linear(v.data(), hidden, quant_buf, layer.v_proj, H, H, &quant_scale);

    // Store K, V in cache for this sequence
    float* k_cache = engine->kv_cache->get_key(seq_id, layer_idx);
    float* v_cache = engine->kv_cache->get_value(seq_id, layer_idx);

    std::memcpy(k_cache + pos * heads * head_dim, k.data(), H * sizeof(float));
    std::memcpy(v_cache + pos * heads * head_dim, v.data(), H * sizeof(float));

    // Multi-head attention
    std::vector<float> attn_out(H, 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int h = 0; h < heads; h++) {
        const float* q_head = q.data() + h * head_dim;

        std::vector<float> scores(pos + 1);
        for (int p = 0; p <= pos; p++) {
            const float* k_p = k_cache + p * heads * head_dim + h * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q_head[d] * k_p[d];
            }
            scores[p] = score * scale;
        }

        softmax(scores.data(), pos + 1);

        float* out_head = attn_out.data() + h * head_dim;
        for (int p = 0; p <= pos; p++) {
            const float* v_p = v_cache + p * heads * head_dim + h * head_dim;
            for (int d = 0; d < head_dim; d++) {
                out_head[d] += scores[p] * v_p[d];
            }
        }
    }

    bitnet_linear(output, attn_out.data(), quant_buf, layer.o_proj, H, H, &quant_scale);
}

// =============================================================================
// MLP Forward
// =============================================================================

static void mlp_forward_batched(BitNetBatchEngine* engine, int layer_idx,
                                float* hidden, float* output) {
    auto& cfg = engine->config;
    auto& layer = engine->layers[layer_idx];

    const int H = cfg.hidden_size;
    const int I = cfg.intermediate_size;

    std::vector<float> gate(I), up(I);
    int8_t* quant_buf = engine->quant_buffer.data();
    float quant_scale;

    bitnet_linear(gate.data(), hidden, quant_buf, layer.gate_proj, I, H, &quant_scale);
    bitnet_linear(up.data(), hidden, quant_buf, layer.up_proj, I, H, &quant_scale);

    silu(gate.data(), I);
    elementwise_mul(gate.data(), gate.data(), up.data(), I);

    bitnet_linear(output, gate.data(), quant_buf, layer.down_proj, H, I, &quant_scale);
}

// =============================================================================
// Forward Pass for Single Token in Batch
// =============================================================================

static void forward_one_token_batched(
    BitNetBatchEngine* engine,
    int32_t token,
    int32_t pos,
    bitnet_seq_id seq_id,
    float* logits_out
) {
    auto& cfg = engine->config;
    const int H = cfg.hidden_size;

    // Use scratch buffers
    float* hidden = engine->hidden_states.data();
    float* residual = engine->residual.data();
    float* attn_out = engine->attn_output.data();
    float* mlp_out = engine->mlp_output.data();

    // Embedding lookup
    embed_lookup(hidden, engine->embed_tokens.fp32_data.data(), token, H);

    // Process layers
    for (int l = 0; l < cfg.num_hidden_layers; l++) {
        auto& layer = engine->layers[l];

        std::memcpy(residual, hidden, H * sizeof(float));

        rms_norm(hidden, hidden, layer.input_layernorm.fp32_data.data(),
                 H, cfg.rms_norm_eps);

        attention_forward_batched(engine, l, token, pos, seq_id, hidden, attn_out);

        for (int i = 0; i < H; i++) {
            hidden[i] = residual[i] + attn_out[i];
        }

        std::memcpy(residual, hidden, H * sizeof(float));

        rms_norm(hidden, hidden, layer.post_attention_layernorm.fp32_data.data(),
                 H, cfg.rms_norm_eps);

        mlp_forward_batched(engine, l, hidden, mlp_out);

        for (int i = 0; i < H; i++) {
            hidden[i] = residual[i] + mlp_out[i];
        }
    }

    // Final norm
    rms_norm(hidden, hidden, engine->final_norm.fp32_data.data(),
             H, cfg.rms_norm_eps);

    // LM head
    if (logits_out) {
        if (engine->lm_head.is_packed) {
            int8_t* quant_buf = engine->quant_buffer.data();
            float quant_scale;
            bitnet_linear(logits_out, hidden, quant_buf, engine->lm_head,
                          cfg.vocab_size, H, &quant_scale);
        } else {
            fp32_linear(logits_out, hidden, engine->lm_head.fp32_data.data(),
                        cfg.vocab_size, H);
        }
    }
}

// =============================================================================
// Model Loading (Placeholder)
// =============================================================================

static bool load_batch_model(BitNetBatchEngine* engine, const char* model_path) {
    auto& cfg = engine->config;
    const int H = cfg.hidden_size;
    const int I = cfg.intermediate_size;
    const int V = cfg.vocab_size;
    const int L = cfg.num_hidden_layers;

    engine->embed_tokens.fp32_data.resize(V * H);
    engine->embed_tokens.is_packed = false;

    engine->final_norm.fp32_data.resize(H, 1.0f);

    engine->lm_head.fp32_data.resize(V * H);
    engine->lm_head.is_packed = false;

    engine->layers.resize(L);
    for (int l = 0; l < L; l++) {
        auto& layer = engine->layers[l];

        layer.q_proj.packed_data.resize(H * H / 4);
        layer.q_proj.is_packed = true;
        layer.q_proj.scale = 1.0f;

        layer.k_proj.packed_data.resize(H * H / 4);
        layer.k_proj.is_packed = true;
        layer.k_proj.scale = 1.0f;

        layer.v_proj.packed_data.resize(H * H / 4);
        layer.v_proj.is_packed = true;
        layer.v_proj.scale = 1.0f;

        layer.o_proj.packed_data.resize(H * H / 4);
        layer.o_proj.is_packed = true;
        layer.o_proj.scale = 1.0f;

        layer.gate_proj.packed_data.resize(I * H / 4);
        layer.gate_proj.is_packed = true;
        layer.gate_proj.scale = 1.0f;

        layer.up_proj.packed_data.resize(I * H / 4);
        layer.up_proj.is_packed = true;
        layer.up_proj.scale = 1.0f;

        layer.down_proj.packed_data.resize(H * I / 4);
        layer.down_proj.is_packed = true;
        layer.down_proj.scale = 1.0f;

        layer.input_layernorm.fp32_data.resize(H, 1.0f);
        layer.post_attention_layernorm.fp32_data.resize(H, 1.0f);
    }

    return true;
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
    config.num_threads = 0;
    return config;
}

BitNetBatchEngine* bitnet_batch_engine_create(
    const char* model_path,
    const BitNetBatchConfig* config
) {
    try {
        auto engine = new BitNetBatchEngine();

        // Apply config
        if (config) {
            engine->batch_config = *config;
        } else {
            engine->batch_config = bitnet_batch_config_default();
        }

        auto& cfg = engine->config;
        auto& bcfg = engine->batch_config;

        // Create multi-sequence KV cache
        engine->kv_cache = std::make_unique<MultiSeqKVCache>(
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.head_dim,
            bcfg.n_ctx_per_seq,
            bcfg.max_sequences
        );

        if (!engine->kv_cache->data) {
            set_batch_error("Failed to allocate KV cache");
            delete engine;
            return nullptr;
        }

        // Allocate scratch buffers
        engine->hidden_states.resize(cfg.hidden_size);
        engine->residual.resize(cfg.hidden_size);
        engine->attn_output.resize(cfg.hidden_size);
        engine->mlp_output.resize(cfg.hidden_size);
        engine->logits_buffer.resize(bcfg.max_batch_size * cfg.vocab_size);
        engine->quant_buffer.resize(std::max(cfg.hidden_size, cfg.intermediate_size));
        engine->logits_indices.resize(bcfg.max_batch_size);

        engine->rng.seed(42);

        if (!load_batch_model(engine, model_path)) {
            delete engine;
            return nullptr;
        }

        engine->is_loaded = true;
        return engine;

    } catch (const std::exception& e) {
        set_batch_error(std::string("Batch engine creation failed: ") + e.what());
        return nullptr;
    }
}

void bitnet_batch_engine_destroy(BitNetBatchEngine* engine) {
    delete engine;
}

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

    for (int32_t s = 0; s < n_seq_ids && s < batch->_n_seq_max; s++) {
        batch->seq_id[i][s] = seq_ids[s];
    }

    batch->n_tokens++;
}

bitnet_seq_id bitnet_seq_alloc(BitNetBatchEngine* engine) {
    if (!engine) return -1;
    std::lock_guard<std::mutex> lock(engine->engine_mutex);
    return engine->kv_cache->alloc_seq();
}

void bitnet_seq_free(BitNetBatchEngine* engine, bitnet_seq_id seq_id) {
    if (!engine) return;
    std::lock_guard<std::mutex> lock(engine->engine_mutex);
    engine->kv_cache->free_seq(seq_id);
}

int bitnet_seq_get_info(
    BitNetBatchEngine* engine,
    bitnet_seq_id seq_id,
    BitNetSeqInfo* info
) {
    if (!engine || !info || seq_id < 0 ||
        seq_id >= engine->batch_config.max_sequences) {
        return -1;
    }

    info->seq_id = seq_id;
    info->state = engine->kv_cache->seq_active[seq_id]
        ? SEQ_STATE_DECODING : SEQ_STATE_IDLE;
    info->position = engine->kv_cache->seq_positions[seq_id];
    info->prompt_len = 0;  // TODO: track this
    info->generated_count = 0;  // TODO: track this
    return 0;
}

int32_t bitnet_seq_active_count(BitNetBatchEngine* engine) {
    if (!engine) return 0;
    return engine->kv_cache->active_count();
}

int32_t bitnet_seq_available_slots(BitNetBatchEngine* engine) {
    if (!engine) return 0;
    return engine->kv_cache->available_slots();
}

int bitnet_batch_decode(
    BitNetBatchEngine* engine,
    const BitNetBatch* batch
) {
    if (!engine || !batch || !engine->is_loaded) {
        set_batch_error("Invalid engine or batch");
        return -1;
    }

    std::lock_guard<std::mutex> lock(engine->engine_mutex);

    engine->n_logits = 0;

    // Process each token in the batch
    for (int32_t i = 0; i < batch->n_tokens; i++) {
        int32_t token = batch->token[i];
        int32_t pos = batch->pos[i];
        bitnet_seq_id seq_id = batch->seq_id[i][0];  // Primary sequence

        // Compute logits if requested
        float* logits_out = nullptr;
        if (batch->logits[i]) {
            logits_out = engine->logits_buffer.data() +
                         engine->n_logits * engine->config.vocab_size;
            engine->logits_indices[engine->n_logits] = i;
            engine->n_logits++;
        }

        forward_one_token_batched(engine, token, pos, seq_id, logits_out);

        // Update sequence position
        engine->kv_cache->seq_positions[seq_id] = pos + 1;
    }

    return 0;
}

const float* bitnet_get_logits_ith(
    BitNetBatchEngine* engine,
    int32_t batch_idx
) {
    if (!engine) return nullptr;

    // Find which logits slot corresponds to this batch index
    for (int32_t i = 0; i < engine->n_logits; i++) {
        if (engine->logits_indices[i] == batch_idx) {
            return engine->logits_buffer.data() + i * engine->config.vocab_size;
        }
    }
    return nullptr;
}

int32_t bitnet_batch_sample(
    BitNetBatchEngine* engine,
    int32_t batch_idx,
    const SamplingParams* params
) {
    if (!engine || !params) return -1;

    const float* logits = bitnet_get_logits_ith(engine, batch_idx);
    if (!logits) return -1;

    auto& cfg = engine->config;

    if (params->temperature <= 0.0f) {
        return std::distance(logits,
            std::max_element(logits, logits + cfg.vocab_size));
    }

    std::vector<float> probs(cfg.vocab_size);
    for (int v = 0; v < cfg.vocab_size; v++) {
        probs[v] = logits[v] / params->temperature;
    }
    softmax(probs.data(), cfg.vocab_size);

    float top_p = params->top_p > 0.0f ? params->top_p : 1.0f;
    return sample_top_p(probs.data(), cfg.vocab_size, top_p, engine->rng);
}

int bitnet_kv_cache_seq_rm(
    BitNetBatchEngine* engine,
    bitnet_seq_id seq_id,
    int32_t p0,
    int32_t p1
) {
    if (!engine) return -1;
    std::lock_guard<std::mutex> lock(engine->engine_mutex);

    if (seq_id < 0) {
        // Clear all
        for (int32_t i = 0; i < engine->batch_config.max_sequences; i++) {
            engine->kv_cache->clear_seq(i);
        }
    } else {
        engine->kv_cache->clear_seq(seq_id);
    }
    return 0;
}

void bitnet_kv_cache_seq_cp(
    BitNetBatchEngine* engine,
    bitnet_seq_id seq_id_src,
    bitnet_seq_id seq_id_dst,
    int32_t p0,
    int32_t p1
) {
    if (!engine) return;
    std::lock_guard<std::mutex> lock(engine->engine_mutex);
    engine->kv_cache->copy_seq(seq_id_src, seq_id_dst, p0, p1);
}

int32_t bitnet_kv_cache_seq_pos_max(
    BitNetBatchEngine* engine,
    bitnet_seq_id seq_id
) {
    if (!engine || seq_id < 0 ||
        seq_id >= engine->batch_config.max_sequences) {
        return -1;
    }
    return engine->kv_cache->seq_positions[seq_id];
}

int32_t bitnet_kv_cache_used_cells(BitNetBatchEngine* engine) {
    if (!engine) return 0;
    int32_t total = 0;
    for (int32_t i = 0; i < engine->batch_config.max_sequences; i++) {
        if (engine->kv_cache->seq_active[i]) {
            total += engine->kv_cache->seq_positions[i];
        }
    }
    return total;
}

int32_t bitnet_kv_cache_capacity(BitNetBatchEngine* engine) {
    if (!engine) return 0;
    return engine->batch_config.max_sequences *
           engine->batch_config.n_ctx_per_seq;
}

void bitnet_kv_cache_clear(BitNetBatchEngine* engine) {
    if (!engine) return;
    std::lock_guard<std::mutex> lock(engine->engine_mutex);
    for (int32_t i = 0; i < engine->batch_config.max_sequences; i++) {
        engine->kv_cache->clear_seq(i);
    }
}

int32_t bitnet_batch_eos_token(BitNetBatchEngine* engine) {
    return engine ? engine->config.eos_token_id : -1;
}

int32_t bitnet_batch_vocab_size(BitNetBatchEngine* engine) {
    return engine ? engine->config.vocab_size : 0;
}

int32_t bitnet_batch_max_ctx_per_seq(BitNetBatchEngine* engine) {
    return engine ? engine->batch_config.n_ctx_per_seq : 0;
}

}  // extern "C"
