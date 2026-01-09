/**
 * BitNet C++ Inference Engine Implementation
 *
 * Self-contained inference engine for 1.58-bit BitNet models.
 * Uses sgl-kernel SIMD operations for matrix multiplication.
 *
 * Model format: HuggingFace safetensors with BitNet quantization
 */

#include "bitnet_engine.h"
#include "kv_cache.h"
#include "sglkernel_loader.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <omp.h>

// SIMD intrinsics for optimized fp32_linear
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// Use SIMD kernels when available
#ifdef USE_SIMD_KERNELS
#include "bitnet_gemv.h"
#else
// Scalar fallback for standalone testing (no SIMD)
namespace sgl_kernel { namespace bitnet {
    inline void quantize_activations_i8(int K, int8_t* out, const float* input, float* scale) {
        float max_val = 0.0f;
        for (int i = 0; i < K; i++) {
            float abs_val = std::abs(input[i]);
            if (abs_val > max_val) max_val = abs_val;
        }
        *scale = max_val / 127.0f;
        float inv_scale = (max_val > 0) ? 127.0f / max_val : 0.0f;
        for (int i = 0; i < K; i++) {
            int val = static_cast<int>(std::round(input[i] * inv_scale));
            out[i] = static_cast<int8_t>(std::max(-127, std::min(127, val)));
        }
    }

    // SIMD-compatible block-interleaved unpacking
    // Block size: 128 elements = 32 packed bytes
    // byte[j].bits[6:7] = weight for activation[j+0]
    // byte[j].bits[4:5] = weight for activation[j+32]
    // byte[j].bits[2:3] = weight for activation[j+64]
    // byte[j].bits[0:1] = weight for activation[j+96]
    inline void bitnet_vec_dot_i2_i8(int K, float* out, const uint8_t* packed_weights, const int8_t* quant_input) {
        constexpr int QK_BLOCK = 128;  // Block size for SIMD packing
        int sum = 0;
        int num_blocks = K / QK_BLOCK;

        for (int block = 0; block < num_blocks; block++) {
            int base_w = block * 32;   // 32 packed bytes per 128-element block
            int base_a = block * 128;  // 128 activations per block

            for (int j = 0; j < 32; j++) {
                uint8_t packed = packed_weights[base_w + j];
                // Extract 4 weights from positions j, j+32, j+64, j+96 within block
                int w0 = static_cast<int>((packed >> 6) & 0x03) - 1;  // bits 6-7 → activation[j+0]
                int w1 = static_cast<int>((packed >> 4) & 0x03) - 1;  // bits 4-5 → activation[j+32]
                int w2 = static_cast<int>((packed >> 2) & 0x03) - 1;  // bits 2-3 → activation[j+64]
                int w3 = static_cast<int>((packed >> 0) & 0x03) - 1;  // bits 0-1 → activation[j+96]

                sum += w0 * static_cast<int>(quant_input[base_a + j + 0]);
                sum += w1 * static_cast<int>(quant_input[base_a + j + 32]);
                sum += w2 * static_cast<int>(quant_input[base_a + j + 64]);
                sum += w3 * static_cast<int>(quant_input[base_a + j + 96]);
            }
        }
        *out = static_cast<float>(sum);
    }
}}
#endif

// Thread-local error message
static thread_local std::string g_last_error;

// Model configuration (BitNet-b1.58-2B-4T)
struct ModelConfig {
    int32_t vocab_size = 152064;
    int32_t hidden_size = 2560;
    int32_t intermediate_size = 6912;
    int32_t num_hidden_layers = 26;
    int32_t num_attention_heads = 20;
    int32_t num_key_value_heads = 20;
    int32_t head_dim = 128;  // hidden_size / num_attention_heads
    int32_t max_position_embeddings = 4096;
    float rms_norm_eps = 1e-6f;
    int32_t bos_token_id = 151643;
    int32_t eos_token_id = 151645;
    int32_t pad_token_id = 151643;
};

// Weight tensor (packed 2-bit for linear layers, fp32/fp16 for others)
struct WeightTensor {
    std::vector<uint8_t> packed_data;  // For BitNet layers (2-bit packed)
    std::vector<float> fp32_data;      // For embedding, norms, etc.
    std::vector<int32_t> shape;
    float scale = 1.0f;
    bool is_packed = false;
};

// Layer weights
struct LayerWeights {
    // Attention
    WeightTensor q_proj;
    WeightTensor k_proj;
    WeightTensor v_proj;
    WeightTensor o_proj;

    // MLP
    WeightTensor gate_proj;
    WeightTensor up_proj;
    WeightTensor down_proj;

    // Norms
    WeightTensor input_layernorm;
    WeightTensor post_attention_layernorm;

    // SubLN (sub-layer normalization)
    WeightTensor attn_sub_norm;  // Applied after attn output, before O proj
    WeightTensor ffn_sub_norm;   // Applied after gate*up, before down proj
};

// Engine implementation
struct BitNetEngine {
    ModelConfig config;
    std::vector<LayerWeights> layers;
    WeightTensor embed_tokens;
    WeightTensor lm_head;
    WeightTensor final_norm;

    std::unique_ptr<KVCache> kv_cache;

    // Scratch buffers (pre-allocated)
    std::vector<float> hidden_states;
    std::vector<float> residual;
    std::vector<float> attn_output;
    std::vector<float> mlp_output;
    std::vector<float> logits;
    std::vector<int8_t> quant_buffer;
    float quant_scale;

    // Random generator for sampling
    std::mt19937 rng;

    // Current sequence length
    int32_t seq_len = 0;

    bool is_loaded = false;
};

// ============================================================================
// Helper Functions
// ============================================================================

static void set_error(const std::string& msg) {
    g_last_error = msg;
}

// RMS normalization
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

// SiLU activation (not used for SubLN models)
static void silu(float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

// ReLU² activation: relu(x)^2 (used in BitNet/SubLN models)
static void relu_squared(float* x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 0.0f) x[i] = 0.0f;
        x[i] = x[i] * x[i];
    }
}

// Element-wise multiply
static void elementwise_mul(float* output, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = a[i] * b[i];
    }
}

// Softmax
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
static int32_t sample_top_p(const float* probs, int vocab_size, float top_p,
                            std::mt19937& rng) {
    // Sort indices by probability
    std::vector<std::pair<float, int32_t>> sorted_probs;
    sorted_probs.reserve(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        sorted_probs.emplace_back(probs[i], i);
    }
    std::sort(sorted_probs.begin(), sorted_probs.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    // Find cutoff
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

    // Renormalize and sample
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

// ============================================================================
// BitNet Matrix Operations (using sgl-kernel)
// ============================================================================

// BitNet linear layer: output = input @ weight.T * scale
static void bitnet_linear(float* output, const float* input, int8_t* quant_input,
                          const WeightTensor& weight, int M, int K,
                          float* quant_scale) {
    using namespace sgl_kernel::bitnet;

    // Quantize input activations to INT8
    quantize_activations_i8(K, quant_input, input, quant_scale);

    // Call GEMV for each output row
    // weight shape: [M, K/4] (packed 2-bit)
    // The scalar fallback already handles {0,1,2} -> {-1,0,1} conversion
    const int K_packed = K / 4;  // 4 weights per byte
    const float scale_factor = weight.scale * (*quant_scale);

    // Parallelize across output rows (each row is independent)
    #pragma omp parallel for schedule(static) if(M >= 64)
    for (int m = 0; m < M; m++) {
        bitnet_vec_dot_i2_i8(K, &output[m],
                             weight.packed_data.data() + m * K_packed,
                             quant_input);
        // Apply weight scale and activation scale
        output[m] *= scale_factor;
    }
}

// Regular fp32 linear layer (for embeddings, lm_head)
// Optimized with SIMD and OpenMP parallelization
static void fp32_linear(float* output, const float* input, const float* weight,
                        int M, int K) {
#if defined(__AVX512F__)
    // AVX-512: Process 16 floats at a time
    #pragma omp parallel for schedule(static) if(M >= 64)
    for (int m = 0; m < M; m++) {
        const float* w_row = weight + m * K;
        __m512 sum_vec = _mm512_setzero_ps();

        int k = 0;
        for (; k + 16 <= K; k += 16) {
            __m512 w = _mm512_loadu_ps(w_row + k);
            __m512 x = _mm512_loadu_ps(input + k);
            sum_vec = _mm512_fmadd_ps(w, x, sum_vec);
        }

        float sum = _mm512_reduce_add_ps(sum_vec);

        // Handle remaining elements
        for (; k < K; k++) {
            sum += w_row[k] * input[k];
        }
        output[m] = sum;
    }
#elif defined(__AVX2__) && defined(__FMA__)
    // AVX2 + FMA: Process 8 floats at a time
    #pragma omp parallel for schedule(static) if(M >= 64)
    for (int m = 0; m < M; m++) {
        const float* w_row = weight + m * K;
        __m256 sum_vec0 = _mm256_setzero_ps();
        __m256 sum_vec1 = _mm256_setzero_ps();

        int k = 0;
        // 2x unrolled loop for better latency hiding
        for (; k + 16 <= K; k += 16) {
            __m256 w0 = _mm256_loadu_ps(w_row + k);
            __m256 w1 = _mm256_loadu_ps(w_row + k + 8);
            __m256 x0 = _mm256_loadu_ps(input + k);
            __m256 x1 = _mm256_loadu_ps(input + k + 8);
            sum_vec0 = _mm256_fmadd_ps(w0, x0, sum_vec0);
            sum_vec1 = _mm256_fmadd_ps(w1, x1, sum_vec1);
        }
        for (; k + 8 <= K; k += 8) {
            __m256 w = _mm256_loadu_ps(w_row + k);
            __m256 x = _mm256_loadu_ps(input + k);
            sum_vec0 = _mm256_fmadd_ps(w, x, sum_vec0);
        }

        // Horizontal sum
        __m256 sum_vec = _mm256_add_ps(sum_vec0, sum_vec1);
        __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
        __m128 lo = _mm256_castps256_ps128(sum_vec);
        __m128 sum4 = _mm_add_ps(hi, lo);
        sum4 = _mm_add_ps(sum4, _mm_shuffle_ps(sum4, sum4, _MM_SHUFFLE(2, 3, 0, 1)));
        sum4 = _mm_add_ps(sum4, _mm_shuffle_ps(sum4, sum4, _MM_SHUFFLE(1, 0, 3, 2)));
        float sum = _mm_cvtss_f32(sum4);

        // Handle remaining elements
        for (; k < K; k++) {
            sum += w_row[k] * input[k];
        }
        output[m] = sum;
    }
#else
    // Scalar fallback with OpenMP
    #pragma omp parallel for schedule(static) if(M >= 64)
    for (int m = 0; m < M; m++) {
        const float* w_row = weight + m * K;
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += input[k] * w_row[k];
        }
        output[m] = sum;
    }
#endif
}

// Embedding lookup
static void embed_lookup(float* output, const float* embed_table,
                         const int32_t* input_ids, int num_tokens, int hidden_size) {
    for (int t = 0; t < num_tokens; t++) {
        int32_t token_id = input_ids[t];
        std::memcpy(output + t * hidden_size,
                    embed_table + token_id * hidden_size,
                    hidden_size * sizeof(float));
    }
}

// ============================================================================
// Attention Forward (with GQA and SubLN support)
// ============================================================================

static void attention_forward(BitNetEngine* engine, int layer_idx, int pos,
                              float* hidden, float* output) {
    auto& cfg = engine->config;
    auto& layer = engine->layers[layer_idx];

    const int H = cfg.hidden_size;
    const int heads = cfg.num_attention_heads;
    const int head_dim = cfg.head_dim;
    const int kv_heads = cfg.num_key_value_heads;
    const int kv_dim = kv_heads * head_dim;  // K and V dimension (smaller for GQA)
    const int heads_per_kv = heads / kv_heads;  // How many Q heads share each KV head

    std::vector<float> q(H), k(kv_dim), v(kv_dim);
    int8_t* quant_buf = engine->quant_buffer.data();
    float quant_scale;

    // Q projection: hidden_size -> hidden_size
    bitnet_linear(q.data(), hidden, quant_buf, layer.q_proj, H, H, &quant_scale);
    // K, V projections: hidden_size -> kv_dim (GQA)
    bitnet_linear(k.data(), hidden, quant_buf, layer.k_proj, kv_dim, H, &quant_scale);
    bitnet_linear(v.data(), hidden, quant_buf, layer.v_proj, kv_dim, H, &quant_scale);

    // Store K, V in cache (only kv_dim per position)
    float* k_cache = kv_cache_get_key(engine->kv_cache.get(), layer_idx);
    float* v_cache = kv_cache_get_value(engine->kv_cache.get(), layer_idx);

    // Copy to position in cache (note: using kv_dim, not H)
    std::memcpy(k_cache + pos * kv_dim, k.data(), kv_dim * sizeof(float));
    std::memcpy(v_cache + pos * kv_dim, v.data(), kv_dim * sizeof(float));

    // Multi-head attention with GQA
    std::vector<float> attn_out(H, 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int h = 0; h < heads; h++) {
        const float* q_head = q.data() + h * head_dim;
        const int kv_h = h / heads_per_kv;  // Which KV head this Q head uses

        // Compute attention scores for all positions up to pos
        std::vector<float> scores(pos + 1);
        for (int p = 0; p <= pos; p++) {
            const float* k_p = k_cache + p * kv_dim + kv_h * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q_head[d] * k_p[d];
            }
            scores[p] = score * scale;
        }

        // Softmax
        softmax(scores.data(), pos + 1);

        // Weighted sum of values
        float* out_head = attn_out.data() + h * head_dim;
        for (int p = 0; p <= pos; p++) {
            const float* v_p = v_cache + p * kv_dim + kv_h * head_dim;
            for (int d = 0; d < head_dim; d++) {
                out_head[d] += scores[p] * v_p[d];
            }
        }
    }

    // Apply attn_sub_norm before O projection (if available)
    if (!layer.attn_sub_norm.fp32_data.empty()) {
        rms_norm(attn_out.data(), attn_out.data(),
                 layer.attn_sub_norm.fp32_data.data(), H, cfg.rms_norm_eps);
    }

    // Output projection
    bitnet_linear(output, attn_out.data(), quant_buf, layer.o_proj, H, H, &quant_scale);
}

// ============================================================================
// MLP Forward (with ReLU² and SubLN support)
// ============================================================================

static void mlp_forward(BitNetEngine* engine, int layer_idx, float* hidden, float* output) {
    auto& cfg = engine->config;
    auto& layer = engine->layers[layer_idx];

    const int H = cfg.hidden_size;
    const int I = cfg.intermediate_size;

    std::vector<float> gate(I), up(I), mlp_hidden(I);
    int8_t* quant_buf = engine->quant_buffer.data();
    float quant_scale;

    // Gate and up projections
    bitnet_linear(gate.data(), hidden, quant_buf, layer.gate_proj, I, H, &quant_scale);
    bitnet_linear(up.data(), hidden, quant_buf, layer.up_proj, I, H, &quant_scale);

    // ReLU²(gate) * up (BitNet uses ReLU² instead of SiLU)
    relu_squared(gate.data(), I);
    elementwise_mul(mlp_hidden.data(), gate.data(), up.data(), I);

    // Apply ffn_sub_norm before down projection (if available)
    if (!layer.ffn_sub_norm.fp32_data.empty()) {
        rms_norm(mlp_hidden.data(), mlp_hidden.data(),
                 layer.ffn_sub_norm.fp32_data.data(), I, cfg.rms_norm_eps);
    }

    // Down projection
    bitnet_linear(output, mlp_hidden.data(), quant_buf, layer.down_proj, H, I, &quant_scale);
}

// ============================================================================
// Full Forward Pass
// ============================================================================

static void forward_one_token(BitNetEngine* engine, int32_t token_id, int pos,
                              float* logits_out) {
    auto& cfg = engine->config;
    const int H = cfg.hidden_size;

    // Get hidden state buffer
    float* hidden = engine->hidden_states.data();
    float* residual = engine->residual.data();
    float* attn_out = engine->attn_output.data();
    float* mlp_out = engine->mlp_output.data();

    // Embedding lookup
    embed_lookup(hidden, engine->embed_tokens.fp32_data.data(),
                 &token_id, 1, H);

    // Process each layer
    for (int l = 0; l < cfg.num_hidden_layers; l++) {
        auto& layer = engine->layers[l];

        // Save residual
        std::memcpy(residual, hidden, H * sizeof(float));

        // Pre-attention norm
        rms_norm(hidden, hidden, layer.input_layernorm.fp32_data.data(),
                 H, cfg.rms_norm_eps);

        // Self-attention
        attention_forward(engine, l, pos, hidden, attn_out);

        // Residual connection
        for (int i = 0; i < H; i++) {
            hidden[i] = residual[i] + attn_out[i];
        }

        // Save residual
        std::memcpy(residual, hidden, H * sizeof(float));

        // Post-attention norm
        rms_norm(hidden, hidden, layer.post_attention_layernorm.fp32_data.data(),
                 H, cfg.rms_norm_eps);

        // MLP
        mlp_forward(engine, l, hidden, mlp_out);

        // Residual connection
        for (int i = 0; i < H; i++) {
            hidden[i] = residual[i] + mlp_out[i];
        }
    }

    // Final norm
    rms_norm(hidden, hidden, engine->final_norm.fp32_data.data(),
             H, cfg.rms_norm_eps);

    // LM head (output projection to vocabulary)
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

// ============================================================================
// Model Loading - sgl-kernel binary format
// ============================================================================

static bool load_weight_tensor(
    const sgl_kernel::SGLKernelModelLoader& loader,
    const std::string& name,
    WeightTensor& tensor,
    bool expect_packed = false
) {
    auto info = loader.get_tensor_info(name);
    if (!info) {
        set_error("Tensor not found: " + name);
        return false;
    }

    auto data = loader.load_tensor_data(name);
    if (data.empty()) {
        set_error("Failed to load tensor data: " + name);
        return false;
    }

    tensor.scale = info->scale;
    tensor.is_packed = (info->dtype == sgl_kernel::DType::UINT8);

    if (tensor.is_packed) {
        tensor.packed_data = std::move(data);
        tensor.shape.assign(info->shape.begin(), info->shape.end());
    } else {
        // Convert to float32
        size_t num_floats = data.size() / sizeof(float);
        tensor.fp32_data.resize(num_floats);
        std::memcpy(tensor.fp32_data.data(), data.data(), data.size());
        tensor.shape.assign(info->shape.begin(), info->shape.end());
    }

    return true;
}

static bool load_model(BitNetEngine* engine, const char* model_path) {
    using namespace sgl_kernel;

    SGLKernelModelLoader loader;
    if (!loader.load(model_path)) {
        set_error("Failed to load model: " + loader.error());
        return false;
    }

    // Update engine config from model
    auto& model_cfg = loader.config();
    auto& cfg = engine->config;

    cfg.vocab_size = model_cfg.vocab_size;
    cfg.hidden_size = model_cfg.hidden_size;
    cfg.intermediate_size = model_cfg.intermediate_size;
    cfg.num_hidden_layers = model_cfg.num_hidden_layers;
    cfg.num_attention_heads = model_cfg.num_attention_heads;
    cfg.num_key_value_heads = model_cfg.num_key_value_heads;
    cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
    cfg.max_position_embeddings = model_cfg.max_position_embeddings;
    cfg.rms_norm_eps = model_cfg.rms_norm_eps;
    cfg.bos_token_id = model_cfg.bos_token_id;
    cfg.eos_token_id = model_cfg.eos_token_id;
    cfg.pad_token_id = model_cfg.pad_token_id;

    // Load embedding
    if (!load_weight_tensor(loader, "model.embed_tokens.weight", engine->embed_tokens)) {
        return false;
    }

    // Load final norm
    if (!load_weight_tensor(loader, "model.norm.weight", engine->final_norm)) {
        return false;
    }

    // Load lm_head (may be tied to embed_tokens)
    if (loader.get_tensor_info("lm_head.weight")) {
        if (!load_weight_tensor(loader, "lm_head.weight", engine->lm_head)) {
            return false;
        }
    } else {
        // Tied embeddings: reuse embed_tokens for lm_head
        // Note: lm_head will use the same data as embed_tokens
        engine->lm_head = engine->embed_tokens;
    }

    // Load layer weights
    engine->layers.resize(cfg.num_hidden_layers);
    for (int l = 0; l < cfg.num_hidden_layers; l++) {
        auto& layer = engine->layers[l];
        std::string prefix = "model.layers." + std::to_string(l) + ".";

        // Attention projections (packed)
        if (!load_weight_tensor(loader, prefix + "self_attn.q_proj.weight", layer.q_proj, true)) return false;
        if (!load_weight_tensor(loader, prefix + "self_attn.k_proj.weight", layer.k_proj, true)) return false;
        if (!load_weight_tensor(loader, prefix + "self_attn.v_proj.weight", layer.v_proj, true)) return false;
        if (!load_weight_tensor(loader, prefix + "self_attn.o_proj.weight", layer.o_proj, true)) return false;

        // MLP projections (packed)
        if (!load_weight_tensor(loader, prefix + "mlp.gate_proj.weight", layer.gate_proj, true)) return false;
        if (!load_weight_tensor(loader, prefix + "mlp.up_proj.weight", layer.up_proj, true)) return false;
        if (!load_weight_tensor(loader, prefix + "mlp.down_proj.weight", layer.down_proj, true)) return false;

        // Norms (fp32)
        if (!load_weight_tensor(loader, prefix + "input_layernorm.weight", layer.input_layernorm)) return false;
        if (!load_weight_tensor(loader, prefix + "post_attention_layernorm.weight", layer.post_attention_layernorm)) return false;

        // SubLN norms (optional - only present in SubLN models)
        if (loader.get_tensor_info(prefix + "self_attn.attn_sub_norm.weight")) {
            load_weight_tensor(loader, prefix + "self_attn.attn_sub_norm.weight", layer.attn_sub_norm);
        }
        if (loader.get_tensor_info(prefix + "mlp.ffn_sub_norm.weight")) {
            load_weight_tensor(loader, prefix + "mlp.ffn_sub_norm.weight", layer.ffn_sub_norm);
        }
    }

    return true;
}

// ============================================================================
// Public C API Implementation
// ============================================================================

extern "C" {

BitNetEngine* bitnet_engine_create(const char* model_path, const BitNetConfig* config) {
    try {
        auto engine = new BitNetEngine();

        // Load model weights FIRST - this populates config from model file
        if (!load_model(engine, model_path)) {
            delete engine;
            return nullptr;
        }

        auto& cfg = engine->config;

        // Apply user configuration overrides if provided
        if (config) {
            if (config->max_seq_len > 0) {
                cfg.max_position_embeddings = config->max_seq_len;
            }
            if (config->num_threads > 0) {
                // Store for later use
            }
        }

        // Create KV cache with dimensions from loaded model (use kv_heads for GQA)
        engine->kv_cache.reset(
            kv_cache_create(cfg.num_hidden_layers, cfg.num_key_value_heads,
                           cfg.head_dim, cfg.max_position_embeddings)
        );
        if (!engine->kv_cache) {
            set_error("Failed to create KV cache");
            delete engine;
            return nullptr;
        }

        // Allocate scratch buffers with dimensions from loaded model
        engine->hidden_states.resize(cfg.hidden_size);
        engine->residual.resize(cfg.hidden_size);
        engine->attn_output.resize(cfg.hidden_size);
        engine->mlp_output.resize(cfg.hidden_size);
        engine->logits.resize(cfg.vocab_size);
        engine->quant_buffer.resize(std::max(cfg.hidden_size, cfg.intermediate_size));

        // Initialize RNG
        engine->rng.seed(42);

        engine->is_loaded = true;
        return engine;

    } catch (const std::exception& e) {
        set_error(std::string("Engine creation failed: ") + e.what());
        return nullptr;
    }
}

void bitnet_engine_destroy(BitNetEngine* engine) {
    delete engine;
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
    if (!engine || !engine->is_loaded) {
        set_error("Engine not initialized");
        return -1;
    }

    try {
        auto& cfg = engine->config;

        // Reset KV cache
        kv_cache_reset(engine->kv_cache.get());
        engine->seq_len = 0;

        // Prefill: process all input tokens
        for (int i = 0; i < num_input_tokens; i++) {
            forward_one_token(engine, input_ids[i], i, engine->logits.data());
        }
        engine->seq_len = num_input_tokens;

        // Decode: generate new tokens
        std::vector<int32_t> generated;
        generated.reserve(params->max_tokens);

        for (int i = 0; i < params->max_tokens; i++) {
            int32_t next_token;

            if (params->temperature <= 0.0f) {
                // Greedy decoding
                next_token = std::distance(
                    engine->logits.begin(),
                    std::max_element(engine->logits.begin(), engine->logits.end())
                );
            } else {
                // Apply temperature
                std::vector<float> probs(cfg.vocab_size);
                for (int v = 0; v < cfg.vocab_size; v++) {
                    probs[v] = engine->logits[v] / params->temperature;
                }
                softmax(probs.data(), cfg.vocab_size);

                // Sample with top-p
                float top_p = params->top_p > 0.0f ? params->top_p : 1.0f;
                next_token = sample_top_p(probs.data(), cfg.vocab_size,
                                          top_p, engine->rng);
            }

            generated.push_back(next_token);

            // Check for EOS
            if (next_token == cfg.eos_token_id) {
                break;
            }

            // Generate next logits
            forward_one_token(engine, next_token, engine->seq_len,
                             engine->logits.data());
            engine->seq_len++;
        }

        // Fill result
        result->num_tokens = generated.size();
        result->output_ids = static_cast<int32_t*>(
            malloc(generated.size() * sizeof(int32_t))
        );
        std::memcpy(result->output_ids, generated.data(),
                    generated.size() * sizeof(int32_t));
        result->logits = nullptr;
        result->logits_size = 0;

        return 0;

    } catch (const std::exception& e) {
        set_error(std::string("Generation failed: ") + e.what());
        return -1;
    }
}

int bitnet_prefill(
    BitNetEngine* engine,
    const int32_t* input_ids,
    int32_t num_tokens
) {
    if (!engine || !engine->is_loaded) {
        set_error("Engine not initialized");
        return -1;
    }

    try {
        // Reset and process all tokens
        kv_cache_reset(engine->kv_cache.get());
        for (int i = 0; i < num_tokens; i++) {
            forward_one_token(engine, input_ids[i], i, engine->logits.data());
        }
        engine->seq_len = num_tokens;
        kv_cache_set_seq_len(engine->kv_cache.get(), num_tokens);
        return 0;

    } catch (const std::exception& e) {
        set_error(std::string("Prefill failed: ") + e.what());
        return -1;
    }
}

int bitnet_decode_step(
    BitNetEngine* engine,
    int32_t position,
    const SamplingParams* params,
    int32_t* output_id
) {
    if (!engine || !engine->is_loaded) {
        set_error("Engine not initialized");
        return -1;
    }

    try {
        // Use logits from previous forward pass
        auto& cfg = engine->config;

        if (params->temperature <= 0.0f) {
            *output_id = std::distance(
                engine->logits.begin(),
                std::max_element(engine->logits.begin(), engine->logits.end())
            );
        } else {
            std::vector<float> probs(cfg.vocab_size);
            for (int v = 0; v < cfg.vocab_size; v++) {
                probs[v] = engine->logits[v] / params->temperature;
            }
            softmax(probs.data(), cfg.vocab_size);
            float top_p = params->top_p > 0.0f ? params->top_p : 1.0f;
            *output_id = sample_top_p(probs.data(), cfg.vocab_size,
                                       top_p, engine->rng);
        }

        // Forward pass for next position
        forward_one_token(engine, *output_id, position, engine->logits.data());
        engine->seq_len = position + 1;
        kv_cache_set_seq_len(engine->kv_cache.get(), engine->seq_len);

        return 0;

    } catch (const std::exception& e) {
        set_error(std::string("Decode step failed: ") + e.what());
        return -1;
    }
}

void bitnet_reset_cache(BitNetEngine* engine) {
    if (engine && engine->kv_cache) {
        kv_cache_reset(engine->kv_cache.get());
        engine->seq_len = 0;
    }
}

int32_t bitnet_vocab_size(BitNetEngine* engine) {
    return engine ? engine->config.vocab_size : 0;
}

int32_t bitnet_hidden_size(BitNetEngine* engine) {
    return engine ? engine->config.hidden_size : 0;
}

int32_t bitnet_num_layers(BitNetEngine* engine) {
    return engine ? engine->config.num_hidden_layers : 0;
}

int32_t bitnet_max_seq_len(BitNetEngine* engine) {
    return engine ? engine->config.max_position_embeddings : 0;
}

int bitnet_get_num_kv_heads(BitNetEngine* engine) {
    return engine ? engine->config.num_key_value_heads : 0;
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
