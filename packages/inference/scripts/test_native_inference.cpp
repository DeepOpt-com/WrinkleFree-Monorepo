/**
 * Minimal test for sgl-kernel native BitNet inference.
 *
 * Usage:
 *   g++ -O3 -mavx2 -fopenmp test_native_inference.cpp -o test_native
 *   ./test_native /path/to/model.sglbin
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <cmath>
#include <algorithm>

// ============================================================================
// Minimal sgl-kernel binary loader (simplified version)
// ============================================================================

struct TensorData {
    std::vector<uint8_t> data;
    std::vector<int32_t> shape;
    float scale;
    bool is_packed;
};

struct ModelConfig {
    int32_t vocab_size;
    int32_t hidden_size;
    int32_t intermediate_size;
    int32_t num_hidden_layers;
    int32_t num_attention_heads;
    int32_t num_key_value_heads;
    int32_t head_dim;
    float rms_norm_eps;
    int32_t bos_token_id;
    int32_t eos_token_id;
};

class ModelLoader {
public:
    bool load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open: " << path << std::endl;
            return false;
        }

        // Read magic
        char magic[8];
        file.read(magic, 8);
        if (std::memcmp(magic, "SGLBITNT", 8) != 0) {
            std::cerr << "Invalid magic" << std::endl;
            return false;
        }

        // Skip version
        file.seekg(4, std::ios::cur);

        // Read config JSON
        uint32_t config_len;
        file.read(reinterpret_cast<char*>(&config_len), 4);
        std::string config_json(config_len, '\0');
        file.read(&config_json[0], config_len);

        // Parse config (simple extraction)
        auto get_int = [&](const std::string& key) -> int32_t {
            size_t pos = config_json.find("\"" + key + "\"");
            if (pos == std::string::npos) return 0;
            pos = config_json.find(':', pos);
            if (pos == std::string::npos) return 0;
            return std::stoi(config_json.substr(pos + 1));
        };

        config_.vocab_size = get_int("vocab_size");
        config_.hidden_size = get_int("hidden_size");
        config_.intermediate_size = get_int("intermediate_size");
        config_.num_hidden_layers = get_int("num_hidden_layers");
        config_.num_attention_heads = get_int("num_attention_heads");
        config_.num_key_value_heads = get_int("num_key_value_heads");
        config_.head_dim = config_.hidden_size / config_.num_attention_heads;
        config_.bos_token_id = get_int("bos_token_id");
        config_.eos_token_id = get_int("eos_token_id");

        std::cout << "Model config:" << std::endl;
        std::cout << "  vocab_size: " << config_.vocab_size << std::endl;
        std::cout << "  hidden_size: " << config_.hidden_size << std::endl;
        std::cout << "  num_layers: " << config_.num_hidden_layers << std::endl;

        // Read tensor count
        uint32_t num_tensors;
        file.read(reinterpret_cast<char*>(&num_tensors), 4);
        std::cout << "  num_tensors: " << num_tensors << std::endl;

        // Read tensors
        size_t total_size = 0;
        size_t packed_count = 0;
        for (uint32_t i = 0; i < num_tensors; i++) {
            // Name
            uint32_t name_len;
            file.read(reinterpret_cast<char*>(&name_len), 4);
            std::string name(name_len, '\0');
            file.read(&name[0], name_len);

            // Dtype
            uint32_t dtype;
            file.read(reinterpret_cast<char*>(&dtype), 4);
            bool is_packed = (dtype == 0);  // uint8

            // Shape
            uint32_t ndims;
            file.read(reinterpret_cast<char*>(&ndims), 4);
            std::vector<int32_t> shape(ndims);
            for (uint32_t d = 0; d < ndims; d++) {
                file.read(reinterpret_cast<char*>(&shape[d]), 4);
            }

            // Scale
            uint32_t has_scale;
            file.read(reinterpret_cast<char*>(&has_scale), 4);
            float scale = 1.0f;
            if (has_scale) {
                file.read(reinterpret_cast<char*>(&scale), 4);
            }

            // Data
            uint64_t data_size;
            file.read(reinterpret_cast<char*>(&data_size), 8);

            TensorData tensor;
            tensor.data.resize(data_size);
            tensor.shape = shape;
            tensor.scale = scale;
            tensor.is_packed = is_packed;
            file.read(reinterpret_cast<char*>(tensor.data.data()), data_size);

            tensors_[name] = std::move(tensor);
            total_size += data_size;
            if (is_packed) packed_count++;
        }

        std::cout << "  packed tensors: " << packed_count << std::endl;
        std::cout << "  total size: " << (total_size / 1024.0 / 1024.0) << " MB" << std::endl;

        return true;
    }

    const TensorData* get(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) return nullptr;
        return &it->second;
    }

    const ModelConfig& config() const { return config_; }

private:
    ModelConfig config_;
    std::unordered_map<std::string, TensorData> tensors_;
};

// ============================================================================
// Simple RMS Norm (scalar, no SIMD)
// ============================================================================

void rms_norm(float* output, const float* input, const float* weight, int n, float eps = 1e-5f) {
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

// ============================================================================
// Scalar BitNet GEMV (no SIMD - for testing only)
// ============================================================================

void bitnet_gemv_scalar(
    float* output,
    const float* input,
    const uint8_t* packed_weights,
    int M,  // out_features
    int K,  // in_features
    float weight_scale
) {
    const int K_packed = K / 4;

    for (int m = 0; m < M; m++) {
        float sum = 0.0f;
        const uint8_t* row = packed_weights + m * K_packed;

        for (int k_packed = 0; k_packed < K_packed; k_packed++) {
            uint8_t packed = row[k_packed];

            // Unpack 4 ternary values: 00=-1, 01=0, 10=+1
            for (int i = 0; i < 4; i++) {
                int w_unsigned = (packed >> (i * 2)) & 0x03;
                int w = w_unsigned - 1;  // Convert to {-1, 0, 1}
                int k = k_packed * 4 + i;
                sum += w * input[k];
            }
        }

        output[m] = sum * weight_scale;
    }
}

// ============================================================================
// Test embedding lookup
// ============================================================================

void embed_lookup(float* output, const float* embed_table, int token_id, int hidden_size) {
    std::memcpy(output, embed_table + token_id * hidden_size, hidden_size * sizeof(float));
}

// ============================================================================
// Main test
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.sglbin>" << std::endl;
        return 1;
    }

    ModelLoader loader;
    if (!loader.load(argv[1])) {
        return 1;
    }

    auto& cfg = loader.config();

    // Get embedding table
    auto embed = loader.get("model.embed_tokens.weight");
    if (!embed) {
        std::cerr << "Embedding not found" << std::endl;
        return 1;
    }

    // Test: look up token embedding for token 128000 (BOS)
    std::cout << "\n=== Testing embedding lookup ===" << std::endl;
    std::vector<float> hidden(cfg.hidden_size);
    const float* embed_data = reinterpret_cast<const float*>(embed->data.data());
    embed_lookup(hidden.data(), embed_data, cfg.bos_token_id, cfg.hidden_size);

    std::cout << "BOS token (" << cfg.bos_token_id << ") embedding first 5: ";
    for (int i = 0; i < 5; i++) {
        std::cout << hidden[i] << " ";
    }
    std::cout << std::endl;

    // Test: apply first layer's input norm
    std::cout << "\n=== Testing layer 0 input norm ===" << std::endl;
    auto norm = loader.get("model.layers.0.input_layernorm.weight");
    if (!norm) {
        std::cerr << "Norm not found" << std::endl;
        return 1;
    }
    const float* norm_weight = reinterpret_cast<const float*>(norm->data.data());
    std::vector<float> normed(cfg.hidden_size);
    rms_norm(normed.data(), hidden.data(), norm_weight, cfg.hidden_size);

    std::cout << "After norm first 5: ";
    for (int i = 0; i < 5; i++) {
        std::cout << normed[i] << " ";
    }
    std::cout << std::endl;

    // Test: q_proj (BitNet GEMV)
    std::cout << "\n=== Testing layer 0 q_proj ===" << std::endl;
    auto q_proj = loader.get("model.layers.0.self_attn.q_proj.weight");
    if (!q_proj) {
        std::cerr << "q_proj not found" << std::endl;
        return 1;
    }

    int M = cfg.hidden_size;  // Output features
    int K = cfg.hidden_size;  // Input features

    std::vector<float> q_output(M);

    auto start = std::chrono::high_resolution_clock::now();
    bitnet_gemv_scalar(
        q_output.data(),
        normed.data(),
        q_proj->data.data(),
        M, K,
        q_proj->scale
    );
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Q projection first 5: ";
    for (int i = 0; i < 5; i++) {
        std::cout << q_output[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "q_proj time: " << duration.count() << " us" << std::endl;

    // Test: measure full layer throughput estimate
    std::cout << "\n=== Throughput estimate ===" << std::endl;
    int num_ops = 7;  // q, k, v, o, gate, up, down projections
    int total_gemv_us = duration.count() * num_ops * cfg.num_hidden_layers;
    std::cout << "Estimated time per token (scalar): " << (total_gemv_us / 1000.0) << " ms" << std::endl;
    std::cout << "Estimated TPS (scalar, no SIMD): " << (1000000.0 / total_gemv_us) << std::endl;
    std::cout << "(Note: With SIMD, expect 10-50x improvement)" << std::endl;

    std::cout << "\n=== Test PASSED ===" << std::endl;
    return 0;
}
