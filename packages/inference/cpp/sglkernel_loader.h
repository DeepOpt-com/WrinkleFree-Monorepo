/**
 * sgl-kernel Binary Model Loader
 *
 * Loads BitNet models from the .bin format produced by convert_to_sglkernel.py
 *
 * Format:
 *   [8 bytes]  Magic: "SGLBITNT"
 *   [4 bytes]  Version: 1
 *   [4 bytes]  Config JSON length
 *   [N bytes]  Config JSON
 *   [4 bytes]  Number of tensors
 *   For each tensor:
 *       [4 bytes]  Name length
 *       [N bytes]  Name (UTF-8)
 *       [4 bytes]  Dtype (0=uint8, 1=float32, 2=float16, 3=bfloat16)
 *       [4 bytes]  Number of dimensions
 *       [dims x 4] Shape
 *       [4 bytes]  Scale present flag
 *       [4 bytes]  Scale value (float32, if present)
 *       [8 bytes]  Data size in bytes
 *       [N bytes]  Raw tensor data
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>

namespace sgl_kernel {

constexpr char MAGIC[] = "SGLBITNT";
constexpr uint32_t VERSION = 1;

// Dtype enum matching Python side
enum class DType : uint32_t {
    UINT8 = 0,
    FLOAT32 = 1,
    FLOAT16 = 2,
    BFLOAT16 = 3
};

struct TensorInfo {
    std::string name;
    DType dtype;
    std::vector<int32_t> shape;
    bool has_scale;
    float scale;
    size_t data_offset;  // Offset in file
    size_t data_size;
};

struct ModelConfig {
    int32_t vocab_size = 0;
    int32_t hidden_size = 0;
    int32_t intermediate_size = 0;
    int32_t num_hidden_layers = 0;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t max_position_embeddings = 4096;
    float rms_norm_eps = 1e-6f;
    int32_t bos_token_id = 0;
    int32_t eos_token_id = 0;
    int32_t pad_token_id = 0;
};

class SGLKernelModelLoader {
public:
    SGLKernelModelLoader() = default;

    bool load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            error_ = "Failed to open file: " + path;
            return false;
        }

        // Read magic
        char magic[8];
        file.read(magic, 8);
        if (std::memcmp(magic, MAGIC, 8) != 0) {
            error_ = "Invalid magic number";
            return false;
        }

        // Read version
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), 4);
        if (version != VERSION) {
            error_ = "Unsupported version: " + std::to_string(version);
            return false;
        }

        // Read config JSON
        uint32_t config_len;
        file.read(reinterpret_cast<char*>(&config_len), 4);
        std::string config_json(config_len, '\0');
        file.read(&config_json[0], config_len);

        if (!parse_config(config_json)) {
            return false;
        }

        // Read tensor count
        uint32_t num_tensors;
        file.read(reinterpret_cast<char*>(&num_tensors), 4);

        // Read tensor metadata and data
        tensors_.reserve(num_tensors);
        for (uint32_t i = 0; i < num_tensors; i++) {
            TensorInfo info;

            // Name
            uint32_t name_len;
            file.read(reinterpret_cast<char*>(&name_len), 4);
            info.name.resize(name_len);
            file.read(&info.name[0], name_len);

            // Dtype
            uint32_t dtype;
            file.read(reinterpret_cast<char*>(&dtype), 4);
            info.dtype = static_cast<DType>(dtype);

            // Shape
            uint32_t ndims;
            file.read(reinterpret_cast<char*>(&ndims), 4);
            info.shape.resize(ndims);
            for (uint32_t d = 0; d < ndims; d++) {
                file.read(reinterpret_cast<char*>(&info.shape[d]), 4);
            }

            // Scale
            uint32_t has_scale;
            file.read(reinterpret_cast<char*>(&has_scale), 4);
            info.has_scale = (has_scale == 1);
            if (info.has_scale) {
                file.read(reinterpret_cast<char*>(&info.scale), 4);
            } else {
                info.scale = 1.0f;
            }

            // Data size and offset
            uint64_t data_size;
            file.read(reinterpret_cast<char*>(&data_size), 8);
            info.data_size = data_size;
            info.data_offset = file.tellg();

            // Skip data (we'll mmap or read later)
            file.seekg(data_size, std::ios::cur);

            tensor_map_[info.name] = tensors_.size();
            tensors_.push_back(std::move(info));
        }

        file_path_ = path;
        return true;
    }

    const TensorInfo* get_tensor_info(const std::string& name) const {
        auto it = tensor_map_.find(name);
        if (it == tensor_map_.end()) {
            return nullptr;
        }
        return &tensors_[it->second];
    }

    // Load tensor data into a pre-allocated buffer
    bool load_tensor_data(const std::string& name, void* buffer, size_t buffer_size) const {
        auto info = get_tensor_info(name);
        if (!info) {
            return false;
        }
        if (buffer_size < info->data_size) {
            return false;
        }

        std::ifstream file(file_path_, std::ios::binary);
        if (!file) {
            return false;
        }

        file.seekg(info->data_offset);
        file.read(reinterpret_cast<char*>(buffer), info->data_size);
        return file.good();
    }

    // Load tensor data as a new vector
    std::vector<uint8_t> load_tensor_data(const std::string& name) const {
        auto info = get_tensor_info(name);
        if (!info) {
            return {};
        }

        std::vector<uint8_t> data(info->data_size);
        if (!load_tensor_data(name, data.data(), data.size())) {
            return {};
        }
        return data;
    }

    const ModelConfig& config() const { return config_; }
    const std::string& error() const { return error_; }
    const std::vector<TensorInfo>& tensors() const { return tensors_; }

private:
    bool parse_config(const std::string& json) {
        // Simple JSON parsing for config values
        // Note: This is a minimal parser for our specific config format

        auto get_int = [&](const std::string& key) -> int32_t {
            size_t pos = json.find("\"" + key + "\"");
            if (pos == std::string::npos) return 0;
            pos = json.find(':', pos);
            if (pos == std::string::npos) return 0;
            return std::stoi(json.substr(pos + 1));
        };

        auto get_float = [&](const std::string& key) -> float {
            size_t pos = json.find("\"" + key + "\"");
            if (pos == std::string::npos) return 0.0f;
            pos = json.find(':', pos);
            if (pos == std::string::npos) return 0.0f;
            return std::stof(json.substr(pos + 1));
        };

        config_.vocab_size = get_int("vocab_size");
        config_.hidden_size = get_int("hidden_size");
        config_.intermediate_size = get_int("intermediate_size");
        config_.num_hidden_layers = get_int("num_hidden_layers");
        config_.num_attention_heads = get_int("num_attention_heads");
        config_.num_key_value_heads = get_int("num_key_value_heads");
        config_.max_position_embeddings = get_int("max_position_embeddings");
        config_.rms_norm_eps = get_float("rms_norm_eps");
        config_.bos_token_id = get_int("bos_token_id");
        config_.eos_token_id = get_int("eos_token_id");
        config_.pad_token_id = get_int("pad_token_id");

        // Validation
        if (config_.vocab_size == 0 || config_.hidden_size == 0) {
            error_ = "Invalid config: vocab_size or hidden_size is 0";
            return false;
        }

        return true;
    }

    std::string file_path_;
    std::string error_;
    ModelConfig config_;
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, size_t> tensor_map_;
};

}  // namespace sgl_kernel
