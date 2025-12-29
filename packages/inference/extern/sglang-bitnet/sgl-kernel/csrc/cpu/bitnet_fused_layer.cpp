/**
 * BitNet Fused Layer Forward - reduces Python overhead by fusing operations.
 *
 * This implements a fused forward pass through one BitNet transformer layer,
 * doing all operations in C++ to minimize Python dispatch overhead.
 *
 * Target: Reduce per-layer Python calls from ~15 to 1.
 */

#include <ATen/ATen.h>
#include <torch/all.h>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../bitnet/bitnet_gemv.h"

namespace sgl_kernel {
namespace bitnet {

/**
 * RMS Normalization in-place.
 */
void rms_norm_inplace(
    float* data,
    const float* weight,
    int64_t hidden_size,
    float eps
) {
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < hidden_size; i++) {
        sum_sq += data[i] * data[i];
    }
    float rms = 1.0f / std::sqrt(sum_sq / hidden_size + eps);
    for (int64_t i = 0; i < hidden_size; i++) {
        data[i] = data[i] * rms * weight[i];
    }
}

/**
 * Quantize activations to INT8 with dynamic scaling.
 */
float quantize_to_int8(
    int8_t* output,
    const float* input,
    int64_t n
) {
    // Find max absolute value
    float max_val = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float abs_val = std::fabs(input[i]);
        if (abs_val > max_val) max_val = abs_val;
    }

    if (max_val < 1e-6f) max_val = 1.0f;
    float scale = max_val / 127.0f;
    float inv_scale = 127.0f / max_val;

    // Quantize
    for (int64_t i = 0; i < n; i++) {
        float scaled = input[i] * inv_scale;
        int32_t rounded = static_cast<int32_t>(std::round(scaled));
        output[i] = static_cast<int8_t>(std::max(-128, std::min(127, rounded)));
    }

    return scale;
}

/**
 * BitNet linear with fused quantization.
 *
 * Does: output = scale * (W @ quantize(input))
 *
 * The kernel computes sum((w+1) * a) where w+1 is encoded {0,1,2} for {-1,0,+1}.
 * We need to subtract the activation sum: result = kernel_out - weight_scale * sum(a)
 */
void bitnet_linear_fused(
    float* output,
    const float* input,
    const uint8_t* packed_weights,
    float weight_scale,
    int64_t in_features,
    int64_t out_features
) {
    // Allocate temporary buffer for quantized input
    std::vector<int8_t> input_int8(in_features);
    float act_scale = quantize_to_int8(input_int8.data(), input, in_features);

    // Compute activation sum for encoding correction
    float act_sum = 0.0f;
    for (int64_t i = 0; i < in_features; i++) {
        act_sum += static_cast<float>(input_int8[i]);
    }

    int64_t packed_in_features = in_features / 4;

    // Compute each output element
    #pragma omp parallel for
    for (int64_t i = 0; i < out_features; i++) {
        float result;
        bitnet_vec_dot_i2_i8(
            in_features,
            &result,
            packed_weights + i * packed_in_features,
            input_int8.data()
        );
        // Apply correction and scaling
        // Kernel returns: weight_scale * sum((w+1)*a), but w+1 is encoded as {0,1,2}
        // So: kernel_out = weight_scale * (sum(w*a) + sum(a))
        // We want: weight_scale * act_scale * sum(w*a)
        // Therefore: output = (kernel_out - weight_scale * sum(a)) * act_scale
        output[i] = (result - weight_scale * act_sum) * act_scale;
    }
}

/**
 * ReLU² activation: relu(x)^2.
 */
void relu_squared_inplace(float* data, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        float x = data[i];
        if (x < 0) x = 0;
        data[i] = x * x;
    }
}

/**
 * Element-wise multiply.
 */
void mul_inplace(float* a, const float* b, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        a[i] *= b[i];
    }
}

/**
 * Fused MLP forward - optimized to minimize quantization overhead.
 *
 * Does:
 *   gate = bitnet_linear(input, gate_proj)
 *   up = bitnet_linear(input, up_proj)
 *   hidden = relu(gate)^2 * up
 *   hidden = rms_norm(hidden, ffn_sub_norm)
 *   output = bitnet_linear(hidden, down_proj)
 *
 * Key optimization: quantize input once, reuse for gate and up.
 */
void bitnet_mlp_fused(
    float* output,
    const float* input,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    const uint8_t* down_weight,
    float gate_scale,
    float up_scale,
    float down_scale,
    const float* ffn_sub_norm,
    int64_t hidden_size,
    int64_t intermediate_size,
    float eps
) {
    // Allocate temporary buffers
    std::vector<float> gate(intermediate_size);
    std::vector<float> up(intermediate_size);
    std::vector<float> hidden(intermediate_size);
    std::vector<int8_t> input_int8(hidden_size);
    std::vector<int8_t> hidden_int8(intermediate_size);

    int64_t packed_hidden = hidden_size / 4;
    int64_t packed_inter = intermediate_size / 4;

    // Quantize input ONCE (shared by gate and up)
    float input_scale = quantize_to_int8(input_int8.data(), input, hidden_size);
    float input_sum = 0.0f;
    for (int64_t i = 0; i < hidden_size; i++) {
        input_sum += static_cast<float>(input_int8[i]);
    }

    // Gate and Up projections in parallel (using shared quantized input)
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            #pragma omp parallel for
            for (int64_t i = 0; i < intermediate_size; i++) {
                float result;
                bitnet_vec_dot_i2_i8(hidden_size, &result,
                    gate_weight + i * packed_hidden, input_int8.data());
                gate[i] = (result - gate_scale * input_sum) * input_scale;
            }
        }

        #pragma omp section
        {
            for (int64_t i = 0; i < intermediate_size; i++) {
                float result;
                bitnet_vec_dot_i2_i8(hidden_size, &result,
                    up_weight + i * packed_hidden, input_int8.data());
                up[i] = (result - up_scale * input_sum) * input_scale;
            }
        }
    }

    // ReLU² * up
    for (int64_t i = 0; i < intermediate_size; i++) {
        float g = gate[i];
        if (g < 0) g = 0;
        hidden[i] = g * g * up[i];
    }

    // FFN sub-norm
    rms_norm_inplace(hidden.data(), ffn_sub_norm, intermediate_size, eps);

    // Quantize hidden for down projection
    float hidden_scale = quantize_to_int8(hidden_int8.data(), hidden.data(), intermediate_size);
    float hidden_sum = 0.0f;
    for (int64_t i = 0; i < intermediate_size; i++) {
        hidden_sum += static_cast<float>(hidden_int8[i]);
    }

    // Down projection
    #pragma omp parallel for
    for (int64_t i = 0; i < hidden_size; i++) {
        float result;
        bitnet_vec_dot_i2_i8(intermediate_size, &result,
            down_weight + i * packed_inter, hidden_int8.data());
        output[i] = (result - down_scale * hidden_sum) * hidden_scale;
    }
}

}  // namespace bitnet
}  // namespace sgl_kernel

// ============================================================================
// Torch bindings
// ============================================================================

/**
 * Fused MLP forward for BitNet layer.
 *
 * This replaces 3 separate linear calls + activations with a single C++ call.
 */
at::Tensor bitnet_mlp_forward_cpu(
    at::Tensor& input,
    at::Tensor& gate_weight,
    at::Tensor& up_weight,
    at::Tensor& down_weight,
    double gate_scale,
    double up_scale,
    double down_scale,
    at::Tensor& ffn_sub_norm,
    double eps
) {
    TORCH_CHECK(input.dim() == 1 || (input.dim() == 2 && input.size(0) == 1),
        "Input must be 1D or [1, hidden_size]");

    int64_t hidden_size = input.numel();
    int64_t intermediate_size = gate_weight.size(0);

    // Convert input to float
    auto input_f32 = input.to(torch::kFloat32).contiguous();
    auto output = torch::zeros({hidden_size}, torch::kFloat32);

    sgl_kernel::bitnet::bitnet_mlp_fused(
        output.data_ptr<float>(),
        input_f32.data_ptr<float>(),
        gate_weight.data_ptr<uint8_t>(),
        up_weight.data_ptr<uint8_t>(),
        down_weight.data_ptr<uint8_t>(),
        static_cast<float>(gate_scale),
        static_cast<float>(up_scale),
        static_cast<float>(down_scale),
        ffn_sub_norm.to(torch::kFloat32).contiguous().data_ptr<float>(),
        hidden_size,
        intermediate_size,
        static_cast<float>(eps)
    );

    return output;
}

/**
 * Fused QKV projection for BitNet layer.
 *
 * Does Q, K, V projections in parallel with shared quantization.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> bitnet_qkv_forward_cpu(
    at::Tensor& input,
    at::Tensor& q_weight,
    at::Tensor& k_weight,
    at::Tensor& v_weight,
    double q_scale,
    double k_scale,
    double v_scale
) {
    int64_t hidden_size = input.numel();
    int64_t q_size = q_weight.size(0);
    int64_t kv_size = k_weight.size(0);

    // Convert input to float and contiguous
    auto input_f32 = input.to(torch::kFloat32).contiguous();

    // Allocate outputs
    auto q_out = torch::zeros({q_size}, torch::kFloat32);
    auto k_out = torch::zeros({kv_size}, torch::kFloat32);
    auto v_out = torch::zeros({kv_size}, torch::kFloat32);

    // Quantize input once (shared across Q, K, V)
    std::vector<int8_t> input_int8(hidden_size);
    float act_scale = sgl_kernel::bitnet::quantize_to_int8(
        input_int8.data(), input_f32.data_ptr<float>(), hidden_size
    );

    // Compute activation sum for encoding correction
    float act_sum = 0.0f;
    for (int64_t i = 0; i < hidden_size; i++) {
        act_sum += static_cast<float>(input_int8[i]);
    }

    int64_t packed_in_features = hidden_size / 4;

    // Q, K, V projections in parallel
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            float* q_ptr = q_out.data_ptr<float>();
            const uint8_t* w_ptr = q_weight.data_ptr<uint8_t>();
            #pragma omp parallel for
            for (int64_t i = 0; i < q_size; i++) {
                float result;
                sgl_kernel::bitnet::bitnet_vec_dot_i2_i8(
                    hidden_size, &result,
                    w_ptr + i * packed_in_features,
                    input_int8.data()
                );
                q_ptr[i] = (result - q_scale * act_sum) * act_scale * q_scale;
            }
        }

        #pragma omp section
        {
            float* k_ptr = k_out.data_ptr<float>();
            const uint8_t* w_ptr = k_weight.data_ptr<uint8_t>();
            for (int64_t i = 0; i < kv_size; i++) {
                float result;
                sgl_kernel::bitnet::bitnet_vec_dot_i2_i8(
                    hidden_size, &result,
                    w_ptr + i * packed_in_features,
                    input_int8.data()
                );
                k_ptr[i] = (result - k_scale * act_sum) * act_scale * k_scale;
            }
        }

        #pragma omp section
        {
            float* v_ptr = v_out.data_ptr<float>();
            const uint8_t* w_ptr = v_weight.data_ptr<uint8_t>();
            for (int64_t i = 0; i < kv_size; i++) {
                float result;
                sgl_kernel::bitnet::bitnet_vec_dot_i2_i8(
                    hidden_size, &result,
                    w_ptr + i * packed_in_features,
                    input_int8.data()
                );
                v_ptr[i] = (result - v_scale * act_sum) * act_scale * v_scale;
            }
        }
    }

    return std::make_tuple(q_out, k_out, v_out);
}
