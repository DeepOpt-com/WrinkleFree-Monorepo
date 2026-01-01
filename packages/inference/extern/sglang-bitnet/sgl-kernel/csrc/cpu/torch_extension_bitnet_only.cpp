/**
 * Minimal torch extension with only BitNet kernels.
 *
 * This avoids the PyTorch 2.9.1 brgemm/exp_u20 API incompatibility issues
 * by only including BitNet-specific ops that don't use those APIs.
 *
 * Ops provided:
 *   - bitnet_gemv_cpu: Single-token decode (8x faster than gemm)
 *   - bitnet_gemm_cpu: Batched decode
 *   - bitnet_quantize_activations_cpu: INT8 activation quantization
 *   - bitnet_mlp_forward_cpu: Fused MLP (3 linear + activations)
 *   - bitnet_qkv_forward_cpu: Fused QKV projection
 */

#include <ATen/ATen.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>

#include "../bitnet/bitnet_gemv.h"

// REGISTER_EXTENSION macro for PyModule creation
#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)
#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

#define REGISTER_EXTENSION(NAME)                                                                      \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                                            \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                                                  \
  }

// Forward declarations from bitnet_fused_layer.cpp
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
);

std::tuple<at::Tensor, at::Tensor, at::Tensor> bitnet_qkv_forward_cpu(
    at::Tensor& input,
    at::Tensor& q_weight,
    at::Tensor& k_weight,
    at::Tensor& v_weight,
    double q_scale,
    double k_scale,
    double v_scale
);

// ============================================================================
// BitNet GEMV/GEMM wrappers
// ============================================================================

at::Tensor bitnet_gemv_cpu(
    at::Tensor& packed_weights,
    at::Tensor& activations,
    double scale
) {
    TORCH_CHECK(packed_weights.dtype() == torch::kUInt8, "packed_weights must be uint8");
    TORCH_CHECK(activations.dtype() == torch::kInt8, "activations must be int8");
    TORCH_CHECK(packed_weights.dim() == 2, "packed_weights must be 2D [out_features, in_features/4]");
    TORCH_CHECK(activations.dim() == 1, "activations must be 1D [in_features]");

    int64_t out_features = packed_weights.size(0);
    int64_t packed_in_features = packed_weights.size(1);
    int64_t in_features = packed_in_features * 4;

    TORCH_CHECK(activations.size(0) == in_features,
        "activations.size(0) must equal packed_weights.size(1) * 4");

    auto output = torch::zeros({out_features}, torch::kFloat32);

    const uint8_t* w_ptr = packed_weights.data_ptr<uint8_t>();
    const int8_t* a_ptr = activations.data_ptr<int8_t>();
    float* o_ptr = output.data_ptr<float>();

    // Compute each output element
    #pragma omp parallel for
    for (int64_t i = 0; i < out_features; i++) {
        float result;
        sgl_kernel::bitnet::bitnet_vec_dot_i2_i8(
            in_features,
            &result,
            w_ptr + i * packed_in_features,
            a_ptr
        );
        o_ptr[i] = result * static_cast<float>(scale);
    }

    return output;
}

at::Tensor bitnet_gemm_cpu(
    at::Tensor& packed_weights,
    at::Tensor& activations,
    double scale
) {
    TORCH_CHECK(packed_weights.dtype() == torch::kUInt8, "packed_weights must be uint8");
    TORCH_CHECK(activations.dtype() == torch::kInt8, "activations must be int8");
    TORCH_CHECK(packed_weights.dim() == 2, "packed_weights must be 2D");
    TORCH_CHECK(activations.dim() == 2, "activations must be 2D [batch, in_features]");

    int64_t out_features = packed_weights.size(0);
    int64_t packed_in_features = packed_weights.size(1);
    int64_t in_features = packed_in_features * 4;
    int64_t batch_size = activations.size(0);

    TORCH_CHECK(activations.size(1) == in_features,
        "activations.size(1) must equal packed_weights.size(1) * 4");

    auto output = torch::zeros({batch_size, out_features}, torch::kFloat32);

    const uint8_t* w_ptr = packed_weights.data_ptr<uint8_t>();
    const int8_t* a_ptr = activations.data_ptr<int8_t>();
    float* o_ptr = output.data_ptr<float>();

    sgl_kernel::bitnet::TileConfig config;
    sgl_kernel::bitnet::bitnet_gemm_i2_i8(
        out_features,
        batch_size,
        in_features,
        o_ptr,
        w_ptr,
        a_ptr,
        static_cast<float>(scale),
        config
    );

    return output;
}

std::tuple<at::Tensor, at::Tensor> bitnet_quantize_activations_cpu(at::Tensor& input) {
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    auto flat = input.flatten();
    int64_t n = flat.numel();

    auto output = torch::zeros({n}, torch::kInt8);
    float scale = 1.0f;

    sgl_kernel::bitnet::quantize_activations_i8(
        n,
        output.data_ptr<int8_t>(),
        flat.data_ptr<float>(),
        &scale
    );

    auto scale_tensor = torch::tensor({scale}, torch::kFloat32);
    return std::make_tuple(output.reshape(input.sizes()), scale_tensor);
}

// Check if BitNet kernels are available
bool bitnet_check_kernel_available() {
    auto caps = sgl_kernel::bitnet::detect_cpu_capabilities();
    return caps.has_avx2 || caps.has_avx512 || caps.has_neon;
}

std::string bitnet_get_cpu_capabilities() {
    auto caps = sgl_kernel::bitnet::detect_cpu_capabilities();
    std::string result;
    if (caps.has_avx2) result += "AVX2 ";
    if (caps.has_avx512) result += "AVX512 ";
    if (caps.has_neon) result += "NEON ";
    if (caps.has_dotprod) result += "DotProd ";
    if (result.empty()) result = "None";
    return result;
}

// ============================================================================
// RMSNorm - minimal implementation for BitNet inference
// ============================================================================

at::Tensor rmsnorm_cpu(at::Tensor& input, at::Tensor& weight, double eps) {
    TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dimension");

    auto input_f32 = input.to(torch::kFloat32).contiguous();
    auto weight_f32 = weight.to(torch::kFloat32).contiguous();

    int64_t hidden_size = input_f32.size(-1);
    int64_t batch_size = input_f32.numel() / hidden_size;

    auto output = torch::empty_like(input_f32);

    const float* in_ptr = input_f32.data_ptr<float>();
    const float* w_ptr = weight_f32.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; b++) {
        const float* x = in_ptr + b * hidden_size;
        float* o = out_ptr + b * hidden_size;

        // Compute sum of squares
        float sum_sq = 0.0f;
        for (int64_t i = 0; i < hidden_size; i++) {
            sum_sq += x[i] * x[i];
        }

        // RMS normalization
        float rms = 1.0f / std::sqrt(sum_sq / hidden_size + static_cast<float>(eps));
        for (int64_t i = 0; i < hidden_size; i++) {
            o[i] = x[i] * rms * w_ptr[i];
        }
    }

    return output;
}

// ============================================================================
// SiLU activation - simple implementation for BitNet inference
// ============================================================================

at::Tensor silu_and_mul_cpu(at::Tensor& input) {
    auto x = input.to(torch::kFloat32).contiguous();
    int64_t hidden_size = x.size(-1) / 2;
    int64_t batch_size = x.numel() / (hidden_size * 2);

    auto output = torch::empty({batch_size, hidden_size}, torch::kFloat32);

    const float* in_ptr = x.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; b++) {
        const float* gate = in_ptr + b * hidden_size * 2;
        const float* up = gate + hidden_size;
        float* o = out_ptr + b * hidden_size;

        for (int64_t i = 0; i < hidden_size; i++) {
            float g = gate[i];
            float sigmoid_g = 1.0f / (1.0f + std::exp(-g));
            o[i] = g * sigmoid_g * up[i];  // SiLU(gate) * up
        }
    }

    return output;
}

// ============================================================================
// Register ops
// ============================================================================

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
    // BitNet 1.58-bit quantized GEMV/GEMM
    m.def("bitnet_gemv_cpu(Tensor packed_weights, Tensor activations, float scale) -> Tensor");
    m.impl("bitnet_gemv_cpu", torch::kCPU, &bitnet_gemv_cpu);
    m.def("bitnet_gemm_cpu(Tensor packed_weights, Tensor activations, float scale) -> Tensor");
    m.impl("bitnet_gemm_cpu", torch::kCPU, &bitnet_gemm_cpu);
    m.def("bitnet_quantize_activations_cpu(Tensor input) -> (Tensor, Tensor)");
    m.impl("bitnet_quantize_activations_cpu", torch::kCPU, &bitnet_quantize_activations_cpu);

    // BitNet fused operations for reduced Python overhead
    m.def("bitnet_mlp_forward_cpu(Tensor input, Tensor gate_weight, Tensor up_weight, Tensor down_weight, "
          "float gate_scale, float up_scale, float down_scale, Tensor ffn_sub_norm, float eps) -> Tensor");
    m.impl("bitnet_mlp_forward_cpu", torch::kCPU, &bitnet_mlp_forward_cpu);
    m.def("bitnet_qkv_forward_cpu(Tensor input, Tensor q_weight, Tensor k_weight, Tensor v_weight, "
          "float q_scale, float k_scale, float v_scale) -> (Tensor, Tensor, Tensor)");
    m.impl("bitnet_qkv_forward_cpu", torch::kCPU, &bitnet_qkv_forward_cpu);

    // Kernel availability check
    m.def("bitnet_check_kernel_available() -> bool");
    m.def("bitnet_get_cpu_capabilities() -> str");

    // Minimal ops needed for inference
    m.def("rmsnorm_cpu(Tensor input, Tensor weight, float eps) -> Tensor");
    m.impl("rmsnorm_cpu", torch::kCPU, &rmsnorm_cpu);
    m.def("silu_and_mul_cpu(Tensor input) -> Tensor");
    m.impl("silu_and_mul_cpu", torch::kCPU, &silu_and_mul_cpu);
}

TORCH_LIBRARY_IMPL(sgl_kernel, CatchAll, m) {
    m.impl("bitnet_check_kernel_available", &bitnet_check_kernel_available);
    m.impl("bitnet_get_cpu_capabilities", &bitnet_get_cpu_capabilities);
}

REGISTER_EXTENSION(common_ops)
