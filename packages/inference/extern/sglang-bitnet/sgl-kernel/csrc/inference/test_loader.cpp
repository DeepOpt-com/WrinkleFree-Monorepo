// Debug test for model loading
#include <cstdio>
#include "sglkernel_loader.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model.bin>\n", argv[0]);
        return 1;
    }

    printf("Loading model: %s\n", argv[1]);

    sgl_kernel::SGLKernelModelLoader loader;
    if (!loader.load(argv[1])) {
        printf("Failed to load: %s\n", loader.error().c_str());
        return 1;
    }

    printf("\n=== Model Config ===\n");
    auto& cfg = loader.config();
    printf("vocab_size: %d\n", cfg.vocab_size);
    printf("hidden_size: %d\n", cfg.hidden_size);
    printf("intermediate_size: %d\n", cfg.intermediate_size);
    printf("num_hidden_layers: %d\n", cfg.num_hidden_layers);
    printf("num_attention_heads: %d\n", cfg.num_attention_heads);
    printf("num_key_value_heads: %d\n", cfg.num_key_value_heads);
    printf("max_position_embeddings: %d\n", cfg.max_position_embeddings);
    printf("rms_norm_eps: %e\n", cfg.rms_norm_eps);

    printf("\n=== Tensors (%zu total) ===\n", loader.tensors().size());
    size_t total_bytes = 0;
    for (const auto& tensor : loader.tensors()) {
        const char* dtype_str = "???";
        switch (tensor.dtype) {
            case sgl_kernel::DType::UINT8: dtype_str = "uint8"; break;
            case sgl_kernel::DType::FLOAT32: dtype_str = "fp32"; break;
            case sgl_kernel::DType::FLOAT16: dtype_str = "fp16"; break;
            case sgl_kernel::DType::BFLOAT16: dtype_str = "bf16"; break;
        }

        printf("%s: dtype=%s, shape=[", tensor.name.c_str(), dtype_str);
        for (size_t i = 0; i < tensor.shape.size(); i++) {
            printf("%d", tensor.shape[i]);
            if (i < tensor.shape.size() - 1) printf(", ");
        }
        printf("], size=%zu", tensor.data_size);
        if (tensor.has_scale) {
            printf(", scale=%.6f", tensor.scale);
        }
        printf("\n");
        total_bytes += tensor.data_size;
    }

    printf("\nTotal tensor bytes: %zu (%.2f MB)\n", total_bytes, total_bytes / 1024.0 / 1024.0);

    return 0;
}
