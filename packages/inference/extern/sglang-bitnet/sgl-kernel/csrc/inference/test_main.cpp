// Simple test for BitNet inference engine
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>

#include "bitnet_engine.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model.sglbin> [num_tokens]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    int num_generate = argc > 2 ? atoi(argv[2]) : 50;

    printf("Loading model: %s\n", model_path);

    BitNetConfig config = {
        .max_seq_len = 2048,
        .num_threads = 0,  // auto
        .kv_cache_size = 256
    };

    BitNetEngine* engine = bitnet_engine_create(model_path, &config);
    if (!engine) {
        printf("Failed to load model: %s\n", bitnet_get_error());
        return 1;
    }

    printf("Model loaded successfully!\n");
    printf("  Vocab size: %d\n", bitnet_vocab_size(engine));
    printf("  Hidden size: %d\n", bitnet_hidden_size(engine));
    printf("  Num layers: %d\n", bitnet_num_layers(engine));
    printf("  Max seq len: %d\n", bitnet_max_seq_len(engine));

    // Debug: print more config
    extern "C" int bitnet_get_num_kv_heads(BitNetEngine* engine);
    printf("  Num KV heads: %d\n", bitnet_get_num_kv_heads(engine));

    // Token IDs for "Hello, my name is" (from Qwen tokenizer)
    int32_t input_ids[] = {9707, 11, 856, 836, 374};
    int num_input = 5;

    printf("\nGenerating with %d input tokens, max %d new tokens...\n",
           num_input, num_generate);

    SamplingParams params = {
        .temperature = 0.0f,  // greedy for reproducibility
        .top_p = 1.0f,
        .top_k = 0,
        .repetition_penalty = 1.0f,
        .max_tokens = num_generate
    };

    auto start = std::chrono::high_resolution_clock::now();

    GenerationResult result;
    int status = bitnet_generate(engine, input_ids, num_input, &params, &result);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    if (status != 0) {
        printf("Generation failed: %s\n", bitnet_get_error());
        bitnet_engine_destroy(engine);
        return 1;
    }

    printf("\nGenerated %d tokens in %.2f seconds\n", result.num_tokens, elapsed);
    printf("Throughput: %.2f tokens/sec\n", result.num_tokens / elapsed);

    printf("\nOutput token IDs:\n");
    for (int i = 0; i < result.num_tokens && i < 100; i++) {
        printf("%d ", result.output_ids[i]);
        if ((i + 1) % 20 == 0) printf("\n");
    }
    printf("\n");

    bitnet_free_result(&result);
    bitnet_engine_destroy(engine);

    return 0;
}
