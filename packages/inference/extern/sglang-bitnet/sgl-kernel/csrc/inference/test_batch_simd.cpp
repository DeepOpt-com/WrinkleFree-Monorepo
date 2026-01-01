/**
 * Test program for BitNet SIMD Batch Engine
 *
 * Tests:
 * 1. Model loading from .bin format
 * 2. Single token forward pass
 * 3. Multi-token generation with KV cache
 * 4. Performance benchmark
 */

#include "bitnet_batch.h"
#include "bitnet_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>

// Simple timer
class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
};

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "/home/gcpuser/models/dlm-bitnet-2b.bin";

    printf("=== BitNet SIMD Batch Engine Test ===\n\n");

    // Test 1: Create engine
    printf("1. Loading model from: %s\n", model_path);
    Timer timer;
    timer.start();

    BitNetBatchConfig config = bitnet_batch_config_default();
    config.max_sequences = 1;  // Start with just 1 sequence for debugging
    config.n_ctx_per_seq = 256;
    config.num_threads = 8;
    config.max_batch_size = 32;

    BitNetBatchEngine* engine = bitnet_batch_engine_create(model_path, &config);
    if (!engine) {
        printf("ERROR: Failed to create engine: %s\n", bitnet_batch_get_error());
        return 1;
    }
    printf("   Model loaded in %.2f ms\n", timer.elapsed_ms());

    // Print model info
    printf("\n2. Model info:\n");
    printf("   Vocab size: %d\n", bitnet_batch_vocab_size(engine));
    printf("   Hidden size: %d\n", bitnet_batch_n_embd(engine));
    printf("   Max sequences: %d\n", bitnet_batch_max_sequences(engine));
    printf("   Context per seq: %d\n", bitnet_batch_max_ctx_per_seq(engine));

    // Test 2: Allocate a sequence
    printf("\n3. Testing sequence allocation...\n");
    bitnet_seq_id seq = bitnet_seq_alloc(engine);
    if (seq < 0) {
        printf("ERROR: Failed to allocate sequence\n");
        bitnet_batch_engine_destroy(engine);
        return 1;
    }
    printf("   Allocated sequence ID: %d\n", seq);

    // Test 3: Single token forward pass
    printf("\n4. Testing single token forward pass...\n");

    // Create a batch with a single token (BOS token)
    BitNetBatch* batch = bitnet_batch_init(32, 4);
    if (!batch) {
        printf("ERROR: Failed to create batch\n");
        bitnet_batch_engine_destroy(engine);
        return 1;
    }

    // Add BOS token at position 0
    int32_t bos_token = 151643;  // BitNet BOS token
    bitnet_batch_add(batch, bos_token, 0, &seq, 1, 1);  // output logits = true

    timer.start();
    int result = bitnet_batch_decode(engine, batch);
    double single_token_ms = timer.elapsed_ms();

    if (result != 0) {
        printf("ERROR: Decode failed: %s\n", bitnet_batch_get_error());
        bitnet_batch_free(batch);
        bitnet_batch_engine_destroy(engine);
        return 1;
    }
    printf("   Single token forward: %.2f ms\n", single_token_ms);

    // Get logits and find top token
    const float* logits = bitnet_get_logits_ith(engine, 0);
    if (!logits) {
        printf("ERROR: Failed to get logits\n");
    } else {
        int32_t top_token = 0;
        float max_logit = logits[0];
        for (int i = 1; i < bitnet_batch_vocab_size(engine); i++) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                top_token = i;
            }
        }
        printf("   Top token: %d (logit: %.4f)\n", top_token, max_logit);
    }

    // Test 4: Multi-token generation benchmark
    printf("\n5. Running generation benchmark (32 tokens)...\n");

    // Reset sequence
    bitnet_seq_free(engine, seq);
    seq = bitnet_seq_alloc(engine);

    std::vector<int32_t> generated_tokens;
    generated_tokens.push_back(bos_token);

    int num_tokens = 32;
    timer.start();

    for (int i = 0; i < num_tokens; i++) {
        bitnet_batch_clear(batch);
        int32_t token = generated_tokens.back();
        int32_t pos = generated_tokens.size() - 1;
        bitnet_batch_add(batch, token, pos, &seq, 1, 1);

        if (bitnet_batch_decode(engine, batch) != 0) {
            printf("ERROR: Decode failed at step %d\n", i);
            break;
        }

        // Sample next token (greedy)
        BitNetSamplingParams params = {0.0f, 1.0f, 0, 1.0f, 100};  // Greedy
        int32_t next = bitnet_batch_sample(engine, 0, &params);

        if (next < 0) {
            printf("ERROR: Sampling failed at step %d\n", i);
            break;
        }

        generated_tokens.push_back(next);

        // Check for EOS
        if (bitnet_batch_is_eos(engine, next)) {
            printf("   EOS reached at step %d\n", i + 1);
            break;
        }
    }

    double total_ms = timer.elapsed_ms();
    int tokens_generated = generated_tokens.size() - 1;
    double tok_per_sec = tokens_generated / (total_ms / 1000.0);

    printf("\n   Generated %d tokens in %.2f ms\n", tokens_generated, total_ms);
    printf("   Speed: %.2f tok/s\n", tok_per_sec);

    printf("\n   Generated token IDs: ");
    for (size_t i = 0; i < generated_tokens.size() && i < 10; i++) {
        printf("%d ", generated_tokens[i]);
    }
    if (generated_tokens.size() > 10) printf("...");
    printf("\n");

    // Cleanup
    bitnet_batch_free(batch);
    bitnet_seq_free(engine, seq);
    bitnet_batch_engine_destroy(engine);

    printf("\n=== Test Complete ===\n");
    return 0;
}
