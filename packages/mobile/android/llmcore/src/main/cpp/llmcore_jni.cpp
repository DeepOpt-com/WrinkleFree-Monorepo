/**
 * JNI bindings for llama.cpp inference engine
 *
 * This provides a thin wrapper around llama.cpp for Android.
 * Uses TL1 kernels optimized for ARM NEON.
 */

#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <memory>

#include "llama.h"

#define LOG_TAG "LLMCore"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Engine state
struct LLMEngine {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_sampler* sampler = nullptr;
    int n_ctx = 2048;
    int n_threads = 4;
};

extern "C" {

/**
 * Initialize the llama backend (call once on app startup)
 */
JNIEXPORT void JNICALL
Java_com_wrinklefree_llmcore_LLMEngine_initBackend(JNIEnv* env, jclass clazz) {
    llama_backend_init();
    LOGI("llama.cpp backend initialized");
}

/**
 * Create a new LLM engine with the given model path
 *
 * @param modelPath Path to the GGUF model file (TL1 quantized for ARM)
 * @param nThreads Number of threads for inference (default: 4)
 * @param nCtx Context size (default: 2048)
 * @return Engine handle (pointer as long) or 0 on failure
 */
JNIEXPORT jlong JNICALL
Java_com_wrinklefree_llmcore_LLMEngine_create(
    JNIEnv* env, jobject thiz,
    jstring modelPath, jint nThreads, jint nCtx
) {
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    LOGI("Loading model: %s", path);

    auto* engine = new LLMEngine();
    engine->n_threads = nThreads > 0 ? nThreads : 4;
    engine->n_ctx = nCtx > 0 ? nCtx : 2048;

    // Model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU only on Android

    // Load model
    engine->model = llama_model_load_from_file(path, model_params);
    env->ReleaseStringUTFChars(modelPath, path);

    if (!engine->model) {
        LOGE("Failed to load model");
        delete engine;
        return 0;
    }

    // Context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = engine->n_ctx;
    ctx_params.n_threads = engine->n_threads;
    ctx_params.n_threads_batch = engine->n_threads;

    // Create context
    engine->ctx = llama_init_from_model(engine->model, ctx_params);
    if (!engine->ctx) {
        LOGE("Failed to create context");
        llama_model_free(engine->model);
        delete engine;
        return 0;
    }

    // Create sampler (greedy for speed)
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    engine->sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(engine->sampler, llama_sampler_init_greedy());

    LOGI("Model loaded successfully, ctx=%d, threads=%d", engine->n_ctx, engine->n_threads);
    return reinterpret_cast<jlong>(engine);
}

/**
 * Generate text from a prompt
 *
 * @param handle Engine handle from create()
 * @param prompt Input prompt text
 * @param maxTokens Maximum tokens to generate
 * @return Generated text or null on error
 */
JNIEXPORT jstring JNICALL
Java_com_wrinklefree_llmcore_LLMEngine_generate(
    JNIEnv* env, jobject thiz,
    jlong handle, jstring prompt, jint maxTokens
) {
    auto* engine = reinterpret_cast<LLMEngine*>(handle);
    if (!engine || !engine->ctx) {
        LOGE("Invalid engine handle");
        return nullptr;
    }

    const char* text = env->GetStringUTFChars(prompt, nullptr);
    LOGI("Generating from prompt: %.50s...", text);

    // Tokenize prompt
    std::vector<llama_token> tokens(engine->n_ctx);
    int n_tokens = llama_tokenize(
        engine->model, text, strlen(text),
        tokens.data(), tokens.size(),
        true,  // add_special
        true   // parse_special
    );
    env->ReleaseStringUTFChars(prompt, text);

    if (n_tokens < 0) {
        LOGE("Tokenization failed");
        return nullptr;
    }
    tokens.resize(n_tokens);

    // Clear KV cache for new generation
    llama_kv_cache_clear(engine->ctx);

    // Decode prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(engine->ctx, batch) != 0) {
        LOGE("Prompt decode failed");
        return nullptr;
    }

    // Generate tokens
    std::string output;
    int max_gen = maxTokens > 0 ? maxTokens : 256;

    for (int i = 0; i < max_gen; i++) {
        // Sample next token
        llama_token new_token = llama_sampler_sample(engine->sampler, engine->ctx, -1);

        // Check for end of generation
        if (llama_vocab_is_eog(engine->model, new_token)) {
            break;
        }

        // Decode token to text
        char buf[256];
        int len = llama_token_to_piece(engine->model, new_token, buf, sizeof(buf), 0, true);
        if (len > 0) {
            output.append(buf, len);
        }

        // Prepare next batch
        batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(engine->ctx, batch) != 0) {
            LOGE("Decode failed at token %d", i);
            break;
        }
    }

    LOGI("Generated %zu chars", output.size());
    return env->NewStringUTF(output.c_str());
}

/**
 * Get generation statistics
 *
 * @param handle Engine handle
 * @return Tokens per second (last generation)
 */
JNIEXPORT jfloat JNICALL
Java_com_wrinklefree_llmcore_LLMEngine_getTokensPerSecond(
    JNIEnv* env, jobject thiz, jlong handle
) {
    auto* engine = reinterpret_cast<LLMEngine*>(handle);
    if (!engine || !engine->ctx) {
        return 0.0f;
    }

    // Get timing info
    llama_perf_context_data perf = llama_perf_context(engine->ctx);
    if (perf.t_eval_ms > 0) {
        return (float)perf.n_eval / (perf.t_eval_ms / 1000.0f);
    }
    return 0.0f;
}

/**
 * Free the engine and release all resources
 */
JNIEXPORT void JNICALL
Java_com_wrinklefree_llmcore_LLMEngine_destroy(
    JNIEnv* env, jobject thiz, jlong handle
) {
    auto* engine = reinterpret_cast<LLMEngine*>(handle);
    if (engine) {
        if (engine->sampler) {
            llama_sampler_free(engine->sampler);
        }
        if (engine->ctx) {
            llama_free(engine->ctx);
        }
        if (engine->model) {
            llama_model_free(engine->model);
        }
        delete engine;
        LOGI("Engine destroyed");
    }
}

/**
 * Free the llama backend (call on app shutdown)
 */
JNIEXPORT void JNICALL
Java_com_wrinklefree_llmcore_LLMEngine_freeBackend(JNIEnv* env, jclass clazz) {
    llama_backend_free();
    LOGI("llama.cpp backend freed");
}

} // extern "C"
