package com.wrinklefree.llmcore

import android.os.Build
import android.util.Log
import java.io.Closeable

/**
 * LLM inference engine for Android using llama.cpp with TL1 kernels.
 *
 * On ARM64 devices: Uses native llama.cpp for real inference.
 * On x86/x86_64 (emulators): Falls back to mock mode automatically.
 *
 * Example usage:
 * ```kotlin
 * if (LLMEngine.isNativeAvailable) {
 *     LLMEngine.initBackend()
 *     val engine = LLMEngine.load("/path/to/model-tl1.gguf")
 *     val response = engine.generate("Hello, how are you?")
 *     engine.close()
 *     LLMEngine.freeBackend()
 * }
 * ```
 */
class LLMEngine private constructor(private var handle: Long) : Closeable {

    /**
     * Generate text from a prompt.
     *
     * @param prompt Input text prompt
     * @param maxTokens Maximum tokens to generate (default: 256)
     * @return Generated text, or null on error
     */
    fun generate(prompt: String, maxTokens: Int = 256): String? {
        check(handle != 0L) { "Engine has been closed" }
        return generate(handle, prompt, maxTokens)
    }

    /**
     * Get the tokens per second from the last generation.
     */
    val tokensPerSecond: Float
        get() {
            check(handle != 0L) { "Engine has been closed" }
            return getTokensPerSecond(handle)
        }

    /**
     * Release the engine resources.
     */
    override fun close() {
        if (handle != 0L) {
            destroy(handle)
            handle = 0L
        }
    }

    protected fun finalize() {
        close()
    }

    // Native methods
    private external fun generate(handle: Long, prompt: String, maxTokens: Int): String?
    private external fun getTokensPerSecond(handle: Long): Float
    private external fun destroy(handle: Long)

    companion object {
        private const val TAG = "LLMEngine"

        /**
         * Whether native library is available (ARM64 only).
         * On x86 emulators this is false - use mock mode instead.
         */
        val isNativeAvailable: Boolean

        init {
            // Only try to load native library on ARM devices
            val isArm = Build.SUPPORTED_ABIS.any { it.startsWith("arm") }
            isNativeAvailable = if (isArm) {
                try {
                    System.loadLibrary("llmcore")
                    Log.i(TAG, "Loaded llmcore native library")
                    true
                } catch (e: UnsatisfiedLinkError) {
                    Log.e(TAG, "Failed to load llmcore: ${e.message}")
                    false
                }
            } else {
                Log.i(TAG, "Non-ARM device detected (${Build.SUPPORTED_ABIS.joinToString()}), native library not available")
                false
            }
        }

        /**
         * Initialize the llama.cpp backend.
         * Call once on application startup.
         */
        @JvmStatic
        external fun initBackend()

        /**
         * Free the llama.cpp backend.
         * Call on application shutdown.
         */
        @JvmStatic
        external fun freeBackend()

        /**
         * Load a model and create an engine instance.
         *
         * @param modelPath Path to the GGUF model file (TL1 quantized for ARM)
         * @param nThreads Number of threads for inference (default: 4)
         * @param nCtx Context size (default: 2048)
         * @return LLMEngine instance, or null on failure
         */
        @JvmStatic
        fun load(
            modelPath: String,
            nThreads: Int = 4,
            nCtx: Int = 2048
        ): LLMEngine? {
            val handle = create(modelPath, nThreads, nCtx)
            return if (handle != 0L) {
                LLMEngine(handle)
            } else {
                Log.e(TAG, "Failed to load model: $modelPath")
                null
            }
        }

        @JvmStatic
        private external fun create(modelPath: String, nThreads: Int, nCtx: Int): Long
    }
}
