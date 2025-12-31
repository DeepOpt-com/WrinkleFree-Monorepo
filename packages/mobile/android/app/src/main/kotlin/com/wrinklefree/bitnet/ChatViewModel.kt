package com.wrinklefree.bitnet

import android.os.Build
import android.util.Log
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.wrinklefree.llmcore.LLMEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * ViewModel for managing chat state and LLM inference.
 */
class ChatViewModel : ViewModel() {
    companion object {
        private const val TAG = "ChatViewModel"

        /**
         * Check if we're running on an ARM device (real inference)
         * or x86/x86_64 (mock mode for emulator)
         */
        val isArmDevice: Boolean
            get() = Build.SUPPORTED_ABIS.any { it.startsWith("arm") }
    }

    private val _messages = mutableStateListOf<ChatMessage>()
    val messages: List<ChatMessage> = _messages

    private var _isGenerating by mutableStateOf(false)
    val isGenerating: Boolean get() = _isGenerating

    private var _isModelLoaded by mutableStateOf(false)
    val isModelLoaded: Boolean get() = _isModelLoaded

    private var _modelLoadError by mutableStateOf<String?>(null)
    val modelLoadError: String? get() = _modelLoadError

    private var engine: LLMEngine? = null

    /**
     * Load the LLM model from the given path.
     * On x86 emulators, this enables mock mode instead.
     */
    fun loadModel(modelPath: String) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                if (!isArmDevice) {
                    // Mock mode for x86 emulator
                    Log.i(TAG, "Running on x86 - enabling mock mode")
                    withContext(Dispatchers.Main) {
                        _isModelLoaded = true
                        _messages.add(ChatMessage(
                            content = "Mock mode enabled (x86 emulator detected). Responses are simulated.",
                            isUser = false
                        ))
                    }
                    return@launch
                }

                Log.i(TAG, "Loading model from: $modelPath")
                LLMEngine.initBackend()
                engine = LLMEngine.load(modelPath, nThreads = 4, nCtx = 2048)

                withContext(Dispatchers.Main) {
                    if (engine != null) {
                        _isModelLoaded = true
                        Log.i(TAG, "Model loaded successfully")
                    } else {
                        _modelLoadError = "Failed to load model"
                        Log.e(TAG, "Failed to load model")
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error loading model", e)
                withContext(Dispatchers.Main) {
                    _modelLoadError = e.message ?: "Unknown error"
                }
            }
        }
    }

    /**
     * Send a message and generate a response.
     */
    fun sendMessage(text: String) {
        if (text.isBlank() || _isGenerating) return

        _messages.add(ChatMessage(content = text.trim(), isUser = true))
        _isGenerating = true

        viewModelScope.launch(Dispatchers.IO) {
            try {
                val (response, tps) = if (!isArmDevice) {
                    // Mock mode for emulator testing
                    kotlinx.coroutines.delay(500) // Simulate generation time
                    val mockResponse = generateMockResponse(text)
                    Pair(mockResponse, 30.0f)
                } else {
                    // Real inference on ARM device
                    val result = engine?.generate(text, maxTokens = 256) ?: "Error: Engine not loaded"
                    val speed = engine?.tokensPerSecond ?: 0f
                    Pair(result, speed)
                }

                withContext(Dispatchers.Main) {
                    _messages.add(ChatMessage(
                        content = response,
                        isUser = false,
                        tokensPerSecond = tps
                    ))
                    _isGenerating = false
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error generating response", e)
                withContext(Dispatchers.Main) {
                    _messages.add(ChatMessage(
                        content = "Error: ${e.message}",
                        isUser = false,
                        isError = true
                    ))
                    _isGenerating = false
                }
            }
        }
    }

    private fun generateMockResponse(prompt: String): String {
        val responses = listOf(
            "This is a mock response for testing. The real model would generate actual content here.",
            "Mock mode active! Your prompt was: \"${prompt.take(50)}...\"",
            "Testing the UI flow. On a real ARM device, you'd see actual LLM output.",
            "The BitNet 2B model can generate ~30 tok/s on Snapdragon 8 Gen 3.",
            "Mock response generated successfully. Deploy to a real device for actual inference."
        )
        return responses.random()
    }

    override fun onCleared() {
        super.onCleared()
        engine?.close()
        if (isArmDevice) {
            LLMEngine.freeBackend()
        }
    }
}
