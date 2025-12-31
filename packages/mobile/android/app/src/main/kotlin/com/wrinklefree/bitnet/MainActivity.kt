package com.wrinklefree.bitnet

import android.os.Bundle
import android.os.Environment
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import com.wrinklefree.bitnet.ui.theme.BitNetChatTheme
import java.io.File

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            BitNetChatTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    val viewModel: ChatViewModel = viewModel()

                    // Try to load model from common locations
                    androidx.compose.runtime.LaunchedEffect(Unit) {
                        val modelPaths = listOf(
                            // Internal app storage
                            File(filesDir, "model.gguf").absolutePath,
                            // External storage (for ADB push)
                            "/data/local/tmp/bitnet/ggml-model-i2_s.gguf",
                            "/data/local/tmp/model.gguf",
                            // Downloads folder
                            File(
                                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
                                "ggml-model-i2_s.gguf"
                            ).absolutePath
                        )

                        val modelPath = modelPaths.firstOrNull { File(it).exists() }
                        if (modelPath != null) {
                            viewModel.loadModel(modelPath)
                        } else if (!ChatViewModel.isArmDevice) {
                            // On x86 emulator, just enable mock mode
                            viewModel.loadModel("")
                        }
                    }

                    ChatScreen(viewModel = viewModel)
                }
            }
        }
    }
}
