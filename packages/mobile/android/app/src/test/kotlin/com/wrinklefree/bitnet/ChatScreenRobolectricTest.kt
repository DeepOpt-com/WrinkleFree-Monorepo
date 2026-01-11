package com.wrinklefree.bitnet

import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import com.wrinklefree.bitnet.ui.theme.BitNetChatTheme
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Local JVM tests using Robolectric - no emulator needed!
 * Run with: ./gradlew :app:testDebugUnitTest
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], manifest = Config.NONE)
class ChatScreenRobolectricTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun `chat screen displays title and input`() {
        val viewModel = ChatViewModel()
        // Enable mock mode for testing
        forceLoadMockMode(viewModel)

        composeTestRule.setContent {
            BitNetChatTheme {
                ChatScreen(viewModel = viewModel)
            }
        }

        // Verify input field exists
        composeTestRule.onNodeWithTag("chatInput")
            .assertIsDisplayed()

        // Verify send button exists
        composeTestRule.onNodeWithTag("sendButton")
            .assertExists()
    }

    @Test
    fun `send button is disabled when input is empty`() {
        val viewModel = ChatViewModel()
        forceLoadMockMode(viewModel)

        composeTestRule.setContent {
            BitNetChatTheme {
                ChatScreen(viewModel = viewModel)
            }
        }

        // Send button should be disabled with empty input
        composeTestRule.onNodeWithTag("sendButton")
            .assertIsNotEnabled()
    }

    @Test
    fun `typing enables send button`() {
        val viewModel = ChatViewModel()
        forceLoadMockMode(viewModel)

        composeTestRule.setContent {
            BitNetChatTheme {
                ChatScreen(viewModel = viewModel)
            }
        }

        // Type a message
        composeTestRule.onNodeWithTag("chatInput")
            .performTextInput("Hello")

        // Send button should now be enabled
        composeTestRule.onNodeWithTag("sendButton")
            .assertIsEnabled()
    }

    @Test
    fun `sending message adds user message bubble`() {
        val viewModel = ChatViewModel()
        forceLoadMockMode(viewModel)

        composeTestRule.setContent {
            BitNetChatTheme {
                ChatScreen(viewModel = viewModel)
            }
        }

        // Type and send
        composeTestRule.onNodeWithTag("chatInput")
            .performTextInput("Test message")
        composeTestRule.onNodeWithTag("sendButton")
            .performClick()

        // Wait for UI update
        composeTestRule.waitForIdle()

        // User message should appear
        composeTestRule.onNodeWithTag("userMessage")
            .assertIsDisplayed()
    }

    @Test
    fun `empty state shown when no messages`() {
        val viewModel = ChatViewModel()
        forceLoadMockMode(viewModel)

        composeTestRule.setContent {
            BitNetChatTheme {
                ChatScreen(viewModel = viewModel)
            }
        }

        // Empty state should be visible
        composeTestRule.onNodeWithTag("emptyState")
            .assertIsDisplayed()
    }

    /**
     * Helper to trigger mock mode loading
     */
    private fun forceLoadMockMode(viewModel: ChatViewModel) {
        // On x86 (test JVM), this will enable mock mode
        viewModel.loadModel("")
        // Wait for async load
        Thread.sleep(100)
    }
}
