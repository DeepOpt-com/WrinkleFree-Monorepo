package com.wrinklefree.bitnet

import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.wrinklefree.bitnet.ui.theme.BitNetChatTheme
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Instrumented UI tests for ChatScreen using Compose Testing APIs.
 * These tests run on an Android emulator/device.
 *
 * Run with: ./gradlew :app:connectedDebugAndroidTest
 */
@RunWith(AndroidJUnit4::class)
class ChatScreenTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    /**
     * Test that the chat input field is displayed and accepts text.
     */
    @Test
    fun chatInput_displaysAndAcceptsText() {
        val viewModel = ChatViewModel()
        // Force mock mode for testing
        setMockModelLoaded(viewModel)

        composeTestRule.setContent {
            BitNetChatTheme {
                ChatScreen(viewModel = viewModel)
            }
        }

        // Verify input field exists and is displayed
        composeTestRule.onNodeWithTag("chatInput")
            .assertIsDisplayed()

        // Type a message
        composeTestRule.onNodeWithTag("chatInput")
            .performTextInput("Hello BitNet")

        // Verify send button is enabled when there's text
        composeTestRule.onNodeWithTag("sendButton")
            .assertIsEnabled()
    }

    /**
     * Test that sending a message adds it to the chat.
     */
    @Test
    fun sendButton_addsUserMessage() {
        val viewModel = ChatViewModel()
        setMockModelLoaded(viewModel)

        composeTestRule.setContent {
            BitNetChatTheme {
                ChatScreen(viewModel = viewModel)
            }
        }

        // Type a message
        composeTestRule.onNodeWithTag("chatInput")
            .performTextInput("Test message from UI test")

        // Click send
        composeTestRule.onNodeWithTag("sendButton")
            .performClick()

        // Wait for UI to update
        composeTestRule.waitForIdle()

        // Verify user message appears
        composeTestRule.onNodeWithTag("userMessage")
            .assertIsDisplayed()

        composeTestRule.onNodeWithText("Test message from UI test")
            .assertExists()
    }

    /**
     * Test that the empty state is shown when there are no messages.
     */
    @Test
    fun emptyState_showsWhenNoMessages() {
        val viewModel = ChatViewModel()
        setMockModelLoaded(viewModel)

        composeTestRule.setContent {
            BitNetChatTheme {
                ChatScreen(viewModel = viewModel)
            }
        }

        // Verify empty state text is shown
        composeTestRule.onNodeWithTag("emptyState")
            .assertIsDisplayed()
    }

    /**
     * Test that the send button is disabled when input is empty.
     */
    @Test
    fun sendButton_disabledWhenInputEmpty() {
        val viewModel = ChatViewModel()
        setMockModelLoaded(viewModel)

        composeTestRule.setContent {
            BitNetChatTheme {
                ChatScreen(viewModel = viewModel)
            }
        }

        // Verify send button is disabled with empty input
        composeTestRule.onNodeWithTag("sendButton")
            .assertIsNotEnabled()
    }

    /**
     * Test that assistant messages display TPS.
     */
    @Test
    fun assistantMessage_displaysTPS() {
        val viewModel = ChatViewModel()
        setMockModelLoaded(viewModel)

        composeTestRule.setContent {
            BitNetChatTheme {
                ChatScreen(viewModel = viewModel)
            }
        }

        // Send a message to trigger response
        composeTestRule.onNodeWithTag("chatInput")
            .performTextInput("Hello")
        composeTestRule.onNodeWithTag("sendButton")
            .performClick()

        // Wait for mock generation (500ms delay + some buffer)
        composeTestRule.waitUntil(timeoutMillis = 2000) {
            composeTestRule.onAllNodesWithTag("assistantMessage")
                .fetchSemanticsNodes().isNotEmpty()
        }

        // Verify assistant message with TPS is displayed
        composeTestRule.onNodeWithTag("assistantMessage")
            .assertIsDisplayed()

        composeTestRule.onNodeWithTag("tpsDisplay")
            .assertIsDisplayed()
    }

    /**
     * Test message list scrolls correctly with multiple messages.
     */
    @Test
    fun messageList_scrollsWithMultipleMessages() {
        val viewModel = ChatViewModel()
        setMockModelLoaded(viewModel)

        composeTestRule.setContent {
            BitNetChatTheme {
                ChatScreen(viewModel = viewModel)
            }
        }

        // Send multiple messages
        repeat(3) { i ->
            composeTestRule.onNodeWithTag("chatInput")
                .performTextInput("Message $i")
            composeTestRule.onNodeWithTag("sendButton")
                .performClick()

            // Wait for response
            composeTestRule.waitUntil(timeoutMillis = 2000) {
                composeTestRule.onAllNodesWithTag("assistantMessage")
                    .fetchSemanticsNodes().size > i
            }
        }

        // Verify message list exists and has content
        composeTestRule.onNodeWithTag("messageList")
            .assertIsDisplayed()

        // Verify we have multiple user messages
        composeTestRule.onAllNodesWithTag("userMessage")
            .assertCountEquals(3)
    }

    /**
     * Helper to set the ViewModel to a "model loaded" state for testing.
     * Uses reflection since the property is private.
     */
    private fun setMockModelLoaded(viewModel: ChatViewModel) {
        // Trigger mock mode loading by calling loadModel with empty path
        // On x86 emulators this automatically enables mock mode
        viewModel.loadModel("")
    }
}
