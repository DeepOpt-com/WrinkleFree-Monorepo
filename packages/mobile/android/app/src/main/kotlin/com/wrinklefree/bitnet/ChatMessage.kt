package com.wrinklefree.bitnet

import java.util.UUID

/**
 * Represents a single message in the chat conversation.
 */
data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val content: String,
    val isUser: Boolean,
    val timestamp: Long = System.currentTimeMillis(),
    val tokensPerSecond: Float? = null,
    val isError: Boolean = false
)
