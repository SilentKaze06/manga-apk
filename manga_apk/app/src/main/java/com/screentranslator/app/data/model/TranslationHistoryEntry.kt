package com.screentranslator.app.data.model

import java.util.UUID

/**
 * A single translation session entry stored in history.
 * originalText and translatedText are combined from all OCR blocks.
 * imagePath points to the saved screenshot with translated overlay.
 */
data class TranslationHistoryEntry(
    val id: String = UUID.randomUUID().toString(),
    val timestamp: Long = System.currentTimeMillis(),
    val originalText: String,
    val translatedText: String,
    val targetLanguage: String,
    val imagePath: String? = null
)
