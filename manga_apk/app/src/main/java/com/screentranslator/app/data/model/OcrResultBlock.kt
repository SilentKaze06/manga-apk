package com.screentranslator.app.data.model

import android.graphics.Rect

/**
 * Represents a recognized block of text on the screen,
 * its pixel bounds, detected source language, and translated text.
 */
data class OcrResultBlock(
    val id: String,
    val originalText: String,
    val boundingBox: Rect,
    val sourceLanguage: String? = null,
    val translatedText: String? = null,
    val isMangaBubble: Boolean = false,
    val isVertical: Boolean = false,
    val bubbleBgColor: Int = android.graphics.Color.WHITE,
    val bubbleTextColor: Int = android.graphics.Color.BLACK
)
