package com.screentranslator.app.translation

import com.screentranslator.app.data.model.TranslationEngineType

/**
 * Common contract for translation engine backends.
 */
interface TranslationEngine {
    val engineType: TranslationEngineType

    /**
     * Translates [text] from [sourceLang] (or auto-detected if null) to [targetLang].
     * Returns [Result.success] with the translated text, or [Result.failure] with error details.
     */
    suspend fun translate(
        text: String,
        sourceLang: String?,
        targetLang: String
    ): Result<String>
}
