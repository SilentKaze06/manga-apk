package com.screentranslator.app.translation

import android.util.Log
import com.screentranslator.app.data.model.TranslationEngineType
import com.screentranslator.app.data.network.LibreTranslateRequest
import com.screentranslator.app.data.network.NetworkClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.withContext

/**
 * Online translation engine using LibreTranslate.
 * Supports public mirrors and self-hosted instances with optional API keys.
 */
class LibreTranslateEngine(
    private val urlProvider: () -> String,
    private val apiKeyProvider: () -> String
) : TranslationEngine {

    override val engineType: TranslationEngineType = TranslationEngineType.LIBRE_TRANSLATE

    companion object {
        // ✅ TUZATILDI: Matn uzunligi chegarasi
        private const val MAX_INPUT_LENGTH = 5000
    }

    override suspend fun translate(
        text: String,
        sourceLang: String?,
        targetLang: String
    ): Result<String> = withContext(Dispatchers.IO) {
        if (text.isBlank()) return@withContext Result.success(text)

        val chunks = TranslationTextChunks.split(text, MAX_INPUT_LENGTH)
        if (chunks.size > 1) {
            val translated = mutableListOf<String>()
            for (chunk in chunks) {
                val result = translate(chunk, sourceLang, targetLang)
                if (result.isFailure) return@withContext result
                translated += result.getOrThrow()
            }
            return@withContext Result.success(translated.joinToString("\n"))
        }
        val inputText = text

        val baseUrl = urlProvider().trim().ifEmpty { "https://libretranslate.com" }
        val apiKey = apiKeyProvider().trim().ifEmpty { null }

        val normalizedSource = com.screentranslator.app.data.model.MangaSourceLanguage.normalizeToIso(sourceLang)
        val source = normalizedSource ?: "auto"
        val target = targetLang.lowercase()

        if (source == target) {
            return@withContext Result.success(text)
        }

        try {
            val api = NetworkClient.createLibreTranslateApi(baseUrl)
            val request = LibreTranslateRequest(
                q = inputText,
                source = source,
                target = target,
                apiKey = apiKey
            )

            val response = api.translate(request)
            if (response.isSuccessful) {
                val translated = response.body()?.translatedText
                if (!translated.isNullOrEmpty()) {
                    Result.success(translated)
                } else {
                    Result.failure(Exception("LibreTranslate returned an empty translation."))
                }
            } else {
                val errorBody = response.errorBody()?.string()
                val errorMsg = "LibreTranslate failed (HTTP ${response.code()}): ${errorBody ?: response.message()}"
                Log.w("LibreTranslateEngine", errorMsg)
                Result.failure(Exception(errorMsg))
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e("LibreTranslateEngine", "Network request error: ${e.message}", e)
            Result.failure(Exception("LibreTranslate server connection error: ${e.localizedMessage ?: e.message}"))
        }
    }
}
