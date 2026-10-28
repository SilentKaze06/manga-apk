package com.screentranslator.app.translation

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.Toast
import com.screentranslator.app.data.model.TranslationEngineType
import com.screentranslator.app.data.network.NetworkClient
import com.screentranslator.app.data.preferences.UserSettings
import com.screentranslator.app.util.NetworkUtils

/**
 * Orchestrates translation calls across engines (MyMemory, LibreTranslate, DeepL).
 * Uses MyMemory as the automatic fallback if an engine fails.
 */
class TranslationManager(
    private val context: Context,
    initialSettings: UserSettings
) {
    @Volatile
    private var currentSettings: UserSettings = initialSettings

    private val myMemoryEngine = MyMemoryTranslationEngine(
        myMemoryApi = NetworkClient.myMemoryApi
    )

    private var cacheGeneration = 0L

    // In-memory thread-safe LRU cache for translations to eliminate redundant network calls
    // Key: "$effectiveSource|$targetLangCode|$text" -> Value: translatedText
    private val translationCache = java.util.Collections.synchronizedMap(
        object : java.util.LinkedHashMap<String, String>(100, 0.75f, true) {
            override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, String>?): Boolean {
                return size > 300 // Keep last 300 translations in memory
            }
        }
    )

    private val mainHandler = Handler(Looper.getMainLooper())
    private var lastToastTime = 0L
    private var lastToastMessage = ""

    @Synchronized
    fun updateSettings(settings: UserSettings) {
        if (currentSettings != settings) {
            cacheGeneration++
            translationCache.clear()
            this.currentSettings = settings
        }
    }

    /**
     * Translates [text] into [targetLangCode], utilizing the chosen engine,
     * with in-memory caching and automatic fallback to MyMemory if the selected engine fails.
     */
    suspend fun translate(
        text: String,
        sourceLang: String?,
        targetLangCode: String
    ): Result<String> {
        val cleanText = text.trim()
        if (cleanText.isBlank()) return Result.success(text)

        // Ensure source language is normalized to ISO 639-1 (e.g. jpn -> ja, kor -> ko) or detected from script
        val effectiveSource = com.screentranslator.app.data.model.MangaSourceLanguage.normalizeToIso(sourceLang)
            ?: com.screentranslator.app.ocr.MangaBubbleDetector.detectLanguageFromText(cleanText)
            ?: "en"

        // Check memory cache first (0ms latency for repeated texts/phrases)
        val (settings, generation) = synchronized(this) { currentSettings to cacheGeneration }
        val cacheKey = "$generation|${settings.engineType}|$effectiveSource|$targetLangCode|$cleanText"
        translationCache[cacheKey]?.let { cached ->
            return Result.success(cached)
        }

        if (!NetworkUtils.isNetworkAvailable(context)) {
            showToastOnMain("Internet aloqasi yo'q.")
            return Result.failure(Exception("Internet aloqasi mavjud emas"))
        }

        val selectedType = settings.engineType
        val activeEngine = getEngine(settings)

        val primaryResult = activeEngine.translate(cleanText, effectiveSource, targetLangCode)
        if (primaryResult.isSuccess) {
            synchronized(this) {
                if (generation == cacheGeneration) primaryResult.getOrNull()?.let { translationCache[cacheKey] = it }
            }
            return primaryResult
        }

        // If primary engine failed and it wasn't already MyMemory, trigger fallback to MyMemory
        if (selectedType != TranslationEngineType.MY_MEMORY) {
            val error = primaryResult.exceptionOrNull()
            Log.w("TranslationManager", "Dvigatel ${selectedType.name} xatosi: ${error?.message}. MyMemory'ga o'tilmoqda.")
            showToastOnMain("${selectedType.title} xatosi. MyMemory orqali tarjima qilinmoqda.")

            val fallbackResult = myMemoryEngine.translate(cleanText, effectiveSource, targetLangCode)
            if (fallbackResult.isSuccess) {
                // A temporary fallback must not mask a recovered selected provider in the cache.
                return fallbackResult
            }
        }

        return primaryResult
    }

    private fun getEngine(settings: UserSettings): TranslationEngine {
        return when (settings.engineType) {
            TranslationEngineType.MY_MEMORY -> myMemoryEngine
            TranslationEngineType.LIBRE_TRANSLATE -> LibreTranslateEngine(
                urlProvider = { settings.libreTranslateUrl }, apiKeyProvider = { settings.libreTranslateApiKey }
            )
            TranslationEngineType.DEEPL -> DeepLTranslationEngine(
                deepLApi = NetworkClient.deepLApi, apiKeyProvider = { settings.deeplApiKey }
            )
        }
    }

    private fun showToastOnMain(message: String) {
        val now = System.currentTimeMillis()
        synchronized(this) {
            if (message == lastToastMessage && now - lastToastTime < 3000L) {
                return
            }
            lastToastTime = now
            lastToastMessage = message
        }
        mainHandler.post {
            Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
        }
    }

    fun release() {
        // No local resources to close
    }
}
