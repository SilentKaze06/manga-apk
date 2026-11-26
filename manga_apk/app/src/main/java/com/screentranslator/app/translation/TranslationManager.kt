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
        val enginesChanged = currentSettings.engineType != settings.engineType ||
                currentSettings.targetLanguage != settings.targetLanguage ||
                currentSettings.deeplApiKey != settings.deeplApiKey ||
                currentSettings.libreTranslateUrl != settings.libreTranslateUrl ||
                currentSettings.libreTranslateApiKey != settings.libreTranslateApiKey
        if (enginesChanged) {
            cacheGeneration++
            translationCache.clear()
        }
        this.currentSettings = settings
    }

    private fun cacheKey(
        generation: Long,
        engineType: TranslationEngineType,
        sourceLang: String,
        targetLang: String,
        text: String
    ): String = "$generation|${engineType.name}|$sourceLang|$targetLang|$text"

    /**
     * Translates [text] into [targetLangCode], utilizing the chosen engine,
     * with in-memory caching keyed by the actual executing engine and automatic fallback to MyMemory.
     */
    suspend fun translate(
        text: String,
        sourceLang: String?,
        targetLangCode: String
    ): Result<String> {
        val cleanText = text.trim()
        if (cleanText.isBlank()) return Result.success(text)

        // Preserve "auto" when source language is unspecified so DeepL/LibreTranslate can auto-detect
        val effectiveSource = com.screentranslator.app.data.model.MangaSourceLanguage.normalizeToIso(sourceLang)
            ?: "auto"

        // Check memory cache first (0ms latency for repeated texts/phrases)
        val (settings, generation) = synchronized(this) { currentSettings to cacheGeneration }
        val selectedType = settings.engineType
        val primaryKey = cacheKey(generation, selectedType, effectiveSource, targetLangCode, cleanText)
        translationCache[primaryKey]?.let { cached ->
            return Result.success(cached)
        }

        if (!NetworkUtils.isNetworkAvailable(context)) {
            showToastOnMain("Internet aloqasi yo'q.")
            return Result.failure(Exception("Internet aloqasi mavjud emas"))
        }

        val activeEngine = getEngine(settings)

        val primaryResult = activeEngine.translate(cleanText, effectiveSource, targetLangCode)
        if (primaryResult.isSuccess) {
            synchronized(this) {
                if (generation == cacheGeneration) primaryResult.getOrNull()?.let { translationCache[primaryKey] = it }
            }
            return primaryResult
        }

        val error = primaryResult.exceptionOrNull()
        Log.w("TranslationManager", "Dvigatel ${selectedType.name} xatosi: ${error?.message}")

        if (selectedType != TranslationEngineType.MY_MEMORY) {
            if (selectedType == TranslationEngineType.DEEPL && error is IllegalArgumentException) {
                showToastOnMain("DeepL $targetLangCode tilini qo'llamaydi; MyMemory ishlatilmoqda.")
            }
            val fallbackKey = cacheKey(generation, TranslationEngineType.MY_MEMORY, effectiveSource, targetLangCode, cleanText)
            translationCache[fallbackKey]?.let { cachedFallback ->
                return Result.success(cachedFallback)
            }
            Log.i("TranslationManager", "Asosiy dvigatel muvaffaqiyatsiz bo'ldi. MyMemory dvigateliga avtomatik o'tilmoqda")
            val fallbackResult = myMemoryEngine.translate(cleanText, effectiveSource, targetLangCode)
            if (fallbackResult.isSuccess) {
                synchronized(this) {
                    if (generation == cacheGeneration) fallbackResult.getOrNull()?.let { translationCache[fallbackKey] = it }
                }
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
