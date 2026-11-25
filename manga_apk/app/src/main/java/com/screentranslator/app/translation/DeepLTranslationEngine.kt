package com.screentranslator.app.translation

import android.util.Log
import com.screentranslator.app.data.model.TranslationEngineType
import com.screentranslator.app.data.network.DeepLApi
import com.screentranslator.app.data.network.DeepLRequest
import com.screentranslator.app.data.network.NetworkClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.withContext

/**
 * Online neural translation engine using DeepL Free API.
 * Endpoint: api-free.deepl.com
 */
class DeepLTranslationEngine(
    private val deepLApi: DeepLApi,
    private val apiKeyProvider: () -> String
) : TranslationEngine {

    override val engineType: TranslationEngineType = TranslationEngineType.DEEPL

    companion object {
        // ✅ TUZATILDI: Bir so'rovda maksimal belgilar soni chegarasi
        // DeepL bepul tarif: oyiga 500,000 belgi — haddan ziyod sarflashni oldini olish
        private const val MAX_INPUT_LENGTH = 4000
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

        val apiKey = apiKeyProvider().trim()
        if (apiKey.isEmpty()) {
            return@withContext Result.failure(
                IllegalStateException("DeepL API kaliti kiritilmagan. Iltimos, Sozlamalar bo'limiga kiriting.")
            )
        }

        val mappedTarget = mapToDeepLTargetLanguage(targetLang)
        if (mappedTarget == null) {
            return@withContext Result.failure(
                IllegalArgumentException("DeepL does not support target language '$targetLang'.")
            )
        }

        val mappedSource = sourceLang?.let { mapToDeepLLanguage(it) }
        val mappedTargetBase = mappedTarget.substringBefore("-")
        if (mappedSource != null && mappedSource.equals(mappedTargetBase, ignoreCase = true)) {
            return@withContext Result.success(text)
        }

        try {
            val authHeader = "DeepL-Auth-Key $apiKey"
            val request = DeepLRequest(
                text = listOf(inputText),
                targetLang = mappedTarget,
                sourceLang = mappedSource
            )

            // DeepL Free API keys end with ':fx', while DeepL Pro API keys route to api.deepl.com
            val targetApi = if (apiKey.endsWith(":fx")) deepLApi else NetworkClient.deepLProApi
            val response = targetApi.translate(authHeader, request)

            if (response.isSuccessful) {
                val translations = response.body()?.translations
                val resultText = translations?.firstOrNull()?.text
                if (!resultText.isNullOrEmpty()) {
                    Result.success(resultText)
                } else {
                    Result.failure(Exception("DeepL returned an empty response."))
                }
            } else {
                val code = response.code()
                val errorMsg = when (code) {
                    403 -> "DeepL API key is invalid or unauthorized (HTTP 403)."
                    456 -> "DeepL Free translation quota limit reached for this month (HTTP 456)."
                    429 -> "DeepL rate limit exceeded (HTTP 429). Please slow down."
                    else -> "DeepL translation failed with HTTP $code: ${response.message()}"
                }
                Log.w("DeepLTranslationEngine", errorMsg)
                Result.failure(Exception(errorMsg))
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e("DeepLTranslationEngine", "Network call failed: ${e.message}", e)
            Result.failure(Exception("DeepL network connection error: ${e.localizedMessage ?: e.message}"))
        }
    }

    private fun mapToDeepLTargetLanguage(code: String): String? {
        val norm = code.lowercase().trim()
        if (norm in listOf("en", "eng", "en-us")) return "EN-US"
        if (norm == "en-gb") return "EN-GB"
        if (norm in listOf("pt", "por", "pt-br")) return "PT-BR"
        if (norm == "pt-pt") return "PT-PT"
        return mapToDeepLLanguage(code)
    }

    private fun mapToDeepLLanguage(code: String): String? {
        return when (code.lowercase().trim()) {
            "ru", "rus" -> "RU"
            "en", "eng" -> "EN"
            "es", "spa" -> "ES"
            "de", "deu", "ger" -> "DE"
            "fr", "fra", "fre" -> "FR"
            "tr", "tur" -> "TR"
            "zh", "chs", "cht", "zh-cn", "zh-tw" -> "ZH"
            "ja", "jpn" -> "JA"
            "ko", "kor" -> "KO"
            "ar", "ara" -> "AR"
            "uz", "uzb" -> null // Uzbek is not supported on DeepL Free; triggers auto-fallback
            else -> if (code.length == 2) code.uppercase() else null
        }
    }
}
