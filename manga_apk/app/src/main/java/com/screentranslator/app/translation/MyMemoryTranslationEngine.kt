package com.screentranslator.app.translation

import android.util.Log
import com.screentranslator.app.data.model.TranslationEngineType
import com.screentranslator.app.data.network.MyMemoryApi
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.withContext

/**
 * Online translation engine using MyMemory Translation Memory API.
 * Free public API without registration.
 */
class MyMemoryTranslationEngine(
    private val myMemoryApi: MyMemoryApi
) : TranslationEngine {

    override val engineType: TranslationEngineType = TranslationEngineType.MY_MEMORY

    companion object {
        // ✅ TUZATILDI: MyMemory GET so'rovi uchun URL uzunlik chegarasi oldini olish
        private const val MAX_INPUT_LENGTH = 500
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

        val normalizedSrc = com.screentranslator.app.data.model.MangaSourceLanguage.normalizeToIso(sourceLang)
            ?: com.screentranslator.app.ocr.MangaBubbleDetector.detectLanguageFromText(text)
            ?: "en"
        val src = normalizedSrc.lowercase()
        val tgt = targetLang.lowercase()

        if (src == tgt) {
            return@withContext Result.success(text)
        }

        val langPair = "$src|$tgt"

        try {
            val response = myMemoryApi.translate(
                query = inputText,
                langPair = langPair
            )

            if (response.isSuccessful) {
                val body = response.body()
                val translated = body?.responseData?.translatedText
                if (body != null && body.responseStatus == 200 && !translated.isNullOrEmpty()) {
                    Result.success(translated)
                } else {
                    val details = body?.responseDetails ?: "Quota or translation error"
                    Result.failure(Exception("MyMemory error: $details"))
                }
            } else {
                val code = response.code()
                val msg = if (code == 429) "MyMemory daily request limit reached." else "MyMemory HTTP $code: ${response.message()}"
                Log.w("MyMemoryEngine", msg)
                Result.failure(Exception(msg))
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e("MyMemoryEngine", "Network call failed: ${e.message}", e)
            Result.failure(Exception("MyMemory network error: ${e.localizedMessage ?: e.message}"))
        }
    }
}
