package com.screentranslator.app.ocr

import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions
import com.google.mlkit.vision.text.japanese.JapaneseTextRecognizerOptions
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.screentranslator.app.data.model.MangaSourceLanguage
import com.screentranslator.app.data.model.OcrResultBlock
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.tasks.await
import kotlinx.coroutines.withContext
import java.util.UUID

/**
 * High-performance On-Device Optical Character Recognition (OCR) engine powered by Google ML Kit.
 * Supports offline zero-latency recognition across Latin, Japanese (vertical Tategaki & horizontal),
 * Chinese, and Korean scripts. Russian/Cyrillic requires server-side OCR when enabled.
 */
class OcrProcessor {

    companion object {
        private const val TAG = "OcrProcessor"
        private const val TARGET_MAX_DIMENSION = 2400
        private const val MIN_SHORT_SIDE_PX = 1080
        private const val MAX_SCALED_DIMENSION = 4096

        internal fun computeScaledDimensions(width: Int, height: Int): Pair<Int, Int> {
            val maxDim = maxOf(width, height)
            val minDim = minOf(width, height)
            if (maxDim <= TARGET_MAX_DIMENSION) {
                return width to height
            }
            var ratio = TARGET_MAX_DIMENSION.toFloat() / maxDim.toFloat()
            if (minDim >= MIN_SHORT_SIDE_PX && minDim * ratio < MIN_SHORT_SIDE_PX) {
                ratio = (MIN_SHORT_SIDE_PX.toFloat() / minDim.toFloat()).coerceAtMost(1.0f)
            }
            if (maxDim * ratio > MAX_SCALED_DIMENSION) {
                ratio = MAX_SCALED_DIMENSION.toFloat() / maxDim.toFloat()
            }
            if (ratio >= 0.99f) {
                return width to height
            }
            val targetW = (width * ratio).toInt().coerceAtLeast(1)
            val targetH = (height * ratio).toInt().coerceAtLeast(1)
            return targetW to targetH
        }

        internal fun scoreAutoLanguageText(text: String, language: String): Double {
            val letters = text.filter { it.isLetter() }
            if (letters.isEmpty()) return 0.0
            val latin = letters.count { it in 'a'..'z' || it in 'A'..'Z' || it in '\u00C0'..'\u024F' }
            // Exclude '\u30FC' ('ー') from syllabic kana so a stray prolonged mark never disqualifies "zh".
            val kana = letters.count { it in '\u3041'..'\u3096' || it in '\u30A1'..'\u30FA' || it in '\uFF66'..'\uFF9D' }
            val han = letters.count { it in '\u3400'..'\u4DBF' || it in '\u4E00'..'\u9FFF' }
            val hangul = letters.count { it in '\uAC00'..'\uD7AF' || it in '\u1100'..'\u11FF' }
            val expected: Double = when (language) {
                "ja" -> if (kana > 0) (kana * 2 + han).toDouble() else han * 0.4
                "zh" -> if (kana == 0) han.toDouble() else 0.0
                "ko" -> hangul.toDouble()
                else -> latin.toDouble()
            }
            if (expected <= 0.0) return 0.0
            val purity = minOf(1.0, expected / letters.length)
            return purity * 100.0 + minOf(expected, 100.0) * 0.01 + minOf(letters.length, 100) * 0.001
        }

        internal fun shouldEarlyBreakAutoLanguage(text: String, language: String, score: Double): Boolean {
            if (score <= 75.0) return false
            val letters = text.filter { it.isLetter() }
            if (letters.isEmpty()) return false
            val kana = letters.count { it in '\u3041'..'\u3096' || it in '\u30A1'..'\u30FA' || it in '\uFF66'..'\uFF9D' }
            val han = letters.count { it in '\u3400'..'\u4DBF' || it in '\u4E00'..'\u9FFF' }
            val hangul = letters.count { it in '\uAC00'..'\uD7AF' || it in '\u1100'..'\u11FF' }
            val latin = letters.count { it in 'a'..'z' || it in 'A'..'Z' || it in '\u00C0'..'\u024F' }
            return when (language) {
                "ja" -> kana > 0 && (kana + han) > latin
                "ko" -> hangul > 0 && hangul >= latin
                "zh" -> kana == 0 && han > 0 && han >= latin
                else -> latin > 0 && kana == 0 && han == 0 && hangul == 0
            }
        }
    }

    // Lazy cached ML Kit Recognizers for maximum throughput and zero reinitialization overhead
    private var latinRecognizer: TextRecognizer? = null
    private var japaneseRecognizer: TextRecognizer? = null
    private var chineseRecognizer: TextRecognizer? = null
    private var koreanRecognizer: TextRecognizer? = null
    private val ocrMutex = Mutex()

    @Volatile
    private var released = false

    @Synchronized
    private fun getLatinRecognizer(): TextRecognizer {
        return latinRecognizer ?: TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS).also {
            latinRecognizer = it
        }
    }

    @Synchronized
    private fun getJapaneseRecognizer(): TextRecognizer {
        return japaneseRecognizer ?: TextRecognition.getClient(JapaneseTextRecognizerOptions.Builder().build()).also {
            japaneseRecognizer = it
        }
    }

    @Synchronized
    private fun getChineseRecognizer(): TextRecognizer {
        return chineseRecognizer ?: TextRecognition.getClient(ChineseTextRecognizerOptions.Builder().build()).also {
            chineseRecognizer = it
        }
    }

    @Synchronized
    private fun getKoreanRecognizer(): TextRecognizer {
        return koreanRecognizer ?: TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build()).also {
            koreanRecognizer = it
        }
    }

    private data class ScaledBitmapInfo(
        val bitmap: Bitmap,
        val scaleFactorX: Float,
        val scaleFactorY: Float,
        val isCloned: Boolean
    )

    private fun getOptimizedOcrBitmap(original: Bitmap): ScaledBitmapInfo {
        val (targetW, targetH) = computeScaledDimensions(original.width, original.height)
        if (targetW == original.width && targetH == original.height) {
            return ScaledBitmapInfo(original, 1.0f, 1.0f, false)
        }
        val scaled = Bitmap.createScaledBitmap(original, targetW, targetH, true)
        val factorX = original.width.toFloat() / scaled.width.toFloat()
        val factorY = original.height.toFloat() / scaled.height.toFloat()
        return ScaledBitmapInfo(scaled, factorX, factorY, true)
    }

    /**
     * Processes a screen capture bitmap using on-device Google ML Kit:
     * 1. Respects the configured source language in both Standard and Manga modes.
     * 2. In Manga auto mode, scores candidate CJK/Latin recognizers without premature Japanese bias.
     * 3. Preserves high resolution (maxDim ~2400 / short side >= 1080px) so small text is retained.
     * 4. Returns empty list gracefully for Russian/Cyrillic (unsupported offline by bundled ML Kit).
     */
    suspend fun processImage(
        bitmap: Bitmap,
        ocrLanguage: String? = null,
        isMangaMode: Boolean = false
    ): List<OcrResultBlock> = withContext(Dispatchers.Default) {
        if (released || bitmap.isRecycled || bitmap.width <= 0 || bitmap.height <= 0) {
            if (!released) Log.e(TAG, "Yaroqsiz bitmap berildi.")
            return@withContext emptyList()
        }

        val normalizedLang = MangaSourceLanguage.normalizeToIso(ocrLanguage)

        // ML Kit bundled text recognition does not support Cyrillic/Russian on-device.
        val isRussianExplicit = normalizedLang == "ru" || ocrLanguage.equals("rus", ignoreCase = true)
        if (isRussianExplicit) {
            Log.d(TAG, "Ruscha (Kirill) matn oflayn ML Kit tomonidan qo'llab-quvvatlanmaydi; server OCR talab qilinadi.")
            return@withContext emptyList()
        }

        return@withContext try {
            val mlBlocks = runMlKitRecognition(bitmap, normalizedLang, isMangaMode)
            if (mlBlocks.isNotEmpty()) {
                Log.d(TAG, "ML Kit orqali ${mlBlocks.size} ta matn bloki aniqlandi (Oflayn).")
            }
            mlBlocks
        } catch (e: Exception) {
            if (e is CancellationException) throw e
            Log.w(TAG, "ML Kit OCR xatosi: ${e.message}")
            emptyList()
        }
    }

    /**
     * Runs on-device ML Kit recognition based on requested language and mode.
     */
    private suspend fun runMlKitRecognition(
        bitmap: Bitmap,
        lang: String?,
        isMangaMode: Boolean
    ): List<OcrResultBlock> = ocrMutex.withLock {
        if (released) return emptyList()
        val scaledInfo = getOptimizedOcrBitmap(bitmap)
        val scaleX = scaledInfo.scaleFactorX
        val scaleY = scaledInfo.scaleFactorY
        var activeTask: Task<Text>? = null

        try {
            val inputImage = InputImage.fromBitmap(scaledInfo.bitmap, 0)
            if (isMangaMode && (lang.isNullOrBlank() || lang == "auto")) {
                val candidateOrder = listOf("ja", "ko", "zh", "en")
                var bestBlocks = emptyList<OcrResultBlock>()
                var bestScore = -1.0
                var bestCandidate: String? = null
                for (candidateLang in candidateOrder) {
                    if (released) return emptyList()
                    val recognizer = when (candidateLang) {
                        "ja" -> getJapaneseRecognizer()
                        "en" -> getLatinRecognizer()
                        "ko" -> getKoreanRecognizer()
                        "zh" -> getChineseRecognizer()
                        else -> getJapaneseRecognizer()
                    }
                    val task = recognizer.process(inputImage)
                    activeTask = task
                    val result = task.await()
                    val candidateBlocks = extractBlocks(
                        visionText = result,
                        origWidth = bitmap.width,
                        origHeight = bitmap.height,
                        scaleX = scaleX,
                        scaleY = scaleY,
                        lang = candidateLang,
                        isMangaMode = true
                    )
                    val combinedText = candidateBlocks.joinToString(" ") { it.originalText }
                    val score = scoreAutoLanguageText(combinedText, candidateLang)
                    if (score > bestScore) {
                        bestScore = score
                        bestBlocks = candidateBlocks
                        bestCandidate = candidateLang
                    }
                    if (score >= bestScore && shouldEarlyBreakAutoLanguage(combinedText, candidateLang, score)) {
                        bestCandidate = candidateLang
                        break
                    }
                }
                evictUnusedCjkRecognizers(bestCandidate)
                return bestBlocks
            }

            val primaryRecognizer = if (isMangaMode) {
                when (lang) {
                    "ja" -> getJapaneseRecognizer()
                    "zh" -> getChineseRecognizer()
                    "ko" -> getKoreanRecognizer()
                    "en", "uz" -> getLatinRecognizer()
                    else -> getJapaneseRecognizer()
                }
            } else {
                when (lang) {
                    "ja" -> getJapaneseRecognizer()
                    "zh" -> getChineseRecognizer()
                    "ko" -> getKoreanRecognizer()
                    else -> getLatinRecognizer()
                }
            }

            val task = primaryRecognizer.process(inputImage)
            activeTask = task
            val visionText = task.await()
            return extractBlocks(
                visionText = visionText,
                origWidth = bitmap.width,
                origHeight = bitmap.height,
                scaleX = scaleX,
                scaleY = scaleY,
                lang = lang,
                isMangaMode = isMangaMode
            )
        } finally {
            val inFlight = activeTask
            if (inFlight != null && !inFlight.isComplete) {
                inFlight.addOnCompleteListener {
                    if (scaledInfo.isCloned && !scaledInfo.bitmap.isRecycled) {
                        scaledInfo.bitmap.recycle()
                    }
                    if (released) {
                        closeAllRecognizers()
                    }
                }
            } else {
                if (scaledInfo.isCloned && !scaledInfo.bitmap.isRecycled) {
                    scaledInfo.bitmap.recycle()
                }
                if (released) {
                    closeAllRecognizers()
                }
            }
        }
    }

    private fun evictUnusedCjkRecognizers(keepLang: String?) {
        if (keepLang != "ja") {
            try { japaneseRecognizer?.close() } catch (e: Exception) { Log.w(TAG, "Error closing japanese recognizer: ${e.message}") }
            japaneseRecognizer = null
        }
        if (keepLang != "zh") {
            try { chineseRecognizer?.close() } catch (e: Exception) { Log.w(TAG, "Error closing chinese recognizer: ${e.message}") }
            chineseRecognizer = null
        }
        if (keepLang != "ko") {
            try { koreanRecognizer?.close() } catch (e: Exception) { Log.w(TAG, "Error closing korean recognizer: ${e.message}") }
            koreanRecognizer = null
        }
    }

    private fun extractBlocks(
        visionText: com.google.mlkit.vision.text.Text,
        origWidth: Int,
        origHeight: Int,
        scaleX: Float,
        scaleY: Float,
        lang: String?,
        isMangaMode: Boolean
    ): List<OcrResultBlock> {
        val blocks = mutableListOf<OcrResultBlock>()

        if (isMangaMode) {
            for (block in visionText.textBlocks) {
                for (line in block.lines) {
                    val lineText = line.text.trim()
                    if (lineText.isBlank()) continue

                    val box = line.boundingBox ?: continue
                    val validBox = Rect(
                        (box.left * scaleX).toInt().coerceIn(0, origWidth),
                        (box.top * scaleY).toInt().coerceIn(0, origHeight),
                        (box.right * scaleX).toInt().coerceIn(0, origWidth),
                        (box.bottom * scaleY).toInt().coerceIn(0, origHeight)
                    )

                    if (validBox.width() <= 2 || validBox.height() <= 2) continue
                    val isVertical = validBox.height() > validBox.width() * 1.25f

                    blocks.add(
                        OcrResultBlock(
                            id = UUID.randomUUID().toString(),
                            originalText = lineText,
                            boundingBox = validBox,
                            sourceLanguage = lang,
                            isVertical = isVertical
                        )
                    )
                }
            }
        } else {
            for (block in visionText.textBlocks) {
                val box = block.boundingBox ?: continue
                val validBox = Rect(
                    (box.left * scaleX).toInt().coerceIn(0, origWidth),
                    (box.top * scaleY).toInt().coerceIn(0, origHeight),
                    (box.right * scaleX).toInt().coerceIn(0, origWidth),
                    (box.bottom * scaleY).toInt().coerceIn(0, origHeight)
                )

                if (validBox.width() <= 2 || validBox.height() <= 2) continue

                val combinedText = block.lines
                    .map { it.text.trim() }
                    .filter { it.isNotBlank() }
                    .joinToString(" ")

                if (combinedText.isBlank()) continue

                val isVertical = validBox.height() > validBox.width() * 1.3f
                blocks.add(
                    OcrResultBlock(
                        id = UUID.randomUUID().toString(),
                        originalText = combinedText,
                        boundingBox = validBox,
                        sourceLanguage = lang,
                        isVertical = isVertical
                    )
                )
            }
        }

        return blocks
    }

    @Synchronized
    fun release() {
        released = true
        if (ocrMutex.tryLock()) {
            try {
                closeAllRecognizers()
            } finally {
                ocrMutex.unlock()
            }
        }
    }

    @Synchronized
    private fun closeAllRecognizers() {
        try { latinRecognizer?.close() } catch (e: Exception) { Log.w(TAG, "Error closing latin recognizer: ${e.message}") }
        try { japaneseRecognizer?.close() } catch (e: Exception) { Log.w(TAG, "Error closing japanese recognizer: ${e.message}") }
        try { chineseRecognizer?.close() } catch (e: Exception) { Log.w(TAG, "Error closing chinese recognizer: ${e.message}") }
        try { koreanRecognizer?.close() } catch (e: Exception) { Log.w(TAG, "Error closing korean recognizer: ${e.message}") }
        latinRecognizer = null
        japaneseRecognizer = null
        chineseRecognizer = null
        koreanRecognizer = null
    }
}
