package com.screentranslator.app.ocr

import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions
import com.google.mlkit.vision.text.japanese.JapaneseTextRecognizerOptions
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.screentranslator.app.data.model.MangaSourceLanguage
import com.screentranslator.app.data.model.OcrResultBlock
import com.screentranslator.app.data.network.NetworkClient
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.tasks.await
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream
import java.util.UUID

/**
 * High-performance Optical Character Recognition (OCR) engine.
 * Primary: Google ML Kit (on-device, offline, zero latency ~50ms, full resolution, vertical CJK & Latin).
 * Fallback: OCR.space Cloud API (for Cyrillic/Russian or when on-device returns empty).
 */
class OcrProcessor {

    companion object {
        private const val TAG = "OcrProcessor"
        private const val MAX_FILE_SIZE_BYTES = 950 * 1024 // OCR.space limit ~1MB
    }

    // Lazy cached ML Kit Recognizers for maximum throughput and zero reinitialization overhead
    private var latinRecognizer: TextRecognizer? = null
    private var japaneseRecognizer: TextRecognizer? = null
    private var chineseRecognizer: TextRecognizer? = null
    private var koreanRecognizer: TextRecognizer? = null

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

    data class ProcessedImage(
        val bytes: ByteArray,
        val scaleFactorX: Float,
        val scaleFactorY: Float
    )

    private data class ScaledBitmapInfo(
        val bitmap: Bitmap,
        val scaleFactorX: Float,
        val scaleFactorY: Float,
        val isCloned: Boolean
    )

    private fun getOptimizedOcrBitmap(original: Bitmap): ScaledBitmapInfo {
        val maxDim = maxOf(original.width, original.height)
        val targetMax = 1600 // Optimal resolution for fast ML Kit on-device inference without losing small characters on tall 20:9 screens
        if (maxDim <= targetMax) {
            return ScaledBitmapInfo(original, 1.0f, 1.0f, false)
        }
        val ratio = targetMax.toFloat() / maxDim.toFloat()
        val targetW = (original.width * ratio).toInt().coerceAtLeast(960.coerceAtMost(original.width))
        val targetH = (original.height * ratio).toInt().coerceAtLeast(100)
        val scaled = Bitmap.createScaledBitmap(original, targetW, targetH, true)
        val factorX = original.width.toFloat() / scaled.width.toFloat()
        val factorY = original.height.toFloat() / scaled.height.toFloat()
        return ScaledBitmapInfo(scaled, factorX, factorY, true)
    }

    /**
     * Processes a screen capture bitmap:
     * 1. Attempts fast on-device Google ML Kit text recognition with exact pixel bounds.
     * 2. Uses Latin recognizer by default in Standard Mode (30-50ms) and CJK models in Manga Mode.
     * 3. Supports vertical text orientation (Tategaki) natively.
     * 4. Groups paragraphs in Standard mode to prevent 40+ network translation requests.
     * 5. Falls back to OCR.space API if on-device yields no blocks or for Cyrillic.
     */
    suspend fun processImage(
        bitmap: Bitmap,
        apiKey: String = "",
        ocrLanguage: String? = null,
        isMangaMode: Boolean = false
    ): List<OcrResultBlock> = withContext(Dispatchers.Default) {
        if (bitmap.isRecycled || bitmap.width <= 0 || bitmap.height <= 0) {
            Log.e(TAG, "Yaroqsiz bitmap berildi.")
            return@withContext emptyList()
        }

        val normalizedLang = MangaSourceLanguage.normalizeToIso(ocrLanguage)

        // ─── 1. Google ML Kit On-Device Recognition ───────────────────────────
        // For Russian / Cyrillic, ML Kit bundled does not support Cyrillic on-device, so skip directly to OCR.space
        val isRussianExplicit = normalizedLang == "ru" || ocrLanguage.equals("rus", ignoreCase = true)

        if (!isRussianExplicit) {
            try {
                val mlBlocks = runMlKitRecognition(bitmap, normalizedLang, isMangaMode)
                if (mlBlocks.isNotEmpty()) {
                    Log.d(TAG, "ML Kit orqali ${mlBlocks.size} ta matn bloki aniqlandi (Oflayn).")
                    return@withContext mlBlocks
                }
            } catch (e: Exception) {
                if (e is CancellationException) throw e
                Log.w(TAG, "ML Kit OCR xatosi, OCR.space zaxirasiga o'tiladi: ${e.message}")
            }
        }

        // ─── 2. OCR.space Cloud Fallback ──────────────────────────────────────
        try {
            val fallbackBlocks = runOcrSpace(bitmap, apiKey, ocrLanguage)
            if (fallbackBlocks.isNotEmpty()) {
                Log.d(TAG, "OCR.space orqali ${fallbackBlocks.size} ta matn bloki aniqlandi.")
                return@withContext fallbackBlocks
            }
        } catch (e: Exception) {
            if (e is CancellationException) throw e
            Log.e(TAG, "OCR.space zaxirasi ham xato berdi: ${e.message}", e)
        }

        emptyList()
    }

    /**
     * Runs on-device ML Kit recognition based on requested language and mode.
     * Downscales the image for 4x-5x faster inference, then scales coordinates back.
     */
    private suspend fun runMlKitRecognition(
        bitmap: Bitmap,
        lang: String?,
        isMangaMode: Boolean
    ): List<OcrResultBlock> {
        val scaledInfo = getOptimizedOcrBitmap(bitmap)
        val inputImage = InputImage.fromBitmap(scaledInfo.bitmap, 0)
        val scaleX = scaledInfo.scaleFactorX
        val scaleY = scaledInfo.scaleFactorY

        try {
            if (isMangaMode && (lang.isNullOrBlank() || lang == "auto")) {
                val candidateOrder = listOf("ja", "en", "ko", "zh")
                var bestBlocks = emptyList<OcrResultBlock>()
                var bestScore = -1.0
                for (candidateLang in candidateOrder) {
                    val recognizer = when (candidateLang) {
                        "ja" -> getJapaneseRecognizer()
                        "en" -> getLatinRecognizer()
                        "ko" -> getKoreanRecognizer()
                        "zh" -> getChineseRecognizer()
                        else -> getJapaneseRecognizer()
                    }
                    val result = recognizer.process(inputImage).await()
                    val candidateBlocks = extractBlocks(
                        visionText = result,
                        origWidth = bitmap.width,
                        origHeight = bitmap.height,
                        scaleX = scaleX,
                        scaleY = scaleY,
                        lang = candidateLang,
                        isMangaMode = true
                    )
                    val score = scoreAutoLanguage(candidateBlocks, candidateLang)
                    if (score > bestScore) {
                        bestScore = score
                        bestBlocks = candidateBlocks
                    }
                    if (score > 65.0) {
                        return bestBlocks
                    }
                }
                return bestBlocks
            }

            // Pick primary recognizer:
            // Standard mode: ALWAYS start with Latin recognizer (super fast: 30-50ms)
            // Manga mode: Use language-specific (ja, zh, ko) or Japanese for auto
            val primaryRecognizer = if (isMangaMode) {
                when (lang) {
                    "ja" -> getJapaneseRecognizer()
                    "zh" -> getChineseRecognizer()
                    "ko" -> getKoreanRecognizer()
                    "en", "uz" -> getLatinRecognizer()
                    else -> getJapaneseRecognizer() // Japanese is default in manga
                }
            } else {
                when (lang) {
                    "ja" -> getJapaneseRecognizer()
                    "zh" -> getChineseRecognizer()
                    "ko" -> getKoreanRecognizer()
                    else -> getLatinRecognizer() // Standard mode default: Latin recognizer (30-50ms)
                }
            }

            val visionText = primaryRecognizer.process(inputImage).await()
            var blocks = extractBlocks(
                visionText = visionText,
                origWidth = bitmap.width,
                origHeight = bitmap.height,
                scaleX = scaleX,
                scaleY = scaleY,
                lang = lang,
                isMangaMode = isMangaMode
            )

            return blocks
        } finally {
            if (scaledInfo.isCloned && !scaledInfo.bitmap.isRecycled) {
                scaledInfo.bitmap.recycle()
            }
        }
    }

    private fun scoreAutoLanguage(blocks: List<OcrResultBlock>, language: String): Double {
        val letters = blocks.joinToString(" ") { it.originalText }.filter { it.isLetter() }
        if (letters.isEmpty()) return 0.0
        val latin = letters.count { it in 'a'..'z' || it in 'A'..'Z' }
        val kana = letters.count { it in '\u3040'..'\u30ff' }
        val han = letters.count { it in '\u3400'..'\u9fff' }
        val hangul = letters.count { it in '\uac00'..'\ud7af' || it in '\u1100'..'\u11ff' }
        val expected = when (language) {
            "ja" -> kana * 3 + han / 3
            "zh" -> if (kana == 0) han else 0
            "ko" -> hangul
            else -> latin
        }
        val purity = minOf(1.0, expected.toDouble() / letters.length)
        return purity * 100.0 + minOf(expected, 100) * 0.01 + minOf(letters.length, 100) * 0.001
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
            // In Manga mode, extract lines individually so MangaBubbleDetector can cluster
            // vertical Tategaki columns (right-to-left) or horizontal lines into speech balloons
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
            // In Standard mode, keep cohesive paragraphs (TextBlocks).
            // This groups related lines into single coherent blocks, which:
            // 1. Drastically reduces network translation requests from 40+ to 3-6
            // 2. Improves translation grammar and sentence flow significantly
            // 3. Renders clean, readable overlay cards
            for (block in visionText.textBlocks) {
                val box = block.boundingBox ?: continue
                val validBox = Rect(
                    (box.left * scaleX).toInt().coerceIn(0, origWidth),
                    (box.top * scaleY).toInt().coerceIn(0, origHeight),
                    (box.right * scaleX).toInt().coerceIn(0, origWidth),
                    (box.bottom * scaleY).toInt().coerceIn(0, origHeight)
                )

                if (validBox.width() <= 2 || validBox.height() <= 2) continue

                // Join lines cleanly
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
                        sourceLanguage = lang ?: "en",
                        isVertical = isVertical
                    )
                )
            }
        }

        return blocks
    }

    /**
     * Fallback to OCR.space API.
     */
    private suspend fun runOcrSpace(
        bitmap: Bitmap,
        apiKey: String,
        ocrLanguage: String?
    ): List<OcrResultBlock> = withContext(Dispatchers.IO) {
        val validKey = apiKey.trim()
        if (validKey.isBlank()) {
            Log.w(TAG, "OCR.space zaxirasi uchun shaxsiy API kaliti sozlanmagan.")
            return@withContext emptyList()
        }
        val processedImage = prepareJpeg(bitmap)
        if (processedImage.bytes.isEmpty() || processedImage.bytes.size > MAX_FILE_SIZE_BYTES) {
            Log.w(TAG, "OCR.space uchun JPEG faylini ${MAX_FILE_SIZE_BYTES / 1024} KB dan kichik qilish imkoni bo'lmadi.")
            return@withContext emptyList()
        }

        val requestFile = processedImage.bytes.toRequestBody("image/jpeg".toMediaTypeOrNull())
        val filePart = MultipartBody.Part.createFormData("file", "screenshot.jpg", requestFile)

        val isOverlay = "true".toRequestBody("text/plain".toMediaTypeOrNull())
        val scale = "true".toRequestBody("text/plain".toMediaTypeOrNull())

        // For Russian/Cyrillic, use engine 1 which has mature Cyrillic support
        val isRussian = ocrLanguage?.lowercase() in listOf("ru", "rus")
        val engineStr = if (isRussian) "1" else "2"
        val engine = engineStr.toRequestBody("text/plain".toMediaTypeOrNull())

        val effectiveLang = when {
            isRussian -> "rus"
            ocrLanguage.isNullOrBlank() || ocrLanguage.equals("auto", ignoreCase = true) -> "auto"
            else -> ocrLanguage.lowercase().trim()
        }
        val langBody = effectiveLang.toRequestBody("text/plain".toMediaTypeOrNull())

        val response = NetworkClient.ocrSpaceApi.parseImage(
            apiKey = validKey,
            file = filePart,
            isOverlayRequired = isOverlay,
            scale = scale,
            language = langBody,
            ocrEngine = engine
        )

        if (!response.isSuccessful) {
            Log.e(TAG, "OCR.space HTTP xatosi: ${response.code()} ${response.message()}")
            return@withContext emptyList()
        }

        val body = response.body()
        if (body == null || body.isErrored == true) {
            val err = body?.errorMessage?.joinToString(", ") ?: body?.errorDetails ?: "Noma'lum xato"
            Log.e(TAG, "OCR.space xatosi: $err")
            return@withContext emptyList()
        }

        val blocks = mutableListOf<OcrResultBlock>()
        val parsedResults = body.parsedResults ?: emptyList()
        val scaleX = processedImage.scaleFactorX
        val scaleY = processedImage.scaleFactorY

        for (result in parsedResults) {
            val overlay = result.textOverlay
            val lines = overlay?.lines

            if (!lines.isNullOrEmpty()) {
                for (line in lines) {
                    val words = line.words ?: emptyList()
                    val lineText = (line.lineText ?: words.joinToString(" ") { it.wordText ?: "" }).trim()
                    if (lineText.isBlank()) continue

                    val box = if (words.isNotEmpty()) {
                        val minLeft = (words.minOf { it.left } * scaleX).toInt().coerceIn(0, bitmap.width)
                        val minTop = (words.minOf { it.top } * scaleY).toInt().coerceIn(0, bitmap.height)
                        val maxRight = (words.maxOf { it.left + it.width } * scaleX).toInt().coerceIn(minLeft, bitmap.width)
                        val maxBottom = (words.maxOf { it.top + it.height } * scaleY).toInt().coerceIn(minTop, bitmap.height)
                        Rect(minLeft, minTop, maxRight, maxBottom)
                    } else {
                        val top = ((line.minTop ?: 0f) * scaleY).toInt().coerceIn(0, bitmap.height)
                        val height = ((line.maxHeight ?: 30f) * scaleY).toInt().coerceAtLeast(20)
                        val bottom = (top + height).coerceIn(top, bitmap.height)
                        Rect(0, top, bitmap.width, bottom)
                    }

                    if (box.width() > 2 && box.height() > 2) {
                        val isVert = box.height() > box.width() * 1.25f
                        blocks.add(
                            OcrResultBlock(
                                id = UUID.randomUUID().toString(),
                                originalText = lineText,
                                boundingBox = box,
                                sourceLanguage = if (isRussian) "ru" else null,
                                isVertical = isVert
                            )
                        )
                    }
                }
            } else if (!result.parsedText.isNullOrBlank()) {
                val text = result.parsedText.trim()
                blocks.add(
                    OcrResultBlock(
                        id = UUID.randomUUID().toString(),
                        originalText = text,
                        boundingBox = Rect(20, 100, bitmap.width - 20, 250),
                        sourceLanguage = if (isRussian) "ru" else null
                    )
                )
            }
        }

        blocks
    }

    private fun prepareJpeg(bitmap: Bitmap): ProcessedImage {
        val maxDim = maxOf(bitmap.width, bitmap.height)
        var targetMax = minOf(maxDim, 1200)
        var lastTargetMax = -1
        var currentScaledBitmap: Bitmap? = null
        var quality = 75
        var bytes: ByteArray
        var scaledWidth = bitmap.width
        var scaledHeight = bitmap.height

        try {
            while (true) {
                if (targetMax != lastTargetMax) {
                    if (currentScaledBitmap != null && currentScaledBitmap !== bitmap) {
                        currentScaledBitmap.recycle()
                    }
                    val scale = targetMax.toFloat() / maxDim.toFloat()
                    val targetW = (bitmap.width * scale).toInt().coerceAtLeast(100)
                    val targetH = (bitmap.height * scale).toInt().coerceAtLeast(100)
                    currentScaledBitmap = if (scale < 0.99f) {
                        Bitmap.createScaledBitmap(bitmap, targetW, targetH, true)
                    } else {
                        bitmap
                    }
                    lastTargetMax = targetMax
                    scaledWidth = currentScaledBitmap.width
                    scaledHeight = currentScaledBitmap.height
                }

                val activeBitmap = currentScaledBitmap ?: bitmap
                val stream = ByteArrayOutputStream()
                activeBitmap.compress(Bitmap.CompressFormat.JPEG, quality, stream)
                bytes = stream.toByteArray()

                if (bytes.size <= MAX_FILE_SIZE_BYTES || (targetMax <= 480 && quality <= 45)) break
                if (quality > 45) {
                    quality -= 10
                } else {
                    targetMax = (targetMax * 0.85f).toInt().coerceAtLeast(480)
                    quality = 75
                }
            }
        } finally {
            if (currentScaledBitmap != null && currentScaledBitmap !== bitmap && !currentScaledBitmap.isRecycled) {
                currentScaledBitmap.recycle()
            }
        }

        val factorX = bitmap.width.toFloat() / scaledWidth.toFloat()
        val factorY = bitmap.height.toFloat() / scaledHeight.toFloat()
        return ProcessedImage(bytes, factorX, factorY)
    }

    @Synchronized
    fun release() {
        try {
            latinRecognizer?.close()
            japaneseRecognizer?.close()
            chineseRecognizer?.close()
            koreanRecognizer?.close()
        } catch (ignored: Exception) {}
        latinRecognizer = null
        japaneseRecognizer = null
        chineseRecognizer = null
        koreanRecognizer = null
    }
}
