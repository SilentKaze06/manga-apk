package com.screentranslator.app.ocr

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Rect
import com.screentranslator.app.data.model.MangaSourceLanguage
import com.screentranslator.app.data.model.OcrResultBlock
import java.util.UUID
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Intelligent line clustering and speech bubble detection for Manga, Manhwa, and Comics.
 * Groups vertically and horizontally proximate text lines inside speech balloons
 * into unified dialogue sentences, computes bubble bounds, supports vertical text (Tategaki),
 * accurately detects background colors, and normalizes ISO source language codes.
 */
object MangaBubbleDetector {

    private val JAPANESE_OR_CHINESE_REGEX = Regex("[\\p{IsHiragana}\\p{IsKatakana}\\p{IsHan}]")
    private val KOREAN_REGEX = Regex("[\\p{IsHangul}]")

    /**
     * Detects ISO 639-1 language code based on character scripts.
     */
    fun detectLanguageFromText(text: String, fallback: String? = null): String? {
        val clean = text.trim()
        if (clean.isEmpty()) return fallback

        // Japanese has unique kana scripts (Hiragana or Katakana)
        if (clean.any { it in '\u3040'..'\u309F' || it in '\u30A0'..'\u30FF' }) {
            return "ja"
        }
        // Korean has unique Hangul syllables and jamo
        if (clean.any { it in '\uAC00'..'\uD7AF' || it in '\u1100'..'\u11FF' || it in '\u3130'..'\u318F' }) {
            return "ko"
        }
        // Chinese characters (Hanzi / Kanji) without Japanese Kana
        if (clean.any { it in '\u4E00'..'\u9FFF' || it in '\u3400'..'\u4DBF' }) {
            return "zh"
        }
        // Cyrillic (Russian)
        if (clean.any { it in '\u0400'..'\u04FF' }) {
            return "ru"
        }
        // Latin / English
        if (clean.any { it in 'a'..'z' || it in 'A'..'Z' }) {
            return "en"
        }
        return fallback
    }

    /**
     * Clusters recognized OCR text lines into full manga speech bubbles.
     */
    fun clusterIntoBubbles(
        rawBlocks: List<OcrResultBlock>,
        bitmap: Bitmap?,
        sourceLangCode: String? = null
    ): List<OcrResultBlock> {
        val validBlocks = rawBlocks.filter { cleanOcrNoise(it.originalText).isNotBlank() }
        if (validBlocks.isEmpty()) return emptyList()

        val normalizedSourceLang = MangaSourceLanguage.normalizeToIso(sourceLangCode)

        val n = validBlocks.size
        if (n == 1) {
            val single = validBlocks[0]
            val cleanedText = cleanOcrNoise(single.originalText)
            val effectiveLang = normalizedSourceLang ?: detectLanguageFromText(cleanedText)
            val (bgColor, txtColor) = detectBubbleColors(single.boundingBox, bitmap)
            val isVert = single.isVertical || (single.boundingBox.height() > single.boundingBox.width() * 1.2f)
            val paddedBox = expandRect(single.boundingBox, bitmap, padX = if (isVert) 16 else 10, padY = 8)
            return listOf(
                single.copy(
                    originalText = cleanedText,
                    boundingBox = paddedBox,
                    sourceLanguage = effectiveLang,
                    isMangaBubble = true,
                    isVertical = isVert,
                    bubbleBgColor = bgColor,
                    bubbleTextColor = txtColor
                )
            )
        }

        // 1. Build adjacency list for line grouping (graph components)
        val adj = Array(n) { mutableListOf<Int>() }
        for (i in 0 until n) {
            val boxA = validBlocks[i].boundingBox
            for (j in i + 1 until n) {
                val boxB = validBlocks[j].boundingBox
                if (areLinesInSameBubble(boxA, boxB)) {
                    adj[i].add(j)
                    adj[j].add(i)
                }
            }
        }

        // 2. Find connected components (each represents a speech bubble)
        val visited = BooleanArray(n)
        val clusteredBlocks = mutableListOf<OcrResultBlock>()

        for (i in 0 until n) {
            if (visited[i]) continue

            val clusterIndices = mutableListOf<Int>()
            val queue = ArrayDeque<Int>()
            queue.add(i)
            visited[i] = true

            while (queue.isNotEmpty()) {
                val curr = queue.removeFirst()
                clusterIndices.add(curr)
                for (neighbor in adj[curr]) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true
                        queue.add(neighbor)
                    }
                }
            }

            val rawBubbleLines = clusterIndices.map { validBlocks[it] }

            val isVert = isVerticalBubble(rawBubbleLines)

            // 3. Sort lines in this bubble according to text orientation:
            // Vertical text (Japanese/Chinese Tategaki) is read RIGHT to LEFT!
            // Horizontal text (Yokogaki, Western, Korean) is read TOP to BOTTOM!
            val bubbleLines = if (isVert) {
                rawBubbleLines.sortedByDescending { it.boundingBox.centerX() }
            } else {
                rawBubbleLines.sortedBy { it.boundingBox.top }
            }

            // 4. Combine text into full dialogue sentence
            val mergedText = mergeDialogueText(bubbleLines)
            if (mergedText.isBlank()) continue

            // Determine effective ISO source language for this speech bubble
            val effectiveLang = normalizedSourceLang ?: detectLanguageFromText(mergedText)

            // 5. Compute merged bounding box for the whole bubble
            val minL = bubbleLines.minOf { it.boundingBox.left }
            val minT = bubbleLines.minOf { it.boundingBox.top }
            val maxR = bubbleLines.maxOf { it.boundingBox.right }
            val maxB = bubbleLines.maxOf { it.boundingBox.bottom }

            val rawBubbleBox = Rect(minL, minT, maxR, maxB)
            val paddedBox = expandRect(rawBubbleBox, bitmap, padX = if (isVert) 16 else 14, padY = 8)

            // 6. Detect background & text color from original screen capture
            val (bgColor, txtColor) = detectBubbleColors(rawBubbleBox, bitmap)

            clusteredBlocks.add(
                OcrResultBlock(
                    id = UUID.randomUUID().toString(),
                    originalText = mergedText,
                    boundingBox = paddedBox,
                    sourceLanguage = effectiveLang,
                    isMangaBubble = true,
                    isVertical = isVert,
                    bubbleBgColor = bgColor,
                    bubbleTextColor = txtColor
                )
            )
        }

        return clusteredBlocks
    }

    /**
     * Determines whether two lines belong to the same speech bubble.
     * Supports both horizontal lines (Yokogaki) and vertical text columns (Tategaki).
     */
    private fun areLinesInSameBubble(boxA: Rect, boxB: Rect): Boolean {
        val avgHeight = (boxA.height() + boxB.height()) / 2f
        val avgWidth = (boxA.width() + boxB.width()) / 2f

        // Case 1: Vertical text columns arranged side-by-side (Tategaki)
        val yOverlap = min(boxA.bottom, boxB.bottom) - max(boxA.top, boxB.top)
        val horizGap = max(0, max(boxA.left, boxB.left) - min(boxA.right, boxB.right))
        val isVerticalPair = (boxA.height() > boxA.width() * 1.15f) ||
                (boxB.height() > boxB.width() * 1.15f)

        if (isVerticalPair && yOverlap > avgHeight * 0.35f) {
            val maxAllowedHorizGap = max(avgWidth * 0.9f, 32f)
            if (horizGap <= maxAllowedHorizGap) {
                return true
            }
        }

        // Case 2: Horizontal text lines stacked vertically (Yokogaki)
        val vertGap = max(0, max(boxA.top, boxB.top) - min(boxA.bottom, boxB.bottom))
        val maxAllowedVertGap = max(avgHeight * 1.3f, 26f)
        if (vertGap > maxAllowedVertGap) {
            return false
        }

        val horizOverlap = min(boxA.right, boxB.right) - max(boxA.left, boxB.left)
        val centerDistX = abs(boxA.centerX() - boxB.centerX())
        val maxAllowedCenterDist = avgWidth * 0.75f + 35f

        return horizOverlap > -25 || centerDistX <= maxAllowedCenterDist
    }

    /**
     * Checks if lines in a bubble represent vertical text orientation.
     */
    private fun isVerticalBubble(lines: List<OcrResultBlock>): Boolean {
        if (lines.isEmpty()) return false
        val tallLineCount = lines.count { it.isVertical || (it.boundingBox.height() > it.boundingBox.width() * 1.15f) }
        if (tallLineCount * 2 >= lines.size) return true

        if (lines.size >= 2) {
            var sideBySidePairs = 0
            for (i in 0 until lines.size - 1) {
                val a = lines[i].boundingBox
                val b = lines[i + 1].boundingBox
                val yOverlap = min(a.bottom, b.bottom) - max(a.top, b.top)
                val minH = min(a.height(), b.height())
                if (minH > 0 && yOverlap.toFloat() / minH > 0.45f) {
                    sideBySidePairs++
                }
            }
            if (sideBySidePairs * 2 >= lines.size - 1) return true
        }
        return false
    }

    /**
     * Intelligently merges lines into a cohesive sentence:
     * - Japanese & Chinese: merges without spaces.
     * - Korean: merges with spaces between words.
     * - Western / Cyrillic: reconnects hyphenated words, merges with spaces.
     */
    private fun mergeDialogueText(lines: List<OcrResultBlock>): String {
        if (lines.isEmpty()) return ""
        if (lines.size == 1) return cleanOcrNoise(lines[0].originalText)

        val sb = StringBuilder()
        for (i in lines.indices) {
            val raw = cleanOcrNoise(lines[i].originalText)
            if (raw.isBlank()) continue

            if (sb.isEmpty()) {
                sb.append(raw)
            } else {
                val isCJK = JAPANESE_OR_CHINESE_REGEX.containsMatchIn(raw) || JAPANESE_OR_CHINESE_REGEX.containsMatchIn(sb.toString())
                val isKorean = KOREAN_REGEX.containsMatchIn(raw) || KOREAN_REGEX.containsMatchIn(sb.toString())

                if (isCJK && !isKorean) {
                    // Japanese & Chinese: no spaces between lines
                    sb.append(raw)
                } else {
                    // Korean, English, Russian, Latin: space between words / lines
                    if (sb.endsWith("-")) {
                        // De-hyphenate broken words: "trans-" + "lation" -> "translation"
                        sb.deleteCharAt(sb.length - 1)
                        sb.append(raw)
                    } else {
                        sb.append(" ").append(raw)
                    }
                }
            }
        }
        return sb.toString().trim()
    }

    /**
     * Removes common stray OCR artifacts and noise from speech bubble dialogue.
     */
    private fun cleanOcrNoise(text: String): String {
        var s = text.trim()
        // Strip common edge noise like stray vertical bars or lone quotes while preserving trailing hyphens
        s = s.replace(Regex("^[|!~_—\\-]+\\s*"), "")
        s = s.replace(Regex("\\s*[|~_—]+$"), "")
        return s.trim()
    }

    /**
     * Expands rect by padding while staying within screen bitmap bounds.
     */
    private fun expandRect(box: Rect, bitmap: Bitmap?, padX: Int, padY: Int): Rect {
        val maxWidth = bitmap?.width ?: 2000
        val maxHeight = bitmap?.height ?: 3000

        val left = max(0, box.left - padX)
        val top = max(0, box.top - padY)
        val right = min(maxWidth, box.right + padX)
        val bottom = min(maxHeight, box.bottom + padY)

        return Rect(left, top, right, bottom)
    }

    /**
     * Analyzes screenshot pixel colors inside the speech bubble margins
     * to determine if bubble background is light (white bubble) or dark (black bubble).
     * Avoids outer rectangular corners that often lie in external manga artwork,
     * and avoids the center pixel where text ink resides.
     */
    fun detectBubbleColors(box: Rect, bitmap: Bitmap?): Pair<Int, Int> {
        val defaultWhiteBg = Color.WHITE
        val defaultBlackText = Color.parseColor("#111827") // Slate 900 for sharp contrast

        val defaultDarkBg = Color.parseColor("#18181B")   // Zinc 900
        val defaultWhiteText = Color.WHITE

        if (bitmap == null || bitmap.isRecycled || box.width() <= 0 || box.height() <= 0) {
            return Pair(defaultWhiteBg, defaultBlackText)
        }

        var tempBitmap: Bitmap? = null
        val readableBitmap: Bitmap = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O &&
            bitmap.config == Bitmap.Config.HARDWARE
        ) {
            try {
                bitmap.copy(Bitmap.Config.ARGB_8888, false)?.also { tempBitmap = it } ?: bitmap
            } catch (e: Exception) {
                bitmap
            }
        } else {
            bitmap
        }

        try {
            val bWidth = readableBitmap.width
            val bHeight = readableBitmap.height

            // Sample points strictly inside the interior perimeter margins of the bubble
            val insetX = (box.width() * 0.12f).toInt().coerceIn(2, 20)
            val insetY = (box.height() * 0.12f).toInt().coerceIn(2, 20)

            val samplePoints = listOf(
                Pair(box.centerX(), box.top + insetY),
                Pair(box.centerX(), box.bottom - insetY),
                Pair(box.left + insetX, box.centerY()),
                Pair(box.right - insetX, box.centerY()),
                Pair(box.left + insetX, box.top + insetY),
                Pair(box.right - insetX, box.top + insetY),
                Pair(box.left + insetX, box.bottom - insetY),
                Pair(box.right - insetX, box.bottom - insetY)
            )

            var totalLuminance = 0.0
            var validSamples = 0
            var brightSamples = 0

            for ((x, y) in samplePoints) {
                if (x in 0 until bWidth && y in 0 until bHeight) {
                    val pixel = readableBitmap.getPixel(x, y)
                    val r = Color.red(pixel)
                    val g = Color.green(pixel)
                    val b = Color.blue(pixel)
                    val lum = 0.299 * r + 0.587 * g + 0.114 * b
                    totalLuminance += lum
                    if (lum > 128) brightSamples++
                    validSamples++
                }
            }

            if (validSamples > 0) {
                val avgLuminance = totalLuminance / validSamples
                // If more than 40% of samples are bright or avg luminance > 115, it's a light speech bubble
                val isLightBubble = (brightSamples.toFloat() / validSamples >= 0.4f) || (avgLuminance >= 115.0)
                return if (isLightBubble) {
                    Pair(defaultWhiteBg, defaultBlackText)
                } else {
                    Pair(defaultDarkBg, defaultWhiteText)
                }
            }
        } catch (e: Exception) {
            // Fallback
        } finally {
            if (tempBitmap != null && !tempBitmap!!.isRecycled) {
                tempBitmap!!.recycle()
            }
        }

        return Pair(defaultWhiteBg, defaultBlackText)
    }
}
