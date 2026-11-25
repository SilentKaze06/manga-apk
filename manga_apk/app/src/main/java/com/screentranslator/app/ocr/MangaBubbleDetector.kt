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

        // Japanese has unique syllabic kana scripts (Hiragana or Katakana), excluding middle dot '・' and prolonged mark 'ー'
        if (clean.any { it in '\u3041'..'\u3096' || it in '\u30A1'..'\u30FA' || it in '\uFF66'..'\uFF9D' }) {
            return "ja"
        }
        // Korean has unique Hangul syllables and jamo
        if (clean.any { it in '\uAC00'..'\uD7AF' || it in '\u1100'..'\u11FF' || it in '\u3130'..'\u318F' }) {
            return "ko"
        }
        // Han characters (Kanji / Hanzi): preserve explicit Chinese fallback if present
        if (clean.any { it in '\u4E00'..'\u9FFF' || it in '\u3400'..'\u4DBF' }) {
            return if (fallback == "zh") "zh" else "ja"
        }
        // Cyrillic (Russian)
        if (clean.any { it in '\u0400'..'\u04FF' }) {
            return "ru"
        }
        // Latin / English
        if (clean.any { it in 'a'..'z' || it in 'A'..'Z' || it in '\u00C0'..'\u024F' }) {
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
            val effectiveLang = normalizedSourceLang ?: detectLanguageFromText(cleanedText, fallback = single.sourceLanguage)
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

            // Determine effective ISO source language for this speech bubble, preserving rawBlock hint
            val blockLangFallback = bubbleLines
                .mapNotNull { it.sourceLanguage }
                .groupingBy { it }
                .eachCount()
                .maxByOrNull { it.value }
                ?.key
            val effectiveLang = normalizedSourceLang ?: detectLanguageFromText(mergedText, fallback = blockLangFallback)

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
        return areLinesInSameBubbleCoords(
            boxA.left, boxA.top, boxA.right, boxA.bottom,
            boxB.left, boxB.top, boxB.right, boxB.bottom
        )
    }

    internal fun areLinesInSameBubbleCoords(
        aLeft: Int, aTop: Int, aRight: Int, aBottom: Int,
        bLeft: Int, bTop: Int, bRight: Int, bBottom: Int
    ): Boolean {
        val widthA = max(1, aRight - aLeft)
        val heightA = max(1, aBottom - aTop)
        val widthB = max(1, bRight - bLeft)
        val heightB = max(1, bBottom - bTop)

        val avgHeight = (heightA + heightB) / 2f
        val avgWidth = (widthA + widthB) / 2f
        val minHeight = min(heightA, heightB).toFloat()
        val minWidth = min(widthA, widthB).toFloat()

        val isAVertical = heightA > widthA * 1.15f
        val isBVertical = heightB > widthB * 1.15f
        val isVerticalPair = isAVertical || isBVertical

        // Case 1: Vertical text columns arranged side-by-side (Tategaki)
        val yOverlap = min(aBottom, bBottom) - max(aTop, bTop)
        val horizGap = max(0, max(aLeft, bLeft) - min(aRight, bRight))

        if (isVerticalPair && yOverlap > minHeight * 0.45f) {
            val maxColWidth = max(widthA, widthB).toFloat()
            val maxAllowedHorizGap = max(avgWidth * 1.35f, maxColWidth * 1.15f).coerceAtLeast(16f)
            return horizGap <= maxAllowedHorizGap
        }

        // Case 2: Split segments of a single vertical column stacked top-to-bottom
        if (isAVertical && isBVertical) {
            val vertGap = max(0, max(aTop, bTop) - min(aBottom, bBottom))
            val horizOverlap = min(aRight, bRight) - max(aLeft, bLeft)
            return horizOverlap >= minWidth * 0.4f && vertGap <= max(widthA, widthB) * 1.5f
        }

        // Case 3: Horizontal text lines stacked vertically (Yokogaki)
        val vertGap = max(0, max(aTop, bTop) - min(aBottom, bBottom))
        val maxAllowedVertGap = avgHeight * 1.3f
        if (vertGap > maxAllowedVertGap) {
            return false
        }

        val horizOverlap = min(aRight, bRight) - max(aLeft, bLeft)
        val centerDistX = abs((aLeft + aRight) / 2f - (bLeft + bRight) / 2f)
        val maxWideW = max(widthA, widthB).toFloat()

        return horizOverlap > -(minHeight * 0.45f) || centerDistX <= maxWideW * 0.55f
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
     * Removes stray edge frame artifacts while preserving short tokens, hyphens, and dashes.
     */
    internal fun cleanOcrNoise(text: String): String {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return ""
        val stripped = trimmed
            .replace(Regex("^[|_]+\\s*"), "")
            .replace(Regex("\\s*[|_]+$"), "")
            .trim()
        return stripped.ifEmpty {
            if (trimmed.length >= 2 && trimmed.all { it == '|' || it == '_' }) "" else trimmed
        }
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
     * Analyzes screenshot pixel colors inside the speech bubble to estimate the actual
     * non-ink background RGB color (preserving colored, cream, gray, and dark bubbles)
     * paired with a high-contrast text color.
     */
    fun detectBubbleColors(box: Rect, bitmap: Bitmap?): Pair<Int, Int> {
        val defaultWhiteBg = 0xFFFFFFFF.toInt()
        val defaultBlackText = 0xFF111827.toInt()

        if (bitmap == null || bitmap.isRecycled || box.width() <= 0 || box.height() <= 0) {
            return Pair(defaultWhiteBg, defaultBlackText)
        }

        var tempBitmap: Bitmap? = null
        val readableBitmap: Bitmap = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O &&
            bitmap.config == Bitmap.Config.HARDWARE
        ) {
            try {
                bitmap.copy(Bitmap.Config.ARGB_8888, false)?.also { tempBitmap = it } ?: bitmap
            } catch (e: Throwable) {
                bitmap
            }
        } else {
            bitmap
        }

        try {
            val bWidth = readableBitmap.width
            val bHeight = readableBitmap.height
            val xStart = box.left.coerceIn(0, bWidth - 1)
            val yStart = box.top.coerceIn(0, bHeight - 1)
            val xEnd = box.right.coerceIn(xStart + 1, bWidth)
            val yEnd = box.bottom.coerceIn(yStart + 1, bHeight)
            val boxW = xEnd - xStart
            val boxH = yEnd - yStart
            if (boxW <= 0 || boxH <= 0) {
                return Pair(defaultWhiteBg, defaultBlackText)
            }

            val insetX = if (boxW > 8) (boxW * 0.06f).toInt().coerceIn(1, 12) else 0
            val insetY = if (boxH > 8) (boxH * 0.06f).toInt().coerceIn(1, 12) else 0
            val innerLeft = (xStart + insetX).coerceAtMost(xEnd - 1)
            val innerRight = (xEnd - 1 - insetX).coerceAtLeast(innerLeft)
            val innerTop = (yStart + insetY).coerceAtMost(yEnd - 1)
            val innerBottom = (yEnd - 1 - insetY).coerceAtLeast(innerTop)

            val cols = min(10, max(3, innerRight - innerLeft + 1))
            val rows = min(10, max(3, innerBottom - innerTop + 1))
            val gridSamples = IntArray(cols * rows)
            val perimeterList = ArrayList<Int>(cols * 2 + rows * 2)
            var idx = 0
            for (r in 0 until rows) {
                val py = if (rows == 1) innerTop else innerTop + (innerBottom - innerTop) * r / (rows - 1)
                for (c in 0 until cols) {
                    val px = if (cols == 1) innerLeft else innerLeft + (innerRight - innerLeft) * c / (cols - 1)
                    val pixel = readableBitmap.getPixel(px, py)
                    gridSamples[idx++] = pixel
                    if (r == 0 || r == rows - 1 || c == 0 || c == cols - 1) {
                        perimeterList.add(pixel)
                    }
                }
            }
            return detectBubbleColorsFromSamples(gridSamples, perimeterList.toIntArray())
        } catch (e: Exception) {
            // Fallback
        } finally {
            if (tempBitmap != null && !tempBitmap!!.isRecycled) {
                tempBitmap!!.recycle()
            }
        }

        return Pair(defaultWhiteBg, defaultBlackText)
    }

    internal fun detectBubbleColorsFromPixels(
        pixels: IntArray,
        width: Int,
        height: Int,
        left: Int,
        top: Int,
        right: Int,
        bottom: Int
    ): Pair<Int, Int> {
        val defaultWhiteBg = 0xFFFFFFFF.toInt()
        val defaultBlackText = 0xFF111827.toInt()
        if (width <= 0 || height <= 0 || pixels.size < width * height) {
            return Pair(defaultWhiteBg, defaultBlackText)
        }
        val xStart = left.coerceIn(0, width - 1)
        val yStart = top.coerceIn(0, height - 1)
        val xEnd = right.coerceIn(xStart + 1, width)
        val yEnd = bottom.coerceIn(yStart + 1, height)
        val boxW = xEnd - xStart
        val boxH = yEnd - yStart
        if (boxW <= 0 || boxH <= 0) return Pair(defaultWhiteBg, defaultBlackText)

        val insetX = if (boxW > 8) (boxW * 0.06f).toInt().coerceIn(1, 12) else 0
        val insetY = if (boxH > 8) (boxH * 0.06f).toInt().coerceIn(1, 12) else 0
        val innerLeft = (xStart + insetX).coerceAtMost(xEnd - 1)
        val innerRight = (xEnd - 1 - insetX).coerceAtLeast(innerLeft)
        val innerTop = (yStart + insetY).coerceAtMost(yEnd - 1)
        val innerBottom = (yEnd - 1 - insetY).coerceAtLeast(innerTop)

        val cols = min(12, max(3, innerRight - innerLeft + 1))
        val rows = min(12, max(3, innerBottom - innerTop + 1))
        val gridSamples = IntArray(cols * rows)
        val perimeterList = ArrayList<Int>(cols * 2 + rows * 2)
        var idx = 0
        for (r in 0 until rows) {
            val py = if (rows == 1) innerTop else innerTop + (innerBottom - innerTop) * r / (rows - 1)
            for (c in 0 until cols) {
                val px = if (cols == 1) innerLeft else innerLeft + (innerRight - innerLeft) * c / (cols - 1)
                val pixel = pixels[py * width + px]
                gridSamples[idx++] = pixel
                if (r == 0 || r == rows - 1 || c == 0 || c == cols - 1) {
                    perimeterList.add(pixel)
                }
            }
        }
        return detectBubbleColorsFromSamples(gridSamples, perimeterList.toIntArray())
    }

    internal fun detectBubbleColorsFromSamples(
        samples: IntArray,
        perimeterSamples: IntArray = intArrayOf()
    ): Pair<Int, Int> {
        val defaultWhiteBg = 0xFFFFFFFF.toInt()
        val defaultBlackText = 0xFF111827.toInt()
        if (samples.isEmpty()) return Pair(defaultWhiteBg, defaultBlackText)

        data class Sample(val r: Int, val g: Int, val b: Int, val lum: Double)

        fun decode(pixel: Int): Sample {
            val r = (pixel ushr 16) and 0xFF
            val g = (pixel ushr 8) and 0xFF
            val b = pixel and 0xFF
            val lum = 0.299 * r + 0.587 * g + 0.114 * b
            return Sample(r, g, b, lum)
        }

        val sorted = samples.map(::decode).sortedBy { it.lum }
        val perimeterMedianLum = if (perimeterSamples.isNotEmpty()) {
            val pSorted = perimeterSamples.map { decode(it).lum }.sorted()
            pSorted[pSorted.size / 2]
        } else {
            sorted[sorted.size / 2].lum
        }

        // The background forms the largest tight luminance cluster (minimum-range 45% window),
        // filtering out both dark text ink on light/colored bubbles and white text on dark boxes.
        val windowSize = max(1, (sorted.size * 0.45).toInt())
        var bestStart = 0
        var bestSpan = Double.MAX_VALUE
        var bestPerimeterDist = Double.MAX_VALUE
        for (i in 0..(sorted.size - windowSize)) {
            val span = sorted[i + windowSize - 1].lum - sorted[i].lum
            val clusterMedLum = sorted[i + windowSize / 2].lum
            val pDist = abs(clusterMedLum - perimeterMedianLum)
            if (span + 1.0 < bestSpan || (abs(span - bestSpan) <= 1.0 && pDist < bestPerimeterDist)) {
                bestSpan = span
                bestStart = i
                bestPerimeterDist = pDist
            }
        }

        val clusterCenterLum = sorted[bestStart + windowSize / 2].lum
        val bgSamples = sorted.filter { abs(it.lum - clusterCenterLum) <= 22.0 }
            .ifEmpty { sorted.subList(bestStart, bestStart + windowSize) }

        val medR = bgSamples.map { it.r }.sorted()[bgSamples.size / 2]
        val medG = bgSamples.map { it.g }.sorted()[bgSamples.size / 2]
        val medB = bgSamples.map { it.b }.sorted()[bgSamples.size / 2]

        val bgColor = (0xFF shl 24) or (medR shl 16) or (medG shl 8) or medB
        val textColor = contrastTextColorForBg(bgColor)
        return Pair(bgColor, textColor)
    }

    internal fun isColorLight(argb: Int): Boolean {
        val r = (argb ushr 16) and 0xFF
        val g = (argb ushr 8) and 0xFF
        val b = argb and 0xFF
        val lum = 0.299 * r + 0.587 * g + 0.114 * b
        return lum >= 128.0
    }

    internal fun contrastTextColorForBg(bgColor: Int): Int {
        fun channelRelLum(v: Int): Double {
            val s = v / 255.0
            return if (s <= 0.03928) s / 12.92 else Math.pow((s + 0.055) / 1.055, 2.4)
        }
        val r = (bgColor ushr 16) and 0xFF
        val g = (bgColor ushr 8) and 0xFF
        val b = bgColor and 0xFF
        val bgRelLum = 0.2126 * channelRelLum(r) + 0.7152 * channelRelLum(g) + 0.0722 * channelRelLum(b)
        // Contrast against #111827 (relLum ~ 0.008) vs #FFFFFF (relLum = 1.0)
        val darkInkRelLum = 0.008
        val contrastWithDark = (bgRelLum + 0.05) / (darkInkRelLum + 0.05)
        val contrastWithWhite = 1.05 / (bgRelLum + 0.05)
        return if (contrastWithDark >= contrastWithWhite) 0xFF111827.toInt() else 0xFFFFFFFF.toInt()
    }
}
