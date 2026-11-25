package com.screentranslator.app.ocr

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MangaBubbleDetectorTest {

    private fun makeBubblePixels(
        width: Int,
        height: Int,
        bgArgb: Int,
        inkArgb: Int,
        inkStripeEvery: Int = 5
    ): IntArray {
        val pixels = IntArray(width * height) { bgArgb }
        for (y in (height / 4) until (height * 3 / 4)) {
            for (x in (width / 4) until (width * 3 / 4)) {
                if ((x + y) % inkStripeEvery == 0) {
                    pixels[y * width + x] = inkArgb
                }
            }
        }
        return pixels
    }

    @Test
    fun detectBubbleColorsPreservesNonWhiteLightColoredBackgrounds() {
        val creamBg = 0xFFF5E6C8.toInt()
        val blackInk = 0xFF111111.toInt()
        val creamPixels = makeBubblePixels(60, 60, creamBg, blackInk)
        val (detectedCream, creamText) = MangaBubbleDetector.detectBubbleColorsFromPixels(
            creamPixels, 60, 60, 0, 0, 60, 60
        )
        assertEquals("Cream background should be preserved, not collapsed to white", creamBg, detectedCream)
        assertNotEquals(0xFFFFFFFF.toInt(), detectedCream)
        assertEquals(0xFF111827.toInt(), creamText)

        val yellowBg = 0xFFFEF08A.toInt()
        val yellowPixels = makeBubblePixels(60, 60, yellowBg, blackInk)
        val (detectedYellow, yellowText) = MangaBubbleDetector.detectBubbleColorsFromPixels(
            yellowPixels, 60, 60, 0, 0, 60, 60
        )
        assertEquals("Yellow background should be preserved", yellowBg, detectedYellow)
        assertEquals(0xFF111827.toInt(), yellowText)
    }

    @Test
    fun detectBubbleColorsPreservesDarkNarrationBoxesWithWhiteText() {
        val darkNavyBg = 0xFF1E293B.toInt()
        val whiteInk = 0xFFFFFFFF.toInt()
        val darkPixels = makeBubblePixels(60, 60, darkNavyBg, whiteInk)
        val (detectedDark, darkText) = MangaBubbleDetector.detectBubbleColorsFromPixels(
            darkPixels, 60, 60, 0, 0, 60, 60
        )
        assertEquals("Dark narration box color should be preserved", darkNavyBg, detectedDark)
        assertEquals(0xFFFFFFFF.toInt(), darkText)
    }

    @Test
    fun areLinesInSameBubbleMergesShortTrailingVerticalColumnAndSeparatesDistinctBubbles() {
        // Tall vertical column + short trailing vertical column inside same bubble
        assertTrue(
            MangaBubbleDetector.areLinesInSameBubbleCoords(
                100, 100, 130, 400,
                60, 110, 90, 150
            )
        )

        // Two separate tall vertical bubbles 120px apart horizontally must NOT merge
        assertFalse(
            MangaBubbleDetector.areLinesInSameBubbleCoords(
                50, 100, 80, 400,
                200, 100, 230, 400
            )
        )

        // Horizontal lines stacked vertically inside same bubble
        assertTrue(
            MangaBubbleDetector.areLinesInSameBubbleCoords(
                100, 100, 260, 130,
                130, 142, 230, 172
            )
        )

        // Two separate horizontal bubbles side-by-side
        assertFalse(
            MangaBubbleDetector.areLinesInSameBubbleCoords(
                50, 100, 170, 130,
                260, 100, 380, 130
            )
        )
    }

    @Test
    fun cleanOcrNoisePreservesShortTokensAndHyphens() {
        assertEquals("-Wait", MangaBubbleDetector.cleanOcrNoise("-Wait"))
        assertEquals("-", MangaBubbleDetector.cleanOcrNoise("-"))
        assertEquals("—", MangaBubbleDetector.cleanOcrNoise("—"))
        assertEquals("Hello", MangaBubbleDetector.cleanOcrNoise("| Hello |"))
        assertEquals("", MangaBubbleDetector.cleanOcrNoise("|||"))
        assertEquals("", MangaBubbleDetector.cleanOcrNoise("___"))
    }

    @Test
    fun detectLanguageFromTextRespectsChineseFallbackAndScriptCues() {
        assertEquals("zh", MangaBubbleDetector.detectLanguageFromText("杀意", fallback = "zh"))
        assertEquals("ja", MangaBubbleDetector.detectLanguageFromText("それは本当ですか"))
        assertEquals("ko", MangaBubbleDetector.detectLanguageFromText("진짜요?"))
        assertEquals("ru", MangaBubbleDetector.detectLanguageFromText("Привет!"))
        assertEquals("en", MangaBubbleDetector.detectLanguageFromText("Let's go!"))
    }
}
