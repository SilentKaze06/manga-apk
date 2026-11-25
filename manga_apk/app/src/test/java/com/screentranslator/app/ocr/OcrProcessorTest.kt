package com.screentranslator.app.ocr

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class OcrProcessorTest {

    @Test
    fun pureHanziScoresChineseHigherThanJapaneseAndDoesNotEarlyBreakJa() {
        val chineseText = "你好世界欢迎来到漫画翻译"
        val jaScore = OcrProcessor.scoreAutoLanguageText(chineseText, "ja")
        val zhScore = OcrProcessor.scoreAutoLanguageText(chineseText, "zh")

        assertTrue("Chinese score ($zhScore) must exceed Japanese score ($jaScore) for pure Hanzi", zhScore > jaScore)
        assertFalse(
            "Japanese must never early-break when kana == 0",
            OcrProcessor.shouldEarlyBreakAutoLanguage(chineseText, "ja", jaScore)
        )
        assertTrue(
            "Chinese should allow early break on pure Hanzi",
            OcrProcessor.shouldEarlyBreakAutoLanguage(chineseText, "zh", zhScore)
        )
    }

    @Test
    fun prolongedSoundMarkDoesNotDisqualifyChinese() {
        val chineseWithProlonged = "你好ー世界"
        val zhScore = OcrProcessor.scoreAutoLanguageText(chineseWithProlonged, "zh")
        val jaScore = OcrProcessor.scoreAutoLanguageText(chineseWithProlonged, "ja")

        assertTrue("Stray 'ー' must not zero out Chinese score (zh=$zhScore, ja=$jaScore)", zhScore > jaScore)
        assertFalse(
            "Japanese must not early-break with only Hanzi and 'ー'",
            OcrProcessor.shouldEarlyBreakAutoLanguage(chineseWithProlonged, "ja", jaScore)
        )
    }

    @Test
    fun japaneseWithKanaBeatsChineseAndAllowsEarlyBreak() {
        val japaneseText = "俺は海賊王になる！"
        val jaScore = OcrProcessor.scoreAutoLanguageText(japaneseText, "ja")
        val zhScore = OcrProcessor.scoreAutoLanguageText(japaneseText, "zh")

        assertTrue("Japanese with kana ($jaScore) must beat Chinese ($zhScore)", jaScore > zhScore)
        assertEquals(0.0, zhScore, 0.001)
        assertTrue(OcrProcessor.shouldEarlyBreakAutoLanguage(japaneseText, "ja", jaScore))
    }

    @Test
    fun koreanAndCyrillicDoNotEarlyBreakJapanese() {
        val koreanText = "안녕하세요 반갑습니다"
        val koScore = OcrProcessor.scoreAutoLanguageText(koreanText, "ko")
        val jaScoreForKo = OcrProcessor.scoreAutoLanguageText(koreanText, "ja")
        assertTrue(koScore > jaScoreForKo)
        assertTrue(OcrProcessor.shouldEarlyBreakAutoLanguage(koreanText, "ko", koScore))

        val cyrillicText = "Привет мир АБВ"
        val jaScoreForRu = OcrProcessor.scoreAutoLanguageText(cyrillicText, "ja")
        assertEquals(0.0, jaScoreForRu, 0.001)
        assertFalse(OcrProcessor.shouldEarlyBreakAutoLanguage(cyrillicText, "ja", jaScoreForRu))
    }

    @Test
    fun computeScaledDimensionsRetainsHighResolutionForSmallText() {
        val (w1, h1) = OcrProcessor.computeScaledDimensions(1440, 3120)
        assertTrue("Width $w1 should stay >= 1080px", w1 >= 1080)
        assertEquals(2400, h1)

        val (w2, h2) = OcrProcessor.computeScaledDimensions(1080, 2400)
        assertEquals(1080, w2)
        assertEquals(2400, h2)

        val (w3, h3) = OcrProcessor.computeScaledDimensions(1080, 3600)
        assertTrue("Tall webtoon width $w3 should stay >= 1080px", w3 >= 1080)
        assertEquals(3600, h3)
    }
}
