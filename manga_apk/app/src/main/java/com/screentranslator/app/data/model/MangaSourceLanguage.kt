package com.screentranslator.app.data.model

/**
 * Source language options for Manga / Comic OCR recognition.
 */
enum class MangaSourceLanguage(
    val code: String,
    val ocrCode: String,
    val displayName: String
) {
    AUTO("auto", "auto", "Avtomatik aniqlash (Auto)"),
    JAPANESE("ja", "jpn", "Yaponcha (日本語 - Manga)"),
    KOREAN("ko", "kor", "Koreyscha (한국어 - Manhwa)"),
    CHINESE("zh", "chs", "Xitoycha (中文 - Manhua)"),
    ENGLISH("en", "eng", "Inglizcha (English Comics)"),
    RUSSIAN("ru", "rus", "Ruscha (Русский komiks)");

    fun toIsoLanguageCode(): String? = when (this) {
        AUTO -> null
        JAPANESE -> "ja"
        KOREAN -> "ko"
        CHINESE -> "zh"
        ENGLISH -> "en"
        RUSSIAN -> "ru"
    }

    companion object {
        fun fromCode(code: String?): MangaSourceLanguage {
            return entries.find { it.code.equals(code, ignoreCase = true) } ?: AUTO
        }

        fun normalizeToIso(code: String?): String? {
            if (code.isNullOrBlank() || code.equals("auto", ignoreCase = true)) return null
            return when (code.lowercase().trim()) {
                "jpn", "ja" -> "ja"
                "kor", "ko" -> "ko"
                "chs", "cht", "zh", "zh-cn", "zh-tw" -> "zh"
                "eng", "en" -> "en"
                "rus", "ru" -> "ru"
                "spa", "es" -> "es"
                "deu", "ger", "de" -> "de"
                "fra", "fre", "fr" -> "fr"
                "tur", "tr" -> "tr"
                "uzb", "uz" -> "uz"
                else -> if (code.length == 2) code.lowercase() else null
            }
        }
    }
}
