package com.screentranslator.app.data.model

/**
 * Supported target languages for translation.
 * Includes Russian, English, Uzbek, and common global languages.
 */
enum class TargetLanguage(
    val code: String,
    val displayName: String
) {
    RUSSIAN("ru", "Russian (Русский)"),
    ENGLISH("en", "English"),
    UZBEK("uz", "Uzbek (Özbek)"),
    SPANISH("es", "Spanish (Español)"),
    GERMAN("de", "German (Deutsch)"),
    FRENCH("fr", "French (Français)"),
    TURKISH("tr", "Turkish (Türkçe)"),
    CHINESE("zh", "Chinese (中文)"),
    JAPANESE("ja", "Japanese (日本語)"),
    KOREAN("ko", "Korean (한국어)"),
    ARABIC("ar", "Arabic (العربية)");

    companion object {
        fun fromCode(code: String): TargetLanguage {
            return entries.find { it.code.equals(code, ignoreCase = true) } ?: UZBEK
        }
    }
}
