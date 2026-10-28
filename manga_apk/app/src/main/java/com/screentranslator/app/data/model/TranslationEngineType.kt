package com.screentranslator.app.data.model

/**
 * Supported translation backends.
 */
enum class TranslationEngineType(
    val title: String,
    val description: String,
    val isOnline: Boolean,
    val requiresApiKey: Boolean,
    val requiresUrl: Boolean
) {
    MY_MEMORY(
        title = "MyMemory API",
        description = "Bepul online tarjima API. Ro'yxatdan o'tmasdan ishlaydi, barcha asosiy tillarni qo'llab-quvvatlaydi.",
        isOnline = true,
        requiresApiKey = false,
        requiresUrl = false
    ),
    LIBRE_TRANSLATE(
        title = "LibreTranslate",
        description = "Ochiq kodli online tarjima API. Ommaviy server yoki o'zingizning serveringiz bilan ishlaydi.",
        isOnline = true,
        requiresApiKey = false,
        requiresUrl = true
    ),
    DEEPL(
        title = "DeepL Free API",
        description = "Yuqori sifatli sun'iy intellekt tarjimasi. deepl.com dan bepul API kalit talab etiladi.",
        isOnline = true,
        requiresApiKey = true,
        requiresUrl = false
    );

    companion object {
        fun fromName(name: String?): TranslationEngineType {
            return entries.find { it.name.equals(name, ignoreCase = true) } ?: MY_MEMORY
        }
    }
}
