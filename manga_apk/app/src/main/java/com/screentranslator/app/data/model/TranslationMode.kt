package com.screentranslator.app.data.model

/**
 * Operating modes for the screen translator.
 */
enum class TranslationMode(
    val id: String,
    val title: String,
    val description: String
) {
    STANDARD(
        id = "standard",
        title = "Standart rejim",
        description = "Ekranni qator-qator o'qiydi (Ilovalar, yangiliklar va tizim matnlari)"
    ),
    MANGA(
        id = "manga",
        title = "Manga / Komiks",
        description = "Dialog bulutlarini (speech bubbles) butun holda birlashtirib o'qiydi va o'sha joyga tahrirlab joylaydi"
    );

    companion object {
        fun fromId(id: String?): TranslationMode {
            return entries.find { it.id.equals(id, ignoreCase = true) } ?: STANDARD
        }
    }
}
