package com.screentranslator.app.data.model

import com.google.gson.annotations.SerializedName

/**
 * Response returned from the Python Manga OCR server for a single bubble (/translate-manga).
 */
data class MangaServerResponse(
    @SerializedName("status")
    val status: String = "success",

    @SerializedName("original_text")
    val originalText: String = "",

    @SerializedName("translated_text")
    val translatedText: String = "",

    @SerializedName("lang")
    val lang: String? = null,

    @SerializedName("detected_source_lang")
    val detectedSourceLang: String? = null,

    @SerializedName("target_lang")
    val targetLang: String? = null
)

/**
 * Text block detected during whole screen manga analysis (/detect-and-translate).
 */
data class MangaDetectedBlock(
    @SerializedName("block_id")
    val blockId: String? = null,

    @SerializedName("left")
    val left: Int = 0,

    @SerializedName("top")
    val top: Int = 0,

    @SerializedName("right")
    val right: Int = 0,

    @SerializedName("bottom")
    val bottom: Int = 0,

    @SerializedName("original_text")
    val originalText: String = "",

    @SerializedName("translated_text")
    val translatedText: String = "",

    @SerializedName("category")
    val category: String? = null,

    @SerializedName("is_dark")
    val isDark: Boolean? = null,

    @SerializedName("bg_color")
    val bgColor: String? = null,

    @SerializedName("text_color")
    val textColor: String? = null
)

/**
 * Response returned from /detect-and-translate for full screen captures.
 */
data class MangaDetectAndTranslateResponse(
    @SerializedName("status")
    val status: String = "success",

    @SerializedName("source_lang")
    val sourceLang: String? = null,

    @SerializedName("blocks_count")
    val blocksCount: Int = 0,

    @SerializedName("blocks")
    val blocks: List<MangaDetectedBlock> = emptyList()
)
