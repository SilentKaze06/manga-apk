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

    @SerializedName("target_lang")
    val targetLang: String? = null
)

/**
 * Text block detected during whole screen manga analysis (/detect-and-translate).
 */
data class MangaDetectedBlock(
    @SerializedName("left")
    val left: Int,

    @SerializedName("top")
    val top: Int,

    @SerializedName("right")
    val right: Int,

    @SerializedName("bottom")
    val bottom: Int,

    @SerializedName("original_text")
    val originalText: String,

    @SerializedName("translated_text")
    val translatedText: String
)

/**
 * Response returned from /detect-and-translate for full screen captures.
 */
data class MangaDetectAndTranslateResponse(
    @SerializedName("status")
    val status: String = "success",

    @SerializedName("blocks_count")
    val blocksCount: Int = 0,

    @SerializedName("blocks")
    val blocks: List<MangaDetectedBlock> = emptyList()
)
