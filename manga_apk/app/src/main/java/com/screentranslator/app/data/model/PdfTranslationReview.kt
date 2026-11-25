package com.screentranslator.app.data.model

import com.google.gson.annotations.SerializedName

data class PdfTranslationReviewPage(
    @SerializedName("page_index") val pageIndex: Int = 0,
    @SerializedName("total_pages") val totalPages: Int = 0,
    @SerializedName("document_mode") val documentMode: String = "book",
    @SerializedName("warning") val warning: String? = null,
    @SerializedName("blocks") val blocks: List<PdfTranslationReviewBlock> = emptyList()
)

data class PdfTranslationReviewBlock(
    @SerializedName("block_index") val blockIndex: Int = 0,
    @SerializedName("original") val original: String = "",
    @SerializedName("translated") val translated: String = "",
    @SerializedName("confidence") val confidence: Float? = null
)

data class PdfTranslationCorrection(
    @SerializedName("page_index") val pageIndex: Int,
    @SerializedName("block_index") val blockIndex: Int,
    @SerializedName("original") val original: String? = null,
    @SerializedName("translated") val translated: String? = null
)
