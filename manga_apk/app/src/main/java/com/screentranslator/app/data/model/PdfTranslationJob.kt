package com.screentranslator.app.data.model

import com.google.gson.annotations.SerializedName

data class PdfTranslationJob(
    @SerializedName("job_id") val jobId: String,
    @SerializedName("filename") val filename: String,
    @SerializedName("status") val status: String,
    @SerializedName("progress") val progress: Int = 0,
    @SerializedName("current_page") val currentPage: Int = 0,
    @SerializedName("total_pages") val totalPages: Int = 0,
    @SerializedName("source_lang") val sourceLang: String = "auto",
    @SerializedName("target_lang") val targetLang: String = "uz",
    @SerializedName("output_mode") val outputMode: String = "translated",
    @SerializedName("document_mode") val documentMode: String = "book",
    @SerializedName("message") val message: String = "",
    @SerializedName("job_token") val jobToken: String? = null,
    @SerializedName("error") val error: String? = null
)
