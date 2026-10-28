package com.screentranslator.app.data.model

data class PdfJobReference(
    val jobId: String,
    val token: String,
    // Null identifies a legacy handle whose originating server is unknown. Never send its token.
    val serverUrl: String? = null,
    val filename: String = ""
)
