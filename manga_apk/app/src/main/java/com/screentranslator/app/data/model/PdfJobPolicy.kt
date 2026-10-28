package com.screentranslator.app.data.model

object PdfJobPolicy {
    fun isRunning(status: String?) = status == "queued" || status == "processing"
    fun canReplace(status: String?) = status == null || status in setOf("completed", "failed", "canceled")
    fun retryDelayMillis(failures: Int): Long = (1800L * (1L shl failures.coerceIn(0, 4))).coerceAtMost(30_000L)
    fun downloadFilename(original: String): String {
        val name = original.substringAfterLast('/').substringAfterLast('\\')
        return name.substringBeforeLast('.', name).ifBlank { "kitob" } + "_tarjima.pdf"
    }
}
