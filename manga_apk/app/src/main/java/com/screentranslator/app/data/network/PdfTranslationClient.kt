package com.screentranslator.app.data.network

import android.content.ContentResolver
import android.net.Uri
import com.screentranslator.app.BuildConfig
import com.google.gson.Gson
import com.screentranslator.app.data.model.PdfTranslationCorrection
import com.screentranslator.app.data.model.PdfTranslationReviewPage
import com.screentranslator.app.data.model.PdfTranslationJob
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.util.concurrent.TimeUnit

object PdfTranslationClient {
    private val gson = Gson()
    private val client = OkHttpClient.Builder()
        .connectTimeout(12, TimeUnit.SECONDS)
        .writeTimeout(5, TimeUnit.MINUTES)
        .readTimeout(60, TimeUnit.SECONDS)
        .retryOnConnectionFailure(true)
        .followRedirects(true)
        .followSslRedirects(true)
        .build()

    private fun baseUrl(value: String): String {
        return ServerEndpoint.normalize(value, BuildConfig.DEBUG)
    }

    suspend fun submit(
        serverUrl: String,
        apiKey: String,
        pdf: File,
        filename: String,
        documentMode: String,
        sourceLang: String,
        targetLang: String,
        outputMode: String
    ): PdfTranslationJob = withContext(Dispatchers.IO) {
        val body = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("file", filename, pdf.asRequestBody("application/pdf".toMediaType()))
            .addFormDataPart("document_mode", documentMode)
            .addFormDataPart("source_lang", sourceLang)
            .addFormDataPart("target_lang", targetLang)
            .addFormDataPart("output_mode", outputMode)
            .build()
        val request = Request.Builder()
            .url("${baseUrl(serverUrl)}/translate-pdf")
            .apply { if (apiKey.isNotBlank()) header("X-API-Key", apiKey) }
            .post(body)
            .build()
        executeJson(request)
    }

    suspend fun status(serverUrl: String, apiKey: String, jobId: String, jobToken: String): PdfTranslationJob = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url("${baseUrl(serverUrl)}/pdf-jobs/$jobId")
            .apply {
                if (apiKey.isNotBlank()) header("X-API-Key", apiKey)
                header("X-PDF-Job-Token", jobToken)
            }
            .get()
            .build()
        executeJson(request)
    }

    suspend fun cancel(serverUrl: String, apiKey: String, jobId: String, jobToken: String): PdfTranslationJob = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url("${baseUrl(serverUrl)}/pdf-jobs/$jobId/cancel")
            .apply {
                if (apiKey.isNotBlank()) header("X-API-Key", apiKey)
                header("X-PDF-Job-Token", jobToken)
            }
            .post(ByteArray(0).toRequestBody(null))
            .build()
        executeJson(request)
    }

    suspend fun resume(serverUrl: String, apiKey: String, jobId: String, jobToken: String): PdfTranslationJob = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url("${baseUrl(serverUrl)}/pdf-jobs/$jobId/resume")
            .apply {
                if (apiKey.isNotBlank()) header("X-API-Key", apiKey)
                header("X-PDF-Job-Token", jobToken)
            }
            .post(ByteArray(0).toRequestBody(null))
            .build()
        executeJson(request)
    }

    suspend fun reviewPage(
        serverUrl: String,
        apiKey: String,
        jobId: String,
        jobToken: String,
        pageIndex: Int
    ): PdfTranslationReviewPage = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url("${baseUrl(serverUrl)}/pdf-jobs/$jobId/review/$pageIndex")
            .apply {
                if (apiKey.isNotBlank()) header("X-API-Key", apiKey)
                header("X-PDF-Job-Token", jobToken)
            }
            .get()
            .build()
        client.newCall(request).readResponse { response ->
            val body = response.body?.string().orEmpty()
            if (!response.isSuccessful) {
                val detail = runCatching { gson.fromJson(body, Map::class.java)["detail"]?.toString() }.getOrNull()
                throw IllegalStateException(detail ?: "Tahrir matnini olib bo'lmadi (${response.code}).")
            }
            gson.fromJson(body, PdfTranslationReviewPage::class.java)
                ?: throw IllegalStateException("Server tahrir matnini qaytarmadi.")
        }
    }

    suspend fun applyCorrections(
        serverUrl: String,
        apiKey: String,
        jobId: String,
        jobToken: String,
        corrections: List<PdfTranslationCorrection>
    ): PdfTranslationJob = withContext(Dispatchers.IO) {
        val payload = gson.toJson(mapOf("corrections" to corrections))
            .toRequestBody("application/json; charset=utf-8".toMediaType())
        val request = Request.Builder()
            .url("${baseUrl(serverUrl)}/pdf-jobs/$jobId/review")
            .apply {
                if (apiKey.isNotBlank()) header("X-API-Key", apiKey)
                header("X-PDF-Job-Token", jobToken)
            }
            .post(payload)
            .build()
        executeJson(request)
    }

    suspend fun download(
        serverUrl: String,
        apiKey: String,
        jobId: String,
        jobToken: String,
        contentResolver: ContentResolver,
        destination: Uri,
        tempDir: File? = null
    ) = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url("${baseUrl(serverUrl)}/pdf-jobs/$jobId/download")
            .apply {
                if (apiKey.isNotBlank()) header("X-API-Key", apiKey)
                header("X-PDF-Job-Token", jobToken)
            }
            .get()
            .build()
        val stagingFile = if (tempDir != null) File.createTempFile("pdf-dl-stage-", ".pdf", tempDir) else null
        try {
            client.newCall(request).readResponse { response ->
                if (!response.isSuccessful) {
                    val detail = response.body?.string().orEmpty()
                    throw IllegalStateException("PDFni yuklab bo'lmadi (${response.code}): $detail")
                }
                val body = response.body ?: throw IllegalStateException("Server bo'sh PDF qaytardi.")
                if (stagingFile != null) {
                    stagingFile.outputStream().use { fileOut ->
                        body.byteStream().use { input -> input.copyTo(fileOut) }
                    }
                } else {
                    val output = contentResolver.openOutputStream(destination, "wt")
                        ?: throw IllegalStateException("Tanlangan joyga fayl yozib bo'lmadi.")
                    body.byteStream().use { input -> output.use { input.copyTo(it) } }
                }
            }
            if (stagingFile != null) {
                if (stagingFile.length() == 0L) {
                    throw IllegalStateException("Server bo'sh PDF qaytardi.")
                }
                val output = contentResolver.openOutputStream(destination, "wt")
                    ?: throw IllegalStateException("Tanlangan joyga fayl yozib bo'lmadi.")
                output.use { outStream ->
                    stagingFile.inputStream().use { inStream ->
                        inStream.copyTo(outStream)
                    }
                }
            }
        } finally {
            stagingFile?.delete()
        }
    }

    private suspend fun executeJson(request: Request): PdfTranslationJob {
        return client.newCall(request).readResponse { response ->
            val body = response.body?.string().orEmpty()
            if (!response.isSuccessful) {
                val detail = runCatching { gson.fromJson(body, Map::class.java)["detail"]?.toString() }.getOrNull()
                throw IllegalStateException(detail ?: "Server xatosi (${response.code}).")
            }
            gson.fromJson(body, PdfTranslationJob::class.java)
                ?: throw IllegalStateException("Server noto'g'ri javob qaytardi.")
        }
    }
}
