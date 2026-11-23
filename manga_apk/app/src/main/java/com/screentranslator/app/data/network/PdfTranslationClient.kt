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
import okhttp3.HttpUrl.Companion.toHttpUrl
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
        .followRedirects(false)
        .followSslRedirects(false)
        .addInterceptor(CrossHostRedirectInterceptor())
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
        val requestUrl = baseUrl(serverUrl).toHttpUrl().newBuilder()
            .addPathSegment("translate-pdf")
            .build()
        val request = Request.Builder()
            .url(requestUrl)
            .apply { if (apiKey.isNotBlank()) header("X-API-Key", apiKey) }
            .post(body)
            .build()
        executeJson(request)
    }

    suspend fun status(serverUrl: String, apiKey: String, jobId: String, jobToken: String): PdfTranslationJob = withContext(Dispatchers.IO) {
        val requestUrl = baseUrl(serverUrl).toHttpUrl().newBuilder()
            .addPathSegment("pdf-jobs")
            .addPathSegment(jobId)
            .build()
        val request = Request.Builder()
            .url(requestUrl)
            .apply {
                if (apiKey.isNotBlank()) header("X-API-Key", apiKey)
                header("X-PDF-Job-Token", jobToken)
            }
            .get()
            .build()
        executeJson(request)
    }

    suspend fun cancel(serverUrl: String, apiKey: String, jobId: String, jobToken: String): PdfTranslationJob = withContext(Dispatchers.IO) {
        val requestUrl = baseUrl(serverUrl).toHttpUrl().newBuilder()
            .addPathSegment("pdf-jobs")
            .addPathSegment(jobId)
            .addPathSegment("cancel")
            .build()
        val request = Request.Builder()
            .url(requestUrl)
            .apply {
                if (apiKey.isNotBlank()) header("X-API-Key", apiKey)
                header("X-PDF-Job-Token", jobToken)
            }
            .post(ByteArray(0).toRequestBody(null))
            .build()
        executeJson(request)
    }

    suspend fun resume(serverUrl: String, apiKey: String, jobId: String, jobToken: String): PdfTranslationJob = withContext(Dispatchers.IO) {
        val requestUrl = baseUrl(serverUrl).toHttpUrl().newBuilder()
            .addPathSegment("pdf-jobs")
            .addPathSegment(jobId)
            .addPathSegment("resume")
            .build()
        val request = Request.Builder()
            .url(requestUrl)
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
        val requestUrl = baseUrl(serverUrl).toHttpUrl().newBuilder()
            .addPathSegment("pdf-jobs")
            .addPathSegment(jobId)
            .addPathSegment("review")
            .addPathSegment(pageIndex.toString())
            .build()
        val request = Request.Builder()
            .url(requestUrl)
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
                throw IllegalStateException("HTTP ${response.code}: " + (detail ?: "Tahrir matnini olib bölmadi (${response.code})."))
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
        val requestUrl = baseUrl(serverUrl).toHttpUrl().newBuilder()
            .addPathSegment("pdf-jobs")
            .addPathSegment(jobId)
            .addPathSegment("review")
            .build()
        val request = Request.Builder()
            .url(requestUrl)
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
        val requestUrl = baseUrl(serverUrl).toHttpUrl().newBuilder()
            .addPathSegment("pdf-jobs")
            .addPathSegment(jobId)
            .addPathSegment("download")
            .build()
        val request = Request.Builder()
            .url(requestUrl)
            .apply {
                if (apiKey.isNotBlank()) header("X-API-Key", apiKey)
                header("X-PDF-Job-Token", jobToken)
            }
            .get()
            .build()
        val stagingFile = if (tempDir != null) {
            File.createTempFile("pdf-dl-stage-", ".pdf", tempDir)
        } else {
            File.createTempFile("pdf-dl-stage-", ".pdf")
        }
        try {
            client.newCall(request).readResponse { response ->
                if (!response.isSuccessful) {
                    val detail = response.body?.string().orEmpty()
                    throw IllegalStateException("HTTP ${response.code}: PDFni yuklab bölmadi: $detail")
                }
                val body = response.body ?: throw IllegalStateException("Server böş PDF qaytardi.")
                stagingFile.outputStream().use { fileOut ->
                    body.byteStream().use { input -> input.copyTo(fileOut) }
                }
            }
            if (stagingFile.length() < 5) {
                throw IllegalStateException("Server böş yoki yaroqsiz PDF qaytardi.")
            }
            val header = ByteArray(5)
            val bytesRead = stagingFile.inputStream().use { it.read(header) }
            if (bytesRead < 5 || String(header, Charsets.US_ASCII) != "%PDF-") {
                throw IllegalStateException("Server yaroqsiz PDF formati qaytardi.")
            }
            val output = try {
                contentResolver.openOutputStream(destination, "wt")
            } catch (e: Exception) {
                null
            } ?: contentResolver.openOutputStream(destination, "w")
              ?: throw IllegalStateException("Tanlangan joyga fayl yozib bölmadi.")

            output.use { outStream ->
                stagingFile.inputStream().use { inStream ->
                    inStream.copyTo(outStream)
                }
            }
        } finally {
            stagingFile.delete()
        }
    }

    private suspend fun executeJson(request: Request): PdfTranslationJob {
        return client.newCall(request).readResponse { response ->
            val body = response.body?.string().orEmpty()
            if (!response.isSuccessful) {
                val detail = runCatching { gson.fromJson(body, Map::class.java)["detail"]?.toString() }.getOrNull()
                throw IllegalStateException("HTTP ${response.code}: " + (detail ?: "Server xatosi (${response.code})."))
            }
            val job = gson.fromJson(body, PdfTranslationJob::class.java)
            if (job == null || job.jobId.isNullOrBlank() || job.status.isNullOrBlank()) {
                throw IllegalStateException("Server notöğri javob qaytardi.")
            }
            job
        }
    }
}
