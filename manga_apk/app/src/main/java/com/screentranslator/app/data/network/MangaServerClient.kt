package com.screentranslator.app.data.network

import android.graphics.Bitmap
import android.util.Log
import com.screentranslator.app.BuildConfig
import kotlinx.coroutines.CancellationException
import com.google.gson.Gson
import com.screentranslator.app.data.model.MangaDetectAndTranslateResponse
import com.screentranslator.app.data.model.MangaServerResponse
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit

/**
 * Client for communicating with the Python Manga OCR server (manga-ocr & PaddleOCR).
 */
object MangaServerClient {

    private const val TAG = "MangaServerClient"
    private val gson = Gson()

    private val httpClient: OkHttpClient by lazy {
        OkHttpClient.Builder()
            .connectTimeout(8, TimeUnit.SECONDS)
            // OCR models may download/load on the first request, and full pages take longer.
            .readTimeout(300, TimeUnit.SECONDS)
            .writeTimeout(25, TimeUnit.SECONDS)
            .retryOnConnectionFailure(true)
            .followRedirects(true)
            .followSslRedirects(true)
            .build()
    }

    private fun normalizeUrl(url: String): String {
        return ServerEndpoint.normalize(url, BuildConfig.DEBUG)
    }

    /**
     * Sends a cropped speech bubble image (and optional extracted text) to /translate-manga.
     * When text is already extracted on-device, passing it allows the server to skip heavy OCR
     * and respond instantly (<200ms) without CPU lockup or timeouts.
     */
    suspend fun translateBubble(
        serverUrl: String,
        bitmap: Bitmap,
        text: String = "",
        sourceLang: String = "ja",
        targetLang: String = "uz",
        apiKey: String = ""
    ): Result<MangaServerResponse> = withContext(Dispatchers.IO) {
        try {
            val base = normalizeUrl(serverUrl)
            val endpoint = "$base/translate-manga"

            val stream = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, 92, stream)
            val imageBytes = stream.toByteArray()

            val bodyBuilder = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "file", "bubble.jpg",
                    imageBytes.toRequestBody("image/jpeg".toMediaTypeOrNull())
                )
                .addFormDataPart("lang", sourceLang)
                .addFormDataPart("target_lang", targetLang)

            if (text.isNotBlank()) {
                bodyBuilder.addFormDataPart("text", text)
            }
            val requestBody = bodyBuilder.build()

            val request = Request.Builder()
                .url(endpoint)
                .apply { if (apiKey.isNotBlank()) header("X-API-Key", apiKey) }
                .post(requestBody)
                .build()

            httpClient.newCall(request).readResponse { response ->
                val bodyString = response.body?.string()
                if (response.isSuccessful && !bodyString.isNullOrBlank()) {
                    val parsed = gson.fromJson(bodyString, MangaServerResponse::class.java)
                    Result.success(parsed)
                } else {
                    val errorMsg = "Server xatosi (${response.code}): ${bodyString ?: "Noma'lum"}"
                    Log.e(TAG, errorMsg)
                    Result.failure(Exception(errorMsg))
                }
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "MangaServer translateBubble xatosi: ${e.message}", e)
            Result.failure(e)
        }
    }

    /**
     * Sends the entire manga screen capture to /detect-and-translate.
     * The server uses PaddleOCR text detection and Manga-OCR recognition to extract and translate
     * all speech bubbles in one pass.
     */
    suspend fun detectAndTranslateScreen(
        serverUrl: String,
        bitmap: Bitmap,
        sourceLang: String = "ja",
        targetLang: String = "uz",
        apiKey: String = ""
    ): Result<MangaDetectAndTranslateResponse> = withContext(Dispatchers.IO) {
        try {
            val base = normalizeUrl(serverUrl)
            val endpoint = "$base/detect-and-translate"

            val stream = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, stream)
            val imageBytes = stream.toByteArray()

            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "file", "screen.jpg",
                    imageBytes.toRequestBody("image/jpeg".toMediaTypeOrNull())
                )
                .addFormDataPart("lang", sourceLang)
                .addFormDataPart("target_lang", targetLang)
                .build()

            val request = Request.Builder()
                .url(endpoint)
                .apply { if (apiKey.isNotBlank()) header("X-API-Key", apiKey) }
                .post(requestBody)
                .build()

            httpClient.newCall(request).readResponse { response ->
                val bodyString = response.body?.string()
                if (response.isSuccessful && !bodyString.isNullOrBlank()) {
                    val parsed = gson.fromJson(bodyString, MangaDetectAndTranslateResponse::class.java)
                    Result.success(parsed)
                } else {
                    val errorMsg = "Server xatosi (${response.code})"
                    Result.failure(Exception(errorMsg))
                }
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "MangaServer detectAndTranslateScreen xatosi: ${e.message}", e)
            Result.failure(e)
        }
    }

    /**
     * Tests server connectivity via /health.
     */
    suspend fun checkHealth(serverUrl: String, apiKey: String = ""): Result<String> = withContext(Dispatchers.IO) {
        try {
            val base = normalizeUrl(serverUrl)
            val endpoint = "$base/health"

            val request = Request.Builder()
                .url(endpoint)
                .apply { if (apiKey.isNotBlank()) header("X-API-Key", apiKey) }
                .get()
                .build()

            httpClient.newCall(request).readResponse { response ->
                val bodyString = response.body?.string()
                if (response.isSuccessful && !bodyString.isNullOrBlank()) {
                    Result.success(bodyString)
                } else {
                    Result.failure(Exception("HTTP ${response.code} javobi qaytdi"))
                }
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
