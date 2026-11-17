package com.screentranslator.app.data.network

import com.screentranslator.app.BuildConfig
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

/**
 * Singleton factory for Retrofit API clients.
 * Configured with timeouts and logging.
 */
object NetworkClient {

    private const val TIMEOUT_SECONDS = 12L

    private val okHttpClient: OkHttpClient by lazy {
        val builder = OkHttpClient.Builder()
            .connectTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .readTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .writeTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .followRedirects(true)
            .followSslRedirects(true)

        // ✅ TUZATILDI: HTTP loglash faqat debug rejimida ishlaydi
        // Release versiyada tarmoq so'rovlari Logcat'ga yozilmaydi
        if (BuildConfig.DEBUG) {
            val logging = HttpLoggingInterceptor().apply {
                level = HttpLoggingInterceptor.Level.BASIC
            }
            builder.addInterceptor(logging)
        }

        builder.build()
    }

    val deepLApi: DeepLApi by lazy {
        Retrofit.Builder()
            .baseUrl("https://api-free.deepl.com/")
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(DeepLApi::class.java)
    }

    val myMemoryApi: MyMemoryApi by lazy {
        Retrofit.Builder()
            .baseUrl("https://api.mymemory.translated.net/")
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(MyMemoryApi::class.java)
    }

    val ocrSpaceApi: OcrSpaceApi by lazy {
        Retrofit.Builder()
            .baseUrl("https://api.ocr.space/")
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(OcrSpaceApi::class.java)
    }

    private val libreTranslateApiCache = java.util.concurrent.ConcurrentHashMap<String, LibreTranslateApi>()

    /**
     * Builds a LibreTranslate API client for a custom or default URL, cached by sanitized URL.
     */
    fun createLibreTranslateApi(baseUrl: String): LibreTranslateApi {
        val sanitizedUrl = ServerEndpoint.normalize(baseUrl, BuildConfig.DEBUG) + "/"
        return libreTranslateApiCache.computeIfAbsent(sanitizedUrl) { url ->
            Retrofit.Builder()
                .baseUrl(url)
                .client(okHttpClient)
                .addConverterFactory(GsonConverterFactory.create())
                .build()
                .create(LibreTranslateApi::class.java)
        }
    }
}
