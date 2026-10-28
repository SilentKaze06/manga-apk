package com.screentranslator.app.data.network

import com.google.gson.annotations.SerializedName
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.POST

data class LibreTranslateRequest(
    @SerializedName("q") val q: String,
    @SerializedName("source") val source: String = "auto",
    @SerializedName("target") val target: String,
    @SerializedName("format") val format: String = "text",
    @SerializedName("api_key") val apiKey: String? = null
)

data class LibreTranslateResponse(
    @SerializedName("translatedText") val translatedText: String?,
    @SerializedName("error") val error: String? = null
)

interface LibreTranslateApi {
    @POST("translate")
    suspend fun translate(
        @Body request: LibreTranslateRequest
    ): Response<LibreTranslateResponse>
}
