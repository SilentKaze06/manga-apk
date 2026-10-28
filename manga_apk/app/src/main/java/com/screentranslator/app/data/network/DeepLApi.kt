package com.screentranslator.app.data.network

import com.google.gson.annotations.SerializedName
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.Header
import retrofit2.http.POST

data class DeepLRequest(
    @SerializedName("text") val text: List<String>,
    @SerializedName("target_lang") val targetLang: String,
    @SerializedName("source_lang") val sourceLang: String? = null
)

data class DeepLTranslationItem(
    @SerializedName("detected_source_language") val detectedSourceLanguage: String,
    @SerializedName("text") val text: String
)

data class DeepLResponse(
    @SerializedName("translations") val translations: List<DeepLTranslationItem>
)

interface DeepLApi {
    @POST("v2/translate")
    suspend fun translate(
        @Header("Authorization") authHeader: String,
        @Body request: DeepLRequest
    ): Response<DeepLResponse>
}
