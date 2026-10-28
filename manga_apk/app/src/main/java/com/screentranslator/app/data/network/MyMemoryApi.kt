package com.screentranslator.app.data.network

import com.google.gson.annotations.SerializedName
import retrofit2.Response
import retrofit2.http.GET
import retrofit2.http.Query

data class MyMemoryResponseData(
    @SerializedName("translatedText") val translatedText: String,
    @SerializedName("match") val match: Double?
)

data class MyMemoryResponse(
    @SerializedName("responseData") val responseData: MyMemoryResponseData?,
    @SerializedName("responseStatus") val responseStatus: Int,
    @SerializedName("responseDetails") val responseDetails: String?
)

interface MyMemoryApi {
    @GET("get")
    suspend fun translate(
        @Query("q") query: String,
        @Query("langpair") langPair: String // e.g., "en|ru"
    ): Response<MyMemoryResponse>
}
