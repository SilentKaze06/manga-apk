package com.screentranslator.app.data.network

import com.google.gson.annotations.SerializedName
import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.Response
import retrofit2.http.Header
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

/**
 * Retrofit interface for the OCR.space Free/Pro API.
 * Endpoint: https://api.ocr.space/parse/image
 */
interface OcrSpaceApi {

    @Multipart
    @POST("parse/image")
    suspend fun parseImage(
        @Header("apikey") apiKey: String,
        @Part file: MultipartBody.Part,
        @Part("isOverlayRequired") isOverlayRequired: RequestBody,
        @Part("scale") scale: RequestBody? = null,
        @Part("language") language: RequestBody? = null,
        @Part("OCREngine") ocrEngine: RequestBody? = null
    ): Response<OcrSpaceResponse>
}

data class OcrSpaceResponse(
    @SerializedName("ParsedResults") val parsedResults: List<OcrSpaceParsedResult>?,
    @SerializedName("OCRExitCode") val exitCode: Int?,
    @SerializedName("IsErroredOnProcessing") val isErrored: Boolean?,
    @SerializedName("ErrorMessage") val errorMessage: List<String>?,
    @SerializedName("ErrorDetails") val errorDetails: String?
)

data class OcrSpaceParsedResult(
    @SerializedName("TextOverlay") val textOverlay: OcrSpaceTextOverlay?,
    @SerializedName("ParsedText") val parsedText: String?,
    @SerializedName("ErrorMessage") val errorMessage: String?,
    @SerializedName("ErrorDetails") val errorDetails: String?
)

data class OcrSpaceTextOverlay(
    @SerializedName("Lines") val lines: List<OcrSpaceLine>?,
    @SerializedName("HasOverlay") val hasOverlay: Boolean?,
    @SerializedName("Message") val message: String?
)

data class OcrSpaceLine(
    @SerializedName("LineText") val lineText: String?,
    @SerializedName("Words") val words: List<OcrSpaceWord>?,
    @SerializedName("MaxHeight") val maxHeight: Float?,
    @SerializedName("MinTop") val minTop: Float?
)

data class OcrSpaceWord(
    @SerializedName("WordText") val wordText: String?,
    @SerializedName("Left") val left: Float,
    @SerializedName("Top") val top: Float,
    @SerializedName("Height") val height: Float,
    @SerializedName("Width") val width: Float
)
