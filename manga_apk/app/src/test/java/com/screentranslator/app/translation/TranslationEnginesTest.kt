package com.screentranslator.app.translation

import com.screentranslator.app.data.network.DeepLApi
import com.screentranslator.app.data.network.DeepLRequest
import com.screentranslator.app.data.network.DeepLResponse
import com.screentranslator.app.data.network.MyMemoryApi
import com.screentranslator.app.data.network.MyMemoryResponse
import com.screentranslator.app.data.network.MyMemoryResponseData
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import retrofit2.Response

class TranslationEnginesTest {

    @Test
    fun deepLSkipsNetworkCallWhenSourceEqualsTarget() = runBlocking {
        var called = false
        val api = object : DeepLApi {
            override suspend fun translate(authKey: String, request: DeepLRequest): Response<DeepLResponse> {
                called = true
                throw AssertionError("Network should not be called when source == target")
            }
        }
        val engine = DeepLTranslationEngine(api) { "test-key:fx" }
        val result = engine.translate("Hello world", "en", "en")
        assertTrue(result.isSuccess)
        assertEquals("Hello world", result.getOrNull())
        assertEquals(false, called)
    }

    @Test
    fun myMemoryUnescapesHtmlEntitiesInTranslatedText() = runBlocking {
        val api = object : MyMemoryApi {
            override suspend fun translate(query: String, langPair: String): Response<MyMemoryResponse> {
                return Response.success(
                    MyMemoryResponse(
                        MyMemoryResponseData("Don&#39;t say &quot;Hello &amp; Bye&quot; &lt;ok&gt;", null),
                        200,
                        null
                    )
                )
            }
        }
        val result = MyMemoryTranslationEngine(api).translate("Salom", "uz", "en").getOrThrow()
        assertEquals("Don't say \"Hello & Bye\" <ok>", result)
    }

    @Test
    fun cjkChunkSplittingBreaksAtSentencePunctuationAndPreservesAllChars() {
        val sentence = "これは日本語のテストです。次の文もとても重要です！"
        val text = sentence.repeat(12)
        val chunks = TranslationTextChunks.split(text, 120)

        assertTrue(chunks.size > 1)
        assertEquals(text, chunks.joinToString(""))
        chunks.forEach { chunk ->
            assertTrue(chunk.toByteArray(Charsets.UTF_8).size <= 120)
        }
        assertTrue(
            "Chunks should break cleanly after Japanese punctuation when possible",
            chunks.first().endsWith("。") || chunks.first().endsWith("！")
        )
    }
}
