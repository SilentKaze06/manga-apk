package com.screentranslator.app.translation

import com.screentranslator.app.data.network.MyMemoryApi
import com.screentranslator.app.data.network.MyMemoryResponse
import com.screentranslator.app.data.network.MyMemoryResponseData
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import retrofit2.Response

class TranslationTextChunksTest {
    @Test fun multilingualChunksKeepEveryCharacterAndRespectByteLimit() {
        val text = "Hello 日本語 안녕하세요 😀 O‘zbekcha. ".repeat(100)
        val parts = TranslationTextChunks.split(text, 500)
        assertEquals(text, parts.joinToString(""))
        assertTrue(parts.size > 1)
        parts.forEach {
            assertTrue(it.toByteArray(Charsets.UTF_8).size <= 500)
            assertFalse(Character.isLowSurrogate(it.first()))
            assertFalse(Character.isHighSurrogate(it.last()))
        }
    }
    @Test fun longTranslationSendsAndReturnsEveryChunk() = runBlocking {
        val inputs = mutableListOf<String>()
        val api = object : MyMemoryApi {
            override suspend fun translate(query: String, langPair: String): Response<MyMemoryResponse> {
                inputs += query
                return Response.success(MyMemoryResponse(MyMemoryResponseData("T${inputs.size}", null), 200, null))
            }
        }
        val text = "A long paragraph. ".repeat(120)
        val result = MyMemoryTranslationEngine(api).translate(text, "en", "uz").getOrThrow()
        assertEquals(text, inputs.joinToString(""))
        assertEquals(inputs.indices.joinToString("\n") { "T${it + 1}" }, result)
    }
    @Test fun failedChunkDoesNotReturnPartialSuccess() = runBlocking {
        var calls = 0
        val api = object : MyMemoryApi {
            override suspend fun translate(query: String, langPair: String): Response<MyMemoryResponse> {
                calls++
                return Response.success(MyMemoryResponse(MyMemoryResponseData("part", null), if (calls == 2) 403 else 200, "quota"))
            }
        }
        assertTrue(MyMemoryTranslationEngine(api).translate("a".repeat(1200), "en", "uz").isFailure)
        assertEquals(2, calls)
    }
    @Test fun cancellationIsNotConvertedToTranslationFailure() {
        val api = object : MyMemoryApi {
            override suspend fun translate(query: String, langPair: String): Response<MyMemoryResponse> = throw CancellationException()
        }
        assertThrows(CancellationException::class.java) {
            runBlocking { MyMemoryTranslationEngine(api).translate("hello", "en", "uz") }
        }
    }
}
