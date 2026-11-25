package com.screentranslator.app.data.network

import okhttp3.Interceptor
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.Protocol
import okhttp3.Request
import okhttp3.Response
import okhttp3.ResponseBody.Companion.toResponseBody
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Test
import java.io.IOException

class CrossHostRedirectInterceptorTest {

    private class FakeChain(
        private val initialRequest: Request,
        private val handler: (Request, Int) -> Response
    ) : Interceptor.Chain {
        private var calls = 0
        val recordedRequests = mutableListOf<Request>()

        override fun request(): Request = initialRequest

        override fun proceed(request: Request): Response {
            recordedRequests += request
            return handler(request, calls++)
        }

        override fun connection() = null
        override fun call() = throw UnsupportedOperationException()
        override fun connectTimeoutMillis() = 1000
        override fun withConnectTimeout(timeout: Int, unit: java.util.concurrent.TimeUnit) = this
        override fun readTimeoutMillis() = 1000
        override fun withReadTimeout(timeout: Int, unit: java.util.concurrent.TimeUnit) = this
        override fun writeTimeoutMillis() = 1000
        override fun withWriteTimeout(timeout: Int, unit: java.util.concurrent.TimeUnit) = this
    }

    private fun response(request: Request, code: Int, location: String? = null): Response {
        val builder = Response.Builder()
            .request(request)
            .protocol(Protocol.HTTP_1_1)
            .code(code)
            .message(if (code in 300..399) "Redirect" else "OK")
            .body("".toResponseBody("text/plain".toMediaType()))
        if (location != null) builder.header("Location", location)
        return builder.build()
    }

    @Test
    fun stripsAuthHeadersOnCrossHostRedirectAndKeepsSameHostHeaders() {
        val initial = Request.Builder()
            .url("https://api.example.com/start")
            .header("Authorization", "Bearer secret")
            .header("X-API-Key", "server-key")
            .header("X-PDF-Job-Token", "job-tok")
            .header("apikey", "ocr-key")
            .build()

        val chain = FakeChain(initial) { req, callIndex ->
            when (callIndex) {
                0 -> response(req, 302, "https://api.example.com/same-host")
                1 -> response(req, 302, "https://cdn.other.com/cross-host")
                else -> response(req, 200)
            }
        }

        val finalResponse = CrossHostRedirectInterceptor().intercept(chain)
        assertEquals(200, finalResponse.code)
        assertEquals(3, chain.recordedRequests.size)

        val sameHostReq = chain.recordedRequests[1]
        assertEquals("Bearer secret", sameHostReq.header("Authorization"))
        assertEquals("server-key", sameHostReq.header("X-API-Key"))

        val crossHostReq = chain.recordedRequests[2]
        assertNull(crossHostReq.header("Authorization"))
        assertNull(crossHostReq.header("X-API-Key"))
        assertNull(crossHostReq.header("X-PDF-Job-Token"))
        assertNull(crossHostReq.header("apikey"))
    }

    @Test
    fun blocksCrossProtocolRedirectAndThrowsOnRedirectLoop() {
        val initial = Request.Builder().url("https://api.example.com/start").build()
        val downgradeChain = FakeChain(initial) { req, _ ->
            response(req, 302, "http://api.example.com/downgrade")
        }
        val downgradeResp = CrossHostRedirectInterceptor().intercept(downgradeChain)
        assertEquals(302, downgradeResp.code)
        assertEquals(1, downgradeChain.recordedRequests.size)

        val loopChain = FakeChain(initial) { req, idx ->
            response(req, 302, "https://api.example.com/hop/$idx")
        }
        assertThrows(IOException::class.java) {
            CrossHostRedirectInterceptor().intercept(loopChain)
        }
    }
}
