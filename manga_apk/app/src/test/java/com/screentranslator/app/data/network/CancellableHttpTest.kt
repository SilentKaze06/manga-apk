package com.screentranslator.app.data.network

import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType
import okhttp3.Protocol
import okhttp3.Request
import okhttp3.Response
import okhttp3.ResponseBody
import okio.Buffer
import okio.BufferedSource
import okio.Source
import okio.Timeout
import okio.buffer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.IOException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class CancellableHttpTest {
    @Test
    fun cancellationDuringBodyReadCancelsCallAndClosesResponse() = runBlocking {
        val reading = CountDownLatch(1)
        val release = CountDownLatch(1)
        val closed = CountDownLatch(1)
        val source = object : Source {
            override fun read(sink: Buffer, byteCount: Long): Long {
                reading.countDown()
                check(release.await(3, TimeUnit.SECONDS)) { "Request was not canceled" }
                throw IOException("Canceled")
            }
            override fun timeout() = Timeout.NONE
            override fun close() { closed.countDown() }
        }
        val call = FakeCall(source.buffer()) { release.countDown() }
        val job = launch(start = CoroutineStart.UNDISPATCHED) {
            call.readResponse { it.body!!.string() }
        }
        assertTrue("Body read must start before cancellation", reading.await(3, TimeUnit.SECONDS))
        job.cancelAndJoin()
        assertTrue(call.isCanceled())
        assertTrue("Response must be closed", closed.await(3, TimeUnit.SECONDS))
    }

    @Test
    fun parsingFailureClosesResponseAndReachesCaller() = runBlocking {
        val closed = CountDownLatch(1)
        val source = object : Source {
            override fun read(sink: Buffer, byteCount: Long) = -1L
            override fun timeout() = Timeout.NONE
            override fun close() { closed.countDown() }
        }
        val failure = runCatching {
            FakeCall(source.buffer()).readResponse<String> { throw IllegalArgumentException("bad response") }
        }.exceptionOrNull()
        assertEquals("bad response", failure?.message)
        assertTrue(closed.await(3, TimeUnit.SECONDS))
    }

    private class FakeCall(
        private val bodySource: BufferedSource,
        private val onCancel: () -> Unit = {}
    ) : Call {
        private val request = Request.Builder().url("https://example.invalid/").build()
        @Volatile private var canceled = false
        @Volatile private var executed = false
        override fun request() = request
        override fun execute(): Response = error("Only async execution is supported")
        override fun enqueue(responseCallback: Callback) {
            executed = true
            Thread {
                val body = object : ResponseBody() {
                    override fun contentType(): MediaType? = null
                    override fun contentLength() = -1L
                    override fun source() = bodySource
                }
                responseCallback.onResponse(this, Response.Builder().request(request)
                    .protocol(Protocol.HTTP_1_1).code(200).message("OK").body(body).build())
            }.apply { isDaemon = true }.start()
        }
        override fun cancel() { canceled = true; onCancel() }
        override fun isExecuted() = executed
        override fun isCanceled() = canceled
        override fun timeout() = Timeout.NONE
        override fun clone(): Call = FakeCall(bodySource, onCancel)
    }
}
