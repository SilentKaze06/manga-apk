package com.screentranslator.app.data.network

import okhttp3.Interceptor
import okhttp3.Response
import java.io.IOException

/**
 * Interceptor that manages HTTP redirects:
 * 1. Enforces followSslRedirects = false (no cross-scheme HTTP <-> HTTPS jumps).
 * 2. Strips sensitive authentication headers (Authorization, X-API-Key, X-PDF-Job-Token, apikey)
 *    when redirecting across different hosts.
 * 3. Throws IOException on redirect loops (>= 10 redirects).
 */
class CrossHostRedirectInterceptor : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        var request = chain.request()
        var response = chain.proceed(request)
        var redirects = 0
        var brokenByPolicy = false
        while (response.isRedirect && redirects < 10) {
            val location = response.header("Location")
            if (location == null) {
                brokenByPolicy = true
                break
            }
            val nextUrl = request.url.resolve(location)
            if (nextUrl == null) {
                brokenByPolicy = true
                break
            }

            // Disable cross-protocol SSL redirects
            if (nextUrl.isHttps != request.url.isHttps) {
                brokenByPolicy = true
                break
            }

            val isCrossHost = !nextUrl.host.equals(request.url.host, ignoreCase = true)
            val newBuilder = request.newBuilder().url(nextUrl)
            if (isCrossHost) {
                newBuilder.removeHeader("Authorization")
                newBuilder.removeHeader("X-API-Key")
                newBuilder.removeHeader("X-PDF-Job-Token")
                newBuilder.removeHeader("apikey")
            }
            if (response.code != 307 && response.code != 308) {
                newBuilder.method("GET", null)
                newBuilder.removeHeader("Transfer-Encoding")
                newBuilder.removeHeader("Content-Length")
                newBuilder.removeHeader("Content-Type")
            }
            request = newBuilder.build()
            response.close()
            response = chain.proceed(request)
            redirects++
        }
        if (!brokenByPolicy && response.isRedirect && redirects >= 10) {
            response.close()
            throw IOException("Too many redirects ($redirects)")
        }
        return response
    }
}
