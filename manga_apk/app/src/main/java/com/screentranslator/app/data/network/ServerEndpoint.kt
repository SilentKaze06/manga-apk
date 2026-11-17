package com.screentranslator.app.data.network

import okhttp3.HttpUrl.Companion.toHttpUrlOrNull

/** Canonical endpoint identity and transport policy, shared by all configurable servers. */
object ServerEndpoint {
    fun normalize(value: String, allowLocalHttp: Boolean = false): String {
        val input = value.trim()
        require(input.isNotEmpty()) { "Server manzilini Server & API bölimida kiriting." }
        val url = (if ("://" in input) input else "https://$input").toHttpUrlOrNull()
            ?: throw IllegalArgumentException("Server manzili notöğri.")
        require(url.username.isEmpty() && url.password.isEmpty() && url.query == null && url.fragment == null) {
            "Server manzilida login, parol, sörov yoki # belgisi bölmasligi kerak."
        }
        require(url.scheme == "https" || (allowLocalHttp && isLocalHost(url.host))) {
            "HTTPS manzil kerak. HTTP faqat sinov versiyasida lokal IP manzillar uçun ruxsat etiladi."
        }
        return url.toString().trimEnd('/')
    }

    fun matches(first: String, second: String, allowLocalHttp: Boolean = false): Boolean =
        runCatching { normalize(first, allowLocalHttp) == normalize(second, allowLocalHttp) }.getOrDefault(false)

    private fun isLocalHost(host: String): Boolean {
        val cleanHost = host.removeSurrounding("[", "]")
        if (cleanHost == "localhost" || cleanHost == "::1") return true
        val parts = cleanHost.split('.').map { it.toIntOrNull() ?: return false }
        if (parts.size != 4 || parts.any { it !in 0..255 }) return false
        return parts[0] == 10 || parts[0] == 127 ||
            (parts[0] == 192 && parts[1] == 168) ||
            (parts[0] == 172 && parts[1] in 16..31) ||
            (parts[0] == 100 && parts[1] in 64..127)
    }
}
