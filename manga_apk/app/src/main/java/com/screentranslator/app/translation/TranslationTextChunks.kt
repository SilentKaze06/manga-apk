package com.screentranslator.app.translation

/** Keeps every character and never splits a UTF-16 surrogate pair; API limits are UTF-8 bytes. */
object TranslationTextChunks {
    fun split(text: String, maxBytes: Int): List<String> {
        require(maxBytes >= 4)
        if (text.isEmpty()) return listOf("")
        val chunks = mutableListOf<String>()
        var start = 0
        while (start < text.length) {
            var end = start
            var bytes = 0
            var boundary = -1
            while (end < text.length) {
                val codePoint = text.codePointAt(end)
                val byteCount = when {
                    codePoint <= 0x7f -> 1
                    codePoint <= 0x7ff -> 2
                    codePoint <= 0xffff -> 3
                    else -> 4
                }
                if (bytes + byteCount > maxBytes) break
                bytes += byteCount
                end += Character.charCount(codePoint)
                if (Character.isWhitespace(codePoint)) boundary = end
            }
            if (end < text.length && boundary > start + (end - start) / 2) end = boundary
            chunks += text.substring(start, end)
            start = end
        }
        return chunks
    }
}
