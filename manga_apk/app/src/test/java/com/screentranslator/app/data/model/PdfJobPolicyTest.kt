package com.screentranslator.app.data.model

import org.junit.Assert.*
import org.junit.Test

class PdfJobPolicyTest {
    @Test fun allTerminalJobsCanBeReplaced() {
        for (status in listOf("completed", "failed", "canceled")) assertTrue(PdfJobPolicy.canReplace(status))
        for (status in listOf("queued", "processing")) {
            assertFalse(PdfJobPolicy.canReplace(status))
            assertTrue(PdfJobPolicy.isRunning(status))
        }
    }
    @Test fun retriesBackOffWithoutOverflow() {
        assertEquals(1800L, PdfJobPolicy.retryDelayMillis(0))
        assertTrue(PdfJobPolicy.retryDelayMillis(2) > PdfJobPolicy.retryDelayMillis(1))
        assertTrue(PdfJobPolicy.retryDelayMillis(Int.MAX_VALUE) <= 30_000L)
    }
    @Test fun originalFilenameIsPreservedForDownload() {
        assertEquals("Manga 01_tarjima.pdf", PdfJobPolicy.downloadFilename("Manga 01.pdf"))
        assertEquals("kitob_tarjima.pdf", PdfJobPolicy.downloadFilename(""))
        assertEquals("kitob_tarjima.pdf", PdfJobPolicy.downloadFilename("folder/kitob.pdf"))
    }
}
