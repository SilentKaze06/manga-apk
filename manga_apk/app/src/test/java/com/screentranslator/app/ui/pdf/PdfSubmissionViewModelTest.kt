package com.screentranslator.app.ui.pdf

import android.app.Application
import android.net.Uri
import com.screentranslator.app.data.model.PdfJobReference
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PdfSubmissionViewModelTest {

    @Test
    fun prepareSave_retainsJobReferenceForScreenRecreation() {
        val app = Application()
        val viewModel = PdfSubmissionViewModel(app)

        val reference = PdfJobReference(
            jobId = "test-job-42",
            token = "secret-token-123",
            serverUrl = "http://10.0.2.2:8000",
            filename = "manga_chapter_1.pdf"
        )

        viewModel.prepareSave(reference)

        assertNotNull(viewModel.pendingSaveJobReference)
        assertEquals("test-job-42", viewModel.pendingSaveJobReference?.jobId)
        assertEquals("secret-token-123", viewModel.pendingSaveJobReference?.token)
        assertEquals("http://10.0.2.2:8000", viewModel.pendingSaveJobReference?.serverUrl)
        assertEquals("manga_chapter_1.pdf", viewModel.pendingSaveJobReference?.filename)

        viewModel.clearPendingSave()
        assertNull(viewModel.pendingSaveJobReference)
    }

    @Test
    fun savePdf_rejectsMismatchedServerEndpoint() {
        val app = Application()
        val viewModel = PdfSubmissionViewModel(app)

        val reference = PdfJobReference(
            jobId = "test-job-42",
            token = "secret-token-123",
            serverUrl = "http://10.0.2.2:8000",
            filename = "manga_chapter_1.pdf"
        )
        viewModel.prepareSave(reference)

        viewModel.savePdf(
            destination = null,
            apiKey = "test-key",
            configuredServerUrl = "http://192.168.1.50:8000"
        )

        assertNotNull(viewModel.downloadError)
        assertTrue(viewModel.downloadError!!.contains("boshqa serverga tegishli"))
    }

    @Test
    fun savePdf_rejectsBlankJobToken() {
        val app = Application()
        val viewModel = PdfSubmissionViewModel(app)

        val reference = PdfJobReference(
            jobId = "test-job-42",
            token = "",
            serverUrl = "http://10.0.2.2:8000",
            filename = "manga_chapter_1.pdf"
        )
        viewModel.prepareSave(reference)

        viewModel.savePdf(
            destination = null,
            apiKey = "test-key",
            configuredServerUrl = "http://10.0.2.2:8000"
        )

        assertEquals("Vazifa kaliti topilmadi.", viewModel.downloadError)
    }

    @Test
    fun savePdf_requiresPendingSaveJobReference() {
        val app = Application()
        val viewModel = PdfSubmissionViewModel(app)

        viewModel.savePdf(
            destination = null,
            apiKey = "test-key",
            configuredServerUrl = "http://10.0.2.2:8000"
        )

        assertEquals("Saqlash uchun faol PDF vazifasi topilmadi.", viewModel.downloadError)
    }
}
