package com.screentranslator.app.ui.pdf

import android.app.Application
import android.net.Uri
import android.provider.OpenableColumns
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.viewModelScope
import com.screentranslator.app.BuildConfig
import com.screentranslator.app.data.model.PdfJobPolicy
import com.screentranslator.app.data.model.PdfJobReference
import com.screentranslator.app.data.model.PdfTranslationCorrection
import com.screentranslator.app.data.model.PdfTranslationJob
import com.screentranslator.app.data.model.PdfTranslationReviewPage
import com.screentranslator.app.data.model.TranslationHistoryEntry
import com.screentranslator.app.data.network.PdfTranslationClient
import com.screentranslator.app.data.network.ServerEndpoint
import com.screentranslator.app.data.preferences.PdfJobRepository
import com.screentranslator.app.data.preferences.TranslationHistoryRepository
import com.screentranslator.app.data.preferences.UserSettings
import com.screentranslator.app.service.PdfTranslationService
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.coroutines.coroutineContext

/** Upload ownership survives tab changes and activity configuration changes. */
class PdfSubmissionViewModel(
    application: Application,
    private val savedStateHandle: SavedStateHandle = SavedStateHandle()
) : AndroidViewModel(application) {
    var busy by mutableStateOf(false)
        private set
    var error by mutableStateOf<String?>(null)
        private set

    var pendingSaveJobReference by mutableStateOf<PdfJobReference?>(
        savedStateHandle.get<String>("pending_job_id")?.let { jobId ->
            PdfJobReference(
                jobId = jobId,
                token = "", // Excluded from SavedStateHandle for security
                serverUrl = savedStateHandle.get<String>("pending_server_url"),
                filename = savedStateHandle.get<String>("pending_filename").orEmpty()
            )
        }
    )
        private set
    var downloading by mutableStateOf(false)
        private set
    var downloadMessage by mutableStateOf<String?>(null)
        private set
    var downloadError by mutableStateOf<String?>(null)
        private set
    var changingJob by mutableStateOf(false)
        private set
    var savingReview by mutableStateOf(false)
        private set
    var jobActionError by mutableStateOf<String?>(null)
        private set
    var jobActionMessage by mutableStateOf<String?>(null)
        private set
    var lastActionJob by mutableStateOf<PdfTranslationJob?>(null)
        private set
    var lastReviewPage by mutableStateOf<PdfTranslationReviewPage?>(null)
        private set
    var reviewPageCleared by mutableStateOf(false)
        private set
    var actionVersion by mutableStateOf(0)
        private set

    fun prepareSave(jobRef: PdfJobReference) {
        pendingSaveJobReference = jobRef
        savedStateHandle["pending_job_id"] = jobRef.jobId
        savedStateHandle["pending_server_url"] = jobRef.serverUrl
        savedStateHandle["pending_filename"] = jobRef.filename
        downloadError = null
        downloadMessage = null
    }

    fun clearPendingSave() {
        pendingSaveJobReference = null
        savedStateHandle.remove<String>("pending_job_id")
        savedStateHandle.remove<String>("pending_server_url")
        savedStateHandle.remove<String>("pending_filename")
    }

    fun clearDownloadStatus() {
        downloadMessage = null
        downloadError = null
    }

    fun clearError() {
        error = null
        jobActionError = null
        jobActionMessage = null
    }

    fun cancelJob(
        serverUrl: String,
        apiKey: String,
        jobId: String,
        jobToken: String
    ) {
        if (changingJob) return
        changingJob = true
        jobActionError = null
        jobActionMessage = null
        viewModelScope.launch {
            try {
                val updated = PdfTranslationClient.cancel(serverUrl, apiKey, jobId, jobToken)
                withContext(NonCancellable) {
                    PdfTranslationService.stop(getApplication())
                }
                lastActionJob = updated
                actionVersion++
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (exception: Exception) {
                jobActionError = exception.message
            } finally {
                changingJob = false
            }
        }
    }

    fun resumeJob(
        serverUrl: String,
        apiKey: String,
        jobId: String,
        jobToken: String,
        filename: String
    ) {
        if (changingJob) return
        changingJob = true
        jobActionError = null
        jobActionMessage = null
        viewModelScope.launch {
            try {
                val updated = PdfTranslationClient.resume(serverUrl, apiKey, jobId, jobToken)
                val ref = PdfJobReference(jobId, jobToken, serverUrl, filename)
                runCatching {
                    PdfTranslationService.start(getApplication(), ref)
                }
                lastActionJob = updated
                actionVersion++
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (exception: Exception) {
                jobActionError = exception.message
            } finally {
                changingJob = false
            }
        }
    }

    fun applyCorrections(
        serverUrl: String,
        apiKey: String,
        jobId: String,
        jobToken: String,
        filename: String,
        pageIndex: Int,
        corrections: List<PdfTranslationCorrection>,
        reviewDrafts: PdfReviewDrafts
    ) {
        if (savingReview) return
        savingReview = true
        jobActionError = null
        jobActionMessage = null
        viewModelScope.launch {
            try {
                val updatedJob = PdfTranslationClient.applyCorrections(
                    serverUrl, apiKey, jobId, jobToken, corrections
                )
                lastActionJob = updatedJob
                reviewDrafts.clear(jobId, pageIndex)
                if (PdfJobPolicy.isRunning(updatedJob.status)) {
                    reviewPageCleared = true
                    lastReviewPage = null
                    val ref = PdfJobReference(updatedJob.jobId, jobToken, serverUrl, filename)
                    runCatching {
                        PdfTranslationService.start(getApplication(), ref)
                    }
                    jobActionMessage = "Tuzatişlar qabul qilindi; PDF qayta yiğilmoqda…"
                } else {
                    reviewPageCleared = false
                    lastReviewPage = PdfTranslationClient.reviewPage(
                        serverUrl, apiKey, jobId, jobToken, pageIndex
                    )
                    jobActionMessage = "Tuzatişlar saqlandi; PDFni qayta yuklab oling."
                }
                actionVersion++
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (exception: Exception) {
                jobActionError = exception.message ?: "Tuzatişlarni saqlab bölmadi."
            } finally {
                savingReview = false
            }
        }
    }

    fun savePdf(
        destination: Uri?,
        apiKey: String,
        configuredServerUrl: String?
    ) {
        val jobRef = pendingSaveJobReference ?: run {
            downloadError = "Saqlaş uçun faol PDF vazifasi topilmadi."
            return
        }
        val serverUrl = jobRef.serverUrl ?: run {
            downloadError = "Vazifa server manzili mavjud emas."
            return
        }
        val jobId = jobRef.jobId
        val initialToken = jobRef.token
        val appContext = getApplication<Application>().applicationContext
        if (configuredServerUrl == null || !ServerEndpoint.matches(serverUrl, configuredServerUrl, BuildConfig.DEBUG)) {
            downloadError = "Vazifa başqa serverga tegişli. Asl server manzilini qaytaring yoki vazifa havolasini tozalang."
            return
        }
        if (initialToken.isBlank() && appContext == null) {
            downloadError = "Vazifa kaliti topilmadi."
            return
        }
        if (destination == null) {
            downloadError = "Tanlangan fayl joyi mavjud emas."
            return
        }

        if (downloading) return
        downloading = true
        downloadError = null
        downloadMessage = null

        viewModelScope.launch {
            try {
                val jobToken = if (initialToken.isNotBlank()) {
                    initialToken
                } else {
                    val persisted = runCatching {
                        PdfJobRepository(appContext!!).activeJob.first()
                    }.getOrNull()
                    val rehydrated = persisted?.takeIf { it.jobId == jobId }?.token.orEmpty()
                    if (rehydrated.isBlank()) {
                        downloadError = "Vazifa kaliti topilmadi."
                        return@launch
                    }
                    pendingSaveJobReference = jobRef.copy(token = rehydrated)
                    rehydrated
                }

                PdfTranslationClient.download(
                    serverUrl = serverUrl,
                    apiKey = apiKey,
                    jobId = jobId,
                    jobToken = jobToken,
                    contentResolver = getApplication<Application>().contentResolver,
                    destination = destination,
                    tempDir = getApplication<Application>().cacheDir
                )
                downloadMessage = "Tarjima PDF tanlangan joyga saqlandi."
                runCatching {
                    TranslationHistoryRepository.getInstance(getApplication()).addEntry(
                        TranslationHistoryEntry(
                            id = java.util.UUID.randomUUID().toString(),
                            originalText = "PDF: ${jobRef.filename.ifBlank { "Hujjat" }}",
                            translatedText = "PDF tarjimasi töliq yakunlanib, qurilmaga saqlandi.",
                            targetLanguage = "PDF",
                            imagePath = null
                        )
                    )
                }
                clearPendingSave()
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (exception: Exception) {
                downloadError = exception.message ?: "PDFni saqlab bölmadi."
            } finally {
                downloading = false
            }
        }
    }

    fun submit(
        uri: Uri,
        settings: UserSettings,
        documentMode: String,
        sourceLang: String,
        bilingual: Boolean,
        onSubmitted: ((PdfJobReference) -> Unit)? = null
    ) {
        if (busy) return
        busy = true
        error = null
        viewModelScope.launch {
            var temporary: File? = null
            try {
                val endpoint = ServerEndpoint.normalize(settings.mangaServerUrl, BuildConfig.DEBUG)
                temporary = File.createTempFile("pdf-translation-", ".pdf", getApplication<Application>().cacheDir)
                val picked = copyPdfToCache(uri, temporary)
                val submitted = PdfTranslationClient.submit(
                    endpoint, settings.mangaServerApiKey, picked.first, picked.second,
                    documentMode, sourceLang, settings.targetLanguage.code,
                    if (bilingual) "bilingual" else "translated"
                )
                val token = submitted.jobToken
                    ?: throw IllegalStateException("Server vazifa kalitini qaytarmadi.")
                val jobRef = PdfJobReference(submitted.jobId, token, endpoint, picked.second)
                // Once received, durably keep the capability even if this ViewModel is cleared.
                withContext(NonCancellable) {
                    PdfJobRepository(getApplication()).save(jobRef)
                }
                runCatching {
                    PdfTranslationService.start(getApplication(), jobRef)
                }.onFailure { e ->
                    android.util.Log.w("PdfSubmissionViewModel", "PdfTranslationService start xatosi: ${e.message}")
                }
                onSubmitted?.invoke(jobRef)
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (exception: Exception) {
                error = exception.message ?: "PDFni tarjimaga yuborib bölmadi."
            } finally {
                temporary?.delete()
                busy = false
            }
        }
    }

    private suspend fun copyPdfToCache(uri: Uri, temp: File): Pair<File, String> = withContext(Dispatchers.IO) {
        val context = getApplication<Application>()
        try {
            val displayName = try {
                context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
                    ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }
                    ?.takeIf { it.isNotBlank() }?.substringAfterLast('/')?.substringAfterLast('\\') ?: "kitob.pdf"
            } catch (e: Exception) {
                "kitob.pdf"
            }

            val input = context.contentResolver.openInputStream(uri)
                ?: throw IllegalStateException("Tanlangan PDF faylni öqib bölmadi.")
            input.use { source ->
                temp.outputStream().use { destination ->
                    val buffer = ByteArray(64 * 1024)
                    var total = 0L
                    while (true) {
                        coroutineContext.ensureActive()
                        val count = source.read(buffer)
                        if (count < 0) break
                        total += count
                        require(total <= 50L * 1024 * 1024) { "PDF 50 MB dan katta bölmasligi kerak." }
                        destination.write(buffer, 0, count)
                    }
                }
            }
            if (temp.length() == 0L) {
                throw IllegalStateException("Tanlangan PDF fayl böş (0 bayt).")
            }
            temp.inputStream().use { stream ->
                val headerBuffer = ByteArray(1024)
                val bytesRead = stream.read(headerBuffer)
                val headerString = if (bytesRead > 0) String(headerBuffer, 0, bytesRead, Charsets.ISO_8859_1) else ""
                require(headerString.contains("%PDF-")) { "Tanlangan fayl PDF formatida emas." }
            }
            temp to displayName
        } catch (exception: Exception) {
            temp.delete()
            throw exception
        }
    }
}
