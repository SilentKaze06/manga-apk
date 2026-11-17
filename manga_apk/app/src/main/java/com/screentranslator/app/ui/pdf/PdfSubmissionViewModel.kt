package com.screentranslator.app.ui.pdf

import android.app.Application
import android.net.Uri
import android.provider.OpenableColumns
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.screentranslator.app.BuildConfig
import com.screentranslator.app.data.model.PdfJobReference
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
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.coroutines.coroutineContext

/** Upload ownership survives tab changes and activity configuration changes. */
class PdfSubmissionViewModel(application: Application) : AndroidViewModel(application) {
    var busy by mutableStateOf(false)
        private set
    var error by mutableStateOf<String?>(null)
        private set

    var pendingSaveJobReference by mutableStateOf<PdfJobReference?>(null)
        private set
    var downloading by mutableStateOf(false)
        private set
    var downloadMessage by mutableStateOf<String?>(null)
        private set
    var downloadError by mutableStateOf<String?>(null)
        private set

    fun prepareSave(jobRef: PdfJobReference) {
        pendingSaveJobReference = jobRef
        downloadError = null
        downloadMessage = null
    }

    fun clearPendingSave() {
        pendingSaveJobReference = null
    }

    fun clearDownloadStatus() {
        downloadMessage = null
        downloadError = null
    }

    fun clearError() {
        error = null
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
        val jobToken = jobRef.token
        if (jobToken.isBlank()) {
            downloadError = "Vazifa kaliti topilmadi."
            return
        }
        if (configuredServerUrl == null || !ServerEndpoint.matches(serverUrl, configuredServerUrl, BuildConfig.DEBUG)) {
            downloadError = "Vazifa başqa serverga tegişli. Asl server manzilini qaytaring yoki vazifa havolasini tozalang."
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
                pendingSaveJobReference = null
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
                PdfTranslationService.start(getApplication(), jobRef)
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
