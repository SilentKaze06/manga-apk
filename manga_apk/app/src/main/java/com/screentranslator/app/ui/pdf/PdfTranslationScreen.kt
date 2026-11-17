package com.screentranslator.app.ui.pdf

import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.MenuBook
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.Cancel
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.CloudDone
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.PictureAsPdf
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.screentranslator.app.BuildConfig
import com.screentranslator.app.data.model.PdfJobPolicy
import com.screentranslator.app.data.model.PdfJobReference
import com.screentranslator.app.data.model.PdfTranslationCorrection
import com.screentranslator.app.data.model.PdfTranslationJob
import com.screentranslator.app.data.model.PdfTranslationReviewBlock
import com.screentranslator.app.data.model.PdfTranslationReviewPage
import com.screentranslator.app.data.network.PdfTranslationClient
import com.screentranslator.app.data.network.ServerEndpoint
import com.screentranslator.app.data.preferences.UserSettings
import com.screentranslator.app.service.PdfTranslationService
import com.screentranslator.app.ui.theme.StudioAccentCyan
import com.screentranslator.app.ui.theme.StudioAccentEmerald
import com.screentranslator.app.ui.theme.StudioBlack
import com.screentranslator.app.ui.theme.StudioBorder
import com.screentranslator.app.ui.theme.StudioBorderFocus
import com.screentranslator.app.ui.theme.StudioBorderSubtle
import com.screentranslator.app.ui.theme.StudioDestructive
import com.screentranslator.app.ui.theme.StudioRaised
import com.screentranslator.app.ui.theme.StudioSurface
import com.screentranslator.app.ui.theme.StudioTextPrimary
import com.screentranslator.app.ui.theme.StudioTextSecondary
import com.screentranslator.app.ui.theme.StudioTextTertiary
import com.screentranslator.app.ui.theme.StudioWarning
import com.screentranslator.app.ui.theme.StudioWhite
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

private val sourceLanguageOptions = listOf(
    "auto" to "Avtomatik aniqlash",
    "en" to "Inglizcha (English)",
    "ru" to "Ruscha (Русский)",
    "fr" to "Fransuzcha (Français)",
    "de" to "Nemischa (Deutsch)",
    "es" to "Ispancha (Español)",
    "pt" to "Portugalcha (Português)",
    "it" to "Italyancha (Italiano)",
    "tr" to "Turkcha (Türkçe)",
    "nl" to "Gollandcha (Nederlands)",
    "pl" to "Polyakcha (Polski)",
    "uz" to "O‘zbekcha (Lotin)",
    "ja" to "Yaponcha (日本語)",
    "ko" to "Koreyscha (한국어)",
    "zh" to "Xitoycha (中文)"
)

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun PdfTranslationScreen(
    settings: UserSettings,
    existingJob: PdfJobReference?,
    onJobChanged: (PdfJobReference?) -> Unit,
    onClose: () -> Unit = {},
    showBackButton: Boolean = false
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val submission: PdfSubmissionViewModel = viewModel()
    val reviewDrafts: PdfReviewDrafts = viewModel()
    var sourceLang by rememberSaveable { mutableStateOf("auto") }
    var documentMode by rememberSaveable { mutableStateOf("comic") }
    var sourceMenuExpanded by remember { mutableStateOf(false) }
    var bilingual by rememberSaveable { mutableStateOf(false) }
    var selectedFileName by rememberSaveable { mutableStateOf("") }
    var currentJob by remember { mutableStateOf<PdfTranslationJob?>(null) }
    var activeJobToken by remember { mutableStateOf(existingJob?.token) }
    var activeJobServerUrl by remember { mutableStateOf(existingJob?.serverUrl) }
    var refreshRequest by remember { mutableStateOf(0) }
    var restoringJob by remember { mutableStateOf(false) }
    val busy = submission.busy
    val downloading = submission.downloading
    val submissionError = submission.error
    val downloadError = submission.downloadError
    val downloadMessage = submission.downloadMessage
    var loadingReview by remember { mutableStateOf(false) }
    var savingReview by remember { mutableStateOf(false) }
    var changingJob by remember { mutableStateOf(false) }
    var reviewPage by remember { mutableStateOf<PdfTranslationReviewPage?>(null) }
    var reviewPageIndex by rememberSaveable { mutableStateOf(0) }
    var message by remember { mutableStateOf<String?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    val editedBlocks = remember(currentJob?.jobId, reviewPage) {
        mutableStateListOf<PdfTranslationReviewBlock>().apply {
            val page = reviewPage
            val jobId = currentJob?.jobId
            addAll(if (page != null && jobId != null) reviewDrafts.get(jobId, page.pageIndex) ?: page.blocks else emptyList())
        }
    }
    val hasUnsavedEdits = reviewPage?.blocks?.let { it != editedBlocks.toList() } == true

    val configuredEndpoint = runCatching { ServerEndpoint.normalize(settings.mangaServerUrl, BuildConfig.DEBUG) }
    val jobEndpointMatches = activeJobServerUrl?.let {
        ServerEndpoint.matches(it, settings.mangaServerUrl, BuildConfig.DEBUG)
    } == true

    fun requireJobEndpoint(): String {
        check(jobEndpointMatches) { "Vazifa boshqa serverga tegishli. Asl server manzilini qaytaring yoki vazifa havolasini tozalang." }
        return activeJobServerUrl!!
    }

    fun clearJob() {
        PdfTranslationService.stop(context)
        currentJob?.jobId?.let(reviewDrafts::clearJob)
        currentJob = null
        activeJobToken = null
        activeJobServerUrl = null
        reviewPage = null
        selectedFileName = ""
        error = null
        onJobChanged(null)
    }

    val backgroundJobState by PdfTranslationService.jobStateFlow.collectAsState()
    LaunchedEffect(backgroundJobState) {
        backgroundJobState?.let { bg ->
            if (activeJobToken != null && bg.jobId == existingJob?.jobId) {
                currentJob = bg
            }
        }
    }

    LaunchedEffect(existingJob, settings.mangaServerUrl, settings.mangaServerApiKey, refreshRequest) {
        val reference = existingJob
        if (reference == null) {
            currentJob = null
            activeJobToken = null
            activeJobServerUrl = null
            PdfTranslationService.stop(context)
            return@LaunchedEffect
        }
        activeJobServerUrl = reference.serverUrl
        activeJobToken = reference.token
        selectedFileName = reference.filename
        if (reference.serverUrl == null || !ServerEndpoint.matches(reference.serverUrl, settings.mangaServerUrl, BuildConfig.DEBUG)) {
            currentJob = null
            error = if (reference.serverUrl == null) "Eski vazifaning server manzili saqlanmagan. Havolani tozalab, PDFni qayta yuboring."
            else "Vazifa ${reference.serverUrl} serveriga tegishli. Asl manzilni qaytaring yoki havolani tozalang."
            return@LaunchedEffect
        }

        PdfTranslationService.start(context, reference)
        var failures = 0
        while (true) {
            restoringJob = currentJob == null
            try {
                val status = PdfTranslationClient.status(reference.serverUrl, settings.mangaServerApiKey, reference.jobId, reference.token)
                currentJob = status
                error = null
                failures = 0
                if (!PdfJobPolicy.isRunning(status.status)) break
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (exception: Exception) {
                failures++
                error = "${exception.message ?: "Vazifa holatini olib bo‘lmadi."} Qayta urinilmoqda."
            } finally {
                restoringJob = false
            }
            delay(PdfJobPolicy.retryDelayMillis(failures))
        }
    }

    LaunchedEffect(currentJob?.jobId, currentJob?.status, activeJobToken, jobEndpointMatches, refreshRequest) {
        val job = currentJob
        val token = activeJobToken
        if (job?.status == "completed" && token != null && jobEndpointMatches) {
            loadingReview = true
            try {
                reviewPageIndex = 0
                reviewPage = PdfTranslationClient.reviewPage(
                    requireJobEndpoint(), settings.mangaServerApiKey, job.jobId, token, 0
                )
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (exception: Exception) {
                error = exception.message ?: "Tahrir oynasini ochib bo'lmadi."
            } finally {
                loadingReview = false
            }
        } else {
            reviewPage = null
        }
    }

    val pdfPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            error = null
            message = null
            submission.submit(uri, settings, documentMode, sourceLang, bilingual)
        }
    }

    val savePdf = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/pdf")) { uri ->
        if (uri != null) {
            submission.clearDownloadStatus()
            val configured = runCatching { ServerEndpoint.normalize(settings.mangaServerUrl, BuildConfig.DEBUG) }.getOrNull()
            submission.savePdf(
                destination = uri,
                apiKey = settings.mangaServerApiKey,
                configuredServerUrl = configured
            )
        } else {
            submission.clearPendingSave()
        }
    }

    if (showBackButton) {
        BackHandler { onClose() }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(StudioBlack)
            .statusBarsPadding()
            .padding(horizontal = 18.dp, vertical = 10.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        // ─── Studio Header ───────────────────────────────────────────────────────────
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
            if (showBackButton) {
                IconButton(onClick = onClose) {
                    Icon(
                        Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Orqaga",
                        tint = StudioTextPrimary
                    )
                }
            }
            Column(modifier = Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        "PDF STUDIO",
                        color = StudioTextPrimary,
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = (-0.3).sp
                    )
                    Spacer(Modifier.width(8.dp))
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(4.dp))
                            .background(StudioAccentEmerald.copy(alpha = 0.12f))
                            .padding(horizontal = 6.dp, vertical = 2.dp)
                    ) {
                        Text(
                            "AI GEMINI",
                            fontSize = 9.sp,
                            fontWeight = FontWeight.Bold,
                            color = StudioAccentEmerald,
                            letterSpacing = 0.5.sp
                        )
                    }
                }
                Text(
                    if (documentMode == "comic") "Komiks, manga va manhwa tarjima laboratoriyasi" else "Kitoblar va ilmiy asarlar tarjima laboratoriyasi",
                    color = StudioTextSecondary,
                    fontSize = 12.sp,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }

        // ─── Real-time Status & Alert Banners ──────────────────────────────────────
        val effectiveError = error ?: submissionError ?: downloadError
        val effectiveMessage = message ?: downloadMessage

        if (!effectiveError.isNullOrBlank()) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(6.dp))
                    .background(StudioDestructive.copy(alpha = 0.15f))
                    .border(1.dp, StudioDestructive.copy(alpha = 0.5f), RoundedCornerShape(6.dp))
                    .padding(12.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.Cancel, contentDescription = null, tint = StudioDestructive, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Text(effectiveError, color = StudioDestructive, fontSize = 12.sp, modifier = Modifier.weight(1f))
                    IconButton(
                        onClick = {
                            error = null
                            submission.clearDownloadStatus()
                        },
                        modifier = Modifier.size(24.dp)
                    ) {
                        Icon(Icons.Default.Close, contentDescription = "Yopish", tint = StudioDestructive, modifier = Modifier.size(14.dp))
                    }
                }
            }
        }

        if (!effectiveMessage.isNullOrBlank()) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(6.dp))
                    .background(StudioAccentEmerald.copy(alpha = 0.15f))
                    .border(1.dp, StudioAccentEmerald.copy(alpha = 0.5f), RoundedCornerShape(6.dp))
                    .padding(12.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.CheckCircle, contentDescription = null, tint = StudioAccentEmerald, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Text(effectiveMessage, color = StudioAccentEmerald, fontSize = 12.sp, modifier = Modifier.weight(1f))
                    IconButton(
                        onClick = {
                            message = null
                            submission.clearDownloadStatus()
                        },
                        modifier = Modifier.size(24.dp)
                    ) {
                        Icon(Icons.Default.Close, contentDescription = "Yopish", tint = StudioAccentEmerald, modifier = Modifier.size(14.dp))
                    }
                }
            }
        }

        // ─── 1. Document Mode Tactile Segmented Switcher ───────────────────────────
        StudioSection(title = "HUJJAT USLUBI") {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(6.dp))
                    .background(StudioRaised)
                    .border(1.dp, StudioBorder, RoundedCornerShape(6.dp))
                    .padding(3.dp),
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                val isBook = documentMode == "book"
                val isComic = documentMode == "comic"

                // Comic Tab
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .clip(RoundedCornerShape(4.dp))
                        .background(if (isComic) StudioWhite else Color.Transparent)
                        .clickable(enabled = currentJob == null && !busy) { documentMode = "comic" }
                        .padding(vertical = 10.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            Icons.AutoMirrored.Filled.MenuBook,
                            contentDescription = null,
                            tint = if (isComic) StudioBlack else StudioTextSecondary,
                            modifier = Modifier.size(15.dp)
                        )
                        Spacer(Modifier.width(6.dp))
                        Text(
                            "Manga / Komiks",
                            fontSize = 12.sp,
                            fontWeight = if (isComic) FontWeight.Bold else FontWeight.Medium,
                            color = if (isComic) StudioBlack else StudioTextSecondary
                        )
                    }
                }

                // Book Tab
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .clip(RoundedCornerShape(4.dp))
                        .background(if (isBook) StudioWhite else Color.Transparent)
                        .clickable(enabled = currentJob == null && !busy) { documentMode = "book" }
                        .padding(vertical = 10.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            Icons.Default.PictureAsPdf,
                            contentDescription = null,
                            tint = if (isBook) StudioBlack else StudioTextSecondary,
                            modifier = Modifier.size(15.dp)
                        )
                        Spacer(Modifier.width(6.dp))
                        Text(
                            "Kitob / Roman",
                            fontSize = 12.sp,
                            fontWeight = if (isBook) FontWeight.Bold else FontWeight.Medium,
                            color = if (isBook) StudioBlack else StudioTextSecondary
                        )
                    }
                }
            }

            Spacer(Modifier.height(8.dp))
            Text(
                if (documentMode == "comic")
                    "Asl rasmlar va personajlar to'liq saqlanadi. Dialog bulutlari (speech bubbles) AI orqali tozalanib, ustiga tarjima qilingan matn kiritiladi."
                else
                    "Matnli va skanerlangan sahifalar aniqlanib, qayta tartiblangan professional o'qiluvchan PDF yaratiladi.",
                color = StudioTextTertiary,
                fontSize = 11.sp,
                lineHeight = 16.sp
            )
        }

        // ─── 2. Language & Translation Flow Configuration ──────────────────────────
        StudioSection(title = "TARJIMA SOZLAMALARI") {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(6.dp))
                    .background(StudioRaised)
                    .border(1.dp, StudioBorder, RoundedCornerShape(6.dp))
                    .clickable { sourceMenuExpanded = true }
                    .padding(horizontal = 14.dp, vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text("Manba tili", fontSize = 10.sp, color = StudioTextTertiary, fontWeight = FontWeight.SemiBold)
                    Text(
                        sourceLanguageOptions.firstOrNull { it.first == sourceLang }?.second ?: "Avtomatik aniqlash",
                        color = StudioTextPrimary,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Medium
                    )
                }
                Icon(Icons.Default.ArrowDropDown, contentDescription = null, tint = StudioTextSecondary)
                DropdownMenu(
                    expanded = sourceMenuExpanded,
                    onDismissRequest = { sourceMenuExpanded = false },
                    modifier = Modifier
                        .background(StudioSurface)
                        .border(1.dp, StudioBorder, RoundedCornerShape(8.dp))
                ) {
                    sourceLanguageOptions.forEach { (code, label) ->
                        DropdownMenuItem(
                            text = { Text(label, color = StudioTextPrimary, fontSize = 13.sp) },
                            onClick = { sourceLang = code; sourceMenuExpanded = false }
                        )
                    }
                }
            }

            Spacer(Modifier.height(8.dp))

            // Bilingual Toggle Row
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(6.dp))
                    .background(StudioRaised)
                    .border(1.dp, StudioBorder, RoundedCornerShape(6.dp))
                    .padding(horizontal = 14.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text("Ikki tilli rejim (Bilingual)", fontSize = 13.sp, color = StudioTextPrimary, fontWeight = FontWeight.Medium)
                    Text("Asl matnni tarjima yonida saqlash", fontSize = 11.sp, color = StudioTextTertiary)
                }
                Switch(
                    checked = bilingual,
                    onCheckedChange = { bilingual = it },
                    enabled = currentJob == null && !busy,
                    colors = SwitchDefaults.colors(
                        checkedThumbColor = StudioWhite,
                        checkedTrackColor = StudioAccentEmerald,
                        uncheckedThumbColor = StudioTextTertiary,
                        uncheckedTrackColor = StudioBorder
                    )
                )
            }
        }

        // ─── 3. Hero Dropzone & Upload Stage / Active Telemetry HUD ─────────────────
        if (currentJob == null) {
            StudioSection(title = "PDF YUKLASH VA ISHGA TUSHIRISH") {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(StudioRaised)
                        .border(1.dp, StudioBorderFocus, RoundedCornerShape(8.dp))
                        .padding(20.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center
                    ) {
                        Box(
                            modifier = Modifier
                                .size(44.dp)
                                .clip(RoundedCornerShape(8.dp))
                                .background(StudioSurface)
                                .border(1.dp, StudioBorder, RoundedCornerShape(8.dp)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                Icons.Default.PictureAsPdf,
                                contentDescription = null,
                                tint = StudioWhite,
                                modifier = Modifier.size(24.dp)
                            )
                        }

                        Spacer(Modifier.height(12.dp))

                        if (restoringJob) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp, color = StudioWhite)
                                Spacer(Modifier.width(8.dp))
                                Text("Avvalgi PDF vazifasi tiklanmoqda…", color = StudioTextPrimary, fontSize = 13.sp)
                            }
                        } else {
                            Text(
                                "PDF faylni tanlang",
                                color = StudioTextPrimary,
                                fontSize = 15.sp,
                                fontWeight = FontWeight.Bold
                            )
                            Spacer(Modifier.height(4.dp))
                            Text(
                                "Maksimal hajm: 50 MB · 500 sahifagacha · AI Gemini Vision",
                                color = StudioTextTertiary,
                                fontSize = 11.sp
                            )
                        }

                        Spacer(Modifier.height(16.dp))

                        // Full-width 52dp thumb-anchored Primary Action Button
                        Button(
                            onClick = { pdfPicker.launch(arrayOf("application/pdf")) },
                            enabled = !busy && !restoringJob && existingJob == null && configuredEndpoint.isSuccess,
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(50.dp),
                            shape = RoundedCornerShape(6.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = StudioWhite,
                                contentColor = StudioBlack,
                                disabledContainerColor = StudioBorder,
                                disabledContentColor = StudioTextTertiary
                            )
                        ) {
                            if (busy) {
                                CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp, color = StudioBlack)
                                Spacer(Modifier.width(8.dp))
                                Text("Serverga uzatilmoqda…", fontWeight = FontWeight.Bold, fontSize = 13.sp)
                            } else {
                                Icon(Icons.Default.FolderOpen, contentDescription = null, modifier = Modifier.size(18.dp))
                                Spacer(Modifier.width(8.dp))
                                Text("PDF Faylni Tanlash", fontWeight = FontWeight.Bold, fontSize = 13.sp)
                            }
                        }

                        if (existingJob != null && !restoringJob) {
                            Spacer(Modifier.height(8.dp))
                            OutlinedButton(
                                onClick = ::clearJob,
                                modifier = Modifier.fillMaxWidth().height(44.dp),
                                shape = RoundedCornerShape(6.dp),
                                border = androidx.compose.foundation.BorderStroke(1.dp, StudioBorder)
                            ) {
                                Text("Saqlangan vazifa havolasini tozalash", color = StudioTextSecondary, fontSize = 12.sp)
                            }
                        }
                    }
                }
            }
        } else {
            // ─── Active Job Telemetry HUD ───────────────────────────────────────────
            val job = currentJob!!
            StudioSection(title = "VAZIFA TELEMETRIYASI") {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(StudioSurface)
                        .border(1.dp, StudioBorder, RoundedCornerShape(8.dp))
                        .padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .clip(CircleShape)
                                .background(
                                    when (job.status) {
                                        "completed" -> StudioAccentEmerald
                                        "processing" -> StudioAccentCyan
                                        "queued" -> StudioWarning
                                        else -> StudioDestructive
                                    }
                                )
                        )
                        Spacer(Modifier.width(8.dp))
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                selectedFileName.ifBlank { job.filename },
                                color = StudioTextPrimary,
                                fontWeight = FontWeight.SemiBold,
                                fontSize = 14.sp,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                            Text(
                                job.message,
                                color = StudioTextSecondary,
                                fontSize = 12.sp
                            )
                        }
                        // Tabular numeric percentage badge
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(4.dp))
                                .background(StudioRaised)
                                .border(1.dp, StudioBorder, RoundedCornerShape(4.dp))
                                .padding(horizontal = 8.dp, vertical = 3.dp)
                        ) {
                            Text(
                                "${job.progress}%",
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold,
                                color = if (job.status == "completed") StudioAccentEmerald else StudioWhite
                            )
                        }
                    }

                    // Telemetry linear progress bar
                    LinearProgressIndicator(
                        progress = { job.progress.coerceIn(0, 100) / 100f },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(4.dp)
                            .clip(RoundedCornerShape(2.dp)),
                        color = if (job.status == "completed") StudioAccentEmerald else StudioWhite,
                        trackColor = StudioRaised
                    )

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            "${job.currentPage} / ${job.totalPages} sahifa bajarildi",
                            color = StudioTextTertiary,
                            fontSize = 11.sp
                        )
                        Text(
                            when (job.status) {
                                "completed" -> "MUVAFFAQ TARJIMA"
                                "processing" -> "AI OCR ISHLAMOQDA"
                                "queued" -> "NAVBATDA"
                                else -> "TO'XTATILGAN"
                            },
                            color = when (job.status) {
                                "completed" -> StudioAccentEmerald
                                "processing" -> StudioAccentCyan
                                else -> StudioTextTertiary
                            },
                            fontSize = 10.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }

                    job.error?.let {
                        Text(
                            it,
                            color = StudioDestructive,
                            fontSize = 11.sp,
                            modifier = Modifier
                                .clip(RoundedCornerShape(4.dp))
                                .background(StudioDestructive.copy(alpha = 0.1f))
                                .padding(8.dp)
                                .fillMaxWidth()
                        )
                    }

                    // Action buttons
                    if (job.status == "processing" || job.status == "queued") {
                        OutlinedButton(
                            onClick = {
                                scope.launch {
                                    val jobToken = activeJobToken ?: return@launch
                                    changingJob = true
                                    try {
                                        PdfTranslationService.stop(context)
                                        currentJob = PdfTranslationClient.cancel(requireJobEndpoint(), settings.mangaServerApiKey, job.jobId, jobToken)
                                        refreshRequest++
                                    } catch (cancelled: CancellationException) {
                                        throw cancelled
                                    } catch (exception: Exception) {
                                        error = exception.message
                                    } finally { changingJob = false }
                                }
                            },
                            enabled = !busy && !changingJob && jobEndpointMatches,
                            modifier = Modifier.fillMaxWidth().height(44.dp),
                            shape = RoundedCornerShape(6.dp),
                            border = androidx.compose.foundation.BorderStroke(1.dp, StudioDestructive.copy(alpha = 0.6f))
                        ) {
                            Icon(Icons.Default.Cancel, contentDescription = null, tint = StudioDestructive, modifier = Modifier.size(16.dp))
                            Spacer(Modifier.width(6.dp))
                            Text("Vazifani To‘xtatish", color = StudioDestructive, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                        }
                    }

                    if (job.status == "completed") {
                        Button(
                            onClick = {
                                val activeToken = activeJobToken ?: ""
                                val activeUrl = activeJobServerUrl ?: ""
                                val filename = selectedFileName.ifBlank { job.filename }
                                submission.prepareSave(
                                    PdfJobReference(
                                        jobId = job.jobId,
                                        token = activeToken,
                                        serverUrl = activeUrl,
                                        filename = filename
                                    )
                                )
                                savePdf.launch(PdfJobPolicy.downloadFilename(filename))
                            },
                            enabled = !downloading && !savingReview && jobEndpointMatches,
                            modifier = Modifier.fillMaxWidth().height(48.dp),
                            shape = RoundedCornerShape(6.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = StudioWhite,
                                contentColor = StudioBlack
                            )
                        ) {
                            Icon(Icons.Default.Download, contentDescription = null, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.width(8.dp))
                            Text(
                                if (downloading) "PDF Saqlanmoqda…" else "Tayyor PDFni Saqlash",
                                fontWeight = FontWeight.Bold,
                                fontSize = 13.sp
                            )
                        }
                    }

                    if (PdfJobPolicy.canReplace(job.status)) {
                        OutlinedButton(
                            onClick = ::clearJob,
                            enabled = !savingReview && !downloading && !changingJob && !hasUnsavedEdits,
                            modifier = Modifier.fillMaxWidth().height(44.dp),
                            shape = RoundedCornerShape(6.dp),
                            border = androidx.compose.foundation.BorderStroke(1.dp, StudioBorder)
                        ) {
                            Icon(Icons.Default.FolderOpen, contentDescription = null, tint = StudioTextSecondary, modifier = Modifier.size(16.dp))
                            Spacer(Modifier.width(6.dp))
                            Text("Boshqa PDF tanlash", color = StudioTextPrimary, fontSize = 12.sp)
                        }
                    }

                    if (job.status == "failed" || job.status == "canceled") {
                        OutlinedButton(
                            onClick = {
                                scope.launch {
                                    val jobToken = activeJobToken ?: return@launch
                                    changingJob = true
                                    try {
                                        currentJob = PdfTranslationClient.resume(requireJobEndpoint(), settings.mangaServerApiKey, job.jobId, jobToken)
                                        val ref = PdfJobReference(job.jobId, jobToken, requireJobEndpoint(), selectedFileName.ifBlank { job.filename })
                                        PdfTranslationService.start(context, ref)
                                        refreshRequest++
                                    } catch (cancelled: CancellationException) {
                                        throw cancelled
                                    } catch (exception: Exception) {
                                        error = exception.message
                                    } finally { changingJob = false }
                                }
                            },
                            enabled = !changingJob && jobEndpointMatches,
                            modifier = Modifier.fillMaxWidth().height(44.dp),
                            shape = RoundedCornerShape(6.dp),
                            border = androidx.compose.foundation.BorderStroke(1.dp, StudioBorderFocus)
                        ) {
                            Icon(Icons.Default.Refresh, contentDescription = null, tint = StudioWhite, modifier = Modifier.size(16.dp))
                            Spacer(Modifier.width(6.dp))
                            Text("Tarjimani Davom Ettirish", color = StudioWhite, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                        }
                    }
                }
            }

            // ─── 4. Page Grid & Review Inspector (When Completed) ───────────────────
            if (job.status == "completed") {
                StudioSection(title = "OCR VA DIALOGLAR INSPEKTORI") {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(8.dp))
                            .background(StudioSurface)
                            .border(1.dp, StudioBorder, RoundedCornerShape(8.dp))
                            .padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Text(
                            "Har bir sahifadagi tanilgan matn va tarjimani bevosita tekshirib, tahrirlashingiz mumkin.",
                            color = StudioTextSecondary,
                            fontSize = 11.sp
                        )

                        // Page Stepper Bar
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            OutlinedButton(
                                onClick = {
                                    val pageNumber = (reviewPage?.pageIndex ?: reviewPageIndex) - 1
                                    val token = activeJobToken ?: return@OutlinedButton
                                    if (pageNumber >= 0) scope.launch {
                                        loadingReview = true
                                        try {
                                            reviewPage = PdfTranslationClient.reviewPage(
                                                requireJobEndpoint(), settings.mangaServerApiKey, job.jobId, token, pageNumber
                                            )
                                            reviewPageIndex = pageNumber
                                        } catch (cancelled: CancellationException) {
                                            throw cancelled
                                        } catch (exception: Exception) {
                                            error = exception.message
                                        } finally { loadingReview = false }
                                    }
                                },
                                enabled = !loadingReview && !savingReview && !hasUnsavedEdits && jobEndpointMatches && (reviewPage?.pageIndex ?: reviewPageIndex) > 0,
                                shape = RoundedCornerShape(4.dp),
                                border = androidx.compose.foundation.BorderStroke(1.dp, StudioBorder)
                            ) { Text("← Oldingi", color = StudioTextPrimary, fontSize = 12.sp) }

                            Text(
                                reviewPage?.let { "${it.pageIndex + 1} / ${it.totalPages} sahifa" } ?: "Yuklanmoqda…",
                                color = StudioWhite,
                                fontWeight = FontWeight.Bold,
                                fontSize = 12.sp,
                                modifier = Modifier.weight(1f),
                                textAlign = androidx.compose.ui.text.style.TextAlign.Center
                            )

                            OutlinedButton(
                                onClick = {
                                    val pageNumber = (reviewPage?.pageIndex ?: reviewPageIndex) + 1
                                    val token = activeJobToken ?: return@OutlinedButton
                                    if (pageNumber < job.totalPages) scope.launch {
                                        loadingReview = true
                                        try {
                                            reviewPage = PdfTranslationClient.reviewPage(
                                                requireJobEndpoint(), settings.mangaServerApiKey, job.jobId, token, pageNumber
                                            )
                                            reviewPageIndex = pageNumber
                                        } catch (cancelled: CancellationException) {
                                            throw cancelled
                                        } catch (exception: Exception) {
                                            error = exception.message
                                        } finally { loadingReview = false }
                                    }
                                },
                                enabled = !loadingReview && !savingReview && !hasUnsavedEdits && jobEndpointMatches && (reviewPage?.pageIndex ?: reviewPageIndex) + 1 < job.totalPages,
                                shape = RoundedCornerShape(4.dp),
                                border = androidx.compose.foundation.BorderStroke(1.dp, StudioBorder)
                            ) { Text("Keyingi →", color = StudioTextPrimary, fontSize = 12.sp) }
                        }

                        if (loadingReview && reviewPage == null) {
                            Box(modifier = Modifier.fillMaxWidth().padding(12.dp), contentAlignment = Alignment.Center) {
                                CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp, color = StudioWhite)
                            }
                        }

                        editedBlocks.forEachIndexed { index, block ->
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clip(RoundedCornerShape(6.dp))
                                    .background(StudioRaised)
                                    .border(1.dp, StudioBorderSubtle, RoundedCornerShape(6.dp))
                                    .padding(12.dp)
                            ) {
                                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Text("DIALOG #${block.blockIndex + 1}", fontSize = 10.sp, fontWeight = FontWeight.Bold, color = StudioTextTertiary)
                                        block.confidence?.let { conf ->
                                            Text(
                                                "OCR: ${(conf * 100).toInt().coerceIn(0, 100)}%",
                                                fontSize = 10.sp,
                                                color = if (conf < 0.6f) StudioDestructive else StudioAccentEmerald,
                                                fontWeight = FontWeight.Bold
                                            )
                                        }
                                    }

                                    OutlinedTextField(
                                        value = block.original,
                                        onValueChange = {
                                            editedBlocks[index] = editedBlocks[index].copy(original = it)
                                            reviewPage?.let { page -> reviewDrafts.save(job.jobId, page.pageIndex, editedBlocks) }
                                        },
                                        label = { Text("Asl matn", fontSize = 11.sp, color = StudioTextTertiary) },
                                        modifier = Modifier.fillMaxWidth(),
                                        colors = OutlinedTextFieldDefaults.colors(
                                            focusedBorderColor = StudioBorderFocus,
                                            unfocusedBorderColor = StudioBorder,
                                            focusedTextColor = StudioTextPrimary,
                                            unfocusedTextColor = StudioTextPrimary,
                                            focusedContainerColor = StudioSurface,
                                            unfocusedContainerColor = StudioSurface
                                        ),
                                        shape = RoundedCornerShape(4.dp)
                                    )

                                    OutlinedTextField(
                                        value = block.translated,
                                        onValueChange = {
                                            editedBlocks[index] = editedBlocks[index].copy(translated = it)
                                            reviewPage?.let { page -> reviewDrafts.save(job.jobId, page.pageIndex, editedBlocks) }
                                        },
                                        label = { Text("Tarjima qilingan matn", fontSize = 11.sp, color = StudioTextTertiary) },
                                        modifier = Modifier.fillMaxWidth(),
                                        colors = OutlinedTextFieldDefaults.colors(
                                            focusedBorderColor = StudioWhite,
                                            unfocusedBorderColor = StudioBorder,
                                            focusedTextColor = StudioTextPrimary,
                                            unfocusedTextColor = StudioTextPrimary,
                                            focusedContainerColor = StudioSurface,
                                            unfocusedContainerColor = StudioSurface
                                        ),
                                        shape = RoundedCornerShape(4.dp)
                                    )
                                }
                            }
                        }

                        if (reviewPage != null && editedBlocks.isNotEmpty()) {
                            val baseline = reviewPage!!.blocks
                            val corrections = editedBlocks.mapIndexedNotNull { index, edited ->
                                val original = baseline.getOrNull(index)
                                if (original == null || (original.original == edited.original && original.translated == edited.translated)) null
                                else PdfTranslationCorrection(
                                    pageIndex = reviewPage!!.pageIndex,
                                    blockIndex = edited.blockIndex,
                                    original = edited.original.takeIf { it != original.original },
                                    translated = edited.translated.takeIf { it != original.translated }
                                )
                            }

                            if (corrections.isNotEmpty()) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                                ) {
                                    OutlinedButton(
                                        onClick = {
                                            reviewPage?.let { page ->
                                                reviewDrafts.clear(job.jobId, page.pageIndex)
                                                editedBlocks.clear()
                                                editedBlocks.addAll(page.blocks)
                                            }
                                        },
                                        enabled = !savingReview && !loadingReview,
                                        modifier = Modifier.weight(1f).height(46.dp),
                                        shape = RoundedCornerShape(6.dp),
                                        border = androidx.compose.foundation.BorderStroke(1.dp, StudioDestructive.copy(alpha = 0.6f))
                                    ) {
                                        Text("Bekor Qilish", color = StudioDestructive, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                                    }

                                    Button(
                                        onClick = {
                                            val token = activeJobToken ?: return@Button
                                            scope.launch {
                                                savingReview = true
                                                error = null
                                                try {
                                                    currentJob = PdfTranslationClient.applyCorrections(
                                                        requireJobEndpoint(), settings.mangaServerApiKey,
                                                        job.jobId, token, corrections
                                                    )
                                                    reviewDrafts.clear(job.jobId, reviewPage!!.pageIndex)
                                                    reviewPage = PdfTranslationClient.reviewPage(
                                                        requireJobEndpoint(), settings.mangaServerApiKey,
                                                        job.jobId, token, reviewPage!!.pageIndex
                                                    )
                                                    message = "Tuzatishlar saqlandi; PDFni qayta yuklab oling."
                                                } catch (cancelled: CancellationException) {
                                                    throw cancelled
                                                } catch (exception: Exception) {
                                                    error = exception.message ?: "Tuzatishlarni saqlab bo'lmadi."
                                                } finally { savingReview = false }
                                            }
                                        },
                                        enabled = !savingReview && !loadingReview && jobEndpointMatches,
                                        modifier = Modifier.weight(1f).height(46.dp),
                                        shape = RoundedCornerShape(6.dp),
                                        colors = ButtonDefaults.buttonColors(
                                            containerColor = StudioWhite,
                                            contentColor = StudioBlack
                                        )
                                    ) {
                                        if (savingReview) CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp, color = StudioBlack)
                                        else Icon(Icons.Default.CheckCircle, contentDescription = null, modifier = Modifier.size(16.dp))
                                        Spacer(Modifier.width(8.dp))
                                        Text("Saqlash", fontWeight = FontWeight.Bold, fontSize = 12.sp)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun StudioSection(
    title: String,
    content: @Composable () -> Unit
) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Text(
            text = title,
            fontSize = 10.sp,
            fontWeight = FontWeight.Bold,
            color = StudioTextTertiary,
            letterSpacing = 1.2.sp,
            modifier = Modifier.padding(start = 2.dp)
        )
        content()
    }
}
