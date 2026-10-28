package com.screentranslator.app.ui.pdf

import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.MenuBook
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.Cancel
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.PictureAsPdf
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.lifecycle.viewmodel.compose.viewModel
import com.screentranslator.app.BuildConfig
import com.screentranslator.app.data.model.PdfJobPolicy
import com.screentranslator.app.data.network.ServerEndpoint
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.screentranslator.app.data.model.PdfTranslationJob
import com.screentranslator.app.data.model.PdfJobReference
import com.screentranslator.app.data.model.PdfTranslationCorrection
import com.screentranslator.app.data.model.PdfTranslationReviewBlock
import com.screentranslator.app.data.model.PdfTranslationReviewPage
import com.screentranslator.app.data.network.PdfTranslationClient
import com.screentranslator.app.data.preferences.UserSettings
import com.screentranslator.app.ui.theme.CyanAccent
import com.screentranslator.app.ui.theme.DarkBackground
import com.screentranslator.app.ui.theme.DarkBorder
import com.screentranslator.app.ui.theme.DarkSurface
import com.screentranslator.app.ui.theme.EmeraldGreen
import com.screentranslator.app.ui.theme.RoseError
import com.screentranslator.app.ui.theme.TextPrimaryDark
import com.screentranslator.app.ui.theme.TextSecondaryDark
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

private val sourceLanguageOptions = listOf(
    "auto" to "Avtomatik aniqlash",
    "en" to "Inglizcha",
    "ru" to "Ruscha",
    "fr" to "Fransuzcha",
    "de" to "Nemischa",
    "es" to "Ispancha",
    "pt" to "Portugalcha",
    "it" to "Italyancha",
    "tr" to "Turkcha",
    "nl" to "Gollandcha",
    "pl" to "Polyakcha",
    "uz" to "O‘zbekcha",
    "ja" to "Yaponcha",
    "ko" to "Koreyscha",
    "zh" to "Xitoycha"
)

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
        currentJob?.jobId?.let(reviewDrafts::clearJob)
        currentJob = null
        activeJobToken = null
        activeJobServerUrl = null
        reviewPage = null
        selectedFileName = ""
        error = null
        onJobChanged(null)
    }

    // One owner for restoration and polling; transient failures retry with bounded backoff.
    LaunchedEffect(existingJob, settings.mangaServerUrl, settings.mangaServerApiKey, refreshRequest) {
        val reference = existingJob
        if (reference == null) {
            currentJob = null
            activeJobToken = null
            activeJobServerUrl = null
            return@LaunchedEffect
        }
        activeJobServerUrl = reference.serverUrl
        activeJobToken = reference.token
        selectedFileName = reference.filename
        if (reference.serverUrl == null || !ServerEndpoint.matches(reference.serverUrl, settings.mangaServerUrl, BuildConfig.DEBUG)) {
            currentJob = null
            error = if (reference.serverUrl == null) "Eski vazifaning server manzili saqlanmagan. Xavfsizlik uchun havolani tozalab, PDFni qayta yuboring."
                else "Vazifa ${reference.serverUrl} serveriga tegishli. Asl manzilni qaytaring yoki havolani tozalang."
            return@LaunchedEffect
        }
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
            .background(DarkBackground)
            .statusBarsPadding()
            .padding(horizontal = 20.dp, vertical = 12.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
            if (showBackButton) {
                IconButton(onClick = onClose) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Orqaga", tint = TextPrimaryDark)
                }
            }
            Column(modifier = Modifier.weight(1f)) {
                Text("PDF tarjima", color = TextPrimaryDark, fontSize = 19.sp, fontWeight = FontWeight.Bold)
                Text(
                    if (documentMode == "book") "Kitoblar va romanlar" else "Manga, manhwa va komiks",
                    color = TextSecondaryDark,
                    style = MaterialTheme.typography.bodySmall
                )
            }
            Icon(Icons.AutoMirrored.Filled.MenuBook, contentDescription = null, tint = CyanAccent, modifier = Modifier.size(24.dp))
        }

        InfoCard {
            Text(
                if (documentMode == "book") "Kitob va roman tarjimasi" else "Manga / manhwa / komiks tarjimasi",
                color = TextPrimaryDark,
                fontWeight = FontWeight.SemiBold
            )
            Spacer(Modifier.height(5.dp))
            Text(
                if (documentMode == "book")
                    "Matnli va skan betlar ajratilib, tarjima qilingan qayta oqimlangan PDF yaratiladi."
                else
                    "Sahifa rasmi saqlanadi; pufak ishonchli aniqlansa tarjima ustiga yoziladi, aks holda tarjima transkriptga qo‘shiladi.",
                color = TextSecondaryDark,
                style = MaterialTheme.typography.bodySmall
            )
            Text(
                "PDF sozlangan serverga yuboriladi; tarjima uchun matn Google xizmatiga uzatiladi. OCR va tarjima natijasini tekshirib chiqing.",
                color = TextSecondaryDark,
                style = MaterialTheme.typography.bodySmall
            )
        }

        InfoCard {
            Text("Hujjat turi", color = TextPrimaryDark, fontWeight = FontWeight.SemiBold)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                FilterChip(
                    selected = documentMode == "book",
                    onClick = { documentMode = "book" },
                    enabled = currentJob == null && !busy,
                    label = { Text("Kitob") }
                )
                FilterChip(
                    selected = documentMode == "comic",
                    onClick = { documentMode = "comic" },
                    enabled = currentJob == null && !busy,
                    label = { Text("Komiks") }
                )
            }
            Text("Manba tili", color = TextPrimaryDark, fontWeight = FontWeight.SemiBold)
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(9.dp))
                    .background(DarkBackground)
                    .clickable { sourceMenuExpanded = true }
                    .padding(horizontal = 12.dp, vertical = 11.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(sourceLanguageOptions.firstOrNull { it.first == sourceLang }?.second ?: "Avtomatik aniqlash", color = TextPrimaryDark, modifier = Modifier.weight(1f))
                Icon(Icons.Default.ArrowDropDown, contentDescription = null, tint = CyanAccent)
                DropdownMenu(expanded = sourceMenuExpanded, onDismissRequest = { sourceMenuExpanded = false }) {
                    sourceLanguageOptions.forEach { (code, label) ->
                        DropdownMenuItem(text = { Text(label) }, onClick = { sourceLang = code; sourceMenuExpanded = false })
                    }
                }
            }
            Text("Tarjima tili: ${settings.targetLanguage.displayName}", color = TextSecondaryDark, style = MaterialTheme.typography.bodySmall)
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                Text("Asl matnni ham qo‘shish", color = TextPrimaryDark, modifier = Modifier.weight(1f))
                Switch(checked = bilingual, onCheckedChange = { bilingual = it }, enabled = currentJob == null && !busy)
            }
        }

        if (currentJob == null) {
            InfoCard {
                Icon(Icons.Default.PictureAsPdf, contentDescription = null, tint = CyanAccent, modifier = Modifier.size(38.dp))
                if (restoringJob) {
                    Spacer(Modifier.height(8.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp, color = CyanAccent)
                        Spacer(Modifier.width(8.dp))
                        Text("Avvalgi PDF vazifasi tiklanmoqda…", color = TextPrimaryDark)
                    }
                } else {
                    Spacer(Modifier.height(8.dp))
                    Text("PDF faylni tanlang", color = TextPrimaryDark, fontWeight = FontWeight.SemiBold)
                    Text("Maksimal hajm: 50 MB · Ko‘pi bilan 500 bet", color = TextSecondaryDark, style = MaterialTheme.typography.bodySmall)
                    Spacer(Modifier.height(10.dp))
                }
                Button(
                    onClick = { pdfPicker.launch(arrayOf("application/pdf")) },
                    enabled = !busy && !restoringJob && existingJob == null && configuredEndpoint.isSuccess,
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(containerColor = CyanAccent, contentColor = Color.White)
                ) {
                    if (busy) CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp, color = Color.White)
                    else Icon(Icons.Default.FolderOpen, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text(if (busy) "Serverga yuborilmoqda…" else "PDF tanlash")
                }
                if (existingJob != null && !restoringJob) {
                    OutlinedButton(onClick = ::clearJob, modifier = Modifier.fillMaxWidth()) {
                        Text("Saqlangan vazifa havolasini tozalash")
                    }
                }
            }
        } else {
            val job = currentJob!!
            InfoCard {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        if (job.status == "completed") Icons.Default.CheckCircle else Icons.Default.PictureAsPdf,
                        contentDescription = null,
                        tint = if (job.status == "completed") EmeraldGreen else CyanAccent,
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(Modifier.width(8.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(selectedFileName.ifBlank { job.filename }, color = TextPrimaryDark, fontWeight = FontWeight.SemiBold, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Text(job.message, color = TextSecondaryDark, style = MaterialTheme.typography.bodySmall)
                    }
                }
                Spacer(Modifier.height(8.dp))
                LinearProgressIndicator(progress = { job.progress.coerceIn(0, 100) / 100f }, modifier = Modifier.fillMaxWidth(), color = CyanAccent)
                Text("${job.currentPage}/${job.totalPages} bet · ${job.progress}%", color = TextSecondaryDark, style = MaterialTheme.typography.labelSmall)
                job.error?.let { Text(it, color = RoseError, style = MaterialTheme.typography.bodySmall) }
                if (job.status == "processing" || job.status == "queued") {
                    OutlinedButton(
                        onClick = {
                            scope.launch {
                                val jobToken = activeJobToken ?: return@launch
                                changingJob = true
                                try {
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
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Icon(Icons.Default.Cancel, contentDescription = null)
                        Spacer(Modifier.width(6.dp))
                        Text("To‘xtatish")
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
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(containerColor = EmeraldGreen, contentColor = Color.White)
                    ) {
                        Icon(Icons.Default.Download, contentDescription = null)
                        Spacer(Modifier.width(7.dp))
                        Text(if (downloading) "Saqlanmoqda…" else "Tarjima PDFni saqlash")
                    }
                }
                if (PdfJobPolicy.canReplace(job.status)) {
                    OutlinedButton(onClick = ::clearJob, enabled = !savingReview && !downloading && !changingJob && !hasUnsavedEdits, modifier = Modifier.fillMaxWidth()) {
                        Icon(Icons.Default.FolderOpen, contentDescription = null)
                        Spacer(Modifier.width(6.dp))
                        Text("Boshqa PDF tanlash")
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
                                    refreshRequest++
                                } catch (cancelled: CancellationException) {
                                    throw cancelled
                                } catch (exception: Exception) {
                                    error = exception.message
                                } finally { changingJob = false }
                            }
                        },
                        enabled = !changingJob && jobEndpointMatches,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Icon(Icons.Default.Refresh, contentDescription = null)
                        Spacer(Modifier.width(6.dp))
                        Text("Davom ettirish")
                    }
                }
            }
            if (job.status == "completed") {
                InfoCard {
                    Text("OCR va tarjimani tekshirish", color = TextPrimaryDark, fontWeight = FontWeight.SemiBold)
                    Text(
                        "Xato tanilgan asl matn yoki tarjimani tuzating. Saqlash PDFni qayta yaratadi.",
                        color = TextSecondaryDark,
                        style = MaterialTheme.typography.bodySmall
                    )
                    if (hasUnsavedEdits) {
                        Text("Betni almashtirishdan oldin tuzatishlarni saqlang yoki bekor qiling.", color = CyanAccent, style = MaterialTheme.typography.bodySmall)
                        OutlinedButton(onClick = {
                            reviewPage?.let { reviewDrafts.clear(job.jobId, it.pageIndex) }
                            editedBlocks.clear()
                            editedBlocks.addAll(reviewPage?.blocks.orEmpty())
                        }, enabled = !savingReview) {
                            Text("Tuzatishlarni bekor qilish")
                        }
                    }
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
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
                            enabled = !loadingReview && !savingReview && !hasUnsavedEdits && jobEndpointMatches && (reviewPage?.pageIndex ?: reviewPageIndex) > 0
                        ) { Text("Oldingi bet") }
                        Text(
                            reviewPage?.let { "${it.pageIndex + 1}/${it.totalPages} bet" } ?: "Bet yuklanmoqda…",
                            color = TextSecondaryDark,
                            modifier = Modifier.weight(1f)
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
                            enabled = !loadingReview && !savingReview && !hasUnsavedEdits && jobEndpointMatches && (reviewPage?.pageIndex ?: reviewPageIndex) + 1 < job.totalPages
                        ) { Text("Keyingi bet") }
                    }
                    reviewPage?.warning?.let {
                        Text(it, color = RoseError, style = MaterialTheme.typography.bodySmall)
                    }
                    if (loadingReview && reviewPage == null) {
                        CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp, color = CyanAccent)
                    }
                    if (reviewPage != null && editedBlocks.isEmpty()) {
                        Text("Bu betda tarjima qilinadigan matn aniqlanmadi.", color = TextSecondaryDark)
                    }
                    editedBlocks.forEachIndexed { index, block ->
                        Spacer(Modifier.height(8.dp))
                        Text("Matn ${block.blockIndex + 1}", color = CyanAccent, fontWeight = FontWeight.SemiBold)
                        block.confidence?.let { confidence ->
                            Text(
                                "OCR ishonchi: ${(confidence * 100).toInt().coerceIn(0, 100)}%",
                                color = if (confidence < 0.6f) RoseError else TextSecondaryDark,
                                style = MaterialTheme.typography.labelSmall
                            )
                        }
                        OutlinedTextField(
                            value = block.original,
                            onValueChange = {
                                editedBlocks[index] = editedBlocks[index].copy(original = it)
                                reviewPage?.let { page -> reviewDrafts.save(job.jobId, page.pageIndex, editedBlocks) }
                            },
                            label = { Text("Asl matn / OCR") },
                            modifier = Modifier.fillMaxWidth(),
                            minLines = 2,
                            enabled = !savingReview && !loadingReview
                        )
                        OutlinedTextField(
                            value = block.translated,
                            onValueChange = {
                                editedBlocks[index] = editedBlocks[index].copy(translated = it)
                                reviewPage?.let { page -> reviewDrafts.save(job.jobId, page.pageIndex, editedBlocks) }
                            },
                            label = { Text("Tarjima") },
                            modifier = Modifier.fillMaxWidth(),
                            minLines = 2,
                            enabled = !savingReview && !loadingReview
                        )
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
                            enabled = corrections.isNotEmpty() && !savingReview && !loadingReview && jobEndpointMatches,
                            modifier = Modifier.fillMaxWidth(),
                            colors = ButtonDefaults.buttonColors(containerColor = CyanAccent, contentColor = Color.White)
                        ) {
                            if (savingReview) CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp, color = Color.White)
                            else Icon(Icons.Default.CheckCircle, contentDescription = null)
                            Spacer(Modifier.width(7.dp))
                            Text(if (savingReview) "Qayta yaratilmoqda…" else "Tuzatishlarni PDFga saqlash")
                        }
                    }
                }
            }
        }

        if (downloading) Row(verticalAlignment = Alignment.CenterVertically) {
            CircularProgressIndicator(modifier = Modifier.size(17.dp), strokeWidth = 2.dp, color = CyanAccent)
            Spacer(Modifier.width(8.dp))
            Text("PDF qurilmaga ko‘chirilmoqda…", color = TextSecondaryDark)
        }
        message?.let { Text(it, color = EmeraldGreen, style = MaterialTheme.typography.bodySmall) }
        submission.downloadMessage?.let { Text(it, color = EmeraldGreen, style = MaterialTheme.typography.bodySmall) }
        error?.let { Text(it, color = RoseError, style = MaterialTheme.typography.bodySmall) }
        submission.downloadError?.let { Text(it, color = RoseError, style = MaterialTheme.typography.bodySmall) }
        submission.error?.let { Text(it, color = RoseError, style = MaterialTheme.typography.bodySmall) }
        configuredEndpoint.exceptionOrNull()?.message?.let { Text(it, color = RoseError, style = MaterialTheme.typography.bodySmall) }
        if (existingJob != null && jobEndpointMatches) {
            OutlinedButton(onClick = { refreshRequest++ }, enabled = !changingJob && !savingReview && !hasUnsavedEdits) {
                Icon(Icons.Default.Refresh, contentDescription = null)
                Spacer(Modifier.width(6.dp))
                Text("Holatni yangilash")
            }
        }

        Spacer(Modifier.height(4.dp))
        Text(
            "Katta kitoblar server orqali tarjima qilinadi. Internet aloqasini uzmang.",
            color = TextSecondaryDark,
            style = MaterialTheme.typography.labelSmall
        )
    }
}

@Composable
private fun InfoCard(content: @Composable () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(DarkSurface)
            .border(1.dp, DarkBorder, RoundedCornerShape(12.dp))
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) { content() }
}

