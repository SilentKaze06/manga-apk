package com.screentranslator.app.ui.settings

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.activity.compose.BackHandler
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.MenuBook
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Language
import androidx.compose.material.icons.filled.Translate
import androidx.compose.material.icons.filled.VpnKey
import androidx.compose.material.icons.filled.ZoomIn
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.screentranslator.app.data.model.MangaSourceLanguage
import com.screentranslator.app.data.model.TargetLanguage
import com.screentranslator.app.data.model.TranslationEngineType
import com.screentranslator.app.data.model.TranslationHistoryEntry
import com.screentranslator.app.data.model.TranslationMode
import com.screentranslator.app.ui.pdf.PdfTranslationScreen
import com.screentranslator.app.ui.theme.StudioAccentCyan
import com.screentranslator.app.ui.theme.StudioAccentEmerald
import com.screentranslator.app.ui.theme.StudioBlack
import com.screentranslator.app.ui.theme.StudioBorder
import com.screentranslator.app.ui.theme.StudioBorderFocus
import com.screentranslator.app.ui.theme.StudioBorderSubtle
import com.screentranslator.app.ui.theme.StudioDestructive
import com.screentranslator.app.ui.theme.StudioRaised
import com.screentranslator.app.ui.theme.StudioSurface
import com.screentranslator.app.ui.theme.StudioSurfaceHover
import com.screentranslator.app.ui.theme.StudioTextPrimary
import com.screentranslator.app.ui.theme.StudioTextSecondary
import com.screentranslator.app.ui.theme.StudioTextTertiary
import com.screentranslator.app.ui.theme.StudioWarning
import com.screentranslator.app.ui.theme.StudioWhite
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

// ─── Studio Navigation Tabs ──────────────────────────────────────────────────
private enum class NavTab(val icon: ImageVector, val label: String) {
    HOME(Icons.AutoMirrored.Filled.MenuBook, "PDF Studio"),
    HISTORY(Icons.Default.History, "Tarix"),
    API(Icons.Default.VpnKey, "Server & API"),
    ABOUT(Icons.Default.Info, "Haqida")
}

// ─── Main Screen Container ───────────────────────────────────────────────────
@Composable
fun SettingsScreen(
    viewModel: SettingsViewModel,
    onStartScreenTranslation: () -> Unit = {},
    onStopScreenTranslation: () -> Unit = {},
    isScreenTranslationRunning: Boolean = false
) {
    val settings by viewModel.userSettings.collectAsStateWithLifecycle()
    val activePdfJob by viewModel.activePdfJob.collectAsStateWithLifecycle()
    val history by viewModel.translationHistory.collectAsStateWithLifecycle()
    val apiSaveSuccess by viewModel.apiSaveSuccess.collectAsStateWithLifecycle()
    val serverHealth by viewModel.serverHealth.collectAsStateWithLifecycle()

    var selectedTab by rememberSaveable { mutableIntStateOf(0) }
    var languageDropdownExpanded by remember { mutableStateOf(false) }
    var viewingImagePath by rememberSaveable { mutableStateOf<String?>(null) }

    BackHandler(enabled = viewingImagePath != null || selectedTab != 0) {
        if (viewingImagePath != null) {
            viewingImagePath = null
        } else if (selectedTab != 0) {
            selectedTab = 0
        }
    }

    // Full-screen image viewer modal dialog
    viewingImagePath?.let { path ->
        FullImageViewerDialog(
            imagePath = path,
            onDismiss = { viewingImagePath = null }
        )
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(StudioBlack)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding()
        ) {
            // Precision Header: Studio Brand & Target Language Selector
            StudioHeader(
                selectedLanguage = settings.targetLanguage,
                expanded = languageDropdownExpanded,
                onExpandChange = { languageDropdownExpanded = it },
                onLanguageSelected = { lang ->
                    viewModel.setTargetLanguage(lang)
                    languageDropdownExpanded = false
                }
            )

            // Animated Tab Content Area
            AnimatedContent(
                targetState = selectedTab,
                transitionSpec = {
                    (fadeIn(tween(220)) + slideInVertically { it / 14 })
                        .togetherWith(fadeOut(tween(160)) + slideOutVertically { -it / 14 })
                },
                label = "tab_content",
                modifier = Modifier.weight(1f)
            ) { tab ->
                when (NavTab.entries[tab]) {
                    NavTab.HOME -> PdfTranslationScreen(
                        settings = settings,
                        existingJob = activePdfJob,
                        onJobChanged = viewModel::savePdfJob,
                        onClose = {},
                        showBackButton = false
                    )
                    NavTab.HISTORY -> HistoryTabContent(
                        history = history,
                        onClearHistory = { viewModel.clearHistory() },
                        onRefreshHistory = { viewModel.refreshHistory() },
                        onViewImage = { viewingImagePath = it }
                    )
                    NavTab.API -> ApiTabContent(
                        settings = settings,
                        apiSaveSuccess = apiSaveSuccess,
                        serverHealth = serverHealth,
                        isScreenTranslationRunning = isScreenTranslationRunning,
                        onStartScreenTranslation = onStartScreenTranslation,
                        onStopScreenTranslation = onStopScreenTranslation,
                        onToggleTranslationMode = viewModel::setTranslationMode,
                        onSelectMangaSourceLanguage = viewModel::setMangaSourceLanguage,
                        onToggleUseMangaServer = { use ->
                            viewModel.updateMangaServerSettings(settings.mangaServerUrl, use, settings.mangaServerApiKey)
                        },
                        onSaveApiSettings = { engine, deepl, libreUrl, libreKey, ocrKey, mangaUrl, useServer, mangaKey ->
                            viewModel.saveApiSettings(engine, deepl, libreUrl, libreKey, ocrKey, mangaUrl, useServer, mangaKey)
                        },
                        onResetSaveSuccess = { viewModel.resetApiSaveSuccess() },
                        onResetHealthStatus = { viewModel.clearServerHealthStatus() },
                        onTestMangaServer = { url, key -> viewModel.testMangaServerConnection(url, key) }
                    )
                    NavTab.ABOUT -> AboutTabContent()
                }
            }

            // Anchored Floating Studio Dock (Bottom Navigation Bar)
            BottomNavBar(
                selectedIndex = selectedTab,
                onTabSelected = {
                    selectedTab = it
                    if (NavTab.entries[it] == NavTab.HISTORY) {
                        viewModel.refreshHistory()
                    }
                }
            )
        }
    }
}

// ─── Studio Header Bar ───────────────────────────────────────────────────────
@Composable
private fun StudioHeader(
    selectedLanguage: TargetLanguage,
    expanded: Boolean,
    onExpandChange: (Boolean) -> Unit,
    onLanguageSelected: (TargetLanguage) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .height(56.dp)
            .background(StudioBlack)
            .border(BorderStroke(1.dp, StudioBorderSubtle))
            .padding(horizontal = 18.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(32.dp)
                    .clip(RoundedCornerShape(6.dp))
                    .background(StudioRaised)
                    .border(1.dp, StudioBorder, RoundedCornerShape(6.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    Icons.Default.Translate,
                    contentDescription = null,
                    tint = StudioWhite,
                    modifier = Modifier.size(16.dp)
                )
            }
            Spacer(modifier = Modifier.width(10.dp))
            Column {
                Text(
                    "MANGA STUDIO",
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold,
                    color = StudioTextPrimary,
                    letterSpacing = 0.5.sp
                )
                Text(
                    "AI PDF Localizer",
                    fontSize = 10.sp,
                    color = StudioTextTertiary
                )
            }
        }

        // Language Selector Pill
        Box {
            Row(
                modifier = Modifier
                    .clip(RoundedCornerShape(6.dp))
                    .background(StudioRaised)
                    .border(1.dp, StudioBorder, RoundedCornerShape(6.dp))
                    .clickable(
                        interactionSource = remember { MutableInteractionSource() },
                        indication = null
                    ) { onExpandChange(true) }
                    .padding(horizontal = 10.dp, vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    Icons.Default.Language,
                    contentDescription = null,
                    tint = StudioTextSecondary,
                    modifier = Modifier.size(14.dp)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = selectedLanguage.displayName.split(" ").first(),
                    fontSize = 12.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = StudioWhite
                )
                Icon(
                    Icons.Default.ArrowDropDown,
                    contentDescription = null,
                    tint = StudioTextTertiary,
                    modifier = Modifier.size(16.dp)
                )
            }

            DropdownMenu(
                expanded = expanded,
                onDismissRequest = { onExpandChange(false) },
                modifier = Modifier
                    .background(StudioSurface)
                    .border(1.dp, StudioBorder, RoundedCornerShape(6.dp))
            ) {
                TargetLanguage.entries.forEach { lang ->
                    DropdownMenuItem(
                        text = {
                            Text(
                                text = lang.displayName,
                                fontSize = 12.sp,
                                color = if (lang == selectedLanguage) StudioWhite else StudioTextSecondary,
                                fontWeight = if (lang == selectedLanguage) FontWeight.Bold else FontWeight.Normal
                            )
                        },
                        onClick = { onLanguageSelected(lang) }
                    )
                }
            }
        }
    }
}

// ─── Tab 2: Tarix (Translation History) ───────────────────────────────────────
@Composable
private fun HistoryTabContent(
    history: List<TranslationHistoryEntry>,
    onClearHistory: () -> Unit,
    onRefreshHistory: () -> Unit,
    onViewImage: (String) -> Unit
) {
    LaunchedEffect(Unit) {
        onRefreshHistory()
    }

    val dateFormat = remember { SimpleDateFormat("dd MMM, HH:mm", Locale.getDefault()) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 18.dp, vertical = 10.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    "TARJIMALAR TARIXI",
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold,
                    color = StudioTextTertiary,
                    letterSpacing = 1.2.sp
                )
                Text(
                    if (history.isEmpty()) "Heç qanday yozuv yöq" else "${history.size} ta tarjima saqlangan",
                    fontSize = 13.sp,
                    color = StudioTextSecondary
                )
            }

            if (history.isNotEmpty()) {
                IconButton(
                    onClick = onClearHistory,
                    modifier = Modifier
                        .size(32.dp)
                        .clip(RoundedCornerShape(6.dp))
                        .background(StudioRaised)
                        .border(1.dp, StudioBorder, RoundedCornerShape(6.dp))
                ) {
                    Icon(
                        Icons.Default.Delete,
                        contentDescription = "Tarixni tozalaş",
                        tint = StudioDestructive,
                        modifier = Modifier.size(16.dp)
                    )
                }
            }
        }

        if (history.isEmpty()) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(bottom = 60.dp),
                contentAlignment = Alignment.Center
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(48.dp)
                            .clip(CircleShape)
                            .background(StudioSurface)
                            .border(1.dp, StudioBorder, CircleShape),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            Icons.Default.History,
                            contentDescription = null,
                            tint = StudioTextTertiary,
                            modifier = Modifier.size(24.dp)
                        )
                    }
                    Text(
                        "Hozirça tarix böş",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = StudioTextSecondary
                    )
                    Text(
                        "Ekran yoki PDF tarjima qilinganda saqlanadi",
                        fontSize = 11.sp,
                        color = StudioTextTertiary
                    )
                }
            }
        } else {
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.spacedBy(10.dp),
                contentPadding = PaddingValues(bottom = 70.dp)
            ) {
                items(history, key = { it.id }) { entry ->
                    HistoryItem(entry, dateFormat, onViewImage)
                }
            }
        }
    }
}

@Composable
private fun HistoryItem(
    entry: TranslationHistoryEntry,
    dateFormat: SimpleDateFormat,
    onViewImage: (String) -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(6.dp))
            .background(StudioSurface)
            .border(1.dp, StudioBorder, RoundedCornerShape(6.dp))
            .padding(14.dp)
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            // Header Row: Time and Target Language Badge
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    dateFormat.format(Date(entry.timestamp)),
                    fontSize = 10.sp,
                    color = StudioTextTertiary,
                    fontWeight = FontWeight.Medium
                )
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(3.dp))
                        .background(StudioRaised)
                        .border(1.dp, StudioBorderSubtle, RoundedCornerShape(3.dp))
                        .padding(horizontal = 6.dp, vertical = 2.dp)
                ) {
                    Text(
                        entry.targetLanguage.uppercase(),
                        fontSize = 9.sp,
                        fontWeight = FontWeight.Bold,
                        color = StudioAccentEmerald
                    )
                }
            }

            // Image snapshot decoded off the main thread with produceState
            val imagePath = entry.imagePath
            val imageFile = remember(imagePath) { imagePath?.let { File(it) } }
            if (imageFile != null && imageFile.exists()) {
                Spacer(modifier = Modifier.height(6.dp))
                val bitmap by produceState<Bitmap?>(initialValue = null, key1 = imageFile.absolutePath) {
                    value = withContext(Dispatchers.IO) {
                        try {
                            val options = BitmapFactory.Options().apply { inSampleSize = 2 }
                            BitmapFactory.decodeFile(imageFile.absolutePath, options)
                        } catch (t: Throwable) {
                            null
                        }
                    }
                }

                if (bitmap != null) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(150.dp)
                            .clip(RoundedCornerShape(4.dp))
                            .border(1.dp, StudioBorder, RoundedCornerShape(4.dp))
                            .clickable { onViewImage(imageFile.absolutePath) }
                    ) {
                        Image(
                            bitmap = bitmap!!.asImageBitmap(),
                            contentDescription = "Skrinşot",
                            contentScale = ContentScale.Crop,
                            modifier = Modifier.fillMaxSize()
                        )
                        Box(
                            modifier = Modifier
                                .align(Alignment.BottomEnd)
                                .padding(6.dp)
                                .clip(RoundedCornerShape(3.dp))
                                .background(StudioBlack.copy(alpha = 0.75f))
                                .padding(4.dp)
                        ) {
                            Icon(
                                Icons.Default.ZoomIn,
                                contentDescription = "Kattalaş",
                                tint = StudioWhite,
                                modifier = Modifier.size(14.dp)
                            )
                        }
                    }
                }
            }

            // Original vs Translated Text
            Text(
                entry.originalText,
                fontSize = 12.sp,
                color = StudioTextSecondary,
                lineHeight = 16.sp,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )

            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(modifier = Modifier.weight(1f).height(1.dp).background(StudioBorderSubtle))
                Text(
                    " TARJIMA ",
                    fontSize = 8.sp,
                    fontWeight = FontWeight.Bold,
                    color = StudioTextTertiary
                )
                Box(modifier = Modifier.weight(1f).height(1.dp).background(StudioBorderSubtle))
            }

            Text(
                entry.translatedText,
                fontSize = 13.sp,
                color = StudioWhite,
                fontWeight = FontWeight.Medium,
                maxLines = 3,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

// ─── Tab 3: Server & API Sozlamalari ──────────────────────────────────────────
@Composable
private fun ApiTabContent(
    settings: com.screentranslator.app.data.preferences.UserSettings,
    apiSaveSuccess: Boolean,
    serverHealth: SettingsViewModel.ServerHealthStatus?,
    isScreenTranslationRunning: Boolean,
    onStartScreenTranslation: () -> Unit,
    onStopScreenTranslation: () -> Unit,
    onToggleTranslationMode: (TranslationMode) -> Unit,
    onSelectMangaSourceLanguage: (MangaSourceLanguage) -> Unit,
    onToggleUseMangaServer: (Boolean) -> Unit,
    onSaveApiSettings: (TranslationEngineType, String, String, String, String, String, Boolean, String) -> Unit,
    onResetSaveSuccess: () -> Unit,
    onResetHealthStatus: () -> Unit,
    onTestMangaServer: (String, String) -> Unit
) {
    var draftUrl by rememberSaveable(settings.mangaServerUrl) { mutableStateOf(settings.mangaServerUrl) }
    var draftKey by rememberSaveable(settings.mangaServerApiKey) { mutableStateOf(settings.mangaServerApiKey) }
    val endpoint = runCatching {
        com.screentranslator.app.data.network.ServerEndpoint.normalize(draftUrl, com.screentranslator.app.BuildConfig.DEBUG)
    }
    val dirty = draftUrl != settings.mangaServerUrl || draftKey != settings.mangaServerApiKey

    LaunchedEffect(draftUrl, draftKey) {
        onResetHealthStatus()
    }

    LaunchedEffect(apiSaveSuccess) {
        if (apiSaveSuccess) { delay(2000); onResetSaveSuccess() }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 18.dp, vertical = 10.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Column {
            Text(
                "SERVER VA API SOZLAMALARI",
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                color = StudioTextTertiary,
                letterSpacing = 1.2.sp
            )
            Text(
                "Masofaviy AI va GPU OCR serverini başqariş",
                fontSize = 13.sp,
                color = StudioTextSecondary
            )
        }

        // ─── Live Screen Translation Card ──────────────────────────────
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(8.dp))
                .background(StudioSurface)
                .border(1.dp, if (isScreenTranslationRunning) StudioAccentEmerald else StudioBorder, RoundedCornerShape(8.dp))
                .padding(14.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        "JONLI EKRAN TARJIMASI",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        color = StudioTextPrimary
                    )
                    Text(
                        if (isScreenTranslationRunning) "Xizmat faol (suzuvçi tugma ekranda)" else "Suzuvçi tugma orqali ekranni tezkor tarjima qiliş",
                        fontSize = 11.sp,
                        color = if (isScreenTranslationRunning) StudioAccentEmerald else StudioTextTertiary
                    )
                }
                Spacer(Modifier.width(8.dp))
                Button(
                    onClick = {
                        if (isScreenTranslationRunning) onStopScreenTranslation()
                        else onStartScreenTranslation()
                    },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (isScreenTranslationRunning) StudioDestructive else StudioWhite,
                        contentColor = if (isScreenTranslationRunning) StudioWhite else StudioBlack
                    ),
                    shape = RoundedCornerShape(6.dp)
                ) {
                    Text(
                        if (isScreenTranslationRunning) "Töxtatiş" else "Başlaş",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }

        // ─── Translation Mode & Manga Options ──────────────────────────
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(8.dp))
                .background(StudioSurface)
                .border(1.dp, StudioBorder, RoundedCornerShape(8.dp))
                .padding(14.dp)
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text(
                    "TARJIMA REJIMI VA MANGA SOZLAMALARI",
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold,
                    color = StudioTextTertiary,
                    letterSpacing = 1.sp
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text("Manga / Komiks rejimi", color = StudioTextPrimary, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                        Text("Dialog bulutlarini ajratiş va vertikal matnni qöllab-quvvatlaş", color = StudioTextTertiary, fontSize = 11.sp)
                    }
                    Switch(
                        checked = settings.translationMode == TranslationMode.MANGA,
                        onCheckedChange = { isManga ->
                            onToggleTranslationMode(if (isManga) TranslationMode.MANGA else TranslationMode.STANDARD)
                        }
                    )
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text("Masofaviy Manga Server", color = StudioTextPrimary, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                        Text("Manga-OCR (ViT) va çuqur dialog modeli", color = StudioTextTertiary, fontSize = 11.sp)
                    }
                    Switch(
                        checked = settings.useMangaServer,
                        onCheckedChange = { onToggleUseMangaServer(it) }
                    )
                }

                // Manga Source Language Selector
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("Asl til (Manga / Manhwa):", color = StudioTextSecondary, fontSize = 11.sp)
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        listOf(
                            MangaSourceLanguage.AUTO to "Auto",
                            MangaSourceLanguage.JAPANESE to "JA",
                            MangaSourceLanguage.KOREAN to "KO",
                            MangaSourceLanguage.CHINESE to "ZH",
                            MangaSourceLanguage.ENGLISH to "EN"
                        ).forEach { (lang, label) ->
                            val isSel = settings.mangaSourceLanguage == lang
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .clip(RoundedCornerShape(4.dp))
                                    .background(if (isSel) StudioWhite else StudioRaised)
                                    .border(1.dp, if (isSel) StudioWhite else StudioBorder, RoundedCornerShape(4.dp))
                                    .clickable { onSelectMangaSourceLanguage(lang) }
                                    .padding(vertical = 8.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    label,
                                    fontSize = 11.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = if (isSel) StudioBlack else StudioWhite
                                )
                            }
                        }
                    }
                }
            }
        }

        // Quick Preset Switcher (No hardcoded live servers or keys)
        Text("TEZKOR SERVER TANLAŞ:", fontSize = 10.sp, fontWeight = FontWeight.Bold, color = StudioTextTertiary, letterSpacing = 1.sp)
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            val isLocal = draftUrl == "http://10.0.2.2:8000"
            Box(
                modifier = Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(6.dp))
                    .background(if (isLocal) StudioWhite else StudioRaised)
                    .border(1.dp, if (isLocal) StudioWhite else StudioBorder, RoundedCornerShape(6.dp))
                    .clickable {
                        draftUrl = "http://10.0.2.2:8000"
                        draftKey = ""
                    }
                    .padding(vertical = 12.dp, horizontal = 8.dp),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        "Lokal Emulator",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        color = if (isLocal) StudioBlack else StudioWhite
                    )
                    Text(
                        "10.0.2.2:8000",
                        fontSize = 10.sp,
                        color = if (isLocal) StudioBlack.copy(alpha = 0.7f) else StudioTextTertiary
                    )
                }
            }

            val isCustom = draftUrl != "http://10.0.2.2:8000" && draftUrl.isNotEmpty()
            Box(
                modifier = Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(6.dp))
                    .background(if (isCustom) StudioWhite else StudioRaised)
                    .border(1.dp, if (isCustom) StudioWhite else StudioBorder, RoundedCornerShape(6.dp))
                    .clickable {
                        if (draftUrl == "http://10.0.2.2:8000") draftUrl = ""
                    }
                    .padding(vertical = 12.dp, horizontal = 8.dp),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        "Maxsus HTTPS Server",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        color = if (isCustom) StudioBlack else StudioWhite
                    )
                    Text(
                        "Masofaviy server",
                        fontSize = 10.sp,
                        color = if (isCustom) StudioBlack.copy(alpha = 0.7f) else StudioTextTertiary
                    )
                }
            }
        }

        // Form Fields
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            StyledTextField(draftUrl, { draftUrl = it }, "Server URL", "https://server.example.com")
            StyledTextField(draftKey, { draftKey = it }, "Server API Kaliti", "API kalitini kiriting (agar talab qilinsa)", isPassword = true)
        }

        endpoint.exceptionOrNull()?.message?.let {
            Text(it, color = StudioDestructive, fontSize = 11.sp)
        }

        // Primary Action: Save Button
        Button(
            onClick = {
                onSaveApiSettings(
                    settings.engineType, settings.deeplApiKey, settings.libreTranslateUrl,
                    settings.libreTranslateApiKey, settings.ocrApiKey, endpoint.getOrThrow(),
                    settings.useMangaServer, draftKey
                )
            },
            enabled = dirty && endpoint.isSuccess,
            modifier = Modifier.fillMaxWidth().height(48.dp),
            shape = RoundedCornerShape(6.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = StudioWhite,
                contentColor = StudioBlack,
                disabledContainerColor = StudioRaised,
                disabledContentColor = StudioTextTertiary
            )
        ) {
            if (apiSaveSuccess) {
                Icon(Icons.Default.Check, contentDescription = null, modifier = Modifier.size(16.dp))
                Spacer(Modifier.width(6.dp))
                Text("Sozlamalar Muvaffaqiyatli Saqlandi!", fontWeight = FontWeight.Bold, fontSize = 12.sp)
            } else {
                Text("Sozlamalarni Saqlaş", fontWeight = FontWeight.Bold, fontSize = 13.sp)
            }
        }

        // Test Connection Button
        OutlinedButton(
            onClick = { onTestMangaServer(endpoint.getOrThrow(), draftKey) },
            enabled = endpoint.isSuccess && serverHealth !is SettingsViewModel.ServerHealthStatus.Checking,
            modifier = Modifier.fillMaxWidth().height(44.dp),
            shape = RoundedCornerShape(6.dp),
            border = BorderStroke(1.dp, StudioBorder)
        ) {
            Text("Server Aloqasini Sinaş", color = StudioWhite, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
        }

        // Connection Health Status
        when (serverHealth) {
            is SettingsViewModel.ServerHealthStatus.Checking -> {
                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 4.dp)) {
                    CircularProgressIndicator(modifier = Modifier.size(14.dp), strokeWidth = 2.dp, color = StudioWhite)
                    Spacer(Modifier.width(8.dp))
                    Text("Server tekşirilmoqda…", color = StudioTextSecondary, fontSize = 11.sp)
                }
            }
            is SettingsViewModel.ServerHealthStatus.Success -> {
                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 4.dp)) {
                    Box(modifier = Modifier.size(6.dp).clip(CircleShape).background(StudioAccentEmerald))
                    Spacer(Modifier.width(8.dp))
                    Text(serverHealth.message, color = StudioAccentEmerald, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
                }
            }
            is SettingsViewModel.ServerHealthStatus.Error -> {
                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 4.dp)) {
                    Box(modifier = Modifier.size(6.dp).clip(CircleShape).background(StudioDestructive))
                    Spacer(Modifier.width(8.dp))
                    Text(serverHealth.error, color = StudioDestructive, fontSize = 11.sp)
                }
            }
            null -> Unit
        }

        Text(
            "Faol PDF vazifasi uni qabul qilgan serverga boğlangan. Server manzilini özgartirsangiz, avvalgi vazifaga qaytiş uçun eski manzilni tiklaş kerak.",
            color = StudioTextTertiary,
            fontSize = 11.sp,
            lineHeight = 16.sp,
            modifier = Modifier.padding(bottom = 60.dp)
        )
    }
}

// ─── Tab 4: Haqida (About Tab) ───────────────────────────────────────────────
@Composable
private fun AboutTabContent() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Box(
            modifier = Modifier
                .size(60.dp)
                .clip(RoundedCornerShape(10.dp))
                .background(StudioRaised)
                .border(1.dp, StudioBorder, RoundedCornerShape(10.dp)),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                Icons.Default.Translate,
                contentDescription = null,
                tint = StudioWhite,
                modifier = Modifier.size(30.dp)
            )
        }

        Text(
            "ScreenTranslator Studio",
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold,
            color = StudioWhite
        )
        Text(
            "Production Build v1.0.0 · Gemini 3.5 Inside",
            fontSize = 11.sp,
            color = StudioAccentEmerald,
            fontWeight = FontWeight.SemiBold
        )

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(6.dp))
                .background(StudioSurface)
                .border(1.dp, StudioBorder, RoundedCornerShape(6.dp))
                .padding(14.dp)
        ) {
            Text(
                "Manga, manhwa, komiks va kitob PDFlarini server yordamida çuqur kontekstni tüşunib tarjima qiluvçi mobil studio. Dialog bulutlarini tozalaş, OCR matnini tekşiriş va naşr sifatidagi PDF yaratiş imkoniyatlarini taqdim etadi.",
                fontSize = 12.sp,
                color = StudioTextSecondary,
                lineHeight = 17.sp,
                textAlign = TextAlign.Center
            )
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            InfoChip("OCR", "Paddle + Manga", modifier = Modifier.weight(1f))
            InfoChip("AI Model", "Gemini 3.5", modifier = Modifier.weight(1f))
            InfoChip("Inpaint", "PyMuPDF", modifier = Modifier.weight(1f))
        }

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 70.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Text(
                "TEXNIK IMKONIYATLAR",
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                color = StudioTextTertiary,
                letterSpacing = 1.2.sp
            )

            listOf(
                "Komiks rejimi: personaj rasmlari saqlanadi, dialog bulutlari tozalanadi",
                "Kitob rejimi: matnli va skaner sahifalar avtomatik reflow qilinadi",
                "Google AI Studio Gemini 3.5 Flash-lite orqali benuqson tarjima",
                "Matnlar qöşilib ketişini avtomatik ajratuvçi leksik filtr",
                "Ilova yopilganda ham fonda davom etuvçi xizmat (Foreground Service)",
                "Töğridan-töğri tahrirlaş va PDFni qayta tiklaş imkoniyati"
            ).forEach { feature ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(4.dp))
                        .background(StudioSurface)
                        .border(1.dp, StudioBorderSubtle, RoundedCornerShape(4.dp))
                        .padding(horizontal = 12.dp, vertical = 10.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Box(modifier = Modifier.size(4.dp).clip(CircleShape).background(StudioWhite))
                    Spacer(Modifier.width(10.dp))
                    Text(feature, fontSize = 11.sp, color = StudioTextPrimary)
                }
            }
        }
    }
}

@Composable
private fun InfoChip(
    title: String,
    value: String,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(6.dp))
            .background(StudioSurface)
            .border(1.dp, StudioBorder, RoundedCornerShape(6.dp))
            .padding(vertical = 10.dp, horizontal = 8.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(title, fontSize = 10.sp, color = StudioTextTertiary, fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(2.dp))
            Text(value, fontSize = 11.sp, color = StudioWhite, fontWeight = FontWeight.SemiBold)
        }
    }
}

// ─── Anchored Studio Bottom Dock ─────────────────────────────────────────────
@Composable
private fun BottomNavBar(
    selectedIndex: Int,
    onTabSelected: (Int) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .height(58.dp)
            .background(StudioBlack)
            .border(BorderStroke(1.dp, StudioBorderSubtle))
            .padding(horizontal = 16.dp),
        horizontalArrangement = Arrangement.SpaceAround,
        verticalAlignment = Alignment.CenterVertically
    ) {
        NavTab.entries.forEachIndexed { index, tab ->
            val isSelected = selectedIndex == index
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center,
                modifier = Modifier
                    .clip(RoundedCornerShape(6.dp))
                    .clickable(
                        interactionSource = remember { MutableInteractionSource() },
                        indication = null
                    ) { onTabSelected(index) }
                    .padding(horizontal = 12.dp, vertical = 6.dp)
            ) {
                Icon(
                    tab.icon,
                    contentDescription = tab.label,
                    tint = if (isSelected) StudioWhite else StudioTextTertiary,
                    modifier = Modifier.size(20.dp)
                )
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    tab.label,
                    fontSize = 10.sp,
                    color = if (isSelected) StudioWhite else StudioTextTertiary,
                    fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal
                )
            }
        }
    }
}

// ─── Precision Text Field ────────────────────────────────────────────────────
@Composable
private fun StyledTextField(
    value: String,
    onValueChange: (String) -> Unit,
    label: String,
    placeholder: String,
    isPassword: Boolean = false
) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(
            text = label,
            fontSize = 11.sp,
            fontWeight = FontWeight.SemiBold,
            color = StudioTextSecondary
        )
        OutlinedTextField(
            value = value,
            onValueChange = onValueChange,
            placeholder = { Text(placeholder, fontSize = 12.sp, color = StudioTextTertiary) },
            singleLine = true,
            visualTransformation = if (isPassword) PasswordVisualTransformation() else VisualTransformation.None,
            modifier = Modifier.fillMaxWidth(),
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = StudioBorderFocus,
                unfocusedBorderColor = StudioBorder,
                focusedTextColor = StudioTextPrimary,
                unfocusedTextColor = StudioTextPrimary,
                focusedContainerColor = StudioSurface,
                unfocusedContainerColor = StudioSurface
            ),
            shape = RoundedCornerShape(6.dp)
        )
    }
}

@Composable
private fun FullImageViewerDialog(
    imagePath: String,
    onDismiss: () -> Unit
) {
    val bitmap by produceState<Bitmap?>(initialValue = null, key1 = imagePath) {
        value = withContext(Dispatchers.IO) {
            try {
                val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                BitmapFactory.decodeFile(imagePath, bounds)
                val maxDim = maxOf(bounds.outWidth, bounds.outHeight)
                var sample = 1
                while (maxDim / sample > 2048) sample *= 2
                val options = BitmapFactory.Options().apply { inSampleSize = sample }
                BitmapFactory.decodeFile(imagePath, options)
            } catch (t: Throwable) {
                null
            }
        }
    }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false)
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(StudioBlack.copy(alpha = 0.95f))
        ) {
            if (bitmap != null) {
                Image(
                    bitmap = bitmap!!.asImageBitmap(),
                    contentDescription = "Töliq rasm",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(top = 48.dp, bottom = 16.dp, start = 8.dp, end = 8.dp)
                )
            } else {
                Text(
                    "Rasmni oçib bölmadi",
                    color = StudioWhite,
                    modifier = Modifier.align(Alignment.Center)
                )
            }

            IconButton(
                onClick = onDismiss,
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(16.dp)
                    .clip(CircleShape)
                    .background(StudioRaised)
            ) {
                Icon(Icons.Default.Close, contentDescription = "Yopiş", tint = StudioWhite)
            }
        }
    }
}
