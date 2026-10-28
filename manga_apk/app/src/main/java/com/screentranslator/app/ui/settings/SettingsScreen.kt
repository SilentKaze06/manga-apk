package com.screentranslator.app.ui.settings

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
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
import androidx.compose.foundation.Image
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
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.DocumentScanner
import androidx.compose.foundation.BorderStroke
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Language
import androidx.compose.material.icons.automirrored.filled.MenuBook
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Translate
import androidx.compose.material.icons.filled.VpnKey
import androidx.compose.material.icons.filled.Wifi
import androidx.compose.material.icons.filled.ZoomIn
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import com.screentranslator.app.data.model.MangaSourceLanguage
import com.screentranslator.app.data.model.TargetLanguage
import com.screentranslator.app.data.model.TranslationEngineType
import com.screentranslator.app.data.model.TranslationHistoryEntry
import com.screentranslator.app.data.model.TranslationMode
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.RadioButton
import androidx.compose.material3.RadioButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import android.graphics.BitmapFactory
import java.io.File
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

import com.screentranslator.app.ui.components.PermissionStatusCard
import com.screentranslator.app.ui.pdf.PdfTranslationScreen
import com.screentranslator.app.ui.theme.AmberWarning
import com.screentranslator.app.ui.theme.CyanAccent
import com.screentranslator.app.ui.theme.DarkBackground
import com.screentranslator.app.ui.theme.DarkBorder
import com.screentranslator.app.ui.theme.DarkSurface
import com.screentranslator.app.ui.theme.DarkSurfaceElevated
import com.screentranslator.app.ui.theme.EmeraldGreen
import com.screentranslator.app.ui.theme.IndigoDark
import com.screentranslator.app.ui.theme.IndigoPrimary
import com.screentranslator.app.ui.theme.RoseError
import com.screentranslator.app.ui.theme.TextPrimaryDark
import com.screentranslator.app.ui.theme.TextSecondaryDark
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.delay

// ─── Navigation tabs ─────────────────────────────────────────────────────────
private enum class NavTab(val icon: ImageVector, val label: String) {
    HOME(Icons.AutoMirrored.Filled.MenuBook, "PDF Tarjima"),
    HISTORY(Icons.Default.History, "Tarix"),
    API(Icons.Default.VpnKey, "Server & API"),
    ABOUT(Icons.Default.Info, "Ma'lumot")
}

// ─── Main Screen ──────────────────────────────────────────────────────────────
@Composable
fun SettingsScreen(
    viewModel: SettingsViewModel
) {
    val settings by viewModel.userSettings.collectAsState()
    val activePdfJob by viewModel.activePdfJob.collectAsState()
    val history by viewModel.translationHistory.collectAsState()
    val apiSaveSuccess by viewModel.apiSaveSuccess.collectAsState()
    val serverHealth by viewModel.serverHealth.collectAsState()

    var selectedTab by remember { mutableIntStateOf(0) }
    var languageDropdownExpanded by remember { mutableStateOf(false) }
    var viewingImagePath by remember { mutableStateOf<String?>(null) }

    // Full screen image viewer modal dialog
    viewingImagePath?.let { path ->
        FullImageViewerDialog(
            imagePath = path,
            onDismiss = { viewingImagePath = null }
        )
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(DarkBackground)
    ) {
        // Ambient glow background orb
        Box(
            modifier = Modifier
                .size(300.dp)
                .align(Alignment.TopCenter)
                .graphicsLayer { translationY = -60f }
                .blur(120.dp)
                .background(
                    brush = Brush.radialGradient(
                        colors = listOf(IndigoPrimary.copy(alpha = 0.3f), Color.Transparent)
                    ),
                    shape = CircleShape
                )
        )

        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding()
        ) {
            // Always: Language selector header
            LanguageSelector(
                selectedLanguage = settings.targetLanguage,
                expanded = languageDropdownExpanded,
                onExpandChange = { languageDropdownExpanded = it },
                onLanguageSelected = { lang ->
                    viewModel.setTargetLanguage(lang)
                    languageDropdownExpanded = false
                }
            )

            // Full-screen tab content (weight=1f fills all space between header and nav)
            AnimatedContent(
                targetState = selectedTab,
                transitionSpec = {
                    (fadeIn(tween(260)) + slideInVertically { it / 10 })
                        .togetherWith(fadeOut(tween(180)) + slideOutVertically { -it / 10 })
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
                        onSaveApiSettings = { engine, deepl, libreUrl, libreKey, ocrKey, mangaUrl, useServer, mangaKey ->
                            viewModel.saveApiSettings(engine, deepl, libreUrl, libreKey, ocrKey, mangaUrl, useServer, mangaKey)
                        },
                        onResetSaveSuccess = { viewModel.resetApiSaveSuccess() },
                        onTestMangaServer = { url, key -> viewModel.testMangaServerConnection(url, key) }
                    )
                    NavTab.ABOUT -> AboutTabContent()
                }
            }


            // Always: Bottom navigation bar
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

// ─── Language Selector Header ─────────────────────────────────────────────────
@Composable
private fun LanguageSelector(
    selectedLanguage: TargetLanguage,
    expanded: Boolean,
    onExpandChange: (Boolean) -> Unit,
    onLanguageSelected: (TargetLanguage) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 20.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(36.dp)
                    .clip(RoundedCornerShape(10.dp))
                    .background(Brush.linearGradient(listOf(IndigoPrimary, CyanAccent))),
                contentAlignment = Alignment.Center
            ) {
                Icon(Icons.Default.Translate, contentDescription = null, tint = Color.White, modifier = Modifier.size(20.dp))
            }
            Spacer(modifier = Modifier.width(10.dp))
            Column {
                Text("AI Manga Tarjimon", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, color = TextPrimaryDark)
                Text("PDF Manga & Book Translator", style = MaterialTheme.typography.labelSmall, color = TextSecondaryDark)
            }
        }

        Box {
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(24.dp))
                    .background(DarkSurface)
                    .border(1.dp, DarkBorder, RoundedCornerShape(24.dp))
                    .clickable(
                        interactionSource = remember { MutableInteractionSource() },
                        indication = null
                    ) { onExpandChange(true) }
                    .padding(horizontal = 14.dp, vertical = 8.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.Language, contentDescription = null, tint = CyanAccent, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = selectedLanguage.displayName.split(" ").first(),
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimaryDark
                    )
                    Icon(Icons.Default.ArrowDropDown, contentDescription = null, tint = TextSecondaryDark, modifier = Modifier.size(18.dp))
                }
            }
            DropdownMenu(
                expanded = expanded,
                onDismissRequest = { onExpandChange(false) },
                modifier = Modifier
                    .background(DarkSurface)
                    .border(1.dp, DarkBorder, RoundedCornerShape(12.dp))
            ) {
                TargetLanguage.entries.forEach { lang ->
                    DropdownMenuItem(
                        text = {
                            Text(
                                text = lang.displayName,
                                style = MaterialTheme.typography.bodyMedium,
                                color = if (lang == selectedLanguage) CyanAccent else TextPrimaryDark,
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

// ─── Tab 1: Bosh sahifa — full screen ─────────────────────────────────────────
@Composable
private fun HomeTabFullContent(
    settings: com.screentranslator.app.data.preferences.UserSettings,
    hasOverlayPermission: Boolean,
    isCaptureServiceRunning: Boolean,
    hasNotificationPermission: Boolean,
    onRequestOverlayPermission: () -> Unit,
    onRequestCapturePermission: () -> Unit,
    onRequestNotificationPermission: () -> Unit,
    onCircleClick: () -> Unit,
    onOpenPdfTranslation: () -> Unit,
    onSelectTranslationMode: (TranslationMode) -> Unit,
    onSelectMangaSourceLanguage: (MangaSourceLanguage) -> Unit
) {
    val allReady = hasOverlayPermission && isCaptureServiceRunning

    Column(modifier = Modifier.fillMaxSize()) {
        // Big clickable circle (takes upper portion)
        CentralCircleButton(
            isActive = settings.isServiceActive,
            allReady = allReady,
            onClick = onCircleClick,
            modifier = Modifier.weight(1f)
        )

        // Scrollable section below circle
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 18.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(12.dp))
                    .background(CyanAccent.copy(alpha = 0.1f))
                    .border(1.dp, CyanAccent.copy(alpha = 0.35f), RoundedCornerShape(12.dp))
                    .clickable(onClick = onOpenPdfTranslation)
                    .padding(14.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.AutoMirrored.Filled.MenuBook, contentDescription = null, tint = CyanAccent, modifier = Modifier.size(24.dp))
                    Spacer(modifier = Modifier.width(10.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text("PDF kitob tarjimasi", color = TextPrimaryDark, fontWeight = FontWeight.SemiBold)
                        Text("Kitob yoki roman PDF faylini tarjima qilish", color = TextSecondaryDark, style = MaterialTheme.typography.bodySmall)
                    }
                    Icon(Icons.Default.ArrowDropDown, contentDescription = null, tint = CyanAccent, modifier = Modifier.size(22.dp))
                }
            }

            // ─── Mode Selector Card (Standart vs Manga) ───────────────────────
            TranslationModeSelectorCard(
                currentMode = settings.translationMode,
                currentMangaLang = settings.mangaSourceLanguage,
                useMangaServer = settings.useMangaServer,
                mangaServerUrl = settings.mangaServerUrl,
                onSelectMode = onSelectTranslationMode,
                onSelectMangaLang = onSelectMangaSourceLanguage
            )

            // Permissions status cards below mode selector
            PermissionStatusCard(
                title = "Overlay ruxsati",
                description = "Boshqa ilovalar ustida ko'rsatish uchun.",
                isGranted = hasOverlayPermission,
                actionButtonText = "Ruxsat berish",
                onActionClick = onRequestOverlayPermission
            )
            PermissionStatusCard(
                title = "Ekran yozib olish",
                description = "Tarjima qilish uchun ekranni suratga oladi.",
                isGranted = isCaptureServiceRunning,
                actionButtonText = "Xizmatni boshlash",
                onActionClick = onRequestCapturePermission
            )
            if (!hasNotificationPermission) {
                PermissionStatusCard(
                    title = "Bildirishnomalar",
                    description = "Android 13+ da xizmatni ishlab turish uchun kerak.",
                    isGranted = false,
                    actionButtonText = "Ruxsat berish",
                    onActionClick = onRequestNotificationPermission
                )
            }
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(10.dp))
                    .background(IndigoPrimary.copy(alpha = 0.12f))
                    .border(1.dp, IndigoPrimary.copy(alpha = 0.25f), RoundedCornerShape(10.dp))
                    .padding(10.dp)
            ) {
                Text(
                    "⚡ Internet yo'q bo'lsa, avtonom ML Kit ga avtomatik o'tiladi.",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondaryDark
                )
            }
            Spacer(modifier = Modifier.height(4.dp))
        }
    }
}

// ─── Mode Selector Card Composable ────────────────────────────────────────────
@Composable
private fun TranslationModeSelectorCard(
    currentMode: TranslationMode,
    currentMangaLang: MangaSourceLanguage,
    useMangaServer: Boolean,
    mangaServerUrl: String,
    onSelectMode: (TranslationMode) -> Unit,
    onSelectMangaLang: (MangaSourceLanguage) -> Unit
) {
    var mangaLangDropdownExpanded by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(DarkSurface)
            .border(
                1.dp,
                if (currentMode == TranslationMode.MANGA) AmberWarning.copy(alpha = 0.45f) else DarkBorder,
                RoundedCornerShape(16.dp)
            )
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    if (currentMode == TranslationMode.MANGA) Icons.AutoMirrored.Filled.MenuBook else Icons.Default.Translate,
                    contentDescription = null,
                    tint = if (currentMode == TranslationMode.MANGA) AmberWarning else CyanAccent,
                    modifier = Modifier.size(20.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    "Ishlash rejimi",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                    color = TextPrimaryDark
                )
            }

            // Mode indicator badge
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(20.dp))
                    .background(
                        if (currentMode == TranslationMode.MANGA) AmberWarning.copy(alpha = 0.15f)
                        else IndigoPrimary.copy(alpha = 0.2f)
                    )
                    .padding(horizontal = 10.dp, vertical = 4.dp)
            ) {
                Text(
                    text = if (currentMode == TranslationMode.MANGA) "Manga rejimi faol" else "Standart rejim faol",
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = if (currentMode == TranslationMode.MANGA) AmberWarning else CyanAccent
                )
            }
        }

        // Segmented Switcher buttons
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(12.dp))
                .background(DarkBackground)
                .padding(4.dp),
            horizontalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            // Standard Mode Button
            val isStandard = currentMode == TranslationMode.STANDARD
            Box(
                modifier = Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(10.dp))
                    .background(
                        if (isStandard) Brush.horizontalGradient(listOf(IndigoPrimary, CyanAccent))
                        else Brush.horizontalGradient(listOf(Color.Transparent, Color.Transparent))
                    )
                    .clickable { onSelectMode(TranslationMode.STANDARD) }
                    .padding(vertical = 10.dp),
                contentAlignment = Alignment.Center
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        Icons.Default.Translate,
                        contentDescription = null,
                        tint = if (isStandard) Color.White else TextSecondaryDark,
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        "Standart",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = if (isStandard) FontWeight.Bold else FontWeight.Normal,
                        color = if (isStandard) Color.White else TextSecondaryDark
                    )
                }
            }

            // Manga Mode Button
            val isManga = currentMode == TranslationMode.MANGA
            Box(
                modifier = Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(10.dp))
                    .background(
                        if (isManga) Brush.horizontalGradient(listOf(AmberWarning, Color(0xFFF59E0B)))
                        else Brush.horizontalGradient(listOf(Color.Transparent, Color.Transparent))
                    )
                    .clickable { onSelectMode(TranslationMode.MANGA) }
                    .padding(vertical = 10.dp),
                contentAlignment = Alignment.Center
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        Icons.AutoMirrored.Filled.MenuBook,
                        contentDescription = null,
                        tint = if (isManga) Color.Black else TextSecondaryDark,
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        "Manga / Komiks",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = if (isManga) FontWeight.Bold else FontWeight.Normal,
                        color = if (isManga) Color.Black else TextSecondaryDark
                    )
                }
            }
        }

        // Manga Mode specific options & instructions
        AnimatedVisibility(visible = currentMode == TranslationMode.MANGA) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 4.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(10.dp))
                        .background(AmberWarning.copy(alpha = 0.08f))
                        .border(1.dp, AmberWarning.copy(alpha = 0.25f), RoundedCornerShape(10.dp))
                        .padding(10.dp)
                ) {
                    Text(
                        "💬 Dialog bulutlari (speech bubbles) butun holda o'qiladi va asl joyiga manga shrifti bilan tahrirlab joylashtiriladi.",
                        style = MaterialTheme.typography.bodySmall,
                        color = TextPrimaryDark
                    )
                }

                // Manga Source Language Selector
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Column {
                        Text(
                            "Manga manba tili:",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = TextPrimaryDark
                        )
                        Text(
                            "Manga, Manhwa yoki Manhua",
                            style = MaterialTheme.typography.labelSmall,
                            color = TextSecondaryDark
                        )
                    }

                    Box {
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(8.dp))
                                .background(DarkSurfaceElevated)
                                .border(1.dp, DarkBorder, RoundedCornerShape(8.dp))
                                .clickable { mangaLangDropdownExpanded = true }
                                .padding(horizontal = 12.dp, vertical = 7.dp)
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text(
                                    text = currentMangaLang.displayName.split(" ").first(),
                                    style = MaterialTheme.typography.labelMedium,
                                    fontWeight = FontWeight.Bold,
                                    color = AmberWarning
                                )
                                Spacer(modifier = Modifier.width(4.dp))
                                Icon(Icons.Default.ArrowDropDown, contentDescription = null, tint = TextSecondaryDark, modifier = Modifier.size(16.dp))
                            }
                        }

                        DropdownMenu(
                            expanded = mangaLangDropdownExpanded,
                            onDismissRequest = { mangaLangDropdownExpanded = false },
                            modifier = Modifier
                                .background(DarkSurface)
                                .border(1.dp, DarkBorder, RoundedCornerShape(12.dp))
                        ) {
                            MangaSourceLanguage.entries.forEach { lang ->
                                DropdownMenuItem(
                                    text = {
                                        Text(
                                            lang.displayName,
                                            style = MaterialTheme.typography.bodySmall,
                                            fontWeight = if (lang == currentMangaLang) FontWeight.Bold else FontWeight.Normal,
                                            color = if (lang == currentMangaLang) AmberWarning else TextPrimaryDark
                                        )
                                    },
                                    onClick = {
                                        onSelectMangaLang(lang)
                                        mangaLangDropdownExpanded = false
                                    }
                                )
                            }
                        }
                    }
                }

                // Manga OCR Server info badge
                if (useMangaServer) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(8.dp))
                            .background(EmeraldGreen.copy(alpha = 0.08f))
                            .border(1.dp, EmeraldGreen.copy(alpha = 0.25f), RoundedCornerShape(8.dp))
                            .padding(horizontal = 10.dp, vertical = 6.dp)
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                Icons.Default.Wifi,
                                contentDescription = null,
                                tint = EmeraldGreen,
                                modifier = Modifier.size(14.dp)
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                "Manga-OCR (ViT) Server faol: $mangaServerUrl",
                                style = MaterialTheme.typography.labelSmall,
                                color = EmeraldGreen,
                                fontWeight = FontWeight.Medium
                            )
                        }
                    }
                }

                // Quick-switch tip
                Text(
                    "💡 Suzib yuruvchi tugmani uzoq bosib tursangiz, Manga va Standart rejimlari darhol almashadi.",
                    style = MaterialTheme.typography.labelSmall,
                    color = TextSecondaryDark
                )
            }
        }
    }
}

// ─── Animated Central Circle (Launch Button) ──────────────────────────────────
@Composable
private fun CentralCircleButton(
    isActive: Boolean,
    allReady: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val infinite = rememberInfiniteTransition(label = "circle")
    val pulseScale by infinite.animateFloat(
        initialValue = 1f,
        targetValue = if (isActive) 1.04f else 1f,
        animationSpec = infiniteRepeatable(tween(1400, easing = FastOutSlowInEasing), RepeatMode.Reverse),
        label = "pulse"
    )
    val glowAlpha by infinite.animateFloat(
        initialValue = 0.2f,
        targetValue = if (isActive) 0.6f else 0.2f,
        animationSpec = infiniteRepeatable(tween(1400, easing = FastOutSlowInEasing), RepeatMode.Reverse),
        label = "glow"
    )
    val rotation by infinite.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(tween(7000, easing = LinearEasing)),
        label = "rot"
    )

    Box(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp),
        contentAlignment = Alignment.Center
    ) {
        // Outer glow
        Box(
            modifier = Modifier
                .size(240.dp)
                .scale(pulseScale + 0.07f)
                .blur(32.dp)
                .background(
                    Brush.radialGradient(
                        listOf(
                            if (isActive) CyanAccent.copy(alpha = glowAlpha) else IndigoPrimary.copy(alpha = 0.18f),
                            Color.Transparent
                        )
                    ),
                    shape = CircleShape
                )
        )
        // Rotating ring (active only)
        if (isActive) {
            Box(
                modifier = Modifier
                    .size(222.dp)
                    .graphicsLayer { rotationZ = rotation }
                    .border(
                        1.dp,
                        Brush.sweepGradient(listOf(
                            CyanAccent.copy(alpha = 0.8f),
                            IndigoPrimary.copy(alpha = 0.4f),
                            Color.Transparent,
                            CyanAccent.copy(alpha = 0.8f)
                        )),
                        CircleShape
                    )
            )
        }
        // Main clickable circle
        Box(
            modifier = Modifier
                .size(200.dp)
                .scale(pulseScale)
                .clip(CircleShape)
                .background(
                    Brush.radialGradient(
                        listOf(
                            if (isActive) IndigoPrimary.copy(alpha = 0.7f) else DarkSurface,
                            if (isActive) DarkBackground.copy(alpha = 0.9f) else DarkSurfaceElevated.copy(alpha = 0.6f)
                        )
                    )
                )
                .border(
                    2.dp,
                    Brush.sweepGradient(
                        if (isActive) listOf(CyanAccent, IndigoPrimary, CyanAccent)
                        else listOf(DarkBorder, DarkSurfaceElevated, DarkBorder)
                    ),
                    CircleShape
                )
                .clickable(
                    interactionSource = remember { MutableInteractionSource() },
                    indication = null,
                    onClick = onClick
                ),
            contentAlignment = Alignment.Center
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Icon(
                    Icons.Default.Translate,
                    contentDescription = "Ishga tushirish",
                    tint = when {
                        isActive -> CyanAccent
                        !allReady -> AmberWarning
                        else -> EmeraldGreen
                    },
                    modifier = Modifier.size(52.dp)
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = when {
                        isActive -> "FAOL"
                        !allReady -> "RUXSAT"
                        else -> "BOSHLASH"
                    },
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.ExtraBold,
                    color = when {
                        isActive -> CyanAccent
                        !allReady -> AmberWarning
                        else -> EmeraldGreen
                    },
                    letterSpacing = 2.sp
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = when {
                        isActive -> "To'xtatish uchun bos"
                        !allReady -> "Ruxsatlarni bering"
                        else -> "Bosib boshlang"
                    },
                    style = MaterialTheme.typography.labelSmall,
                    color = TextSecondaryDark,
                    textAlign = TextAlign.Center
                )
            }
        }
    }
}

// ─── Tab 2: Tarix ─────────────────────────────────────────────────────────────
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

    Column(modifier = Modifier.fillMaxSize()) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp, vertical = 12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                "Tarjima tarixi",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = TextPrimaryDark
            )
            if (history.isNotEmpty()) {
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(8.dp))
                        .background(RoseError.copy(alpha = 0.15f))
                        .border(1.dp, RoseError.copy(alpha = 0.3f), RoundedCornerShape(8.dp))
                        .clickable(
                            interactionSource = remember { MutableInteractionSource() },
                            indication = null,
                            onClick = onClearHistory
                        )
                        .padding(horizontal = 12.dp, vertical = 6.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Delete, contentDescription = null, tint = RoseError, modifier = Modifier.size(14.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Tozalash", style = MaterialTheme.typography.labelSmall, color = RoseError, fontWeight = FontWeight.SemiBold)
                    }
                }
            }
        }

        if (history.isEmpty()) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Icon(Icons.Default.History, contentDescription = null, tint = DarkBorder, modifier = Modifier.size(64.dp))
                    Spacer(modifier = Modifier.height(12.dp))
                    Text("Hali tarjima qilinmagan", style = MaterialTheme.typography.titleMedium, color = TextSecondaryDark)
                    Text(
                        "Suzib yuruvchi tugma orqali tarjima qiling",
                        style = MaterialTheme.typography.bodySmall,
                        color = DarkBorder,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(top = 4.dp, start = 32.dp, end = 32.dp)
                    )
                }
            }
        } else {
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 18.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
                contentPadding = PaddingValues(bottom = 16.dp)
            ) {
                items(history, key = { it.id }) { entry ->
                    HistoryEntryCard(entry = entry, dateFormat = dateFormat, onViewImage = onViewImage)
                }
            }
        }
    }
}

@Composable
private fun HistoryEntryCard(
    entry: TranslationHistoryEntry,
    dateFormat: SimpleDateFormat,
    onViewImage: (String) -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(DarkSurface)
            .border(1.dp, DarkBorder, RoundedCornerShape(14.dp))
            .padding(14.dp)
    ) {
        Column {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(dateFormat.format(Date(entry.timestamp)), style = MaterialTheme.typography.labelSmall, color = TextSecondaryDark)
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(6.dp))
                        .background(CyanAccent.copy(alpha = 0.12f))
                        .padding(horizontal = 8.dp, vertical = 3.dp)
                ) {
                    Text(entry.targetLanguage, style = MaterialTheme.typography.labelSmall, color = CyanAccent)
                }
            }

            // Saved screenshot with translated overlay
            val imagePath = entry.imagePath
            val imageFile = remember(imagePath) { imagePath?.let { File(it) } }
            if (imageFile != null && imageFile.exists()) {
                Spacer(modifier = Modifier.height(10.dp))
                val bitmap = remember(imageFile.absolutePath) {
                    try {
                        val options = BitmapFactory.Options().apply { inSampleSize = 2 }
                        BitmapFactory.decodeFile(imageFile.absolutePath, options)
                    } catch (e: Exception) {
                        null
                    }
                }

                if (bitmap != null) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(160.dp)
                            .clip(RoundedCornerShape(10.dp))
                            .border(1.dp, CyanAccent.copy(alpha = 0.3f), RoundedCornerShape(10.dp))
                            .clickable { onViewImage(imageFile.absolutePath) }
                    ) {
                        Image(
                            bitmap = bitmap.asImageBitmap(),
                            contentDescription = "Tarjima rasmi",
                            contentScale = ContentScale.Crop,
                            modifier = Modifier.fillMaxSize()
                        )
                        Box(
                            modifier = Modifier
                                .align(Alignment.BottomEnd)
                                .padding(8.dp)
                                .clip(RoundedCornerShape(6.dp))
                                .background(Color.Black.copy(alpha = 0.75f))
                                .padding(horizontal = 8.dp, vertical = 4.dp)
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(Icons.Default.ZoomIn, contentDescription = null, tint = CyanAccent, modifier = Modifier.size(14.dp))
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("Rasmni ko'rish", style = MaterialTheme.typography.labelSmall, color = Color.White)
                            }
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(10.dp))
            Text(entry.originalText, style = MaterialTheme.typography.bodySmall, color = TextSecondaryDark, maxLines = 3, overflow = TextOverflow.Ellipsis)
            Row(
                modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(modifier = Modifier.weight(1f).height(1.dp).background(DarkBorder))
                Text(" → ", style = MaterialTheme.typography.labelMedium, color = CyanAccent)
                Box(modifier = Modifier.weight(1f).height(1.dp).background(DarkBorder))
            }
            Text(
                entry.translatedText,
                style = MaterialTheme.typography.bodyMedium,
                color = TextPrimaryDark,
                fontWeight = FontWeight.Medium,
                maxLines = 4,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

@Composable
private fun FullImageViewerDialog(
    imagePath: String,
    onDismiss: () -> Unit
) {
    val bitmap = remember(imagePath) {
        try {
            BitmapFactory.decodeFile(imagePath)
        } catch (e: Exception) {
            null
        }
    }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false)
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black.copy(alpha = 0.94f))
        ) {
            if (bitmap != null) {
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "To'liq rasm",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(top = 48.dp, bottom = 16.dp, start = 8.dp, end = 8.dp)
                )
            } else {
                Text(
                    "Rasmni ochib bo'lmadi",
                    color = Color.White,
                    modifier = Modifier.align(Alignment.Center)
                )
            }

            // Close button
            IconButton(
                onClick = onDismiss,
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(16.dp)
                    .clip(CircleShape)
                    .background(DarkSurfaceElevated.copy(alpha = 0.8f))
            ) {
                Icon(Icons.Default.Close, contentDescription = "Yopish", tint = Color.White)
            }
        }
    }
}

// ─── Tab 3: API — explicit Save button ────────────────────────────────────────
@Composable
private fun ApiTabContent(
    settings: com.screentranslator.app.data.preferences.UserSettings,
    apiSaveSuccess: Boolean,
    serverHealth: SettingsViewModel.ServerHealthStatus?,
    onSaveApiSettings: (TranslationEngineType, String, String, String, String, String, Boolean, String) -> Unit,
    onResetSaveSuccess: () -> Unit,
    onTestMangaServer: (String, String) -> Unit
) {
    var draftUrl by rememberSaveable(settings.mangaServerUrl) { mutableStateOf(settings.mangaServerUrl) }
    var draftKey by rememberSaveable(settings.mangaServerApiKey) { mutableStateOf(settings.mangaServerApiKey) }
    val endpoint = runCatching {
        com.screentranslator.app.data.network.ServerEndpoint.normalize(draftUrl, com.screentranslator.app.BuildConfig.DEBUG)
    }
    val dirty = draftUrl != settings.mangaServerUrl || draftKey != settings.mangaServerApiKey
    LaunchedEffect(apiSaveSuccess) {
        if (apiSaveSuccess) { delay(2000); onResetSaveSuccess() }
    }
    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Text("PDF server sozlamalari", style = MaterialTheme.typography.titleLarge, color = TextPrimaryDark)
        Text(
            "PDF tarjimasini server bajaradi. Matn Google orqali tarjima qilinadi; bu versiyada PDF uchun DeepL, LibreTranslate yoki MyMemory tanlovi mavjud emas.",
            color = TextSecondaryDark, style = MaterialTheme.typography.bodyMedium
        )
        StyledTextField(draftUrl, { draftUrl = it }, "Server URL", "https://server.example")
        StyledTextField(draftKey, { draftKey = it }, "Server API kaliti", "Serverda sozlangan kalit", isPassword = true)
        endpoint.exceptionOrNull()?.message?.let { Text(it, color = RoseError) }
        if (com.screentranslator.app.BuildConfig.DEBUG) {
            Text(
                "Lokal sinov: emulator uchun http://10.0.2.2:8000; telefon uchun kompyuterning lokal IP manzili. HTTP trafik shifrlanmaydi. Internetdagi server uchun HTTPS ishlating.",
                color = AmberWarning, style = MaterialTheme.typography.bodySmall
            )
        } else {
            Text("Server bilan aloqa HTTPS orqali himoyalanadi.", color = TextSecondaryDark)
        }
        Button(
            onClick = {
                onSaveApiSettings(
                    settings.engineType, settings.deeplApiKey, settings.libreTranslateUrl,
                    settings.libreTranslateApiKey, settings.ocrApiKey, endpoint.getOrThrow(),
                    settings.useMangaServer, draftKey
                )
            },
            enabled = dirty && endpoint.isSuccess,
            modifier = Modifier.fillMaxWidth()
        ) { Text(if (apiSaveSuccess) "Saqlandi!" else "Saqlash") }
        Button(
            onClick = { onTestMangaServer(endpoint.getOrThrow(), draftKey) },
            enabled = endpoint.isSuccess && serverHealth !is SettingsViewModel.ServerHealthStatus.Checking,
            modifier = Modifier.fillMaxWidth()
        ) { Text("Aloqani tekshirish") }
        when (serverHealth) {
            is SettingsViewModel.ServerHealthStatus.Checking -> CircularProgressIndicator()
            is SettingsViewModel.ServerHealthStatus.Success -> Text(serverHealth.message, color = EmeraldGreen)
            is SettingsViewModel.ServerHealthStatus.Error -> Text(serverHealth.error, color = RoseError)
            null -> Unit
        }
        Text(
            "Faol PDF vazifasi uni qabul qilgan serverga bog‘langan. Server manzilini o‘zgartirsangiz, avvalgi vazifaga qaytish uchun eski manzilni tiklash kerak.",
            color = TextSecondaryDark, style = MaterialTheme.typography.bodySmall
        )
    }
}

// ─── Tab 4: Ma'lumot ─────────────────────────────────────────────────────────
@Composable
private fun AboutTabContent() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 24.dp, vertical = 20.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Box(
            modifier = Modifier
                .size(72.dp)
                .clip(RoundedCornerShape(20.dp))
                .background(Brush.linearGradient(listOf(IndigoPrimary, CyanAccent))),
            contentAlignment = Alignment.Center
        ) {
            Icon(Icons.Default.Translate, contentDescription = null, tint = Color.White, modifier = Modifier.size(38.dp))
        }

        Text("AI Manga Tarjimon", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold, color = TextPrimaryDark)
        Text("v1.0.0", style = MaterialTheme.typography.labelMedium, color = CyanAccent)

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(14.dp))
                .background(Brush.linearGradient(listOf(IndigoPrimary.copy(alpha = 0.15f), CyanAccent.copy(alpha = 0.08f))))
                .border(1.dp, IndigoPrimary.copy(alpha = 0.3f), RoundedCornerShape(14.dp))
                .padding(16.dp)
        ) {
            Text(
                "Kitob, manga va komiks PDFlarini server orqali tarjima qilish, OCR matnini tekshirish va tuzatish hamda natijani PDF sifatida saqlash uchun ilova. Tarjima aniqligi manba sifatiga va serverdagi OCR imkoniyatlariga bog‘liq.",
                style = MaterialTheme.typography.bodyMedium,
                color = TextSecondaryDark,
                textAlign = TextAlign.Center
            )
        }

        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            InfoChip("PDF", "Tarjima", modifier = Modifier.weight(1f))
            InfoChip("OCR", "Tahrir", modifier = Modifier.weight(1f))
            InfoChip("Natija", "Saqlash", modifier = Modifier.weight(1f))
        }

        listOf(
            "PDF: kitob va komiks rejimlari",
            "Matn tarjimasi: Google, internet orqali",
            "Faol PDF vazifasiga qayta ulanish",
            "OCR va tarjima matnini qo‘lda tuzatish",
            "11 ta maqsad tilini tanlash",
            "Tarix bo‘limi avvalgi ekran tarjimasi yozuvlarini ko‘rsatadi; joriy PDF vazifasi bosh sahifada.",
            "Bu versiyada suzuvchi ekran tarjimasi yoqilmagan."
        ).forEach { feature ->
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .background(DarkSurface)
                    .border(1.dp, DarkBorder, RoundedCornerShape(8.dp))
                    .padding(horizontal = 14.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(feature, style = MaterialTheme.typography.bodySmall, color = TextPrimaryDark)
            }
        }
    }
}

// ─── Bottom Navigation ────────────────────────────────────────────────────────
@Composable
private fun BottomNavBar(selectedIndex: Int, onTabSelected: (Int) -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 10.dp)
            .clip(RoundedCornerShape(28.dp))
            .background(DarkSurface)
            .border(1.dp, DarkBorder, RoundedCornerShape(28.dp))
            .padding(horizontal = 8.dp, vertical = 6.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            NavTab.entries.forEachIndexed { index, tab ->
                NavTabItem(tab = tab, isSelected = selectedIndex == index, onClick = { onTabSelected(index) })
            }
        }
    }
}

@Composable
private fun NavTabItem(tab: NavTab, isSelected: Boolean, onClick: () -> Unit) {
    val scale by animateFloatAsState(
        targetValue = if (isSelected) 1.1f else 1f,
        animationSpec = tween(200),
        label = "navScale"
    )
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier
            .clip(RoundedCornerShape(20.dp))
            .clickable(
                interactionSource = remember { MutableInteractionSource() },
                indication = null,
                onClick = onClick
            )
            .background(
                if (isSelected)
                    Brush.verticalGradient(listOf(IndigoPrimary.copy(alpha = 0.25f), CyanAccent.copy(alpha = 0.1f)))
                else
                    Brush.verticalGradient(listOf(Color.Transparent, Color.Transparent))
            )
            .padding(horizontal = 16.dp, vertical = 8.dp)
            .scale(scale)
    ) {
        Box(
            modifier = Modifier
                .size(36.dp)
                .clip(CircleShape)
                .background(
                    if (isSelected) Brush.radialGradient(listOf(IndigoPrimary, IndigoDark))
                    else Brush.radialGradient(listOf(DarkSurfaceElevated, DarkSurfaceElevated))
                ),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                tab.icon,
                contentDescription = tab.label,
                tint = if (isSelected) Color.White else TextSecondaryDark,
                modifier = Modifier.size(20.dp)
            )
        }
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            tab.label,
            style = MaterialTheme.typography.labelSmall,
            color = if (isSelected) CyanAccent else TextSecondaryDark,
            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
            maxLines = 1
        )
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
@Composable
private fun InfoChip(label: String, value: String, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .clip(RoundedCornerShape(10.dp))
            .background(DarkSurface)
            .border(1.dp, DarkBorder, RoundedCornerShape(10.dp))
            .padding(10.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(value, style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Bold, color = CyanAccent)
        Text(label, style = MaterialTheme.typography.labelSmall, color = TextSecondaryDark)
    }
}

@Composable
private fun StyledTextField(
    value: String,
    onValueChange: (String) -> Unit,
    label: String,
    placeholder: String,
    isPassword: Boolean = false
) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        label = { Text(label, style = MaterialTheme.typography.labelSmall) },
        placeholder = { Text(placeholder, style = MaterialTheme.typography.labelSmall) },
        singleLine = true,
        visualTransformation = if (isPassword) PasswordVisualTransformation() else VisualTransformation.None,
        modifier = Modifier.fillMaxWidth(),
        colors = OutlinedTextFieldDefaults.colors(
            focusedBorderColor = IndigoPrimary,
            unfocusedBorderColor = DarkBorder,
            focusedTextColor = TextPrimaryDark,
            unfocusedTextColor = TextPrimaryDark,
            focusedLabelColor = CyanAccent,
            unfocusedLabelColor = TextSecondaryDark
        )
    )
}
