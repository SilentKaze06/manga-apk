package com.screentranslator.app.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val DarkColorScheme = darkColorScheme(
    primary = StudioWhite,
    onPrimary = StudioBlack,
    primaryContainer = StudioRaised,
    onPrimaryContainer = StudioWhite,
    secondary = StudioAccentEmerald,
    onSecondary = StudioBlack,
    secondaryContainer = StudioSurface,
    onSecondaryContainer = StudioTextPrimary,
    tertiary = StudioAccentCyan,
    onTertiary = StudioBlack,
    background = StudioBlack,
    onBackground = StudioTextPrimary,
    surface = StudioSurface,
    onSurface = StudioTextPrimary,
    surfaceVariant = StudioRaised,
    onSurfaceVariant = StudioTextSecondary,
    outline = StudioBorder,
    outlineVariant = StudioBorderSubtle,
    error = StudioDestructive,
    onError = StudioWhite
)

@Composable
fun ScreenTranslatorTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = DarkColorScheme,
        typography = Typography,
        content = content
    )
}
