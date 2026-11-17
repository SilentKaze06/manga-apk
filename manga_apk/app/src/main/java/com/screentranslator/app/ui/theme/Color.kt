package com.screentranslator.app.ui.theme

import androidx.compose.ui.graphics.Color

// ─── Obsidian Studio Design System Tokens ────────────────────────────────────
// Strict, high-contrast, zero-AI-gradient color palette

val StudioBlack = Color(0xFF090A0C)
val StudioSurface = Color(0xFF121417)
val StudioRaised = Color(0xFF1A1D22)
val StudioSurfaceHover = Color(0xFF22262D)

val StudioBorder = Color(0xFF262930)
val StudioBorderSubtle = Color(0xFF1B1E24)
val StudioBorderFocus = Color(0xFF4A5160)

val StudioWhite = Color(0xFFFFFFFF)
val StudioTextPrimary = Color(0xFFF4F6F8)
val StudioTextSecondary = Color(0xFF949BA6)
val StudioTextTertiary = Color(0xFF606773)

// Semantic Accents
val StudioAccent = Color(0xFFFFFFFF) // High-contrast mono accent for primary CTA
val StudioAccentEmerald = Color(0xFF10B981) // Telemetry: online, completed, active
val StudioAccentCyan = Color(0xFF06B6D4) // Informational telemetry
val StudioWarning = Color(0xFFF59E0B) // Amber alert / pending state
val StudioDestructive = Color(0xFFEF4444) // Error / cancel / delete

// Legacy backward-compatibility aliases
val DarkBackground = StudioBlack
val DarkSurface = StudioSurface
val DarkSurfaceElevated = StudioRaised
val DarkBorder = StudioBorder
val TextPrimaryDark = StudioTextPrimary
val TextSecondaryDark = StudioTextSecondary
val IndigoPrimary = StudioWhite
val IndigoDark = StudioRaised
val CyanAccent = StudioAccentCyan
val EmeraldGreen = StudioAccentEmerald
val AmberWarning = StudioWarning
val RoseError = StudioDestructive
