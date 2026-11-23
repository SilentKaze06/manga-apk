package com.screentranslator.app

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.screentranslator.app.service.FloatingBubbleService
import com.screentranslator.app.service.ScreenCaptureService
import com.screentranslator.app.ui.settings.SettingsScreen
import com.screentranslator.app.ui.settings.SettingsViewModel
import com.screentranslator.app.ui.theme.ScreenTranslatorTheme

class MainActivity : ComponentActivity() {

    private val viewModel: SettingsViewModel by viewModels()

    private val notificationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { /* handled */ }

    private val overlayPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && Settings.canDrawOverlays(this)) {
            requestMediaProjection()
        } else {
            Toast.makeText(this, "Ekranda rasm çiqariş uçun ruxsat berilmadi.", Toast.LENGTH_SHORT).show()
        }
    }

    private val mediaProjectionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK && result.data != null) {
            startScreenCaptureAndBubble(result.resultCode, result.data!!)
        } else {
            Toast.makeText(this, "Ekran tasvirini olişga ruxsat berilmadi.", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            }
        }

        setContent {
            ScreenTranslatorTheme {
                val isBubbleRunning by FloatingBubbleService.isRunningFlow.collectAsStateWithLifecycle()
                val isCaptureRunning by ScreenCaptureService.isRunningFlow.collectAsStateWithLifecycle()
                SettingsScreen(
                    viewModel = viewModel,
                    onStartScreenTranslation = { startScreenTranslationFlow() },
                    onStopScreenTranslation = { stopScreenCaptureAndBubble() },
                    isScreenTranslationRunning = isBubbleRunning && isCaptureRunning
                )
            }
        }
    }

    fun startScreenTranslationFlow() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && !Settings.canDrawOverlays(this)) {
            val intent = Intent(
                Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:$packageName")
            )
            overlayPermissionLauncher.launch(intent)
            return
        }
        requestMediaProjection()
    }

    private fun requestMediaProjection() {
        val mediaProjectionManager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        mediaProjectionLauncher.launch(mediaProjectionManager.createScreenCaptureIntent())
    }

    private fun startScreenCaptureAndBubble(resultCode: Int, resultData: Intent) {
        ScreenCaptureService.startCapture(this, resultCode, resultData)
        val bubbleIntent = Intent(this, FloatingBubbleService::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(bubbleIntent)
        } else {
            startService(bubbleIntent)
        }
    }

    private fun stopScreenCaptureAndBubble() {
        ScreenCaptureService.stop(this)
        stopService(Intent(this, FloatingBubbleService::class.java))
    }

    override fun onResume() {
        super.onResume()
        viewModel.refreshHistory()
    }
}
