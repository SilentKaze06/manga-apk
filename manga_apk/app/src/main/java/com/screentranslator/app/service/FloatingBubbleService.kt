package com.screentranslator.app.service

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.Rect
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.ImageView
import android.widget.Toast
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.screentranslator.app.MainActivity
import com.screentranslator.app.R
import com.screentranslator.app.data.model.MangaDetectedBlock
import com.screentranslator.app.data.model.MangaSourceLanguage
import com.screentranslator.app.data.model.OcrResultBlock
import com.screentranslator.app.data.model.TranslationHistoryEntry
import com.screentranslator.app.data.model.TranslationMode
import com.screentranslator.app.data.network.MangaServerClient
import com.screentranslator.app.data.preferences.TranslationHistoryRepository
import com.screentranslator.app.data.preferences.UserPreferencesRepository
import com.screentranslator.app.data.preferences.UserSettings
import com.screentranslator.app.ocr.MangaBubbleDetector
import com.screentranslator.app.ocr.OcrProcessor
import com.screentranslator.app.overlay.OverlayManager
import com.screentranslator.app.translation.TranslationManager
import com.screentranslator.app.util.DisplayCompat
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.coroutines.withContext
import kotlin.math.abs
import kotlin.math.max

/**
 * Background Service hosting the draggable floating bubble button.
 * Coordinates on-demand screen capture, OCR, translation, and overlay rendering.
 */
class FloatingBubbleService : Service() {

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private lateinit var windowManager: WindowManager
    private lateinit var userPreferencesRepository: UserPreferencesRepository

    private var bubbleView: View? = null
    private var bubbleImageView: ImageView? = null
    private var layoutParams: WindowManager.LayoutParams? = null

    private lateinit var ocrProcessor: OcrProcessor
    private lateinit var translationManager: TranslationManager
    private lateinit var overlayManager: OverlayManager
    private lateinit var historyRepository: TranslationHistoryRepository

    private var currentSettings = UserSettings()
    private var isTranslating = false
    private val mainHandler = Handler(Looper.getMainLooper())

    // ✅ TUZATILDI: statik instance o'rniga bindService ishlatiladi
    private var screenCaptureService: ScreenCaptureService? = null
    private val captureServiceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName, binder: IBinder) {
            screenCaptureService = (binder as ScreenCaptureService.LocalBinder).getService()
            Log.d("FloatingBubbleService", "ScreenCaptureService bog'landi.")
        }
        override fun onServiceDisconnected(name: ComponentName) {
            screenCaptureService = null
            Log.d("FloatingBubbleService", "ScreenCaptureService uzildi.")
        }
    }

    companion object {
        private const val CHANNEL_ID = "floating_bubble_channel"
        private const val NOTIFICATION_ID = 1002
        const val ACTION_STOP = "com.screentranslator.app.ACTION_STOP_BUBBLE"

        private val _isRunningFlow = MutableStateFlow(false)
        val isRunningFlow: StateFlow<Boolean> = _isRunningFlow.asStateFlow()

        @Volatile
        var isRunning = false
            private set(value) {
                field = value
                _isRunningFlow.value = value
            }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startAsForeground()
        if (intent?.action == ACTION_STOP) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                stopForeground(STOP_FOREGROUND_REMOVE)
            } else {
                @Suppress("DEPRECATION")
                stopForeground(true)
            }
            stopSelf()
            return START_NOT_STICKY
        }
        return START_NOT_STICKY
    }

    override fun onCreate() {
        super.onCreate()
        isRunning = true
        createNotificationChannel()
        startAsForeground()
        windowManager = DisplayCompat.getOverlayWindowManager(this)
        userPreferencesRepository = UserPreferencesRepository(applicationContext)
        ocrProcessor = OcrProcessor()
        historyRepository = TranslationHistoryRepository.getInstance(applicationContext)
        translationManager = TranslationManager(
            applicationContext,
            currentSettings
        )
        overlayManager = OverlayManager(applicationContext)

        // ScreenCaptureService ga bog'lanish
        bindService(
            Intent(this, ScreenCaptureService::class.java),
            captureServiceConnection,
            Context.BIND_AUTO_CREATE
        )

        // Listen for setting changes
        serviceScope.launch {
            userPreferencesRepository.userSettingsFlow.collectLatest { settings ->
                currentSettings = settings
                translationManager.updateSettings(settings)
                updateBubbleAppearance(settings.translationMode)
            }
        }

        createFloatingBubble()
        updateBubbleAppearance(currentSettings.translationMode)
    }

    private fun startAsForeground() {
        val notification = createNotification()
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                startForeground(
                    NOTIFICATION_ID,
                    notification,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE
                )
            } else {
                startForeground(NOTIFICATION_ID, notification)
            }
        } catch (e: Exception) {
            Log.e("FloatingBubbleService", "startForeground xatosi: ${e.message}", e)
        }
    }

    private fun createNotification(): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.notification_title))
            .setContentText(getString(R.string.notification_text))
            .setSmallIcon(R.drawable.ic_notification)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                getString(R.string.notification_channel_name),
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = getString(R.string.notification_channel_desc)
                setShowBadge(false)
            }
            val manager = getSystemService(NotificationManager::class.java)
            manager?.createNotificationChannel(channel)
        }
    }

    private fun updateBubbleAppearance(mode: TranslationMode) {
        val view = bubbleImageView ?: return
        if (mode == TranslationMode.MANGA) {
            // Distinct visual tint for Manga mode
            view.setColorFilter(android.graphics.Color.parseColor("#FBBF24")) // Amber gold accent
        } else {
            view.clearColorFilter()
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun createFloatingBubble() {
        val bubbleSize = dpToPx(56)

        val windowType = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
        } else {
            @Suppress("DEPRECATION")
            WindowManager.LayoutParams.TYPE_PHONE
        }

        layoutParams = WindowManager.LayoutParams(
            bubbleSize,
            bubbleSize,
            windowType,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = 20
            y = 350
        }

        val imageView = ImageView(this).apply {
            setImageResource(R.drawable.ic_translate_bubble)
            background = ContextCompat.getDrawable(this@FloatingBubbleService, R.drawable.bg_bubble_gradient)
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            elevation = dpToPx(8).toFloat()
        }
        bubbleImageView = imageView
        bubbleView = imageView

        // Setup dragging and click touch listener
        setupTouchListener(imageView)

        try {
            windowManager.addView(bubbleView, layoutParams)
        } catch (e: Exception) {
            Log.e("FloatingBubbleService", "Floating bubble qo'shib bo'lmadi: ${e.message}", e)
            stopSelf()
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun setupTouchListener(view: View) {
        view.setOnTouchListener(object : View.OnTouchListener {
            private var initialX = 0
            private var initialY = 0
            private var initialTouchX = 0f
            private var initialTouchY = 0f
            private var isClick = false
            private var isLongClickHandled = false

            private val longPressRunnable = Runnable {
                if (isClick && !isLongClickHandled) {
                    isLongClickHandled = true
                    toggleMangaMode()
                }
            }

            override fun onTouch(v: View, event: MotionEvent): Boolean {
                val params = layoutParams ?: return false

                when (event.action) {
                    MotionEvent.ACTION_DOWN -> {
                        initialX = params.x
                        initialY = params.y
                        initialTouchX = event.rawX
                        initialTouchY = event.rawY
                        isClick = true
                        isLongClickHandled = false
                        mainHandler.postDelayed(longPressRunnable, 600L)
                        return true
                    }
                    MotionEvent.ACTION_MOVE -> {
                        val deltaX = (event.rawX - initialTouchX).toInt()
                        val deltaY = (event.rawY - initialTouchY).toInt()

                        // If user moved finger more than threshold, treat as drag
                        if (abs(deltaX) > 10 || abs(deltaY) > 10) {
                            isClick = false
                            mainHandler.removeCallbacks(longPressRunnable)
                        }

                        val (screenWidth, screenHeight) = getScreenDimensions()
                        params.x = (initialX + deltaX).coerceIn(0, screenWidth - dpToPx(56))
                        params.y = (initialY + deltaY).coerceIn(dpToPx(16), screenHeight - dpToPx(80))
                        try {
                            windowManager.updateViewLayout(bubbleView, params)
                        } catch (e: Exception) {
                            Log.d("FloatingBubbleService", "updateViewLayout xatosi: ${e.message}")
                        }
                        return true
                    }
                    MotionEvent.ACTION_UP -> {
                        mainHandler.removeCallbacks(longPressRunnable)
                        if (isClick && !isLongClickHandled) {
                            handleBubbleClick()
                        } else if (!isClick) {
                            snapToNearestEdge(params)
                        }
                        return true
                    }
                    MotionEvent.ACTION_CANCEL -> {
                        mainHandler.removeCallbacks(longPressRunnable)
                        return true
                    }
                }
                return false
            }
        })
    }

    private fun getScreenDimensions(): Pair<Int, Int> {
        val (w, h, _) = DisplayCompat.getRealScreenDimensions(windowManager, resources.displayMetrics.densityDpi)
        return Pair(w, h)
    }

    private fun toggleMangaMode() {
        if (isTranslating) return
        val newMode = if (currentSettings.translationMode == TranslationMode.MANGA) {
            TranslationMode.STANDARD
        } else {
            TranslationMode.MANGA
        }
        serviceScope.launch {
            userPreferencesRepository.updateTranslationMode(newMode)
        }
        vibrateOnce(80L)
        val msg = if (newMode == TranslationMode.MANGA) {
            "Manga rejimi faollaşdi 🎨"
        } else {
            "Standart rejim faollaşdi 📱"
        }
        showToast(msg)
    }

    /**
     * Magnetically snaps the floating bubble to the left or right edge of the screen.
     */
    private fun snapToNearestEdge(params: WindowManager.LayoutParams) {
        val (screenWidth, screenHeight) = getScreenDimensions()
        val middle = screenWidth / 2
        params.x = if (params.x < middle) dpToPx(8) else screenWidth - dpToPx(64)
        params.y = params.y.coerceIn(dpToPx(16), screenHeight - dpToPx(80))
        try {
            windowManager.updateViewLayout(bubbleView, params)
        } catch (e: Exception) {
            Log.d("FloatingBubbleService", "snapToNearestEdge xatosi: ${e.message}")
        }
    }

    override fun onConfigurationChanged(newConfig: android.content.res.Configuration) {
        super.onConfigurationChanged(newConfig)
        overlayManager.clearOverlays()
        layoutParams?.let { params ->
            snapToNearestEdge(params)
        }
    }

    /**
     * Handles tapping the floating bubble:
     * If overlays are active, dismiss them.
     * Otherwise, capture screen, run OCR, translate text, and draw overlays.
     */
    private fun handleBubbleClick() {
        if (isTranslating) return

        if (overlayManager.hasActiveOverlays()) {
            overlayManager.clearOverlays()
            return
        }

        // ✅ TUZATILDI: statik instance o'rniga bindService orqali olingan havola
        val captureService = screenCaptureService
        if (captureService == null || !ScreenCaptureService.isRunning) {
            Toast.makeText(this, "Ekran yozib oliş xizmati işlamayapti. Iltimos, ilovadan qayta işga tüşiring.", Toast.LENGTH_LONG).show()
            return
        }

        isTranslating = true
        bubbleImageView?.alpha = 0.5f

        serviceScope.launch {
            var bitmap: Bitmap? = null
            try {
                // 1. Hide bubble before screenshot so it doesn't appear in captures
                bubbleView?.visibility = View.INVISIBLE
                kotlinx.coroutines.delay(120)
                captureService.drainStaleFrames()
                bitmap = captureService.captureCurrentScreen()
            } finally {
                bubbleView?.visibility = View.VISIBLE
            }

            try {
                if (bitmap == null) {
                    showToast("Ekran tasvirini olib bölmadi.")
                    return@launch
                }

                val isManga = currentSettings.translationMode == TranslationMode.MANGA
                val ocrLang = currentSettings.mangaSourceLanguage.ocrCode
                val targetLang = currentSettings.targetLanguage.code
                val needsServerForRussian = currentSettings.useMangaServer &&
                    currentSettings.mangaSourceLanguage == MangaSourceLanguage.RUSSIAN
                val useMangaServer = currentSettings.useMangaServer && (isManga || needsServerForRussian)

                // 1. Primary path when Manga Server is enabled: Server-side Bubble Detection, Inpainting colors & OCR
                var mangaServerFailure: String? = null
                if (useMangaServer) {
                    val serverUrl = currentSettings.mangaServerUrl
                    val mangaLang = currentSettings.mangaSourceLanguage.code
                    try {
                        val serverDetectRes = MangaServerClient.detectAndTranslateScreen(
                            serverUrl = serverUrl,
                            bitmap = bitmap,
                            sourceLang = mangaLang,
                            targetLang = targetLang,
                            apiKey = currentSettings.mangaServerApiKey
                        )
                        val serverResponse = serverDetectRes.getOrNull()
                        val serverBlocks = serverResponse?.blocks
                        if (!serverBlocks.isNullOrEmpty()) {
                            val responseLang = MangaSourceLanguage.normalizeToIso(serverResponse.sourceLang)
                                ?: currentSettings.mangaSourceLanguage.toIsoLanguageCode()
                            val convertedBlocks = withContext(Dispatchers.Default) {
                                serverBlocks.map { b ->
                                    val rect = Rect(b.left, b.top, b.right, b.bottom)
                                    val colors = resolveServerBlockColors(b, rect, bitmap)
                                    val blockLang = responseLang ?: MangaBubbleDetector.detectLanguageFromText(b.originalText)
                                    OcrResultBlock(
                                        id = java.util.UUID.randomUUID().toString(),
                                        originalText = b.originalText,
                                        translatedText = b.translatedText,
                                        boundingBox = rect,
                                        sourceLanguage = blockLang,
                                        isMangaBubble = isManga,
                                        isVertical = (b.bottom - b.top) > (b.right - b.left) * 1.15f,
                                        bubbleBgColor = colors.first,
                                        bubbleTextColor = colors.second
                                    )
                                }
                            }
                            overlayManager.showOverlays(convertedBlocks)
                            val entryId = java.util.UUID.randomUUID().toString()
                            val combinedOrig = convertedBlocks.joinToString("\n") { it.originalText }
                            val combinedTrans = convertedBlocks.joinToString("\n") { it.translatedText ?: it.originalText }
                            val savedImgPath = try {
                                withContext(Dispatchers.Default) {
                                    val translatedBmp = renderTranslationsOnBitmap(bitmap, convertedBlocks)
                                    if (translatedBmp != null) {
                                        try {
                                            historyRepository.saveHistoryImage(translatedBmp, entryId)
                                        } finally {
                                            translatedBmp.recycle()
                                        }
                                    } else {
                                        null
                                    }
                                }
                            } catch (e: Exception) {
                                if (e is CancellationException) throw e
                                null
                            }

                            try {
                                historyRepository.addEntry(
                                    TranslationHistoryEntry(
                                        id = entryId,
                                        originalText = combinedOrig,
                                        translatedText = combinedTrans,
                                        targetLanguage = currentSettings.targetLanguage.displayName,
                                        imagePath = savedImgPath
                                    )
                                )
                            } catch (e: Exception) {
                                Log.e("FloatingBubbleService", "Tarixga saqlashda xato: ${e.message}")
                            }
                            vibrateOnce(50L)
                            return@launch
                        } else {
                            mangaServerFailure = serverDetectRes.exceptionOrNull()?.localizedMessage
                        }
                    } catch (e: Exception) {
                        if (e is CancellationException) throw e
                        mangaServerFailure = e.localizedMessage ?: "Manga OCR serveriga ulanib bölmadi."
                        Log.w("FloatingBubbleService", "MangaServer detectAndTranslateScreen xatosi: ${e.message}")
                    }
                }

                // 2. On-device ML Kit OCR (respecting configured source language)
                val rawBlocks = ocrProcessor.processImage(
                    bitmap = bitmap,
                    ocrLanguage = ocrLang,
                    isMangaMode = isManga
                )

                if (rawBlocks.isEmpty()) {
                    val emptyMsg = when {
                        mangaServerFailure != null -> "Manga server: $mangaServerFailure"
                        currentSettings.mangaSourceLanguage == MangaSourceLanguage.RUSSIAN && !currentSettings.useMangaServer ->
                            "Ruscha (Kirill) OCR uchun Sozlamalardan Masofaviy Manga Serverni yoqing."
                        else -> "Ekranda matn topilmadi."
                    }
                    showToast(emptyMsg)
                    return@launch
                }

                // In Manga mode, cluster multiple lines into unified speech bubbles
                val blocks = if (isManga) {
                    val preferredIso = currentSettings.mangaSourceLanguage.toIsoLanguageCode()
                    withContext(Dispatchers.Default) {
                        MangaBubbleDetector.clusterIntoBubbles(
                            rawBlocks = rawBlocks,
                            bitmap = bitmap,
                            sourceLangCode = preferredIso
                        )
                    }
                } else {
                    rawBlocks
                }

                if (blocks.isEmpty()) {
                    showToast("Dialog bulutlari topilmadi.")
                    return@launch
                }

                // 3. Translation:
                // When Manga Server is enabled and healthy, crop each bubble and let the server run manga-ocr/PaddleOCR
                // on the crop (omit ML Kit's text so server OCR is never bypassed).
                val translatedBlocks = if (useMangaServer && mangaServerFailure == null) {
                    val mangaLang = currentSettings.mangaSourceLanguage.code
                    val serverUrl = currentSettings.mangaServerUrl
                    val semaphore = Semaphore(3)
                    withContext(Dispatchers.Default) {
                        blocks.map { block ->
                            async {
                                semaphore.withPermit {
                                    try {
                                        val box = block.boundingBox
                                        val cropX = box.left.coerceIn(0, bitmap.width - 1)
                                        val cropY = box.top.coerceIn(0, bitmap.height - 1)
                                        val cropW = box.width().coerceIn(1, bitmap.width - cropX)
                                        val cropH = box.height().coerceIn(1, bitmap.height - cropY)
                                        val bubbleCrop = Bitmap.createBitmap(bitmap, cropX, cropY, cropW, cropH)
                                        val res = try {
                                            MangaServerClient.translateBubble(
                                                serverUrl = serverUrl,
                                                bitmap = bubbleCrop,
                                                text = null,
                                                sourceLang = mangaLang,
                                                targetLang = targetLang,
                                                apiKey = currentSettings.mangaServerApiKey
                                            )
                                        } finally {
                                            if (!bubbleCrop.isRecycled) bubbleCrop.recycle()
                                        }

                                        val serverData = res.getOrNull()
                                        if (serverData != null && serverData.originalText.isNotBlank()) {
                                            val detectedLang = MangaSourceLanguage.normalizeToIso(
                                                serverData.detectedSourceLang ?: serverData.lang
                                            ) ?: block.sourceLanguage
                                            block.copy(
                                                originalText = serverData.originalText,
                                                translatedText = serverData.translatedText.ifBlank { serverData.originalText },
                                                sourceLanguage = detectedLang
                                            )
                                        } else {
                                            // Fallback to local translation
                                            val localRes = translationManager.translate(
                                                text = block.originalText,
                                                sourceLang = block.sourceLanguage,
                                                targetLangCode = targetLang
                                            )
                                            block.copy(translatedText = localRes.getOrNull() ?: block.originalText)
                                        }
                                    } catch (e: Exception) {
                                        if (e is CancellationException) throw e
                                        val localRes = translationManager.translate(
                                            text = block.originalText,
                                            sourceLang = block.sourceLanguage,
                                            targetLangCode = targetLang
                                        )
                                        block.copy(translatedText = localRes.getOrNull() ?: block.originalText)
                                    }
                                }
                            }
                        }.awaitAll()
                    }
                } else {
                    // Standard translation flow across unique text blocks
                    val uniqueTexts = blocks.map { it.originalText }.distinct()
                    val textToTranslation = withContext(Dispatchers.Default) {
                        val semaphore = Semaphore(4)
                        uniqueTexts.map { text ->
                            async {
                                semaphore.withPermit {
                                    val sampleBlock = blocks.first { it.originalText == text }
                                    val result = translationManager.translate(
                                        text = text,
                                        sourceLang = sampleBlock.sourceLanguage,
                                        targetLangCode = targetLang
                                    )
                                    text to (result.getOrNull() ?: text)
                                }
                            }
                        }.awaitAll().toMap()
                    }
                    blocks.map { block ->
                        block.copy(
                            translatedText = textToTranslation[block.originalText] ?: block.originalText
                        )
                    }
                }

                // 4. Draw overlays
                overlayManager.showOverlays(translatedBlocks)

                // 5. Save to history with translated screenshot
                val entryId = java.util.UUID.randomUUID().toString()
                val combinedOriginal = translatedBlocks.joinToString("\n") { it.originalText }
                val combinedTranslated = translatedBlocks.joinToString("\n") { it.translatedText ?: it.originalText }

                // Render translation badges/bubbles on a copy of bitmap and save
                val savedImagePath = try {
                    withContext(Dispatchers.Default) {
                        val translatedBitmap = renderTranslationsOnBitmap(bitmap, translatedBlocks)
                        if (translatedBitmap != null) {
                            try {
                                historyRepository.saveHistoryImage(translatedBitmap, entryId)
                            } finally {
                                translatedBitmap.recycle()
                            }
                        } else {
                            null
                        }
                    }
                } catch (e: Exception) {
                    if (e is CancellationException) throw e
                    Log.w("FloatingBubbleService", "Rasm saqlashda xato: ${e.message}")
                    null
                }

                try {
                    historyRepository.addEntry(
                        TranslationHistoryEntry(
                            id = entryId,
                            originalText = combinedOriginal,
                            translatedText = combinedTranslated,
                            targetLanguage = currentSettings.targetLanguage.displayName,
                            imagePath = savedImagePath
                        )
                    )
                } catch (e: Exception) {
                    Log.e("FloatingBubbleService", "Tarixga saqlashda xato: ${e.message}", e)
                }

                // 6. Haptic feedback — 50ms vibration (minimal battery impact)
                vibrateOnce(50L)

            } catch (e: Exception) {
                if (e is CancellationException) throw e
                Log.e("FloatingBubbleService", "Tarjima jarayonida xato: ${e.message}", e)
                showToast("Tarjima xatosi: ${e.localizedMessage ?: e.message}")
            } finally {
                bitmap?.let {
                    if (!it.isRecycled) it.recycle()
                }
                isTranslating = false
                bubbleImageView?.alpha = 1.0f
            }
        }
    }

    private fun showToast(msg: String) {
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
    }

    /**
     * Short haptic feedback on translation completion.
     * Uses VibrationEffect on API 26+ (more efficient), falls back to legacy on older devices.
     */
    private fun vibrateOnce(durationMs: Long) {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val manager = getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
                manager.defaultVibrator.vibrate(
                    VibrationEffect.createOneShot(durationMs, VibrationEffect.DEFAULT_AMPLITUDE)
                )
            } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                @Suppress("DEPRECATION")
                val vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
                vibrator.vibrate(VibrationEffect.createOneShot(durationMs, VibrationEffect.DEFAULT_AMPLITUDE))
            } else {
                @Suppress("DEPRECATION")
                val vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
                @Suppress("DEPRECATION")
                vibrator.vibrate(durationMs)
            }
        } catch (e: Exception) {
            Log.d("FloatingBubbleService", "Vibrasiya xatosi: ${e.message}")
        }
    }

    private fun dpToPx(dp: Int): Int {
        val density = resources.displayMetrics.density
        return (dp * density).toInt()
    }

    private fun parseHexColorOrNull(hex: String?): Int? {
        val clean = hex?.trim() ?: return null
        if (clean.isEmpty()) return null
        return runCatching {
            Color.parseColor(if (clean.startsWith("#")) clean else "#$clean") or (0xFF shl 24)
        }.getOrNull()
    }

    private fun resolveServerBlockColors(
        block: MangaDetectedBlock,
        rect: Rect,
        bitmap: Bitmap
    ): Pair<Int, Int> {
        val serverBg = parseHexColorOrNull(block.bgColor)
        val serverText = parseHexColorOrNull(block.textColor)
        if (serverBg != null) {
            val resolvedText = serverText ?: when (block.isDark) {
                true -> Color.WHITE
                false -> Color.parseColor("#111827")
                null -> MangaBubbleDetector.contrastTextColorForBg(serverBg)
            }
            return Pair(serverBg, resolvedText)
        }
        val sampled = MangaBubbleDetector.detectBubbleColors(rect, bitmap)
        val resolvedText = serverText ?: when (block.isDark) {
            true -> Color.WHITE
            false -> Color.parseColor("#111827")
            null -> sampled.second
        }
        return Pair(sampled.first, resolvedText)
    }

    private fun buildFittingStaticLayout(
        text: String,
        textPaint: android.text.TextPaint,
        layoutWidth: Int,
        availableHeight: Float,
        alignment: android.text.Layout.Alignment,
        minPx: Float,
        maxPx: Float
    ): android.text.StaticLayout {
        fun createLayout(sizePx: Float): android.text.StaticLayout {
            textPaint.textSize = sizePx
            return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                android.text.StaticLayout.Builder.obtain(text, 0, text.length, textPaint, layoutWidth)
                    .setAlignment(alignment)
                    .setIncludePad(false)
                    .build()
            } else {
                @Suppress("DEPRECATION")
                android.text.StaticLayout(text, textPaint, layoutWidth, alignment, 1f, 0f, false)
            }
        }

        var low = minPx
        var high = maxOf(minPx, maxPx)
        var bestSize = minPx
        repeat(8) {
            val mid = (low + high) / 2f
            val candidate = createLayout(mid)
            if (candidate.height <= availableHeight) {
                bestSize = mid
                low = mid
            } else {
                high = mid
            }
        }
        return createLayout(bestSize)
    }

    private fun renderTranslationsOnBitmap(
        original: Bitmap,
        blocks: List<OcrResultBlock>
    ): Bitmap? {
        var canvasScale = 1.0f
        val resultBitmap = try {
            original.copy(Bitmap.Config.ARGB_8888, true)
        } catch (_: OutOfMemoryError) {
            canvasScale = 0.5f
            try {
                Bitmap.createScaledBitmap(
                    original,
                    max(1, original.width / 2),
                    max(1, original.height / 2),
                    true
                )
            } catch (_: OutOfMemoryError) {
                return null
            }
        } ?: return null

        val canvas = android.graphics.Canvas(resultBitmap)
        if (canvasScale != 1.0f) {
            canvas.scale(canvasScale, canvasScale)
        }

        val cornerPx = (resources.displayMetrics.density * 4f).coerceIn(4f, 12f)
        val bgPaint = android.graphics.Paint().apply {
            color = 0xFF0F172A.toInt()
            style = android.graphics.Paint.Style.FILL
            isAntiAlias = true
        }
        val borderPaint = android.graphics.Paint().apply {
            color = Color.parseColor("#8038BDF8")
            style = android.graphics.Paint.Style.STROKE
            strokeWidth = 3f
            isAntiAlias = true
        }

        for (block in blocks) {
            val text = block.translatedText?.trim() ?: continue
            if (text.isBlank()) continue
            val rect = block.boundingBox

            val drawRect = android.graphics.RectF(
                rect.left.toFloat().coerceIn(0f, original.width.toFloat()),
                rect.top.toFloat().coerceIn(0f, original.height.toFloat()),
                rect.right.toFloat().coerceIn(0f, original.width.toFloat()),
                rect.bottom.toFloat().coerceIn(0f, original.height.toFloat())
            )
            if (drawRect.width() <= 0f || drawRect.height() <= 0f) continue

            val isVert = block.isVertical || (rect.height() > rect.width() * 1.25f)

            if (block.isMangaBubble) {
                val opaqueBg = block.bubbleBgColor or (0xFF shl 24)
                val bubbleBg = android.graphics.Paint().apply {
                    color = opaqueBg
                    style = android.graphics.Paint.Style.FILL
                    isAntiAlias = true
                }
                val bubbleStroke = android.graphics.Paint().apply {
                    color = if (MangaBubbleDetector.isColorLight(opaqueBg)) {
                        Color.parseColor("#CBD5E1")
                    } else {
                        Color.parseColor("#3F3F46")
                    }
                    style = android.graphics.Paint.Style.STROKE
                    strokeWidth = 3f
                    isAntiAlias = true
                }

                // Expand horizontally for vertical original text while keeping the full original drawRect covered
                val bubbleRect = if (isVert) {
                    val desiredW = maxOf(drawRect.width(), (original.width * 0.38f).coerceIn(220f, 560f))
                    val cX = drawRect.centerX()
                    var l = (cX - desiredW / 2f).coerceIn(0f, (original.width - desiredW).coerceAtLeast(0f))
                    if (l > drawRect.left) l = drawRect.left
                    val r = maxOf(l + desiredW, drawRect.right).coerceAtMost(original.width.toFloat())
                    val b = maxOf(drawRect.bottom, drawRect.top + 80f).coerceAtMost(original.height.toFloat())
                    android.graphics.RectF(l, drawRect.top, r, b)
                } else {
                    android.graphics.RectF(drawRect)
                }

                val mangaTextPaint = android.text.TextPaint().apply {
                    color = block.bubbleTextColor
                    isAntiAlias = true
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                }

                val padX = 12f
                val padY = 8f
                val layoutWidth = max(1, (bubbleRect.width() - padX * 2f).toInt())
                val availableHeight = max(16f, bubbleRect.height() - padY * 2f)
                val staticLayout = buildFittingStaticLayout(
                    text = text,
                    textPaint = mangaTextPaint,
                    layoutWidth = layoutWidth,
                    availableHeight = availableHeight,
                    alignment = android.text.Layout.Alignment.ALIGN_CENTER,
                    minPx = 14f,
                    maxPx = 52f
                )

                val neededBottom = bubbleRect.top + staticLayout.height + padY * 2f
                if (neededBottom > bubbleRect.bottom) {
                    bubbleRect.bottom = maxOf(drawRect.bottom, neededBottom).coerceAtMost(original.height.toFloat())
                }

                canvas.drawRoundRect(bubbleRect, cornerPx, cornerPx, bubbleBg)
                canvas.drawRoundRect(bubbleRect, cornerPx, cornerPx, bubbleStroke)

                canvas.save()
                val startY = bubbleRect.centerY() - (staticLayout.height / 2f)
                canvas.translate(bubbleRect.left + padX, startY)
                staticLayout.draw(canvas)
                canvas.restore()
            } else {
                // Standard mode opaque badge rendering covering the full original footprint
                val cardRect = if (isVert) {
                    val desiredW = maxOf(drawRect.width(), (original.width * 0.35f).coerceIn(220f, 520f))
                    val cX = drawRect.centerX()
                    var l = (cX - desiredW / 2f).coerceIn(0f, (original.width - desiredW).coerceAtLeast(0f))
                    if (l > drawRect.left) l = drawRect.left
                    val r = maxOf(l + desiredW, drawRect.right).coerceAtMost(original.width.toFloat())
                    val b = maxOf(drawRect.bottom, drawRect.top + 72f).coerceAtMost(original.height.toFloat())
                    android.graphics.RectF(l, drawRect.top, r, b)
                } else {
                    val pad = 4f
                    android.graphics.RectF(
                        (drawRect.left - pad).coerceAtLeast(0f),
                        (drawRect.top - pad).coerceAtLeast(0f),
                        (drawRect.right + pad).coerceAtMost(original.width.toFloat()),
                        (drawRect.bottom + pad).coerceAtMost(original.height.toFloat())
                    )
                }

                val badgeTextPaint = android.text.TextPaint().apply {
                    color = Color.WHITE
                    isAntiAlias = true
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                }

                val padX = 8f
                val padY = 6f
                val layoutWidth = max(1, (cardRect.width() - padX * 2f).toInt())
                val availableHeight = max(14f, cardRect.height() - padY * 2f)
                val staticLayout = buildFittingStaticLayout(
                    text = text,
                    textPaint = badgeTextPaint,
                    layoutWidth = layoutWidth,
                    availableHeight = availableHeight,
                    alignment = if (isVert) android.text.Layout.Alignment.ALIGN_CENTER else android.text.Layout.Alignment.ALIGN_NORMAL,
                    minPx = 14f,
                    maxPx = 42f
                )

                val neededBottom = cardRect.top + staticLayout.height + padY * 2f
                if (neededBottom > cardRect.bottom) {
                    cardRect.bottom = maxOf(drawRect.bottom, neededBottom).coerceAtMost(original.height.toFloat())
                }

                canvas.drawRoundRect(cardRect, cornerPx, cornerPx, bgPaint)
                canvas.drawRoundRect(cardRect, cornerPx, cornerPx, borderPaint)

                canvas.save()
                val startY = cardRect.centerY() - (staticLayout.height / 2f)
                canvas.translate(cardRect.left + padX, startY)
                staticLayout.draw(canvas)
                canvas.restore()
            }
        }

        return resultBitmap
    }

    override fun onDestroy() {
        isRunning = false
        mainHandler.removeCallbacksAndMessages(null)
        serviceScope.cancel()
        overlayManager.clearOverlays()
        ocrProcessor.release()
        translationManager.release()

        // ✅ Bog'liqlikni uzish
        try {
            unbindService(captureServiceConnection)
        } catch (e: Exception) {
            Log.d("FloatingBubbleService", "unbindService xatosi (normal): ${e.message}")
        }
        screenCaptureService = null

        bubbleView?.let {
            try {
                windowManager.removeView(it)
            } catch (e: Exception) {
                Log.d("FloatingBubbleService", "removeView xatosi: ${e.message}")
            }
        }
        bubbleView = null
        super.onDestroy()
    }
}
