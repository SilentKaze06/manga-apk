package com.screentranslator.app.service

import android.annotation.SuppressLint
import android.app.Service
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.os.Build
import android.os.IBinder
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.DisplayMetrics
import android.util.Log
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.ImageView
import android.widget.Toast
import androidx.core.content.ContextCompat
import android.graphics.Rect
import com.screentranslator.app.R
import com.screentranslator.app.ScreenTranslatorApp
import com.screentranslator.app.data.model.OcrResultBlock
import com.screentranslator.app.data.model.MangaSourceLanguage
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
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.collectLatest
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
        // ✅ TUZATILDI: @Volatile qo'shildi — ko'p oqimli muhitda xavfsiz o'qish
        @Volatile
        var isRunning = false
            private set
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        isRunning = true
        windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        userPreferencesRepository = UserPreferencesRepository(applicationContext)
        ocrProcessor = OcrProcessor()
        historyRepository = TranslationHistoryRepository.getInstance(applicationContext)
        translationManager = TranslationManager(
            applicationContext,
            currentSettings
        )
        overlayManager = OverlayManager(applicationContext)

        // ✅ ScreenCaptureService ga bog'lanish (u allaqachon ishlab turgan bo'lishi kerak)
        bindService(
            Intent(this, ScreenCaptureService::class.java),
            captureServiceConnection,
            0  // Agar service ishlamayotgan bo'lsa — yaratilmaydi
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
        val handler = android.os.Handler(android.os.Looper.getMainLooper())

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
                        handler.postDelayed(longPressRunnable, 600L)
                        return true
                    }
                    MotionEvent.ACTION_MOVE -> {
                        val deltaX = (event.rawX - initialTouchX).toInt()
                        val deltaY = (event.rawY - initialTouchY).toInt()

                        // If user moved finger more than threshold, treat as drag
                        if (abs(deltaX) > 10 || abs(deltaY) > 10) {
                            isClick = false
                            handler.removeCallbacks(longPressRunnable)
                        }

                        params.x = initialX + deltaX
                        params.y = initialY + deltaY
                        try {
                            windowManager.updateViewLayout(bubbleView, params)
                        } catch (e: Exception) {
                            Log.d("FloatingBubbleService", "updateViewLayout xatosi: ${e.message}")
                        }
                        return true
                    }
                    MotionEvent.ACTION_UP -> {
                        handler.removeCallbacks(longPressRunnable)
                        if (isClick && !isLongClickHandled) {
                            handleBubbleClick()
                        } else if (!isClick) {
                            snapToNearestEdge(params)
                        }
                        return true
                    }
                    MotionEvent.ACTION_CANCEL -> {
                        handler.removeCallbacks(longPressRunnable)
                        return true
                    }
                }
                return false
            }
        })
    }

    private fun toggleMangaMode() {
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
            "Manga rejimi faollashdi 🎨"
        } else {
            "Standart rejim faollashdi 📱"
        }
        showToast(msg)
    }

    /**
     * Magnetically snaps the floating bubble to the left or right edge of the screen.
     */
    private fun snapToNearestEdge(params: WindowManager.LayoutParams) {
        // ✅ TUZATILDI: Zamonaviy WindowMetrics API (Android 11+)
        val screenWidth = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            windowManager.currentWindowMetrics.bounds.width()
        } else {
            val displayMetrics = DisplayMetrics()
            @Suppress("DEPRECATION")
            windowManager.defaultDisplay.getMetrics(displayMetrics)
            displayMetrics.widthPixels
        }

        val middle = screenWidth / 2
        params.x = if (params.x < middle) dpToPx(8) else screenWidth - dpToPx(64)
        try {
            windowManager.updateViewLayout(bubbleView, params)
        } catch (e: Exception) {
            Log.d("FloatingBubbleService", "snapToNearestEdge xatosi: ${e.message}")
        }
    }

    /**
     * Handles tapping the floating bubble:
     * If overlays are active, dismiss them.
     * Otherwise, capture screen, run OCR, translate text, and draw overlays.
     */
    private fun handleBubbleClick() {
        if (overlayManager.hasActiveOverlays()) {
            overlayManager.clearOverlays()
            return
        }

        if (isTranslating) return

        // ✅ TUZATILDI: statik instance o'rniga bindService orqali olingan havola
        val captureService = screenCaptureService
        if (captureService == null || !ScreenCaptureService.isRunning) {
            Toast.makeText(this, "Ekran yozib olish xizmati ishlamayapti. Iltimos, ilovadan qayta ishga tushiring.", Toast.LENGTH_LONG).show()
            return
        }

        isTranslating = true
        bubbleImageView?.alpha = 0.5f

        serviceScope.launch {
            var bitmap: Bitmap? = null
            try {
                // 1. Capture screen
                bitmap = captureService.captureCurrentScreen()
                if (bitmap == null) {
                    showToast("Ekran tasvirini olib bo'lmadi.")
                    return@launch
                }

                val isManga = currentSettings.translationMode == TranslationMode.MANGA
                val ocrLang = if (isManga) currentSettings.mangaSourceLanguage.ocrCode else null
                val targetLang = currentSettings.targetLanguage.code
                val useMangaServer = isManga && currentSettings.useMangaServer

                // 2. OCR via on-device ML Kit (optimized & fast) with OCR.space fallback
                var rawBlocks = ocrProcessor.processImage(
                    bitmap = bitmap,
                    apiKey = currentSettings.ocrApiKey,
                    ocrLanguage = ocrLang,
                    isMangaMode = isManga
                )

                // If on-device ML Kit missed vertical stylized manga text (0 blocks found)
                // and Manga OCR Server is enabled, attempt whole-page detection via Python server!
                var mangaServerFailure: String? = null
                if (rawBlocks.isEmpty() && useMangaServer) {
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
                        val serverBlocks = serverDetectRes.getOrNull()?.blocks
                        if (!serverBlocks.isNullOrEmpty()) {
                            val convertedBlocks = serverBlocks.map { b ->
                                OcrResultBlock(
                                    id = java.util.UUID.randomUUID().toString(),
                                    originalText = b.originalText,
                                    translatedText = b.translatedText,
                                    boundingBox = Rect(b.left, b.top, b.right, b.bottom),
                                    sourceLanguage = mangaLang,
                                    isMangaBubble = true,
                                    isVertical = (b.bottom - b.top) > (b.right - b.left) * 1.15f
                                )
                            }
                            overlayManager.showOverlays(convertedBlocks)
                            val entryId = java.util.UUID.randomUUID().toString()
                            val combinedOrig = convertedBlocks.joinToString("\n") { it.originalText }
                            val combinedTrans = convertedBlocks.joinToString("\n") { it.translatedText ?: it.originalText }
                            val savedImgPath = try {
                                val translatedBmp = renderTranslationsOnBitmap(bitmap, convertedBlocks)
                                try {
                                    historyRepository.saveHistoryImage(translatedBmp, entryId)
                                } finally {
                                    translatedBmp.recycle()
                                }
                            } catch (e: Exception) { null }

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
                                ?: "Server matn bloklarini aniqlamadi."
                        }
                    } catch (e: Exception) {
                        mangaServerFailure = e.localizedMessage ?: "Manga OCR serveriga ulanib bo'lmadi."
                        Log.w("FloatingBubbleService", "MangaServer detectAndTranslateScreen xatosi: ${e.message}")
                    }
                }

                if (rawBlocks.isEmpty()) {
                    showToast(mangaServerFailure?.let { "Manga server: $it" } ?: "Ekranda matn topilmadi.")
                    return@launch
                }

                // In Manga mode, cluster multiple lines into unified speech bubbles
                val blocks = if (isManga) {
                    val preferredIso = currentSettings.mangaSourceLanguage.toIsoLanguageCode()
                    MangaBubbleDetector.clusterIntoBubbles(
                        rawBlocks = rawBlocks,
                        bitmap = bitmap,
                        sourceLangCode = preferredIso
                    )
                } else {
                    rawBlocks
                }

                if (blocks.isEmpty()) {
                    showToast("Dialog bulutlari topilmadi.")
                    return@launch
                }

                // 3. Translation:
                // When in Manga Mode with Python Manga OCR Server active:
                // Crop each detected speech bubble from the screenshot and send to Manga-OCR (ViT) & PaddleOCR
                val translatedBlocks = if (useMangaServer) {
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
                                                sourceLang = mangaLang,
                                                targetLang = targetLang,
                                                apiKey = currentSettings.mangaServerApiKey
                                            )
                                        } finally {
                                            if (!bubbleCrop.isRecycled) bubbleCrop.recycle()
                                        }

                                        val serverData = res.getOrNull()
                                        if (serverData != null && serverData.originalText.isNotBlank()) {
                                            block.copy(
                                                originalText = serverData.originalText,
                                                translatedText = serverData.translatedText.ifBlank { serverData.originalText }
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
                    val translatedBitmap = renderTranslationsOnBitmap(bitmap, translatedBlocks)
                    try {
                        historyRepository.saveHistoryImage(translatedBitmap, entryId)
                    } finally {
                        translatedBitmap.recycle()
                    }
                } catch (e: Exception) {
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

    private fun renderTranslationsOnBitmap(
        original: Bitmap,
        blocks: List<com.screentranslator.app.data.model.OcrResultBlock>
    ): Bitmap {
        val resultBitmap = original.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = android.graphics.Canvas(resultBitmap)

        val bgPaint = android.graphics.Paint().apply {
            color = android.graphics.Color.parseColor("#E60F172A") // Dark slate 90% opacity
            style = android.graphics.Paint.Style.FILL
            isAntiAlias = true
        }
        val borderPaint = android.graphics.Paint().apply {
            color = android.graphics.Color.parseColor("#8038BDF8") // Cyan border 50% opacity
            style = android.graphics.Paint.Style.STROKE
            strokeWidth = 3f
            isAntiAlias = true
        }
        val textPaint = android.graphics.Paint().apply {
            color = android.graphics.Color.WHITE
            isAntiAlias = true
            typeface = android.graphics.Typeface.DEFAULT_BOLD
        }

        for (block in blocks) {
            val text = block.translatedText?.trim() ?: continue
            if (text.isBlank()) continue
            val rect = block.boundingBox

            val drawRect = android.graphics.RectF(
                rect.left.toFloat().coerceAtLeast(0f),
                rect.top.toFloat().coerceAtLeast(0f),
                rect.right.toFloat().coerceAtMost(original.width.toFloat()),
                rect.bottom.toFloat().coerceAtMost(original.height.toFloat())
            )

            val isVert = block.isVertical || (rect.height() > rect.width() * 1.25f)

            if (block.isMangaBubble) {
                // In-place Manga speech bubble rendering
                val bubbleBg = android.graphics.Paint().apply {
                    color = block.bubbleBgColor
                    style = android.graphics.Paint.Style.FILL
                    isAntiAlias = true
                }
                val bubbleStroke = android.graphics.Paint().apply {
                    color = if (block.bubbleBgColor == android.graphics.Color.WHITE) {
                        android.graphics.Color.parseColor("#CBD5E1")
                    } else {
                        android.graphics.Color.parseColor("#3F3F46")
                    }
                    style = android.graphics.Paint.Style.STROKE
                    strokeWidth = 3f
                    isAntiAlias = true
                }

                // If original text was vertical, expand horizontally on screenshot for readable layout
                val bubbleRect = if (isVert) {
                    val desiredW = maxOf(drawRect.width(), (original.width * 0.38f).coerceIn(280f, 600f))
                    val cX = drawRect.centerX()
                    val l = (cX - desiredW / 2f).coerceIn(16f, (original.width - desiredW - 16f).coerceAtLeast(16f))
                    val h = maxOf(drawRect.height(), 90f)
                    android.graphics.RectF(l, drawRect.top, l + desiredW, drawRect.top + h)
                } else {
                    drawRect
                }

                canvas.drawRoundRect(bubbleRect, 20f, 20f, bubbleBg)
                canvas.drawRoundRect(bubbleRect, 20f, 20f, bubbleStroke)

                // Typeset centered text
                val charCount = max(1, text.length)
                val area = bubbleRect.width() * bubbleRect.height()
                val targetSize = (kotlin.math.sqrt(area.toDouble() / (charCount * 1.15)) * 1.7f)
                    .toFloat().coerceIn(18f, 36f)

                val mangaTextPaint = android.text.TextPaint().apply {
                    color = block.bubbleTextColor
                    isAntiAlias = true
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                    textSize = targetSize
                }

                val layoutWidth = max(1, (bubbleRect.width() - 24f).toInt())
                val staticLayout = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    android.text.StaticLayout.Builder.obtain(text, 0, text.length, mangaTextPaint, layoutWidth)
                        .setAlignment(android.text.Layout.Alignment.ALIGN_CENTER)
                        .setIncludePad(false)
                        .build()
                } else {
                    @Suppress("DEPRECATION")
                    android.text.StaticLayout(text, mangaTextPaint, layoutWidth, android.text.Layout.Alignment.ALIGN_CENTER, 1f, 0f, false)
                }

                canvas.save()
                val startY = bubbleRect.centerY() - (staticLayout.height / 2f)
                canvas.translate(bubbleRect.left + 12f, startY)
                staticLayout.draw(canvas)
                canvas.restore()
            } else {
                // Standard mode badge rendering
                val cardRect = if (isVert) {
                    val desiredW = maxOf(drawRect.width(), (original.width * 0.35f).coerceIn(240f, 550f))
                    val cX = drawRect.centerX()
                    val l = (cX - desiredW / 2f).coerceIn(16f, (original.width - desiredW - 16f).coerceAtLeast(16f))
                    val h = maxOf(drawRect.height(), 80f)
                    android.graphics.RectF(l, drawRect.top, l + desiredW, drawRect.top + h)
                } else {
                    val padX = 8f
                    val padY = 4f
                    android.graphics.RectF(
                        (rect.left - padX).coerceAtLeast(0f),
                        (rect.top - padY).coerceAtLeast(0f),
                        (rect.right + padX).coerceAtMost(original.width.toFloat()),
                        (rect.bottom + padY).coerceAtMost(original.height.toFloat())
                    )
                }

                canvas.drawRoundRect(cardRect, 12f, 12f, bgPaint)
                canvas.drawRoundRect(cardRect, 12f, 12f, borderPaint)

                val badgeTextPaint = android.text.TextPaint().apply {
                    color = android.graphics.Color.WHITE
                    isAntiAlias = true
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                    textSize = (cardRect.height() * 0.45f).coerceIn(20f, 32f)
                }

                val layoutWidth = max(1, (cardRect.width() - 16f).toInt())
                val staticLayout = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    android.text.StaticLayout.Builder.obtain(text, 0, text.length, badgeTextPaint, layoutWidth)
                        .setAlignment(if (isVert) android.text.Layout.Alignment.ALIGN_CENTER else android.text.Layout.Alignment.ALIGN_NORMAL)
                        .setIncludePad(false)
                        .build()
                } else {
                    @Suppress("DEPRECATION")
                    android.text.StaticLayout(text, badgeTextPaint, layoutWidth, android.text.Layout.Alignment.ALIGN_NORMAL, 1f, 0f, false)
                }

                canvas.save()
                val startY = cardRect.centerY() - (staticLayout.height / 2f)
                canvas.translate(cardRect.left + 8f, startY)
                staticLayout.draw(canvas)
                canvas.restore()
            }
        }

        return resultBitmap
    }

    override fun onDestroy() {
        isRunning = false
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
