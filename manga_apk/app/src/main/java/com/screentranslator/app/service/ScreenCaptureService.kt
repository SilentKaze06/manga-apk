package com.screentranslator.app.service

import android.annotation.SuppressLint
import android.app.Activity
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Binder
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Looper
import android.util.Log
import android.view.WindowManager
import androidx.core.app.NotificationCompat
import com.screentranslator.app.MainActivity
import com.screentranslator.app.R
import com.screentranslator.app.util.DisplayCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.cancel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.coroutines.resume

/**
 * Foreground Service that manages MediaProjection and captures on-demand single-frame screenshots.
 * Complies with Android 10-15 Foreground Service types and MediaProjection lifecycle.
 */
class ScreenCaptureService : Service() {

    inner class LocalBinder : Binder() {
        fun getService(): ScreenCaptureService = this@ScreenCaptureService
    }

    private val binder = LocalBinder()

    private var mediaProjection: MediaProjection? = null
    private var windowManager: WindowManager? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null

    private val captureMutex = Mutex()
    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null
    private var currentWidth = 0
    private var currentHeight = 0
    private var currentDensity = 0

    companion object {
        const val CHANNEL_ID = "screen_translator_capture_channel"
        const val NOTIFICATION_ID = 1001
        const val ACTION_START = "com.screentranslator.app.ACTION_START_CAPTURE"
        const val ACTION_STOP = "com.screentranslator.app.ACTION_STOP_CAPTURE"
        const val EXTRA_RESULT_CODE = "extra_result_code"
        const val EXTRA_RESULT_DATA = "extra_result_data"

        private val _isRunningFlow = MutableStateFlow(false)
        val isRunningFlow: StateFlow<Boolean> = _isRunningFlow.asStateFlow()

        @Volatile
        var isRunning = false
            private set(value) {
                field = value
                _isRunningFlow.value = value
            }

        @Volatile
        private var pendingResultCode: Int = 0
        @Volatile
        private var pendingResultData: Intent? = null

        fun setPendingProjectionData(code: Int, data: Intent) {
            pendingResultCode = code
            pendingResultData = data
        }

        fun startCapture(context: Context, resultCode: Int, resultData: Intent) {
            if (resultCode != Activity.RESULT_OK) {
                Log.w("ScreenCaptureService", "startCapture rejected: resultCode is not RESULT_OK ($resultCode)")
                return
            }
            setPendingProjectionData(resultCode, resultData)
            val intent = Intent(context, ScreenCaptureService::class.java).apply {
                action = ACTION_START
                putExtra(EXTRA_RESULT_CODE, resultCode)
                putExtra(EXTRA_RESULT_DATA, resultData)
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }

        fun stop(context: Context) {
            try {
                val intent = Intent(context, ScreenCaptureService::class.java)
                context.stopService(intent)
            } catch (e: Exception) {
                Log.w("ScreenCaptureService", "stopService error: ${e.message}")
            }
            isRunning = false
        }
    }

    override fun onCreate() {
        super.onCreate()
        windowManager = DisplayCompat.getOverlayWindowManager(this)
        val thread = HandlerThread("ScreenCaptureBackground")
        thread.start()
        backgroundThread = thread
        backgroundHandler = Handler(thread.looper)
        createNotificationChannel()
    }

    override fun onBind(intent: Intent?): IBinder = binder

    private fun stopSelfCleanly() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(STOP_FOREGROUND_REMOVE)
        } else {
            @Suppress("DEPRECATION")
            stopForeground(true)
        }
        stopSelf()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startAsForeground()
        when (intent?.action) {
            ACTION_START -> {
                val resultCode = if (pendingResultCode != 0) pendingResultCode else intent.getIntExtra(EXTRA_RESULT_CODE, 0)
                val resultData: Intent? = pendingResultData ?: run {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                        try {
                            intent.getParcelableExtra(EXTRA_RESULT_DATA, Intent::class.java)
                        } catch (e: Exception) {
                            null
                        }
                    } else {
                        @Suppress("DEPRECATION")
                        intent.getParcelableExtra(EXTRA_RESULT_DATA)
                    }
                }

                // Immediately clear static references after reading to prevent memory leaks
                pendingResultCode = 0
                pendingResultData = null

                if (resultCode == Activity.RESULT_OK && resultData != null) {
                    stopCapture()
                    val initialized = initMediaProjection(resultCode, resultData)
                    if (initialized) {
                        isRunning = true
                    } else {
                        isRunning = false
                        stopSelfCleanly()
                    }
                } else {
                    Log.e("ScreenCaptureService", "resultCode=$resultCode yoki resultData=$resultData yaroqsiz")
                    stopSelfCleanly()
                }
            }
            ACTION_STOP -> {
                serviceScope.launch {
                    try {
                        captureMutex.withLock {
                            stopCapture()
                        }
                    } finally {
                        withContext(Dispatchers.Main) {
                            stopSelfCleanly()
                        }
                    }
                }
            }
            else -> {
                stopSelfCleanly()
            }
        }
        return START_NOT_STICKY
    }

    private fun startAsForeground() {
        val notification = createNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun initMediaProjection(resultCode: Int, resultData: Intent): Boolean {
        return try {
            val projectionManager =
                getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
            val projection = projectionManager.getMediaProjection(resultCode, resultData)
            if (projection == null) {
                Log.e("ScreenCaptureService", "MediaProjection is null!")
                return false
            }
            mediaProjection = projection

            val handler = backgroundHandler ?: mainHandler
            projection.registerCallback(object : MediaProjection.Callback() {
                override fun onStop() {
                    super.onStop()
                    serviceScope.launch {
                        var stoppedActiveProjection = false
                        captureMutex.withLock {
                            if (mediaProjection === projection) {
                                releaseVirtualDisplay()
                                mediaProjection = null
                                isRunning = false
                                stoppedActiveProjection = true
                            }
                        }
                        if (stoppedActiveProjection) {
                            withContext(Dispatchers.Main) {
                                try {
                                    stopService(Intent(this@ScreenCaptureService, FloatingBubbleService::class.java))
                                } catch (ignored: Exception) {}
                                stopSelfCleanly()
                            }
                        }
                    }
                    Log.d("ScreenCaptureService", "MediaProjection stopped by system.")
                }
            }, handler)
            true
        } catch (e: Exception) {
            Log.e("ScreenCaptureService", "initMediaProjection xatosi: ${e.message}", e)
            false
        }
    }

    private fun ensureVirtualDisplay(projection: MediaProjection, width: Int, height: Int, density: Int): ImageReader {
        val existingReader = imageReader
        val existingDisplay = virtualDisplay
        val handler = backgroundHandler ?: mainHandler

        if (existingReader != null && existingDisplay != null) {
            if (currentWidth == width && currentHeight == height) {
                return existingReader
            }
            currentWidth = width
            currentHeight = height
            currentDensity = density

            @SuppressLint("WrongConstant")
            val newReader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2)
            imageReader = newReader

            var resizeSuccess = false
            try {
                existingDisplay.resize(width, height, density)
                existingDisplay.surface = newReader.surface
                resizeSuccess = true
            } catch (e: Exception) {
                Log.w("ScreenCaptureService", "VirtualDisplay resize xatosi, recreating: ${e.message}")
            }

            try {
                existingReader.close()
            } catch (ignored: Exception) {}

            if (!resizeSuccess) {
                try {
                    existingDisplay.release()
                } catch (ignored: Exception) {}
                virtualDisplay = projection.createVirtualDisplay(
                    "ScreenTranslatorCapture",
                    width,
                    height,
                    density,
                    DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                    newReader.surface,
                    null,
                    handler
                )
            }

            return newReader
        }

        currentWidth = width
        currentHeight = height
        currentDensity = density

        @SuppressLint("WrongConstant")
        val reader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2)
        imageReader = reader

        virtualDisplay = projection.createVirtualDisplay(
            "ScreenTranslatorCapture",
            width,
            height,
            density,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            reader.surface,
            null,
            handler
        )

        return reader
    }

    /**
     * Drains any stale image frames buffered in ImageReader to ensure
     * that screenshot capture grabs an up-to-date screen state.
     */
    suspend fun drainStaleFrames() = captureMutex.withLock {
        try {
            var img = imageReader?.acquireLatestImage()
            while (img != null) {
                img.close()
                img = imageReader?.acquireLatestImage()
            }
        } catch (ignored: Exception) {}
    }

    suspend fun drainStaleFrame() = drainStaleFrames()

    /**
     * Captures a single on-demand screenshot of the current screen.
     * Synchronized with captureMutex to prevent concurrent reader access.
     */
    suspend fun captureCurrentScreen(): Bitmap? = captureMutex.withLock {
        withContext(Dispatchers.Default) {
            val projection = mediaProjection ?: run {
                Log.e("ScreenCaptureService", "MediaProjection is null. Cannot capture screen.")
                return@withContext null
            }

            val wm = windowManager ?: return@withContext null
            val (width, height, density) = DisplayCompat.getRealScreenDimensions(
                wm,
                resources.displayMetrics.densityDpi
            )

            if (width <= 0 || height <= 0) return@withContext null

            try {
                val reader = ensureVirtualDisplay(projection, width, height, density)

                delay(40)

                // Try to acquire immediately if a frame has arrived
                var directImage: Image? = null
                try {
                    directImage = reader.acquireLatestImage()
                    if (directImage != null) {
                        val bitmap = convertImageToBitmap(directImage, width, height)
                        if (bitmap != null) {
                            return@withContext bitmap
                        }
                    }
                } catch (ignored: Exception) {
                } finally {
                    directImage?.close()
                }

                // Wait for image frame with timeout, attaching listener before updating surface
                withTimeoutOrNull(1500L) {
                    coroutineScope {
                        val waitDeferred = async {
                            waitForImage(reader, width, height)
                        }
                        // If acquireLatestImage returned null (static screen), refresh surface to prompt frame arrival
                        virtualDisplay?.surface = reader.surface
                        waitDeferred.await()
                    }
                }
            } catch (e: Exception) {
                Log.e("ScreenCaptureService", "Error during screen capture: ${e.message}", e)
                null
            }
        }
    }

    private suspend fun waitForImage(imageReader: ImageReader, width: Int, height: Int): Bitmap? {
        val handler = backgroundHandler ?: mainHandler
        return suspendCancellableCoroutine { continuation ->
            imageReader.setOnImageAvailableListener({ reader ->
                var image: Image? = null
                try {
                    image = reader.acquireLatestImage()
                    if (image != null && continuation.isActive) {
                        reader.setOnImageAvailableListener(null, null)
                        val bitmap = convertImageToBitmap(image, width, height)
                        continuation.resume(bitmap)
                    }
                } catch (e: Exception) {
                    if (continuation.isActive) {
                        reader.setOnImageAvailableListener(null, null)
                        continuation.resume(null)
                    }
                } finally {
                    image?.close()
                }
            }, handler)

            continuation.invokeOnCancellation {
                imageReader.setOnImageAvailableListener(null, null)
            }
        }
    }

    /**
     * Converts raw Image buffer to Android Bitmap considering rowStride and pixelStride padding.
     * Safely reads row-by-row using buffer.position(row * rowStride) for arbitrary alignments,
     * avoiding ByteBuffer.allocateDirect to prevent native memory exhaustion.
     */
    private fun convertImageToBitmap(image: Image, width: Int, height: Int): Bitmap? {
        return try {
            val planes = image.planes
            val buffer = planes[0].buffer
            val pixelStride = planes[0].pixelStride
            val rowStride = planes[0].rowStride

            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

            if (rowStride == width * pixelStride) {
                buffer.position(0)
                bitmap.copyPixelsFromBuffer(buffer)
            } else {
                val rowByteArray = ByteArray(width * pixelStride)
                val rowPixels = IntArray(width)
                for (row in 0 until height) {
                    buffer.position(row * rowStride)
                    buffer.get(rowByteArray, 0, rowByteArray.size)
                    for (col in 0 until width) {
                        val idx = col * pixelStride
                        val r = rowByteArray[idx].toInt() and 0xFF
                        val g = rowByteArray[idx + 1].toInt() and 0xFF
                        val b = rowByteArray[idx + 2].toInt() and 0xFF
                        val a = rowByteArray[idx + 3].toInt() and 0xFF
                        rowPixels[col] = (a shl 24) or (r shl 16) or (g shl 8) or b
                    }
                    bitmap.setPixels(rowPixels, 0, width, 0, row, width, 1)
                }
            }
            bitmap
        } catch (e: Exception) {
            Log.e("ScreenCaptureService", "convertImageToBitmap error: ${e.message}", e)
            null
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

    private fun releaseVirtualDisplay() {
        try {
            imageReader?.setOnImageAvailableListener(null, null)
            virtualDisplay?.release()
            imageReader?.close()
        } catch (ignored: Exception) {}
        virtualDisplay = null
        imageReader = null
    }

    private fun stopCapture() {
        releaseVirtualDisplay()
        try {
            mediaProjection?.stop()
        } catch (ignored: Exception) {}
        mediaProjection = null
        isRunning = false
    }

    override fun onDestroy() {
        isRunning = false
        serviceScope.cancel()
        CoroutineScope(Dispatchers.Default).launch {
            try {
                captureMutex.withLock {
                    stopCapture()
                }
            } catch (ignored: Exception) {
            } finally {
                backgroundThread?.quitSafely()
                backgroundThread = null
                backgroundHandler = null
            }
        }
        super.onDestroy()
    }
}
