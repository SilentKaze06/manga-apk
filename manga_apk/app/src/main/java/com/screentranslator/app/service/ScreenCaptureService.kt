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
import android.os.IBinder
import android.os.Looper
import android.util.DisplayMetrics
import android.util.Log
import android.view.WindowManager
import androidx.core.app.NotificationCompat
import com.screentranslator.app.MainActivity
import com.screentranslator.app.R
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.suspendCancellableCoroutine
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

        // ✅ TUZATILDI: @Volatile qo'shildi — ko'p oqimli muhitda xavfsiz o'qish
        @Volatile
        var isRunning = false
            private set

        @Volatile
        private var pendingResultCode: Int = 0
        @Volatile
        private var pendingResultData: Intent? = null

        fun setPendingProjectionData(code: Int, data: Intent) {
            pendingResultCode = code
            pendingResultData = data
        }
    }

    override fun onCreate() {
        super.onCreate()
        windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        createNotificationChannel()
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> {
                val resultCode = if (pendingResultCode != 0) pendingResultCode else (intent?.getIntExtra(EXTRA_RESULT_CODE, 0) ?: 0)
                val resultData: Intent? = pendingResultData ?: run {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                        try {
                            intent?.getParcelableExtra(EXTRA_RESULT_DATA, Intent::class.java)
                        } catch (e: Exception) {
                            null
                        }
                    } else {
                        @Suppress("DEPRECATION")
                        intent?.getParcelableExtra(EXTRA_RESULT_DATA)
                    }
                }

                if (resultCode == Activity.RESULT_OK && resultData != null) {
                    startAsForeground()
                    val initialized = initMediaProjection(resultCode, resultData)
                    if (initialized) {
                        isRunning = true
                    } else {
                        isRunning = false
                        stopSelf()
                    }
                } else {
                    Log.e("ScreenCaptureService", "resultCode=$resultCode yoki resultData=$resultData yaroqsiz")
                    stopSelf()
                }
            }
            ACTION_STOP -> {
                stopCapture()
                stopSelf()
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

            // Register callback for Android 14+
            projection.registerCallback(object : MediaProjection.Callback() {
                override fun onStop() {
                    super.onStop()
                    releaseVirtualDisplay()
                    mediaProjection = null
                    isRunning = false
                    Log.d("ScreenCaptureService", "MediaProjection stopped by system.")
                }
            }, mainHandler)
            true
        } catch (e: Exception) {
            Log.e("ScreenCaptureService", "initMediaProjection xatosi: ${e.message}", e)
            false
        }
    }

    private fun ensureVirtualDisplay(projection: MediaProjection, width: Int, height: Int, density: Int): ImageReader {
        val existingReader = imageReader
        val existingDisplay = virtualDisplay

        if (existingReader != null && existingDisplay != null) {
            if (currentWidth == width && currentHeight == height) {
                return existingReader
            }
            // Dimensions or density changed: update surface and resize existing VirtualDisplay
            // Android 14 prohibits multiple createVirtualDisplay calls on the same MediaProjection token!
            currentWidth = width
            currentHeight = height
            currentDensity = density

            try {
                existingReader.close()
            } catch (ignored: Exception) {}

            @SuppressLint("WrongConstant")
            val newReader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2)
            imageReader = newReader

            try {
                existingDisplay.resize(width, height, density)
                existingDisplay.surface = newReader.surface
            } catch (e: Exception) {
                Log.w("ScreenCaptureService", "VirtualDisplay resize xatosi: ${e.message}")
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
            mainHandler
        )

        return reader
    }

    /**
     * Captures a single on-demand screenshot of the current screen.
     * Maintains a persistent VirtualDisplay to comply with Android 14+ single-use token constraints.
     */
    @Volatile
    private var lastCapturedBitmap: Bitmap? = null

    suspend fun captureCurrentScreen(): Bitmap? = withContext(Dispatchers.Default) {
        val projection = mediaProjection ?: run {
            Log.e("ScreenCaptureService", "MediaProjection is null. Cannot capture screen.")
            return@withContext null
        }

        val (width, height, density) = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val wm = windowManager ?: return@withContext null
            val bounds = wm.currentWindowMetrics.bounds
            Triple(bounds.width(), bounds.height(), resources.displayMetrics.densityDpi)
        } else {
            val dm = DisplayMetrics()
            @Suppress("DEPRECATION")
            windowManager?.defaultDisplay?.getRealMetrics(dm)
            Triple(dm.widthPixels, dm.heightPixels, dm.densityDpi)
        }

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
                        lastCapturedBitmap?.recycle()
                        lastCapturedBitmap = bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, false)
                        return@withContext bitmap
                    }
                }
            } catch (ignored: Exception) {
            } finally {
                directImage?.close()
            }

            // Wait for image frame with timeout
            val waitedBitmap = withTimeoutOrNull(1500L) {
                waitForImage(reader, width, height)
            }

            if (waitedBitmap != null) {
                lastCapturedBitmap?.recycle()
                lastCapturedBitmap = waitedBitmap.copy(waitedBitmap.config ?: Bitmap.Config.ARGB_8888, false)
                waitedBitmap
            } else {
                // If screen is static and no new frame arrived, reuse last cached frame
                lastCapturedBitmap?.let { cached ->
                    if (!cached.isRecycled) {
                        cached.copy(cached.config ?: Bitmap.Config.ARGB_8888, true)
                    } else null
                }
            }
        } catch (e: Exception) {
            Log.e("ScreenCaptureService", "Error during screen capture: ${e.message}", e)
            lastCapturedBitmap?.let { cached ->
                if (!cached.isRecycled) {
                    cached.copy(cached.config ?: Bitmap.Config.ARGB_8888, true)
                } else null
            }
        }
    }

    private suspend fun waitForImage(imageReader: ImageReader, width: Int, height: Int): Bitmap? {
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
            }, mainHandler)

            continuation.invokeOnCancellation {
                imageReader.setOnImageAvailableListener(null, null)
            }
        }
    }

    /**
     * Converts raw Image buffer to Android Bitmap considering rowStride and pixelStride padding.
     * Guaranteed safe against BufferUnderflowException and buffer shortage on OEM drivers.
     */
    private fun convertImageToBitmap(image: Image, width: Int, height: Int): Bitmap? {
        return try {
            val planes = image.planes
            val buffer = planes[0].buffer
            val pixelStride = planes[0].pixelStride
            val rowStride = planes[0].rowStride
            val rowPadding = rowStride - pixelStride * width

            val effectiveWidth = width + rowPadding / pixelStride
            val requiredBytes = effectiveWidth * height * 4

            buffer.rewind()
            val bitmap = Bitmap.createBitmap(
                effectiveWidth,
                height,
                Bitmap.Config.ARGB_8888
            )

            if (buffer.remaining() < requiredBytes) {
                val tempBuffer = java.nio.ByteBuffer.allocateDirect(requiredBytes)
                tempBuffer.put(buffer)
                tempBuffer.rewind()
                bitmap.copyPixelsFromBuffer(tempBuffer)
            } else {
                bitmap.copyPixelsFromBuffer(buffer)
            }

            // If rowPadding was non-zero, crop to real screen dimensions
            if (rowPadding == 0) {
                bitmap
            } else {
                val cropped = Bitmap.createBitmap(bitmap, 0, 0, width, height)
                bitmap.recycle()
                cropped
            }
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
        lastCapturedBitmap?.recycle()
        lastCapturedBitmap = null
        isRunning = false
    }

    override fun onDestroy() {
        stopCapture()
        super.onDestroy()
    }
}
