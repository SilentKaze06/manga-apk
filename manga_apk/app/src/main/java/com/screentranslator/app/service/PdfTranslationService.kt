package com.screentranslator.app.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import com.screentranslator.app.MainActivity
import com.screentranslator.app.R
import com.screentranslator.app.data.model.PdfJobPolicy
import com.screentranslator.app.data.model.PdfJobReference
import com.screentranslator.app.data.model.PdfTranslationJob
import com.screentranslator.app.data.network.PdfTranslationClient
import com.screentranslator.app.data.preferences.PdfJobRepository
import com.screentranslator.app.data.preferences.UserPreferencesRepository
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * Foreground Service that continues tracking and updating PDF translations
 * even when the app is closed, minimized, or the screen is turned off.
 */
class PdfTranslationService : Service() {

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var pollingJob: Job? = null
    private var activeJobRef: PdfJobReference? = null

    companion object {
        const val CHANNEL_ID = "pdf_translation_progress_channel"
        const val COMPLETE_CHANNEL_ID = "pdf_translation_complete_channel"
        const val NOTIFICATION_ID = 2001
        const val COMPLETE_NOTIFICATION_ID = 2002

        const val ACTION_START = "com.screentranslator.app.ACTION_START_PDF_JOB"
        const val ACTION_STOP = "com.screentranslator.app.ACTION_STOP_PDF_JOB"

        const val EXTRA_JOB_ID = "extra_job_id"
        const val EXTRA_JOB_TOKEN = "extra_job_token"
        const val EXTRA_SERVER_URL = "extra_server_url"
        const val EXTRA_FILENAME = "extra_filename"

        private val _jobStateFlow = MutableStateFlow<PdfTranslationJob?>(null)
        val jobStateFlow: StateFlow<PdfTranslationJob?> = _jobStateFlow.asStateFlow()

        @Volatile
        var isRunning = false
            private set

        fun start(context: Context, jobRef: PdfJobReference) {
            val intent = Intent(context, PdfTranslationService::class.java).apply {
                action = ACTION_START
                putExtra(EXTRA_JOB_ID, jobRef.jobId)
                putExtra(EXTRA_JOB_TOKEN, jobRef.token)
                putExtra(EXTRA_SERVER_URL, jobRef.serverUrl)
                putExtra(EXTRA_FILENAME, jobRef.filename)
            }
            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    context.startForegroundService(intent)
                } else {
                    context.startService(intent)
                }
            } catch (e: Exception) {
                Log.w("PdfTranslationService", "Xizmatni başlaşda xato: ${e.message}")
            }
        }

        fun stop(context: Context) {
            val intent = Intent(context, PdfTranslationService::class.java).apply {
                action = ACTION_STOP
            }
            try {
                context.startService(intent)
            } catch (ignored: Exception) {}
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannels()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val action = intent?.action ?: ACTION_START
        when (action) {
            ACTION_STOP -> {
                stopPolling()
                stopSelf()
                return START_NOT_STICKY
            }
            ACTION_START -> {
                val jobId = intent?.getStringExtra(EXTRA_JOB_ID)
                val token = intent?.getStringExtra(EXTRA_JOB_TOKEN)
                val serverUrl = intent?.getStringExtra(EXTRA_SERVER_URL)
                val filename = intent?.getStringExtra(EXTRA_FILENAME).orEmpty()

                if (!jobId.isNullOrBlank() && !token.isNullOrBlank() && !serverUrl.isNullOrBlank()) {
                    val ref = PdfJobReference(jobId, token, serverUrl, filename)
                    activeJobRef = ref
                    startAsForeground(
                        title = filename.ifBlank { "PDF Tarjima" },
                        text = "Tarjima jarayoni başlanmoqda…",
                        progress = 0,
                        max = 100
                    )
                    startPolling(ref)
                } else {
                    startAsForeground(
                        title = "PDF Tarjima",
                        text = "Tarjima tiklanmoqda…",
                        progress = 0,
                        max = 100
                    )
                    // Try to restore from repository (e.g. after sticky restart)
                    serviceScope.launch {
                        val repo = PdfJobRepository(applicationContext)
                        val ref = repo.activeJob.first()
                        if (ref != null && !ref.serverUrl.isNullOrBlank()) {
                            activeJobRef = ref
                            startPolling(ref)
                        } else {
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                                stopForeground(STOP_FOREGROUND_REMOVE)
                            } else {
                                @Suppress("DEPRECATION")
                                stopForeground(true)
                            }
                            stopSelf()
                        }
                    }
                }
            }
        }
        return START_STICKY
    }

    private fun startAsForeground(title: String, text: String, progress: Int, max: Int) {
        val notification = createProgressNotification(title, text, progress, max)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun startPolling(jobRef: PdfJobReference) {
        pollingJob?.cancel()
        pollingJob = serviceScope.launch {
            isRunning = true
            val userPrefs = UserPreferencesRepository(applicationContext)
            val settings = userPrefs.userSettingsFlow.first()
            val apiKey = settings.mangaServerApiKey
            val serverUrl = jobRef.serverUrl ?: run {
                stopSelf()
                return@launch
            }

            var failures = 0
            while (isActive) {
                try {
                    val status = PdfTranslationClient.status(
                        serverUrl = serverUrl,
                        apiKey = apiKey,
                        jobId = jobRef.jobId,
                        jobToken = jobRef.token
                    )
                    _jobStateFlow.value = status
                    failures = 0

                    val title = jobRef.filename.ifBlank { status.filename }.ifBlank { "PDF Tarjima" }
                    val text = when {
                        status.status == "completed" -> "Barça ${status.totalPages} ta bet tayyor!"
                        status.totalPages > 0 -> "${status.currentPage}/${status.totalPages}-bet tarjima qilindi (${status.progress}%)"
                        else -> status.message.ifBlank { "PDF sahifalari tahlil qilinmoqda…" }
                    }

                    val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
                    notificationManager.notify(
                        NOTIFICATION_ID,
                        createProgressNotification(title, text, status.progress, 100)
                    )

                    if (!PdfJobPolicy.isRunning(status.status)) {
                        if (status.status == "completed") {
                            showCompletionNotification(status)
                        } else if (status.status == "failed") {
                            showErrorNotification(status)
                        }
                        break
                    }
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (e: Exception) {
                    failures++
                    Log.w("PdfTranslationService", "Polling xatosi ($failures): ${e.message}")
                    val msg = e.message.orEmpty()
                    if (msg.contains("404") || msg.contains("401") || msg.contains("403") || failures >= 15) {
                        Log.e("PdfTranslationService", "Polling töxtatildi (kritik xato yoki çeklov): $msg")
                        break
                    }
                }

                delay(PdfJobPolicy.retryDelayMillis(failures))
            }

            if (pollingJob === coroutineContext[Job]) {
                isRunning = false
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                    stopForeground(STOP_FOREGROUND_REMOVE)
                } else {
                    @Suppress("DEPRECATION")
                    stopForeground(true)
                }
                stopSelf()
            }
        }
    }

    private fun stopPolling() {
        pollingJob?.cancel()
        pollingJob = null
        isRunning = false
        _jobStateFlow.value = null
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(STOP_FOREGROUND_REMOVE)
        } else {
            @Suppress("DEPRECATION")
            stopForeground(true)
        }
    }

    private fun createProgressNotification(
        title: String,
        text: String,
        progress: Int,
        max: Int
    ): Notification {
        val contentIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
            },
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        val builder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentIntent(contentIntent)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setCategory(NotificationCompat.CATEGORY_PROGRESS)

        if (max > 0) {
            builder.setProgress(max, progress, false)
        } else {
            builder.setProgress(0, 0, true)
        }

        return builder.build()
    }

    private fun showCompletionNotification(job: PdfTranslationJob) {
        val contentIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
            },
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        val notification = NotificationCompat.Builder(this, COMPLETE_CHANNEL_ID)
            .setContentTitle("✅ PDF tarjima tayyor!")
            .setContentText("${job.filename.ifBlank { "Hujjat" }} töliq tarjima qilindi. Köriş uçun bosing.")
            .setSmallIcon(R.drawable.ic_notification)
            .setContentIntent(contentIntent)
            .setAutoCancel(true)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setDefaults(NotificationCompat.DEFAULT_ALL)
            .build()

        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(COMPLETE_NOTIFICATION_ID, notification)
    }

    private fun showErrorNotification(job: PdfTranslationJob) {
        val contentIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
            },
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        val notification = NotificationCompat.Builder(this, COMPLETE_CHANNEL_ID)
            .setContentTitle("❌ PDF tarjimasi töxtatildi")
            .setContentText(job.error ?: job.message.ifBlank { "Tarjima yakunlanmadi." })
            .setSmallIcon(R.drawable.ic_notification)
            .setContentIntent(contentIntent)
            .setAutoCancel(true)
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .build()

        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(COMPLETE_NOTIFICATION_ID, notification)
    }

    private fun createNotificationChannels() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

            val progressChannel = NotificationChannel(
                CHANNEL_ID,
                getString(R.string.pdf_channel_progress_name),
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = getString(R.string.pdf_channel_progress_desc)
                setShowBadge(false)
            }

            val completeChannel = NotificationChannel(
                COMPLETE_CHANNEL_ID,
                getString(R.string.pdf_channel_complete_name),
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = getString(R.string.pdf_channel_complete_desc)
                setShowBadge(true)
                enableVibration(true)
            }

            manager.createNotificationChannel(progressChannel)
            manager.createNotificationChannel(completeChannel)
        }
    }

    override fun onDestroy() {
        stopPolling()
        serviceScope.cancel()
        super.onDestroy()
    }
}

