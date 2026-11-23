package com.screentranslator.app.overlay

import android.content.Context
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.Rect
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.DisplayMetrics
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.screentranslator.app.R
import com.screentranslator.app.data.model.OcrResultBlock
import kotlin.math.max
import kotlin.math.min

/**
 * Manages rendering translated text cards over original screen coordinates using WindowManager.
 * Uses exact screen-coordinate mapping with FLAG_LAYOUT_IN_SCREEN and cutout awareness
 * so overlays are never displaced downwards by status bars or cut off by screen boundaries.
 */
class OverlayManager(private val context: Context) {

    private val windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
    private val activeViews = mutableListOf<View>()
    private val mainHandler = Handler(Looper.getMainLooper())

    /** Immediate visibility flag so callers are not raced by posted clear/show work. */
    @Volatile
    private var overlaysVisible = false

    /** Bumps on each show/clear so stale posted runnables are ignored. */
    private val generation = java.util.concurrent.atomic.AtomicInteger(0)

    /**
     * Draws translated text overlays directly over each recognized block's screen bounds.
     */
    fun showOverlays(blocks: List<OcrResultBlock>, onDismissRequest: (() -> Unit)? = null) {
        val token = generation.incrementAndGet()
        mainHandler.post {
            if (token != generation.get()) return@post
            internalClear(updateFlag = false)

            var added = 0
            for (block in blocks) {
                val translation = block.translatedText?.trim() ?: continue
                if (translation.isBlank()) continue

                val overlayView = if (block.isMangaBubble) {
                    createMangaOverlayTextView(block, translation) {
                        clearOverlays()
                        onDismissRequest?.invoke()
                    }
                } else {
                    createOverlayTextView(block, translation) {
                        clearOverlays()
                        onDismissRequest?.invoke()
                    }
                }

                val layoutParams = if (block.isMangaBubble) {
                    createMangaLayoutParams(block, translation, overlayView)
                } else {
                    createLayoutParams(block, translation, overlayView)
                }
                try {
                    windowManager.addView(overlayView, layoutParams)
                    synchronized(activeViews) {
                        activeViews.add(overlayView)
                    }
                    added++
                } catch (e: Exception) {
                    Log.w("OverlayManager", "Overlay view qo'shib bo'lmadi: ${e.message}")
                }
            }
            if (token == generation.get()) {
                overlaysVisible = added > 0
            }
        }
    }

    /**
     * Clears all currently visible translation overlays.
     */
    fun clearOverlays() {
        val token = generation.incrementAndGet()
        if (Looper.myLooper() == Looper.getMainLooper()) {
            internalClear()
        } else {
            mainHandler.post {
                if (token == generation.get()) {
                    internalClear()
                }
            }
        }
    }

    private fun internalClear(updateFlag: Boolean = true) {
        synchronized(activeViews) {
            for (view in activeViews) {
                try {
                    windowManager.removeView(view)
                } catch (e: Exception) {
                    Log.w("OverlayManager", "removeView xatosi: ${e.message}")
                }
            }
            activeViews.clear()
        }
        if (updateFlag) {
            overlaysVisible = false
        }
    }

    fun hasActiveOverlays(): Boolean = overlaysVisible

    /**
     * Retrieves the physical screen dimensions matching ScreenCaptureService coordinates.
     */
    private fun getScreenWidth(): Int {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            windowManager.currentWindowMetrics.bounds.width()
        } else {
            val dm = DisplayMetrics()
            @Suppress("DEPRECATION")
            windowManager.defaultDisplay.getRealMetrics(dm)
            dm.widthPixels
        }
    }

    private fun getScreenHeight(): Int {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            windowManager.currentWindowMetrics.bounds.height()
        } else {
            val dm = DisplayMetrics()
            @Suppress("DEPRECATION")
            windowManager.defaultDisplay.getRealMetrics(dm)
            dm.heightPixels
        }
    }

    /**
     * Standard mode overlay: dark card badge over detected line.
     */
    private fun createOverlayTextView(
        block: OcrResultBlock,
        text: String,
        onClick: () -> Unit
    ): TextView {
        val isVert = block.isVertical || (block.boundingBox.height() > block.boundingBox.width() * 1.25f)
        return TextView(context).apply {
            this.text = text
            setTextColor(Color.WHITE)
            background = ContextCompat.getDrawable(context, R.drawable.bg_overlay_card)
            gravity = if (isVert) Gravity.CENTER else (Gravity.CENTER_VERTICAL or Gravity.START)
            setPadding(dpToPx(8), dpToPx(5), dpToPx(8), dpToPx(5))
            includeFontPadding = false

            setSingleLine(false)
            maxLines = 10
            ellipsize = android.text.TextUtils.TruncateAt.END

            val boxHeightDp = pxToDp(block.boundingBox.height())
            val fontSizeSp = when {
                boxHeightDp < 18 -> 11f
                boxHeightDp < 28 -> 12f
                boxHeightDp < 40 -> 13f
                boxHeightDp < 60 -> 14f
                else -> 15f
            }
            setTextSize(TypedValue.COMPLEX_UNIT_SP, fontSizeSp)

            elevation = dpToPx(4).toFloat()
            setOnClickListener { onClick() }
        }
    }

    /**
     * Manga mode overlay: in-place speech bubble replacement with manga typesetting.
     */
    private fun createMangaOverlayTextView(
        block: OcrResultBlock,
        text: String,
        onClick: () -> Unit
    ): TextView {
        val isVert = block.isVertical || (block.boundingBox.height() > block.boundingBox.width() * 1.25f)
        return TextView(context).apply {
            this.text = text
            setTextColor(block.bubbleTextColor)

            val bubbleDrawable = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(14).toFloat()
                setColor(block.bubbleBgColor)
                val strokeColor = if (block.bubbleBgColor == Color.WHITE) {
                    Color.parseColor("#CBD5E1")
                } else {
                    Color.parseColor("#3F3F46")
                }
                setStroke(dpToPx(1.5f), strokeColor)
            }
            background = bubbleDrawable

            gravity = Gravity.CENTER
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setPadding(dpToPx(8), dpToPx(6), dpToPx(8), dpToPx(6))
            includeFontPadding = false

            setSingleLine(false)
            maxLines = 12
            ellipsize = android.text.TextUtils.TruncateAt.END

            val bounds = block.boundingBox
            val effectiveWidthDp = if (isVert) {
                pxToDp(maxOf(bounds.width(), dpToPx(140)))
            } else {
                pxToDp(bounds.width())
            }
            val boxHeightDp = pxToDp(bounds.height())
            val area = max(1, effectiveWidthDp * boxHeightDp)
            val charCount = max(1, text.length)
            val estimatedSp = kotlin.math.sqrt(area.toDouble() / (charCount * 1.15)).toFloat()
            val fontSizeSp = estimatedSp.coerceIn(10f, 17f)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, fontSizeSp)

            elevation = dpToPx(4).toFloat()
            setOnClickListener { onClick() }
        }
    }

    private fun createMangaLayoutParams(
        block: OcrResultBlock,
        text: String,
        overlayView: TextView
    ): WindowManager.LayoutParams {
        val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
        } else {
            @Suppress("DEPRECATION")
            WindowManager.LayoutParams.TYPE_PHONE
        }

        val bounds = block.boundingBox
        val screenW = getScreenWidth()
        val screenH = getScreenHeight()
        val padding = dpToPx(4)

        val isVert = block.isVertical || (bounds.height() > bounds.width() * 1.25f)

        val (targetWidth, posX) = if (isVert) {
            val desiredWidth = (dpToPx(72) + (text.length * dpToPx(1.8f))).toInt()
                .coerceIn(dpToPx(72), dpToPx(150))
            val w = maxOf(bounds.width() + padding * 2, desiredWidth).coerceAtMost(screenW - dpToPx(16))
            val x = (bounds.centerX() - (w / 2)).coerceIn(dpToPx(4), screenW - w - dpToPx(4))
            Pair(w, x)
        } else {
            val desiredW = maxOf(bounds.width() + padding * 2, dpToPx(60)).coerceAtMost(screenW - dpToPx(16))
            val x = (bounds.left - padding).coerceIn(dpToPx(4), screenW - desiredW - dpToPx(4))
            Pair(desiredW, x)
        }

        // Measure view height accurately so translation text is never truncated
        overlayView.measure(
            View.MeasureSpec.makeMeasureSpec(targetWidth, View.MeasureSpec.AT_MOST),
            View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED)
        )
        val measuredH = overlayView.measuredHeight
        val measuredW = overlayView.measuredWidth
        val finalW = maxOf(targetWidth, measuredW).coerceAtMost(screenW - dpToPx(16))
        val finalH = maxOf(bounds.height() + padding * 2, measuredH, dpToPx(32))
            .coerceAtMost(screenH - dpToPx(40))

        // Safety margins so overlays never get pushed under navigation bar or cutouts
        val topSafetyMargin = dpToPx(4)
        val bottomSafetyMargin = dpToPx(28) // Space for navigation bar / gesture pill
        val maxPosY = (screenH - finalH - bottomSafetyMargin).coerceAtLeast(topSafetyMargin)
        val finalY = (bounds.top - padding).coerceIn(topSafetyMargin, maxPosY)
        val finalX = posX.coerceIn(dpToPx(4), screenW - finalW - dpToPx(4))

        return WindowManager.LayoutParams(
            finalW,
            finalH,
            finalX,
            finalY,
            type,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS
            } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
            }
        }
    }

    private fun createLayoutParams(
        block: OcrResultBlock,
        text: String,
        overlayView: TextView
    ): WindowManager.LayoutParams {
        val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
        } else {
            @Suppress("DEPRECATION")
            WindowManager.LayoutParams.TYPE_PHONE
        }

        val bounds = block.boundingBox
        val screenW = getScreenWidth()
        val screenH = getScreenHeight()
        val padding = dpToPx(3)

        val isVert = block.isVertical || (bounds.height() > bounds.width() * 1.25f)

        val (targetWidth, posX) = if (isVert) {
            val desiredWidth = (dpToPx(72) + (text.length * dpToPx(1.8f))).toInt()
                .coerceIn(dpToPx(72), dpToPx(150))
            val w = maxOf(bounds.width() + padding * 2, desiredWidth).coerceAtMost(screenW - dpToPx(16))
            val x = (bounds.centerX() - (w / 2)).coerceIn(dpToPx(4), screenW - w - dpToPx(4))
            Pair(w, x)
        } else {
            // Estimate needed width for horizontal text so it can wrap comfortably
            val estimatedTextW = (text.length * dpToPx(7.5f) + dpToPx(20)).toInt()
            val desiredW = maxOf(bounds.width() + padding * 2, minOf(estimatedTextW, dpToPx(260)), dpToPx(48))
                .coerceAtMost(screenW - dpToPx(16))
            val x = (bounds.left - padding).coerceIn(dpToPx(4), screenW - desiredW - dpToPx(4))
            Pair(desiredW, x)
        }

        // Measure view height accurately so translation text is never truncated
        overlayView.measure(
            View.MeasureSpec.makeMeasureSpec(targetWidth, View.MeasureSpec.AT_MOST),
            View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED)
        )
        val measuredH = overlayView.measuredHeight
        val measuredW = overlayView.measuredWidth
        val finalW = maxOf(targetWidth, measuredW).coerceAtMost(screenW - dpToPx(16))
        val finalH = maxOf(bounds.height() + padding * 2, measuredH, dpToPx(24))
            .coerceAtMost(screenH - dpToPx(40))

        // Safety margins so overlays never get pushed under navigation bar or cutouts
        val topSafetyMargin = dpToPx(4)
        val bottomSafetyMargin = dpToPx(28) // Space for navigation bar / gesture pill
        val maxPosY = (screenH - finalH - bottomSafetyMargin).coerceAtLeast(topSafetyMargin)
        val finalY = (bounds.top - (padding / 2)).coerceIn(topSafetyMargin, maxPosY)
        val finalX = posX.coerceIn(dpToPx(4), screenW - finalW - dpToPx(4))

        return WindowManager.LayoutParams(
            finalW,
            finalH,
            finalX,
            finalY,
            type,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS
            } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
            }
        }
    }

    private fun dpToPx(dp: Float): Int {
        val density = context.resources.displayMetrics.density
        return (dp * density).toInt()
    }

    private fun dpToPx(dp: Int): Int {
        val density = context.resources.displayMetrics.density
        return (dp * density).toInt()
    }

    private fun pxToDp(px: Int): Int {
        val density = context.resources.displayMetrics.density
        return if (density > 0) (px / density).toInt() else px
    }
}
