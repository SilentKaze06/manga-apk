package com.screentranslator.app.overlay

import android.content.Context
import android.graphics.Color
import android.graphics.PixelFormat
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.screentranslator.app.R
import com.screentranslator.app.data.model.OcrResultBlock
import com.screentranslator.app.ocr.MangaBubbleDetector
import com.screentranslator.app.util.DisplayCompat
import kotlin.math.max
import kotlin.math.min

/**
 * Manages rendering translated text cards over original screen coordinates using WindowManager.
 * Uses exact screen-coordinate mapping with FLAG_LAYOUT_IN_SCREEN and cutout awareness
 * so overlays cover the full original text footprint without exposing source glyphs.
 */
class OverlayManager(private val context: Context) {

    private val windowManager = DisplayCompat.getOverlayWindowManager(context)
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
        return DisplayCompat.getRealScreenDimensions(windowManager, context.resources.displayMetrics.densityDpi).first
    }

    private fun getScreenHeight(): Int {
        return DisplayCompat.getRealScreenDimensions(windowManager, context.resources.displayMetrics.densityDpi).second
    }

    /**
     * Standard mode overlay: opaque dark card badge over detected line.
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
            setPadding(dpToPx(6), dpToPx(4), dpToPx(6), dpToPx(4))
            includeFontPadding = false

            setSingleLine(false)
            maxLines = Int.MAX_VALUE
            ellipsize = null

            elevation = dpToPx(4).toFloat()
            setOnClickListener { onClick() }
        }
    }

    /**
     * Manga mode overlay: in-place speech bubble replacement with opaque background and fitted typesetting.
     */
    private fun createMangaOverlayTextView(
        block: OcrResultBlock,
        text: String,
        onClick: () -> Unit
    ): TextView {
        val opaqueBgColor = block.bubbleBgColor or (0xFF shl 24)
        val isLightBg = MangaBubbleDetector.isColorLight(opaqueBgColor)
        return TextView(context).apply {
            this.text = text
            setTextColor(block.bubbleTextColor)

            val bubbleDrawable = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(4).toFloat()
                setColor(opaqueBgColor)
                val strokeColor = if (isLightBg) {
                    Color.parseColor("#CBD5E1")
                } else {
                    Color.parseColor("#3F3F46")
                }
                setStroke(dpToPx(1.5f), strokeColor)
            }
            background = bubbleDrawable

            gravity = Gravity.CENTER
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setPadding(dpToPx(6), dpToPx(4), dpToPx(6), dpToPx(4))
            includeFontPadding = false

            setSingleLine(false)
            maxLines = Int.MAX_VALUE
            ellipsize = null

            elevation = dpToPx(4).toFloat()
            setOnClickListener { onClick() }
        }
    }

    private fun fitTextSizeAndMeasure(
        overlayView: TextView,
        targetWidth: Int,
        idealHeight: Int,
        minSp: Float,
        maxSp: Float
    ): Pair<Int, Int> {
        val widthSpec = View.MeasureSpec.makeMeasureSpec(targetWidth, View.MeasureSpec.EXACTLY)
        val heightSpec = View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED)

        var low = minSp
        var high = maxSp
        var bestSp = minSp

        repeat(8) {
            val mid = (low + high) / 2f
            overlayView.setTextSize(TypedValue.COMPLEX_UNIT_SP, mid)
            overlayView.measure(widthSpec, heightSpec)
            if (overlayView.measuredHeight <= idealHeight) {
                bestSp = mid
                low = mid
            } else {
                high = mid
            }
        }

        overlayView.setTextSize(TypedValue.COMPLEX_UNIT_SP, bestSp)
        overlayView.measure(widthSpec, heightSpec)
        return overlayView.measuredWidth to overlayView.measuredHeight
    }

    private fun buildLayoutParamsForBounds(
        block: OcrResultBlock,
        targetWidth: Int,
        minBoxHeight: Int,
        padding: Int,
        isVert: Boolean,
        screenW: Int,
        screenH: Int,
        overlayView: TextView,
        minSp: Float,
        maxSp: Float
    ): WindowManager.LayoutParams {
        val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
        } else {
            @Suppress("DEPRECATION")
            WindowManager.LayoutParams.TYPE_PHONE
        }

        val bounds = block.boundingBox
        val coveredLeft = bounds.left.coerceIn(0, screenW)
        val coveredTop = bounds.top.coerceIn(0, screenH)
        val coveredRight = bounds.right.coerceIn(coveredLeft, screenW)
        val coveredBottom = bounds.bottom.coerceIn(coveredTop, screenH)

        val idealLeft = (coveredLeft - padding).coerceAtLeast(0)
        val initialW = maxOf(targetWidth, coveredRight - idealLeft).coerceAtMost(screenW)
        val idealHeight = maxOf((coveredBottom - coveredTop) + padding * 2, minBoxHeight)

        val (measuredW, measuredH) = fitTextSizeAndMeasure(
            overlayView = overlayView,
            targetWidth = initialW,
            idealHeight = idealHeight,
            minSp = minSp,
            maxSp = maxSp
        )

        val finalW = maxOf(initialW, measuredW).coerceAtMost(screenW)
        var finalX = if (isVert) {
            (bounds.centerX() - finalW / 2).coerceIn(0, (screenW - finalW).coerceAtLeast(0))
        } else {
            idealLeft.coerceIn(0, (screenW - finalW).coerceAtLeast(0))
        }
        if (finalX > coveredLeft) finalX = coveredLeft
        val adjustedW = maxOf(finalW, coveredRight - finalX).coerceAtMost(screenW - finalX)

        val idealTop = (coveredTop - padding).coerceAtLeast(0)
        val finalH = maxOf(idealHeight, measuredH, coveredBottom - idealTop).coerceAtMost(screenH)
        var finalY = idealTop.coerceIn(0, (screenH - finalH).coerceAtLeast(0))
        if (finalY > coveredTop) finalY = coveredTop
        val adjustedH = maxOf(finalH, coveredBottom - finalY).coerceAtMost(screenH - finalY)

        return WindowManager.LayoutParams(
            adjustedW,
            adjustedH,
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

    private fun createMangaLayoutParams(
        block: OcrResultBlock,
        text: String,
        overlayView: TextView
    ): WindowManager.LayoutParams {
        val bounds = block.boundingBox
        val screenW = getScreenWidth()
        val screenH = getScreenHeight()
        val padding = dpToPx(4)
        val isVert = block.isVertical || (bounds.height() > bounds.width() * 1.25f)

        val targetWidth = if (isVert) {
            val desiredWidth = (dpToPx(72) + (text.length * dpToPx(1.8f))).toInt()
                .coerceIn(dpToPx(72), dpToPx(150))
            maxOf(bounds.width() + padding * 2, desiredWidth).coerceAtMost(screenW)
        } else {
            maxOf(bounds.width() + padding * 2, dpToPx(60)).coerceAtMost(screenW)
        }

        return buildLayoutParamsForBounds(
            block = block,
            targetWidth = targetWidth,
            minBoxHeight = dpToPx(28),
            padding = padding,
            isVert = isVert,
            screenW = screenW,
            screenH = screenH,
            overlayView = overlayView,
            minSp = 8.5f,
            maxSp = 17f
        )
    }

    private fun createLayoutParams(
        block: OcrResultBlock,
        text: String,
        overlayView: TextView
    ): WindowManager.LayoutParams {
        val bounds = block.boundingBox
        val screenW = getScreenWidth()
        val screenH = getScreenHeight()
        val padding = dpToPx(3)
        val isVert = block.isVertical || (bounds.height() > bounds.width() * 1.25f)

        val targetWidth = if (isVert) {
            val desiredWidth = (dpToPx(72) + (text.length * dpToPx(1.8f))).toInt()
                .coerceIn(dpToPx(72), dpToPx(150))
            maxOf(bounds.width() + padding * 2, desiredWidth).coerceAtMost(screenW)
        } else {
            val estimatedTextW = (text.length * dpToPx(7.5f) + dpToPx(20)).toInt()
            maxOf(bounds.width() + padding * 2, minOf(estimatedTextW, dpToPx(260)), dpToPx(48))
                .coerceAtMost(screenW)
        }

        return buildLayoutParamsForBounds(
            block = block,
            targetWidth = targetWidth,
            minBoxHeight = dpToPx(24),
            padding = padding,
            isVert = isVert,
            screenW = screenW,
            screenH = screenH,
            overlayView = overlayView,
            minSp = 8.5f,
            maxSp = 15.5f
        )
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
