package com.screentranslator.app.util

import android.content.Context
import android.hardware.display.DisplayManager
import android.os.Build
import android.util.DisplayMetrics
import android.view.Display
import android.view.WindowManager

/**
 * Display and WindowManager helpers compatible with Android 7-15 (API 24-35).
 * Uses createWindowContext on API 30+ to avoid StrictMode non-visual context violations,
 * and getRealMetrics consistently on API < 30.
 */
object DisplayCompat {

    fun getOverlayWindowManager(context: Context): WindowManager {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val displayManager = context.getSystemService(Context.DISPLAY_SERVICE) as? DisplayManager
            val defaultDisplay = displayManager?.getDisplay(Display.DEFAULT_DISPLAY)
            if (defaultDisplay != null) {
                val windowCtx = context
                    .createDisplayContext(defaultDisplay)
                    .createWindowContext(WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY, null)
                return windowCtx.getSystemService(Context.WINDOW_SERVICE) as WindowManager
            }
        }
        return context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
    }

    fun getRealScreenDimensions(windowManager: WindowManager, fallbackDensityDpi: Int): Triple<Int, Int, Int> {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val bounds = windowManager.currentWindowMetrics.bounds
            Triple(bounds.width(), bounds.height(), fallbackDensityDpi)
        } else {
            val dm = DisplayMetrics()
            @Suppress("DEPRECATION")
            windowManager.defaultDisplay.getRealMetrics(dm)
            Triple(dm.widthPixels, dm.heightPixels, dm.densityDpi)
        }
    }
}
