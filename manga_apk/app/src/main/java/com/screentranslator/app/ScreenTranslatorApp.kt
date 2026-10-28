package com.screentranslator.app

import android.app.Application
import com.screentranslator.app.data.preferences.TranslationHistoryRepository
import com.screentranslator.app.data.preferences.UserPreferencesRepository

/**
 * Main Application class for ScreenTranslator.
 * Exposes application-scoped singletons for history and preferences.
 */
class ScreenTranslatorApp : Application() {

    companion object {
        lateinit var instance: ScreenTranslatorApp
            private set
    }

    val historyRepository: TranslationHistoryRepository by lazy {
        TranslationHistoryRepository.getInstance(this)
    }

    val userPreferencesRepository: UserPreferencesRepository by lazy {
        UserPreferencesRepository(this)
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
    }
}
