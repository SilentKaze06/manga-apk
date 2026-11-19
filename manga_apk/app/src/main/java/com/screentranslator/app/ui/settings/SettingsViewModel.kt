package com.screentranslator.app.ui.settings

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.screentranslator.app.data.model.TargetLanguage
import com.screentranslator.app.data.model.TranslationEngineType
import com.screentranslator.app.data.model.TranslationHistoryEntry
import com.screentranslator.app.data.model.PdfJobReference
import com.screentranslator.app.data.preferences.PdfJobRepository
import com.screentranslator.app.data.preferences.TranslationHistoryRepository
import com.screentranslator.app.data.preferences.UserPreferencesRepository
import com.screentranslator.app.data.preferences.UserSettings
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

class SettingsViewModel(application: Application) : AndroidViewModel(application) {

    private val preferencesRepository = UserPreferencesRepository(application)
    private val historyRepository = TranslationHistoryRepository.getInstance(application)
    private val pdfJobRepository = PdfJobRepository(application)

    val activePdfJob: StateFlow<PdfJobReference?> = pdfJobRepository.activeJob
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), null)

    fun savePdfJob(reference: PdfJobReference?) {
        viewModelScope.launch {
            if (reference == null) pdfJobRepository.clear() else pdfJobRepository.save(reference)
        }
    }

    // ─── User settings ────────────────────────────────────────────────────────
    val userSettings: StateFlow<UserSettings> = preferencesRepository.userSettingsFlow
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = UserSettings()
        )

    // ─── Translation history (Kept in memory with WhileSubscribed) ─────────────
    val translationHistory: StateFlow<List<TranslationHistoryEntry>> =
        historyRepository.historyFlow
            .stateIn(
                scope = viewModelScope,
                started = SharingStarted.WhileSubscribed(5000),
                initialValue = emptyList()
            )

    private var isRefreshingHistory = false
    fun refreshHistory() {
        if (isRefreshingHistory) return
        isRefreshingHistory = true
        viewModelScope.launch {
            try {
                historyRepository.refresh()
            } finally {
                isRefreshingHistory = false
            }
        }
    }

    fun clearHistory() {
        viewModelScope.launch {
            historyRepository.clearHistory()
        }
    }

    // ─── Language & Mode ──────────────────────────────────────────────────────
    fun setTargetLanguage(language: TargetLanguage) {
        viewModelScope.launch {
            preferencesRepository.updateTargetLanguage(language)
        }
    }

    fun setTranslationMode(mode: com.screentranslator.app.data.model.TranslationMode) {
        viewModelScope.launch {
            preferencesRepository.updateTranslationMode(mode)
        }
    }

    fun setMangaSourceLanguage(language: com.screentranslator.app.data.model.MangaSourceLanguage) {
        viewModelScope.launch {
            preferencesRepository.updateMangaSourceLanguage(language)
        }
    }

    // ─── API settings — explicit save only ────────────────────────────────────
    private val _apiSaveSuccess = MutableStateFlow(false)
    val apiSaveSuccess: StateFlow<Boolean> = _apiSaveSuccess.asStateFlow()

    sealed class ServerHealthStatus {
        object Checking : ServerHealthStatus()
        data class Success(val message: String) : ServerHealthStatus()
        data class Error(val error: String) : ServerHealthStatus()
    }

    private val _serverHealth = MutableStateFlow<ServerHealthStatus?>(null)
    val serverHealth: StateFlow<ServerHealthStatus?> = _serverHealth.asStateFlow()

    /**
     * Saves all API/engine settings at once (engine type + API keys + OCR.space key + Manga server).
     * Called only when user presses the explicit "Saqlash" button.
     */
    fun saveApiSettings(
        engineType: TranslationEngineType,
        deeplApiKey: String,
        libreTranslateUrl: String,
        libreTranslateApiKey: String,
        ocrApiKey: String,
        mangaServerUrl: String = "http://10.0.2.2:8000",
        useMangaServer: Boolean = false,
        mangaServerApiKey: String = ""
    ) {
        viewModelScope.launch {
            preferencesRepository.updateEngineType(engineType)
            preferencesRepository.updateDeeplApiKey(deeplApiKey)
            preferencesRepository.updateLibreTranslateSettings(libreTranslateUrl, libreTranslateApiKey)
            preferencesRepository.updateOcrApiKey(ocrApiKey)
            preferencesRepository.updateMangaServerSettings(mangaServerUrl, useMangaServer, mangaServerApiKey)
            _apiSaveSuccess.value = true
        }
    }

    fun updateMangaServerSettings(url: String, enabled: Boolean, apiKey: String? = null) {
        viewModelScope.launch {
            preferencesRepository.updateMangaServerSettings(url, enabled, apiKey)
        }
    }

    fun testMangaServerConnection(url: String, apiKey: String = "") {
        viewModelScope.launch {
            _serverHealth.value = ServerHealthStatus.Checking
            val res = com.screentranslator.app.data.network.MangaServerClient.checkHealth(url, apiKey)
            if (res.isSuccess) {
                val health = runCatching { com.google.gson.Gson().fromJson(res.getOrThrow(), Map::class.java) }.getOrNull()
                _serverHealth.value = when {
                    health == null -> ServerHealthStatus.Error("Server yaroqsiz formatda javob qaytardi.")
                    health["status"] != "ok" && health["status"] != "healthy" -> ServerHealthStatus.Error("Server holati nosoz: ${health["status"] ?: "noma'lum"}")
                    health["pdf_translation_enabled"] == false -> ServerHealthStatus.Error("Serverda PDF tarjimasi öçirilgan. Server sozlamalarini tekşiring.")
                    health["translation_ready"] == false -> ServerHealthStatus.Error("Server javob berdi, lekin tarjima xizmati tayyor emas.")
                    health["api_key_required"] == true && apiKey.isBlank() -> ServerHealthStatus.Error("Server API kalitini kiriting va qayta tekşiring.")
                    else -> {
                        val sysRes = health["system_resources"] as? Map<*, *>
                        val ramHuman = sysRes?.get("human_readable") as? String
                        val ver = health["version"]?.toString() ?: "faol"
                        if (!ramHuman.isNullOrBlank()) {
                            ServerHealthStatus.Success("Server faol ($ver). RAM: $ramHuman")
                        } else {
                            ServerHealthStatus.Success("Server javob berdi ($ver). OCR modellari tayyor.")
                        }
                    }
                }
            } else {
                val err = res.exceptionOrNull()?.localizedMessage ?: "Serverga ulaniş muvaffaqiyatsiz"
                _serverHealth.value = ServerHealthStatus.Error(err)
            }
        }
    }

    fun clearServerHealthStatus() {
        _serverHealth.value = null
    }

    fun resetApiSaveSuccess() {
        _apiSaveSuccess.value = false
    }

    // ─── Service active state ─────────────────────────────────────────────────
    fun setServiceActive(isActive: Boolean) {
        viewModelScope.launch {
            preferencesRepository.updateServiceActive(isActive)
        }
    }
}
