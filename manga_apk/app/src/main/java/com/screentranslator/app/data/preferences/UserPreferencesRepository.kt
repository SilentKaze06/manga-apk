package com.screentranslator.app.data.preferences

import android.content.Context
import android.os.Build
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.emptyPreferences
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.screentranslator.app.data.model.MangaSourceLanguage
import com.screentranslator.app.data.model.TargetLanguage
import com.screentranslator.app.data.model.TranslationEngineType
import com.screentranslator.app.data.model.TranslationMode
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.map
import java.io.IOException

val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "user_settings")

/**
 * Data class holding all user settings.
 */
data class UserSettings(
    val targetLanguage: TargetLanguage = TargetLanguage.UZBEK,
    val engineType: TranslationEngineType = TranslationEngineType.MY_MEMORY,
    val deeplApiKey: String = "",
    val libreTranslateUrl: String = "https://libretranslate.com",
    val libreTranslateApiKey: String = "",
    val ocrApiKey: String = "",
    val isServiceActive: Boolean = false,
    val translationMode: TranslationMode = TranslationMode.STANDARD,
    val mangaSourceLanguage: MangaSourceLanguage = MangaSourceLanguage.AUTO,
    val mangaServerUrl: String = "http://10.0.2.2:8000",
    val mangaServerApiKey: String = "",
    val useMangaServer: Boolean = false
)

/**
 * Repository for persisting and reading user settings via DataStore.
 */
class UserPreferencesRepository(private val context: Context) {

    companion object {
        private val DEFAULT_SETTINGS = UserSettings()
    }

    private object PreferencesKeys {
        val TARGET_LANGUAGE = stringPreferencesKey("target_language")
        val ENGINE_TYPE = stringPreferencesKey("engine_type")
        val DEEPL_API_KEY = stringPreferencesKey("deepl_api_key")
        val LIBRE_TRANSLATE_URL = stringPreferencesKey("libre_translate_url")
        val LIBRE_TRANSLATE_API_KEY = stringPreferencesKey("libre_translate_api_key")
        val OCR_API_KEY = stringPreferencesKey("ocr_api_key")
        val IS_SERVICE_ACTIVE = booleanPreferencesKey("is_service_active")
        val TRANSLATION_MODE = stringPreferencesKey("translation_mode")
        val MANGA_SOURCE_LANGUAGE = stringPreferencesKey("manga_source_language")
        val MANGA_SERVER_URL = stringPreferencesKey("manga_server_url")
        val MANGA_SERVER_API_KEY = stringPreferencesKey("manga_server_api_key")
        val USE_MANGA_SERVER = booleanPreferencesKey("use_manga_server")
    }

    private fun decryptOrRaw(stored: String?): String {
        if (stored.isNullOrEmpty()) return ""
        val decrypted = runCatching { PdfJobTokenVault.decrypt(stored) }.getOrNull()
        if (decrypted != null) return decrypted
        val isBase64Cipher = runCatching {
            val bytes = android.util.Base64.decode(stored, android.util.Base64.NO_WRAP)
            bytes.size > 12 && stored.matches(Regex("^[A-Za-z0-9+/=]+$"))
        }.getOrDefault(false)
        return if (isBase64Cipher) "" else stored
    }

    val userSettingsFlow: Flow<UserSettings> = context.dataStore.data
        .catch { e ->
            if (e is IOException) emit(emptyPreferences()) else throw e
        }
        .map { preferences ->
            val defaults = DEFAULT_SETTINGS
            val targetLangCode = preferences[PreferencesKeys.TARGET_LANGUAGE] ?: defaults.targetLanguage.code
            val engineTypeName = preferences[PreferencesKeys.ENGINE_TYPE] ?: defaults.engineType.name
            val deeplKey = decryptOrRaw(preferences[PreferencesKeys.DEEPL_API_KEY])
            val libreUrl = preferences[PreferencesKeys.LIBRE_TRANSLATE_URL] ?: defaults.libreTranslateUrl
            val libreKey = decryptOrRaw(preferences[PreferencesKeys.LIBRE_TRANSLATE_API_KEY])
            val ocrKey = decryptOrRaw(preferences[PreferencesKeys.OCR_API_KEY])
            val isServiceActive = preferences[PreferencesKeys.IS_SERVICE_ACTIVE] ?: defaults.isServiceActive
            val modeId = preferences[PreferencesKeys.TRANSLATION_MODE] ?: defaults.translationMode.id
            val mangaLangCode = preferences[PreferencesKeys.MANGA_SOURCE_LANGUAGE] ?: defaults.mangaSourceLanguage.code
            val serverUrl = preferences[PreferencesKeys.MANGA_SERVER_URL] ?: defaults.mangaServerUrl
            val serverApiKey = decryptOrRaw(preferences[PreferencesKeys.MANGA_SERVER_API_KEY])
            val useServer = preferences[PreferencesKeys.USE_MANGA_SERVER] ?: defaults.useMangaServer
            val isEmulator = Build.FINGERPRINT.startsWith("generic") ||
                Build.FINGERPRINT.startsWith("unknown") ||
                Build.MODEL.contains("Emulator", ignoreCase = true) ||
                Build.HARDWARE.contains("goldfish", ignoreCase = true) ||
                Build.HARDWARE.contains("ranchu", ignoreCase = true) ||
                Build.HARDWARE.contains("qemu", ignoreCase = true)
            val hasUnusableEmulatorDefault = serverUrl.trimEnd('/') == "http://10.0.2.2:8000" && !isEmulator

            UserSettings(
                targetLanguage = TargetLanguage.fromCode(targetLangCode),
                engineType = TranslationEngineType.fromName(engineTypeName),
                deeplApiKey = deeplKey,
                libreTranslateUrl = libreUrl,
                libreTranslateApiKey = libreKey,
                ocrApiKey = ocrKey,
                isServiceActive = isServiceActive,
                translationMode = TranslationMode.fromId(modeId),
                mangaSourceLanguage = MangaSourceLanguage.fromCode(mangaLangCode),
                mangaServerUrl = serverUrl,
                mangaServerApiKey = serverApiKey,
                useMangaServer = useServer && !hasUnusableEmulatorDefault
            )
        }
        .flowOn(Dispatchers.IO)

    suspend fun updateTargetLanguage(language: TargetLanguage) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.TARGET_LANGUAGE] = language.code
        }
    }

    suspend fun updateTranslationMode(mode: TranslationMode) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.TRANSLATION_MODE] = mode.id
        }
    }

    suspend fun updateMangaSourceLanguage(language: MangaSourceLanguage) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.MANGA_SOURCE_LANGUAGE] = language.code
        }
    }

    suspend fun updateEngineType(engineType: TranslationEngineType) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.ENGINE_TYPE] = engineType.name
        }
    }

    suspend fun updateDeeplApiKey(apiKey: String) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.DEEPL_API_KEY] = if (apiKey.isBlank()) "" else PdfJobTokenVault.encrypt(apiKey.trim())
        }
    }

    suspend fun updateLibreTranslateSettings(url: String, apiKey: String) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.LIBRE_TRANSLATE_URL] = url.trim()
            preferences[PreferencesKeys.LIBRE_TRANSLATE_API_KEY] = if (apiKey.isBlank()) "" else PdfJobTokenVault.encrypt(apiKey.trim())
        }
    }

    suspend fun updateOcrApiKey(apiKey: String) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.OCR_API_KEY] = if (apiKey.isBlank()) "" else PdfJobTokenVault.encrypt(apiKey.trim())
        }
    }

    suspend fun updateMangaServerSettings(url: String, useServer: Boolean, apiKey: String? = null) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.MANGA_SERVER_URL] = url.trim()
            preferences[PreferencesKeys.USE_MANGA_SERVER] = useServer
            if (apiKey != null) {
                preferences[PreferencesKeys.MANGA_SERVER_API_KEY] = if (apiKey.isBlank()) "" else PdfJobTokenVault.encrypt(apiKey.trim())
            }
        }
    }

    suspend fun updateServiceActive(isActive: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.IS_SERVICE_ACTIVE] = isActive
        }
    }
}
