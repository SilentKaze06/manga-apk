package com.screentranslator.app.data.preferences

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.screentranslator.app.data.model.TranslationHistoryEntry
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

private val GSON = Gson()
private const val MAX_HISTORY = 60

/**
 * Singleton repository for persisting translation history via DataStore (JSON string)
 * and managing saved translated screenshot images on internal storage.
 * Maintains an in-memory StateFlow so history updates appear in real-time with zero lag.
 */
class TranslationHistoryRepository(private val context: Context) {

    private val repositoryScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val historyMutationMutex = Mutex()

    private val imagesDir: File by lazy {
        File(context.filesDir, "history_images").apply {
            if (!exists()) mkdirs()
        }
    }

    private object Keys {
        val HISTORY_JSON = stringPreferencesKey("translation_history_json")
    }

    // In-memory cache for instantaneous real-time UI updates
    private val _historyFlow = MutableStateFlow<List<TranslationHistoryEntry>>(emptyList())
    val historyFlow: Flow<List<TranslationHistoryEntry>> = _historyFlow.asStateFlow()

    init {
        // Initialize history from DataStore and continuously observe updates
        repositoryScope.launch {
            try {
                context.dataStore.data.collect { prefs ->
                    val json = prefs[Keys.HISTORY_JSON] ?: "[]"
                    val parsed = parseJson(json)
                    _historyFlow.value = parsed
                }
            } catch (e: Exception) {
                Log.w("HistoryRepo", "DataStore kuzatishda xato: ${e.message}")
            }
        }
    }

    private fun parseJson(json: String): List<TranslationHistoryEntry> {
        return try {
            val type = object : TypeToken<List<TranslationHistoryEntry>>() {}.type
            GSON.fromJson<List<TranslationHistoryEntry>>(json, type) ?: emptyList()
        } catch (e: Exception) {
            emptyList()
        }
    }

    /**
     * Explicitly refreshes history from DataStore.
     */
    suspend fun refresh() = withContext(Dispatchers.IO) {
        historyMutationMutex.withLock {
            try {
                val prefs = context.dataStore.data.first()
                val json = prefs[Keys.HISTORY_JSON] ?: "[]"
                _historyFlow.value = parseJson(json)
            } catch (e: Exception) {
                Log.w("HistoryRepo", "Tarixni yangilaşda xato: ${e.message}")
            }
        }
    }

    /**
     * Saves a translated screenshot bitmap to the internal history directory.
     * Returns the absolute file path, or null if saving fails.
     */
    suspend fun saveHistoryImage(bitmap: Bitmap?, id: String): String? = withContext(Dispatchers.IO) {
        if (bitmap == null || bitmap.isRecycled || bitmap.width <= 0 || bitmap.height <= 0) {
            return@withContext null
        }
        try {
            val file = File(imagesDir, "hist_$id.jpg")
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 80, out)
                out.flush()
            }
            file.absolutePath
        } catch (e: Exception) {
            Log.e("HistoryRepo", "Rasm saqlaşda xato: ${e.message}", e)
            null
        }
    }

    /**
     * Adds an entry to history: immediately updates the in-memory state and persists to DataStore.
     */
    suspend fun addEntry(entry: TranslationHistoryEntry) = withContext(NonCancellable + Dispatchers.IO) {
        historyMutationMutex.withLock {
            var updated: List<TranslationHistoryEntry> = emptyList()
            var evicted: List<TranslationHistoryEntry> = emptyList()
            try {
                context.dataStore.edit { prefs ->
                    val current = parseJson(prefs[Keys.HISTORY_JSON] ?: "[]")
                    updated = (listOf(entry) + current.filterNot { it.id == entry.id }).take(MAX_HISTORY)
                    val keptIds = updated.mapTo(mutableSetOf()) { it.id }
                    evicted = current.filterNot { it.id in keptIds }
                    prefs[Keys.HISTORY_JSON] = GSON.toJson(updated)
                }

                // Publish only after the durable write succeeds, so the UI cannot show
                // a history item that disappears when the service is stopped.
                _historyFlow.value = updated
                evicted.forEach { old ->
                    old.imagePath?.let { path ->
                        try {
                            File(path).delete()
                        } catch (ignored: Exception) {
                        }
                    }
                }
                Log.d("HistoryRepo", "Tarix muvaffaqiyatli saqlandi. Jami yozuvlar: ${updated.size}")
            } catch (e: Exception) {
                Log.e("HistoryRepo", "DataStore ga yozişda xato: ${e.message}", e)
                entry.imagePath?.let { path ->
                    runCatching { File(path).delete() }
                }
            }
        }
    }

    suspend fun clearHistory() = withContext(NonCancellable + Dispatchers.IO) {
        historyMutationMutex.withLock {
            try {
                context.dataStore.edit { prefs ->
                    prefs[Keys.HISTORY_JSON] = "[]"
                }
                _historyFlow.value = emptyList()
                imagesDir.listFiles()?.forEach { it.delete() }
            } catch (e: Exception) {
                Log.w("HistoryRepo", "Tarixni tozalashda xato: ${e.message}")
            }
        }
    }

    companion object {
        @Volatile
        private var INSTANCE: TranslationHistoryRepository? = null

        fun getInstance(context: Context): TranslationHistoryRepository {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: TranslationHistoryRepository(context.applicationContext).also {
                    INSTANCE = it
                }
            }
        }
    }
}
