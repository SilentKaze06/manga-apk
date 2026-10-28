package com.screentranslator.app.data.preferences

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import com.screentranslator.app.data.model.PdfJobReference
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

/** Keeps the server job handle so Android process recreation does not orphan long PDF jobs. */
class PdfJobRepository(private val context: Context) {
    private object Keys {
        val JOB_ID = stringPreferencesKey("active_pdf_job_id")
        val JOB_TOKEN = stringPreferencesKey("active_pdf_job_token")
        val SERVER_URL = stringPreferencesKey("active_pdf_job_server_url")
        val FILENAME = stringPreferencesKey("active_pdf_job_filename")
    }

    val activeJob: Flow<PdfJobReference?> = context.dataStore.data.map { preferences ->
        val jobId = preferences[Keys.JOB_ID]
        val token = preferences[Keys.JOB_TOKEN]?.let(PdfJobTokenVault::decrypt)
        if (!jobId.isNullOrBlank() && !token.isNullOrBlank()) {
            PdfJobReference(jobId, token, preferences[Keys.SERVER_URL], preferences[Keys.FILENAME].orEmpty())
        } else null
    }

    suspend fun save(reference: PdfJobReference) {
        context.dataStore.edit { preferences ->
            preferences[Keys.JOB_ID] = reference.jobId
            preferences[Keys.JOB_TOKEN] = PdfJobTokenVault.encrypt(reference.token)
            reference.serverUrl?.let { preferences[Keys.SERVER_URL] = it }
                ?: preferences.remove(Keys.SERVER_URL)
            preferences[Keys.FILENAME] = reference.filename
        }
    }

    suspend fun clear() {
        context.dataStore.edit { preferences ->
            preferences.remove(Keys.JOB_ID)
            preferences.remove(Keys.JOB_TOKEN)
            preferences.remove(Keys.SERVER_URL)
            preferences.remove(Keys.FILENAME)
        }
    }
}
