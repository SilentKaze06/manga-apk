package com.screentranslator.app.ui.pdf

import androidx.lifecycle.ViewModel
import com.screentranslator.app.data.model.PdfTranslationReviewBlock

/** Unsaved edits survive changing tabs or rotating the activity. They never cross jobs/pages. */
class PdfReviewDrafts : ViewModel() {
    private val drafts = mutableMapOf<Pair<String, Int>, List<PdfTranslationReviewBlock>>()
    fun get(jobId: String, page: Int): List<PdfTranslationReviewBlock>? = drafts[jobId to page]
    fun save(jobId: String, page: Int, blocks: List<PdfTranslationReviewBlock>) {
        drafts[jobId to page] = blocks.toList()
    }
    fun clear(jobId: String, page: Int) { drafts.remove(jobId to page) }
    fun clearJob(jobId: String) { drafts.keys.removeAll { it.first == jobId } }
}
