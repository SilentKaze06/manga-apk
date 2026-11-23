# Google Play Console Permission Declaration & Justification

When submitting **Screen Translator** to the Google Play Console, apps requesting `SYSTEM_ALERT_WINDOW`, `MediaProjection` (with foreground service type `mediaProjection`), `FOREGROUND_SERVICE_SPECIAL_USE`, and `FOREGROUND_SERVICE_DATA_SYNC` undergo strict policy reviews. Below is the ready-to-submit declaration text for your store listing and permission declaration forms.

---

## 1. Justification for `SYSTEM_ALERT_WINDOW` (Display over other apps)

### Google Play Console Prompt:
> *"Describe the core functionality that requires displaying over other apps, and why alternative solutions (such as Notifications or Activity launch) are not feasible."*

### Declaration Statement:
```text
Screen Translator is an on-screen assistive translation utility designed to help users understand foreign-language content inside third-party apps (such as messaging apps, web browsers, social media, manga readers, and document viewers).

The SYSTEM_ALERT_WINDOW permission is used exclusively for two user-facing features:
1. Floating Quick-Action Bubble: A small, user-movable floating action button positioned on the edge of the screen that allows the user to trigger translation with a single tap without navigating away from the app they are currently using.
2. In-Place Translation Overlays: When text is detected and translated, the translated text cards are rendered directly over the coordinates of the original foreign text using WindowManager. This in-place replacement allows users to read translated content seamlessly within its original visual context (preserving paragraph layout, UI labels, and reading flow).

Why alternatives cannot work:
Standard notifications or switching to an Activity would force the user out of their active app, disrupting reading, messaging, or navigation flow. In-place contextual overlay is the fundamental core value proposition of this application.
```

---

## 2. Justification for `MediaProjection` & `FOREGROUND_SERVICE_MEDIA_PROJECTION`

### Google Play Console Prompt:
> *"Explain why MediaProjection and the mediaProjection foreground service type are required for your app's core feature."*

### Declaration Statement:
```text
Screen Translator requires MediaProjection and FOREGROUND_SERVICE_MEDIA_PROJECTION to capture a single static screenshot frame of the user's visible display only when the user explicitly taps the floating translation bubble.

Core Details & Safeguards:
1. User-Initiated Single Frame Capture: Screen capture is strictly episodic and on-demand. The app NEVER conducts continuous screen recording, video streaming, or passive background monitoring. When the user taps the bubble, a single frame is acquired, immediately converted to a bitmap for optical character recognition (OCR), and the VirtualDisplay/ImageReader is immediately closed.
2. Explicit Consent & Visible Indicator: The user must explicitly grant permission via the system MediaProjection dialog ("Start now"). A persistent notification is displayed while capture operations are active, ensuring complete user awareness.
3. Secure Processing: Captured screen frames are converted into coordinates and text blocks. If default on-device mode is selected, recognition runs locally via Google ML Kit.
```

---

## 3. Justification for `FOREGROUND_SERVICE_SPECIAL_USE` (Android 14+, Android 15)

### Google Play Console Prompt:
> *"Explain why the specialUse foreground service type is required and what user-facing feature it powers."*

### Declaration Statement:
```text
Screen Translator utilizes FOREGROUND_SERVICE_SPECIAL_USE for FloatingBubbleService (subtype declared in manifest property android.app.PROPERTY_SPECIAL_USE_FGS_SUBTYPE as "Floating on-screen translation trigger and overlay").

Core Functionality:
The service maintains the lightweight floating action bubble and handles the touch events and floating UI overlay lifecycle across third-party applications.
Because the service must remain responsive to user touch interactions at all times without being terminated by OS power-saving routines while reading or chatting in other apps, a persistent foreground service with a visible ongoing notification is required. No standard foreground service type fits this persistent overlay accessibility controller use case, necessitating the specialUse category under Android 14 and Android 15 guidelines.
```

---

## 4. Justification for `FOREGROUND_SERVICE_DATA_SYNC` (Android 14+, Android 15)

### Google Play Console Prompt:
> *"Explain why the dataSync foreground service type is required and what data synchronization it performs."*

### Declaration Statement:
```text
Screen Translator utilizes FOREGROUND_SERVICE_DATA_SYNC for PdfTranslationService to manage background translation jobs for multi-page PDF documents and manga volumes.

Core Functionality:
1. Long-Running Translation Synchronization: Translating full-length PDF documents involves multi-page OCR, background text translation, typesetting, and generating translated PDFs. This process can take several minutes depending on document page count.
2. Background Job Tracking: PdfTranslationService uploads documents, polls remote translation status, and downloads the resulting translated PDF even when the user switches to other apps or locks their screen.
3. System Compliance: The service displays an ongoing progress notification showing translation percentage. It adheres strictly to Android 15 foreground service limits by implementing onTimeout() to gracefully terminate within the 6-hour dataSync execution window and uses START_NOT_STICKY to prevent unintended automatic restarts.
```

---

## 5. Privacy Policy & Data Safety Disclosures (Remote Modal Uploads & Local Caching)

### Remote Modal OCR & Translation Server Uploads:
```text
When the user configures or selects remote translation processing (such as the Manga OCR server hosted on Modal.com or a custom server endpoint):
- Transmitted Data: The app transmits cropped dialogue bubble image slices, page image segments, or uploaded PDF documents to the designated server over secure HTTPS.
- Purpose: Data is transmitted solely to perform optical character recognition (e.g. Manga-OCR, PaddleOCR) and machine translation.
- Retention: Uploaded image crops and intermediate translation files are processed transiently in temporary server storage and are not stored permanently, sold, or shared with third parties.
- Opt-In: Users can freely toggle between local on-device ML Kit processing and remote server processing in Settings at any time.
```

### Local History & Image Caching (`history_images`):
```text
- Local Storage: When translation history is enabled, translated text and corresponding cropped preview images are cached locally on device internal storage in the app-private directory (files/history_images).
- Backup & Access Protection: The history_images directory is explicitly excluded from Android Auto Backup and cloud extraction rules (via backup_rules.xml and data_extraction_rules.xml). History images remain sandboxed within the app and cannot be accessed by other applications.
- User Control: Users can view, clear individual items, or purge the entire translation history and cached images at any time via app Settings.
```

---

## 6. Ready-to-Publish Privacy Policy Summary

```text
Screen Translator values user privacy and provides full transparency regarding data collection and processing:
1. On-Screen Capture: Screen capture occurs strictly on-demand when explicitly triggered by the user via the floating action bubble. No continuous recording or background screen capture occurs.
2. On-Device Processing by Default: By default, OCR text recognition is executed directly on the user's device using Google ML Kit.
3. Remote Server Translation (Modal / Custom Server): If the user chooses to use the Manga OCR server (e.g. hosted on Modal.com) or remote translation APIs (DeepL, LibreTranslate, MyMemory), selected text snippets or cropped dialogue images/PDFs are sent securely via HTTPS solely for translation execution. Data is never repurposed, monetized, or shared with advertisers.
4. Local History Caching: Translated items and preview snapshots are stored privately on local device storage (history_images), excluded from cloud backups, and can be deleted by the user at any time.
```
