# Google Play Console Permission Declaration & Justification

When submitting **Screen Translator** to the Google Play Console, apps requesting `SYSTEM_ALERT_WINDOW` and `MediaProjection` (with foreground service type `mediaProjection`) undergo strict policy reviews. Below is the ready-to-submit declaration text for your store listing and permission declaration declaration forms.

---

## 1. Justification for `SYSTEM_ALERT_WINDOW` (Display over other apps)

### Google Play Console Prompt:
> *"Describe the core functionality that requires displaying over other apps, and why alternative solutions (such as Notifications or Activity launch) are not feasible."*

### Declaration Statement:
```text
Screen Translator is an on-screen assistive translation utility designed to help users understand foreign-language content inside third-party apps (such as messaging apps, web browsers, social media, and reading apps).

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
Screen Translator requires MediaProjection to capture a single static screenshot frame of the user's visible display only when the user explicitly taps the floating translation bubble.

Core Details & Privacy Safeguards:
1. User-Initiated Single Frame Capture: Screen capture is strictly episodic and on-demand. The app NEVER conducts continuous screen recording, video streaming, or passive background monitoring. When the user taps the bubble, a single frame is acquired, immediately converted to a bitmap for optical character recognition (OCR), and the VirtualDisplay/ImageReader is immediately closed.
2. Explicit Consent & Visible Indicator: The user must explicitly grant permission via the system MediaProjection dialog ("Start now"). A persistent low-priority foreground notification is displayed at all times while the service is active, ensuring complete user awareness.
3. Privacy & Security: OCR and language identification are performed 100% locally on-device using Google ML Kit. Captured screen bitmaps are never stored to disk and are never uploaded to any remote server; they are immediately recycled after text coordinate extraction.
```

---

## 3. Short Summary for User Facing Privacy Policy

```text
Screen Translator captures single-frame screenshots solely when triggered by the user via the floating button to detect on-screen text coordinates. The captured images are processed in volatile memory and discarded immediately. No screenshots or sensitive personal data are ever recorded continuously, stored to device storage, or transmitted to third parties.
```
