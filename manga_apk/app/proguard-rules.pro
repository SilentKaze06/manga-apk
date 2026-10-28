# ─── Umumiy Android ───────────────────────────────────────────────────────────
-keepattributes *Annotation*
-keepattributes SourceFile,LineNumberTable
-keep public class * extends java.lang.Exception

# ─── Retrofit / Gson model klasslari ─────────────────────────────────────────
-keepclassmembers class * {
    @com.google.gson.annotations.SerializedName <fields>;
}
# Tarmoq ma'lumot klasslari (data class) uchun
-keep class com.screentranslator.app.data.network.** { *; }

# ─── DataStore / Preferences ──────────────────────────────────────────────────
-keep class com.screentranslator.app.data.preferences.** { *; }
-keep class com.screentranslator.app.data.model.** { *; }

# ─── OkHttp / Retrofit ────────────────────────────────────────────────────────
-dontwarn okhttp3.**
-dontwarn okio.**
-dontwarn retrofit2.**
-keep class retrofit2.** { *; }
-keepclasseswithmembers class * {
    @retrofit2.http.* <methods>;
}


# ─── Kotlin Coroutines ────────────────────────────────────────────────────────
-keepclassmembernames class kotlinx.** {
    volatile <fields>;
}
