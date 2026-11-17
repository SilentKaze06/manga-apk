# 🎌 AI Manga & Comic Translator (Manga APK)

Ushbu repozitoriy manga, manhwa, manhua va komikslarni sun'iy intellekt yordamida avtomatik tarjima qilish, matn pufakchalarini aniqlash (detection), belgilarni o'qish (OCR) hamda kadr to'qimasini tiklash (inpainting) bo'yicha to'liq ekotizimni o'z ichiga oladi.

---

## 📂 Loyiha Tuzilmasi (Project Architecture)

Loyihada 3 ta asosiy yo'nalish jamlangan:

```text
manga apk/
├── manga_apk/                 # Android Mobil Ilova va Backend Server
│   ├── app/                   # Jetpack Compose + Kotlin Android Ilova (API 24+)
│   ├── gradle/                # Gradle wrapper fayllari
│   ├── server/                # FastAPI asosidagi lokal/onlayn OCR & Tarjima Serveri
│   ├── build.gradle.kts       # Ilova qurish konfiguratsiyasi
│   └── PDF_TRANSLATION_ROADMAP.md
│
├── mashinali oqitish/         # ML Modellar, Datasetlar va O'qitish Skriptlari
│   ├── checkpoints/           # Tayyor o'qitilgan PyTorch modellari (.pt)
│   ├── extracted_data/        # Matnli annotatsiya fayllari
│   ├── manga_ml_datasets/     # Detection, Inpainting va OCR namuna ma'lumotlari
│   ├── models.py              # Neyrotarmoq arxitekturalari (Detector, OCR, Inpainting)
│   ├── manga_pipeline.py      # Yakuniy ishlov berish konveyeri
│   ├── train_and_evaluate.py  # Model o'qitish va baholash skripti
│   └── ML_GUIDE.md            # ML bo'yicha to'liq yo'riqnoma
│
├── hf_dosjanmanga/            # Hugging Face Space (Gradio + ZeroGPU) Deployment
│   ├── app.py                 # Gradio web interfeysi va FastAPI server
│   ├── main.py                # Server API routerlari
│   └── requirements.txt       # Hugging Face bog'liqliklari
│
└── TAHLIL_HISOBOTI.md         # To'liq texnik audit va tahlil hisoboti
```

---

## 📱 1. Android Ilova (`manga_apk/app`)
- **Texnologiyalar:** Kotlin, Jetpack Compose, Coroutines, Flow, Retrofit, DataStore.
- **Imkoniyatlar:**
  - PDF manga va komikslarni yuklash, serverda qayta ishlash va natijani yuklab olish.
  - OCR va tarjima qilingan matnlarni ko'rish va qo'lda tahrirlash (Review & Edit).
  - Ekran tarjimoni (Floating Bubble & Screen Capture) infratuzilmasi.

## ⚡ 2. Backend Server (`manga_apk/server`)
- **Texnologiyalar:** Python 3.10+, FastAPI, Uvicorn, PyTorch, PaddleOCR, Manga-OCR.
- **Asosiy Endpointlar:**
  - `/health` — Server holati va model tekshiruvi.
  - `/translate-pdf` & `/pdf-job-status/{token}` — PDF tarjima vazifalarini boshqarish.
  - `/ocr-manga` & `/translate-manga` — Rasm kadrlarini OCR va tarjima qilish.

## 🧠 3. Mashinaviy O'qitish (`mashinali oqitish`)
1. **`MangaBubbleDetector`**: Nutq va fikr bulutlarini aniqlash (Smooth L1 + CE Loss).
2. **`MangaOCRModel`**: Yapon, Koreys, Xitoy va Lotin matnlarini tanish (Vision + BiLSTM + CTC).
3. **`MangaInpaintingModel`**: Matnni tozalash va manga to'qimasini tiklash (U-Net + Dilated Convolutions).

---

## 🔒 Xavfsizlik va Maxfiylik (Privacy & Security)
Ushbu ochiq repozitoriyga:
- Shaxsiy API kalitlari va tokenlar yuklanmagan (kodda parametr sifatida beriladi).
- Foydalanuvchilarning shaxsiy PDF fayllari va keshlar (`pdf_job_data/`) kiritilmagan.
- Mahalliy muhit va og'ir kutubxonalar (`venv/`, `cloudflared.exe`, `build/`, `*.7z`) `.gitignore` orqali chiqarib tashlangan.

