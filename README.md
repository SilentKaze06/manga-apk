# 🎌 AI Manga & Comic Translator (Manga APK)

Ushbu repozitoriy manga, manhwa, manhua va komikslarni sun'iy intellekt yordamida avtomatik tarjima qilish, matn pufakchalarini aniqlash (detection), belgilarni o'qish (OCR) hamda kadr to'qimasini tiklash (inpainting) bo'yicha to'liq ekotizimni o'z ichiga oladi.

---

## 📂 Loyiha Tuzilmasi (Project Architecture)

Loyihada 3 ta asosiy yo'nalish jamlangan:

```text
manga apk/
├── manga_apk/                        # Android Mobil Ilova va Backend Server
│   ├── app/                          # Jetpack Compose + Kotlin Android Ilova (API 24+)
│   ├── gradle/                       # Gradle wrapper fayllari
│   ├── server/                       # FastAPI asosidagi lokal/onlayn OCR & Tarjima Serveri
│   │   ├── ml/                       # Yagona ML arxitektura va inference konveyeri
│   │   │   ├── models.py             # PyTorch modellar (Detector, OCR, Inpainting)
│   │   │   ├── manga_pipeline.py     # End-to-end Sahifa tozalash, OCR va joylashtirish
│   │   │   ├── detection.py          # Matn bloklari va nutq bulutlarini klasterlash
│   │   │   ├── ocr_engine.py         # Ko'p tilli OCR (ja, ko, zh, en, ru, auto)
│   │   │   ├── cleaning.py           # Adaptiv kadr tozalash (SimpleLama / Telea / fill)
│   │   │   ├── manga_text_cleaner.py # Neyron matn segmentatsiyasi va fon tozalagich
│   │   │   ├── typesetting.py        # Tarjima matnini bulut ichiga joylashtirish
│   │   │   ├── geometry.py           # Bulut konturlari va o'qish tartibi geometriyasi
│   │   │   ├── utils.py              # Umumiy yordamchi funksiyalar va shrift topish
│   │   │   └── ocr_vocab.json        # OCR belgilar lug'ati
│   │   ├── onnx_models/              # Eksport qilingan ONNX modellar
│   │   └── tests/                    # Server API va PDF testlari
│   ├── build.gradle.kts              # Ilova qurish konfiguratsiyasi
│   └── PDF_TRANSLATION_ROADMAP.md
│
├── mashinali oqitish/                # ML Datasetlar, O'qitish va ONNX Eksport Tizimi
│   ├── checkpoints/                  # O'qitilgan PyTorch vaznlari (.pt)
│   ├── manga_ml_datasets/            # Detection, Inpainting, OCR va Characters bazalari
│   ├── tools/                        # Dataset qurish va sifat auditi skriptlari
│   ├── tests/                        # ML o'qitish va pipeline testlari
│   ├── ml_paths.py                   # `manga_apk/server/ml` va `ocr_vocab.json` yo'l resolveri
│   ├── dataset_loader.py             # PyTorch Dataset va DataLoader klasslari
│   ├── train_and_evaluate.py         # Model o'qitish va ValidationGate tekshiruvi
│   ├── export_to_onnx.py             # PyTorch -> ONNX eksport skripti
│   ├── extract_datasets.py           # Annotatsiya arxivlaridan JSONL ajratish
│   ├── validation_gate.py            # Mustaqil validatsiya va atomik promotion
│   ├── app.py                        # Gradio asosidagi interaktiv Manga Studio
│   └── ML_GUIDE.md                   # ML bo'yicha to'liq yo'riqnoma
│
└── tests/                            # Yuqori darajadagi integratsion va regressiya testlari
    └── test_swarm_regression.py
```

---

## 📱 1. Android Ilova (`manga_apk/app`)
- **Texnologiyalar:** Kotlin, Jetpack Compose, Coroutines, Flow, Retrofit, DataStore.
- **Imkoniyatlar:**
  - PDF manga va komikslarni yuklash, serverda qayta ishlash va natijani yuklab olish.
  - OCR va tarjima qilingan matnlarni ko'rish va qo'lda tahrirlash (Review & Edit).
  - Ekran tarjimoni (Floating Bubble & Screen Capture) infratuzilmasi.

## ⚡ 2. Backend Server (`manga_apk/server`)
- **Texnologiyalar:** Python 3.10+, FastAPI, Uvicorn, PyTorch, PaddleOCR, Manga-OCR, ONNX Runtime.
- **Asosiy Endpointlar:**
  - `/health` — Server holati va model tekshiruvi.
  - `/translate-pdf` & `/pdf-job-status/{token}` — PDF tarjima vazifalarini boshqarish.
  - `/ocr-manga`, `/clean-manga`, `/detect-and-translate`, `/translate-manga`, `/translate-page-image` — Rasm kadrlarini aniqlash, OCR, tozalash va tarjima qilish.

## 🧠 3. Mashinaviy O'qitish (`mashinali oqitish`)
1. **`MangaBubbleDetector`**: Nutq, fikr, izoh bulutlari va SFX chegaralarini aniqlash (`BCEWithLogitsLoss + SmoothL1Loss + CrossEntropyLoss`).
2. **`MangaOCRModel`**: Yapon (`ja`), Koreys (`ko`), Xitoy (`zh`), Ingliz (`en`) va Rus (`ru`) matnlarini tanish (CNN + BiLSTM + CTC hamda belgi klassifikatori boshi).
3. **`MangaInpaintingModel`**: Matnni o'chirib, oq, qaymoqrang, qora, rangli, gradient va screentone fon to'qimasini tiklash (Deep U-Net + Dilated Bottleneck).

---

## 🔒 Xavfsizlik va Maxfiylik (Privacy & Security)
Ushbu repozitoriyga:
- Shaxsiy API kalitlari va tokenlar yuklanmagan (muhit o'zgaruvchilari orqali beriladi).
- Foydalanuvchilarning shaxsiy PDF fayllari va keshlar (`pdf_job_data/`) kiritilmagan.
- Mahalliy muhit, og'ir arxivlar va keshlar (`venv/`, `cloudflared.exe`, `build/`, `*.7z`) `.gitignore` orqali chiqarib tashlangan.
