# 🎌 Manga & Comic AI: End-to-End Machine Learning Tizimi

Ushbu loyiha berilgan 3 ta yirik ma'lumotlar to'plami (**Detection**, **OCR**, **Inpainting**) asosida to'liq avtomatlashtirilgan sun'iy intellekt tizimini amalga oshiradi.

---

## 📊 1. Ma'lumotlar To'plami Tahlili (Exploratory Data Analysis)

Arxivdagi 6 ta `.docx` hujjatidan jami **8,124 ta toza annotatsiya** ajratib olindi va `train` (80%), `val` (10%), `test` (10%) qismlarga bo'lindi:

| Vazifa (Task) | Umumiy Hajm | O'quv (Train) | Validatsiya (Val) | Test | Maqsad |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Bubble & Text Detection** | **3,062 sahifa** | 2,449 | 306 | 307 | Nutq bulutlari (`speech_bubble`), fikr bulutlari (`thought_bubble`), matnlar (`text`), SFX chegaralarini aniqlash. |
| **2. Multi-Language OCR** | **2,577 crop** | 2,061 | 257 | 259 | Yapon (`ja`), Koreys (`ko`), Xitoy (`zh`) va Ingliz (`en`) vertikal/gorizontal matnlarini o'qish. |
| **3. Deep Inpainting** | **2,485 triplet** | 1,988 | 248 | 249 | Matnni o'chirib, manga kadrining asl to'qimasini (screentone) tiklash. |

---

## 🧠 2. Neyrotarmoq Arxitekturalari (`models.py`)

1. **`MangaBubbleDetector` (Object Detection)**:
   - 4 pog'onali chuqur konvolyutsion xususiyat ajratuvchi (ResNet/FPN uslubi).
   - Anchor-based ko'p masshtabli boshqaruvchi (Bounding Box regresiyasi + Klassifikatsiya logitlari).
   - Yo'qotish funksiyalari: Smooth L1 Loss + Cross Entropy.

2. **`MangaOCRModel` (Vision + BiLSTM + CTC)**:
   - Konvolyutsion xususiyat qatlami (Spatial feature downsampling).
   - 2 qatlamli ikki yo'nalishli LSTM (Bidirectional LSTM) ketma-ketlik modeli.
   - **2,763 ta noyob belgi**dan iborat lug'at (Kanji, Hangul, Hanzi, Lotin harflari va tinish belgilari).
   - Yo'qotish funksiyasi: `nn.CTCLoss` (Connectionist Temporal Classification).

3. **`MangaInpaintingModel` (Deep U-Net + Dilated Convolutions)**:
   - 4-kanalli kirish: `[RGB Tasvir (3 kanal) + Binar Maska (1 kanal)]`.
   - Skip-connection li U-Net va keng qamrovli (dilated) konvolyutsiya bloklari.
   - Yo'qotish funksiyasi: Valid piksellar uchun $L_1$ + Maska teshiklari uchun $6 \times L_1$ vaznli yo'qotish.

---

## 🚀 3. Tizimni Ishga Tushirish Qo'llanmasi

### A) Ma'lumotlarni tahlil qilish va grafiklarni yaratish (EDA):
```bash
python eda_analysis.py
```
*(Grafiklar `eda_charts/` papkasiga saqlanadi)*

### B) Barcha 3 ta modelni o'qitish va baholash:
```bash
python train_and_evaluate.py
```
*(Og'irliklar `checkpoints/` papkasiga, o'qish egri chiziqlari `training_metrics/` papkasiga saqlanadi)*

### C) End-to-End Pipeline ni sinovdan o'tkazish:
```bash
python manga_pipeline.py
```
*(Natijalar `demo_output/` papkasida hosil bo'ladi)*

### D) Interaktiv Veb Ilova (Gradio Web UI):
**Windows uchun qulay usul (1 marta bosish orqali):**
- Papkadagi `run.bat` yoki `start.bat` faylini sichqoncha bilan ikki marta bosing.

**Yoki konsol (terminal) orqali:**
```bash
python app.py
```
Brauzerda avtomatik ochiladi: `http://127.0.0.1:7860`

### E) Ma'lumotlarni tekshirish va import qilish:
```bash
python -m tools.audit_all_data
python -m tools.inspect_collected_data
```
Ikkinchi buyruq hisobotni `reports/data_inspection_report.json` fayliga saqlaydi.
Bu tekshiruvlar o'quv ma'lumotlarini o'zgartirmaydi.

`extract_datasets.py` — o'quv bazasini qayta yaratish vositasi:
```bash
python extract_datasets.py "C:\path\datasets.zip"
```
ZIP berilmasa, `extracted_data/` ichidagi mavjud matnlar ishlatiladi.
Ushbu import `manga_ml_datasets/` va `ocr_vocab.json` fayllarini qayta yozadi.

### F) 0–9, A–Z, a–z doimiy rasm bazasi:

`manga_ml_datasets/characters/characters.zip` ichida **210 227 ta rasm va
62 ta belgi** saqlanadi. Bu kerakli o'quv fayli; uni tozalash vaqtida
o'chirmang. Arxivdan 168 186 ta rasm o'qitish, 21 065 tasi validatsiya,
20 976 tasi test uchun ajratilgan.

Ilovaning qayta o'qitish tugmasi OCR uchun ushbu bazaning o'quv qismini
yig'ilgan tuzatishlar bilan aralashtirib ishlatadi. Ikkala manba mavjud
bo'lsa, har biridan tanlanish ehtimoli teng. Doimiy baza yig'ilgan
namunalarning 5 000 talik chekloviga kirmaydi. Qisqa siklda butun baza
o'qitilmaydi, tasodifiy namunalar olinadi.

Bu baza yakka belgilar klassifikatori uchun ishlatiladi;
`train_and_evaluate.py --task ocr` satr matnlari bazasi bilan ishlashda
davom etadi. Batafsil: [belgilar bazasi qo'llanmasi](manga_ml_datasets/characters/README.md).

---

---

## 🧠 4. Doimiy O'z-o'zini O'qitish Tizimi (Continuous Self-Learning & Active Feedback)

Dastur qayta ishlangan sahifalardan namunalar yig'adi. Avtomatik OCR,
detektor chegaralari va model tozalagan rasmlar tekshirilmagan hisoblanadi.
O'qitish faqat tasdiqlangan o'quv namunalari va mustaqil validatsiya mavjud
bo'lsa boshlanadi; model faqat tekshiruvlardan o'tganidan keyin almashtiriladi.

```
[Manga Sahifasi Yuklandi] ──► [Pipeline Tahlili (Detection + OCR + Inpainting)]
                                          │
                                          ▼
                         [Self-Learning Data Collector]
                                          │
         ┌────────────────────────────────┼────────────────────────────────┐
         ▼                                ▼                                ▼
  [OCR Matn Qirqimlari]       [Manga Sahifa Kadrlari]           [Inpainting Uchliklari]
  (ocr/crops/*.png)            (detection/pages/*.png)           (raw, mask, clean)
         │                                │                                │
         └────────────────────────────────┬────────────────────────────────┘
                                          │
                   [Inson tasdiqlagan o'quv namunalarini tanlash]
                                          │
                                          ▼
                      [Experience Replay Buffer Trenajyori]
                   (Yangi ma'lumotlar + Asl bazadan namunalar)
                                          │
                                          ▼
                     [Incremental Model Fine-Tuning (AdamW)]
                                          │
                                          ▼
                   [Mustaqil validatsiya: yaxshilanish va regressiyasiz natija]
                                          │
                                          ▼
                   [Checkpointni saqlash va Hot Reload]
```

### Mexanizmning 4 ta asosiy ustuni:

1. **Avtomatik Ma'lumot To'plash (`self_learning_collector.py`)**:
   - Har bir tahlil qilingan manga sahifasidan nutq bulutlari, matn qirqimlari (OCR crops) va tozalash uchliklari (Inpainting triplets) avtomatik ajratilib `self_learning_data/` papkasiga saqlanadi.
   - **Insoniy Teskari Aloqa (Human-in-the-Loop)**: Foydalanuvchi tuzatgan OCR matni tasdiqlangan namuna sifatida saqlanadi. Avtomatik belgilarni segmentlash natijasini ham ko'zdan kechirish kerak; tuzatish detektor chegaralari yoki model yaratgan toza rasmni tasdiqlamaydi.

2. **Lug'at va model mosligi (`ocr_vocab.json`)**:
   - Avtomatik sahifa natijasi ishchi lug'atni kengaytirmaydi. Lug'at indekslari checkpoint bilan mos bo'lishi kerak. Yangi belgilarni qo'shish alohida o'quv ma'lumoti va validatsiyani talab qiladi.

3. **Experience Replay Buffer (`self_learning_trainer.py`)**:
   - OCR qayta o'qitishda doimiy `characters/characters.zip` bazasining o'quv qismi yig'ilgan belgilar bilan aralashtiriladi. Validatsiya va test rasmlari o'qitishga qo'shilmaydi; mavjud lug'at indekslari saqlanadi.

4. **Hot Reloading & Avto-O'qitish Boshqaruvi (`app.py`)**:
   - Foydalanuvchi belgilagan chegara (masalan har 5 ta yangi sahifadan so'ng) yetganda yoki "🚀 Fine-Tune Now" tugmasi bosilganda o'qitish fonda ishga tushadi.
   - Mustaqil tekshiruvdan o'tgan yangi vaznlar operativ xotiraga yuklanadi. Tekshiruv mavjud bo'lmasa yoki natija yomonlashsa, amaldagi checkpoint saqlanadi; xotiraga qayta yuklash xatosi ham aniq xabar qilinadi.

### Mustaqil validatsiya fayllari

`validation/ocr.jsonl`, `validation/detector.jsonl` va
`validation/inpainting.jsonl` fayllari kerak. Har bir vazifa uchun kamida
10 ta haqiqiy, takrorlanmagan va o'qitishda ishlatilmagan rasm bo'lishi
shart. Har satr alohida JSON obyekt; nisbiy rasm yo'llari ushbu ML papkasiga
nisbatan olinadi. `verified` qiymati faqat inson tekshirgan namunaga `true`
qilib qo'yiladi. Bu fayllar avtomatik yaratilmaydi.

OCR namunasi:

```json
{"verified":true,"split":"val","source_type":"real_manga_page","image_path":"validation/images/line_001.png","text":"HELLO"}
```

Matn to'liq va lug'atga ma'lum bo'lishi kerak. 32 ta CTC qadamiga sig'magan
matn (takroriy belgilar orasidagi bo'sh qadamlar ham hisoblanadi) rad etiladi,
kesib tashlanmaydi. OCR tekshiruvi satr CER ko'rsatkichi bilan birga
`characters.zip` arxivining doimiy validatsiya qismidan 256 tagacha belgini
tekshiradi; bu arxiv ham mavjud bo'lishi kerak.

Detektor namunasi (`bbox` piksel koordinatalarida):

```json
{"verified":true,"split":"val","source_type":"real_manga_page","image_path":"validation/images/page_001.png","annotations":[{"category":"speech_bubble","bbox_format":"xyxy","bbox":[20,30,120,160]}]}
```

Sinflar: `speech_bubble`, `thought_bubble`, `narration_box`, `sfx`.
To'plamda kamida bitta musbat annotatsiya bo'lishi kerak. Baholash precision,
recall va F1 bo'yicha, IoU 0.5 da bajariladi.

Inpainting namunasi:

```json
{"verified":true,"split":"val","source_type":"real_manga_page","raw_image":"validation/images/raw_001.png","mask":"validation/images/mask_001.png","ground_truth":"validation/images/clean_001.png","ground_truth_source":"human_cleaned"}
```

`ground_truth_source` faqat `human_cleaned` yoki `paired_original` bo'ladi.
Uchta rasm o'lchami bir xil, maska esa bo'sh bo'lmasligi kerak. Model yaratgan
rasm mustaqil to'g'ri javob sifatida ishlatilmaydi. Baholash maska ichidagi
o'rtacha mutlaq xato (`hole_mae`) bo'yicha o'tadi.

Barcha kerakli ko'rsatkichlar yomonlashmasligi va kamida bittasi yaxshilanishi
shart. `NaN`/cheksiz natijalar yoki vaznlar rad etiladi. Checkpoint avval
vaqtinchalik faylga to'liq yoziladi; eski vaznlar `*_prev.pt` sifatida
saqlanadi. Tekshiruv natijasi `last_promotion_results` va
`training_metrics/self_learning_history.json` da qayd etiladi.

### OCR tili va asl rasmni saqlash

`process_page(..., source_lang=...)` uchun `ja`, `ko`, `zh`, `en`, `auto`
qabul qilinadi. `auto` yapon tilini taxmin qilmaydi: serverdagi ko'p tilli
OCR fallback zarurligini bildiradi. Gradio interfeysida til ochiq tanlanadi.
EasyOCR modeli avtomatik yuklab olinmaydi; mahalliy kesh yoki
`MANGA_EASYOCR_MODEL_DIR` ishlatiladi. Ishonchsiz/bo'sh OCR natijasida asl
matn o'chirilmaydi; ishonchli OCR bo'lganda ham maskadan tashqari piksellar
saqlanadi. Mavjud checkpointlar bo'sh yoki noto'g'ri bo'lsa, tasodifiy yangi
model bilan ishlash o'rniga yuklash xatosi chiqariladi.

---

## 📁 Loyiha Strukturasi

```
mashinali oqitish/
├── extracted_data/            # Datasetlarni qayta yaratish uchun manba matnlar
├── manga_ml_datasets/          # Asl o'quv bazalari
│   └── characters/            # Doimiy 0–9, A–Z, a–z rasm bazasi
│       ├── characters.zip     # 210 227 ta rasm; o'chirmang
│       └── dataset_info.json  # Baza hajmi, sinflar va nazorat summasi
├── self_learning_data/         # Tizim ishlatilgan sari to'planadigan ma'lumotlar bazasi
│   ├── ocr/crops/             # Yig'ilgan matn qirqimlari (.png)
│   ├── detection/pages/       # Yig'ilgan manga sahifalari (.png)
│   ├── inpainting/raw,mask,clean/ # Tozalash uchun fon uchliklari (.png)
│   └── stats.json             # Jonli statistika va avto-o'qitish hisoblagichi
├── checkpoints/               # Saqlangan PyTorch modellari (.pt)
├── training_metrics/          # Konvergentsiya egri chiziqlari va o'rganish tarixi
├── test_chapter_57/           # Ilova namunalari va bob bilan sinash uchun rasmlar
├── recent_samples/            # Interfeysdagi oxirgi yuklangan sahifalar
├── tools/                     # Ma'lumotlar sifatini tekshiruvchi yordamchi skriptlar
├── models.py                  # PyTorch neyrotarmoq arxitekturalari
├── dataset_loader.py          # PyTorch Dataset va DataLoader klasslari
├── train_and_evaluate.py      # Boshlang'ich multi-task o'qitish va validatsiya
├── manga_pipeline.py          # End-to-End inference va avtomatik ma'lumot to'plash pipeline
├── self_learning_collector.py # Ma'lumotlarni yig'uvchi va lug'atni kengaytiruvchi modul
├── self_learning_trainer.py   # Experience Replay orqali modellarni doimiy o'qituvchi dvijok
├── app.py                     # Gradio orqali to'liq Self-Learning Web Studio
└── ocr_vocab.json             # 2,763+ belgidan iborat doimiy kengayuvchi lug'at
```

`demo_output/`, `eda_charts/` va `reports/` natijalar yaratilganda avtomatik
ochiladi. `checkpoints/*_prev.pt` oldingi model holatlari bo'lib, ularni
zaxira sifatida saqlang. O'quv bazalari, `ocr_vocab.json` va namunaviy rasmlar
ilova hamda o'qitish vositalari tomonidan ishlatiladi.

