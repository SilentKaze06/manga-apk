# 🎌 Manga & Comic AI: End-to-End Machine Learning Tizimi

Ushbu modul 3 ta asosiy vazifa (**Detection**, **OCR**, **Inpainting**) uchun ma'lumotlarni tayyorlash, modellarni o'qitish, mustaqil `ValidationGate` orqali tekshirish va `manga_apk/server/onnx_models/` ga eksport qilishni ta'minlaydi.

Model arxitekturalari, inference konveyeri va lug'at fayli (`models.py`, `manga_pipeline.py`, `utils.py`, `ocr_vocab.json`) `manga_apk/server/ml/` papkasida joylashgan va `ml_paths.py` yordamida avtomatik ulanadi.

---

## 📊 1. Ma'lumotlar To'plami (Datasetlar)

`manga_ml_datasets/` ichidagi asosiy o'quv to'plamlari:

| Vazifa (Task) | Umumiy Hajm | Maqsad |
| :--- | :--- | :--- |
| **1. Bubble & Text Detection** | **6 515 sahifa** | `speech_bubble`, `thought_bubble`, `narration_box`, `sfx` (hamda `text_line`/`text_bubble` aliaslari) |
| **2. Multi-Language OCR** | **2 299+ crop** | Yapon (`ja`), Koreys (`ko`), Xitoy (`zh`), Ingliz (`en`), Rus (`ru`) vertikal/gorizontal matnlar |
| **3. Deep Inpainting** | **1 700+ triplet** | Matnni o'chirib, oq/rangli/qora/gradient/screentone fon to'qimasini tiklash (`raw`, `mask`, `clean`) |
| **4. Characters Archive** | **210 227 rasm** | `manga_ml_datasets/characters/characters.zip` — 62 sinf (`0–9`, `A–Z`, `a–z`) |

---

## 🧠 2. Neyrotarmoq Arxitekturalari (`manga_apk/server/ml/models.py`)

1. **`MangaBubbleDetector` (Object Detection)**:
   - 4 bosqichli konvolyutsion backbone (`ConvBlock`, stride=2: `H/16 × W/16`) + `neck` + `1×1` bashorat boshi (`num_classes=4`).
   - Yo'qotish funksiyalari: `BCEWithLogitsLoss` (objectness) + `20 × SmoothL1Loss` (bbox) + `CrossEntropyLoss` (klassifikatsiya).

2. **`MangaOCRModel` (CNN + BiLSTM + Dual Head)**:
   - Konvolyutsion xususiyat ajratuvchi (`3 × MaxPool2d(2, 2)`, kenglik bo'yicha `8x` siqish).
   - Ketma-ketlik boshi: `AdaptiveAvgPool2d((1, None))` + 2 qatlamli BiLSTM + `seq_fc` (`nn.CTCLoss`).
   - Yakka belgi boshi: `AdaptiveAvgPool2d((4, 4))` + `fc` (`nn.CrossEntropyLoss`).
   - Lug'at: `manga_apk/server/ml/ocr_vocab.json` (**3 517+ token**).

3. **`MangaInpaintingModel` (Deep U-Net + Dilated Bottleneck)**:
   - 4-kanalli kirish: `[RGB Tasvir (3 kanal) + Binar Maska (1 kanal)]`.
   - Skip-connection li U-Net va `dilation=2, 4` bottleneck bloklari.
   - Yo'qotish funksiyasi: $4.0 \times L_1(\text{hole}) + 0.5 \times L_1(\text{valid}) + 1.0 \times L_1(\text{composite})$.

---

## 🚀 3. Tizimni Ishga Tushirish Qo'llanmasi

### A) Ma'lumotlarni tahlil qilish (EDA):
```bash
python eda_analysis.py
```
*(Grafiklar `eda_charts/` papkasiga saqlanadi)*

### B) Datasetlarni qurish va sifatini tekshirish:
```bash
python -m tools.build_manga_ocr_dataset
python -m tools.build_manga_inpaint_dataset
python -m tools.audit_all_data
python -m tools.inspect_collected_data
```
Hisobot `reports/data_inspection_report.json` fayliga saqlanadi.

`extract_datasets.py` — `extracted_data/` ichidagi matnli annotatsiyalardan JSONL yaratish vositasi (`extracted_data/` mavjud bo'lmasa yoki bo'sh bo'lsa, mavjud datasetlarni o'chirmasdan xato qaytaradi):
```bash
python extract_datasets.py "C:\path\datasets.zip"
```

### C) Modellarni o'qitish va `ValidationGate` tekshiruvi:
```bash
python train_and_evaluate.py --task all --epochs 10
python train_and_evaluate.py --task ocr --smoke-test
```
- Mavjud `checkpoints/best_manga_*.pt` fayllari faqat `validation/{ocr,detector,inpainting}.jsonl` mustaqil tekshiruvidan o'tgandagina atomik tarzda yangilanadi (aks holda nomzod `candidate_manga_*.pt` ga yoziladi).
- OCR o'qitish bir vaqtning o'zida ham `seq_fc` (CTC), ham `fc` (belgi boshi) ni o'qitadi.

### D) ONNX formatiga eksport qilish:
```bash
python export_to_onnx.py
```
Standart holatda modellar `manga_apk/server/onnx_models/` papkasiga eksport qilinadi (`--output-dir` orqali boshqa papka ham berilishi mumkin).

### E) Interaktiv Veb Studio (Gradio Web UI):
- Windows: `run.bat`
- Terminal: `python app.py` (`http://127.0.0.1:7860`)

---

## 🛡️ 4. Mustaqil Validatsiya (`validation_gate.py`)

`validation/ocr.jsonl`, `validation/detector.jsonl` va `validation/inpainting.jsonl` fayllari talab qilinadi (`verified=true`, `split="val"`, `source_type="real_manga_page"`).

OCR namunasi:
```json
{"verified":true,"split":"val","source_type":"real_manga_page","image_path":"validation/images/line_001.png","text":"HELLO"}
```

Detektor namunasi (`bbox` piksel koordinatalarida):
```json
{"verified":true,"split":"val","source_type":"real_manga_page","image_path":"validation/images/page_001.png","annotations":[{"category":"speech_bubble","bbox_format":"xyxy","bbox":[20,30,120,160]}]}
```

Inpainting namunasi (`ground_truth_source`: `human_cleaned` yoki `paired_original`):
```json
{"verified":true,"split":"val","source_type":"real_manga_page","raw_image":"validation/images/raw_001.png","mask":"validation/images/mask_001.png","ground_truth":"validation/images/clean_001.png","ground_truth_source":"human_cleaned"}
```

---

## 📁 Papka Strukturasi

```text
mashinali oqitish/
├── ml_paths.py                # manga_apk/server/ml/ va ocr_vocab.json yo'l resolveri
├── dataset_loader.py          # PyTorch Dataset va DataLoader klasslari
├── train_and_evaluate.py      # Multi-task o'qitish va ValidationGate promotion
├── export_to_onnx.py          # PyTorch -> ONNX eksport skripti
├── extract_datasets.py        # Annotatsiya matnlaridan dataset ajratish
├── validation_gate.py         # Mustaqil validatsiya va atomik checkpoint almashinuvi
├── eda_analysis.py            # Dataset statistikasi va grafiklari
├── app.py                     # Gradio interaktiv inspeksiya studiyasi
├── tools/                     # Dataset qurish va sifat auditi skriptlari
├── tests/                     # Pytest testlari
├── manga_ml_datasets/         # Asl o'quv bazalari (characters/characters.zip ni o'chirmang)
├── checkpoints/               # O'qitilgan PyTorch modellari (.pt)
└── training_metrics/          # O'qish egri chiziqlari
```
