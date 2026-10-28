---
title: Manga Translator
sdk: docker
app_port: 8080
---

# Manga va kitob PDF tarjima serveri

FastAPI serveri matnli kitob, skan PDF va komikslarni tarjima qiladi. Android hozir PDF oqimidan foydalanadi. PDF matni Google tarjima yo'llariga yuboriladi; Android'dagi eski ekran tarjimoni uchun yozilgan DeepL/LibreTranslate/MyMemory tanlovi PDF provayderini o'zgartirmaydi.

## Ishga tushirish

Windows: `run_server.bat`; Linux: `run_server.sh`. Standart manzil `http://127.0.0.1:8000`, emulatorda `http://10.0.2.2:8000`. Launcher server uchun alohida `venv` va `requirements.txt`dan foydalanadi. Telefonni LAN orqali ulash uchun `MANGA_BIND_HOST=0.0.0.0` va `MANGA_SERVER_API_KEY` belgilang. Kalitni ilovaning server sozlamasiga ham kiriting.

Oddiy lokal loopback rejimida launcher `MANGA_ALLOW_UNAUTHENTICATED=true`ni aniq o'rnatadi. Qo'lda yoki ommaviy muhitda bu flag berilmasa va API kaliti bo'lmasa, tarjima so'rovlari 503 bilan rad etiladi. Release APKga HTTPS endpoint kerak; debug'da aniq mahalliy HTTP ishlatilishi mumkin.

## OCR va modellar

Kitob/skan oqimida PaddleOCR: yapon, koreys, xitoy, ingliz, rus va umumiy Latin modellaridan foydalaniladi. Komiksning asosiy ML yo'li `MANGA_ML_DIR`dagi detector/OCR/inpainting modellarini ishlatadi; standart manzil loyiha yonidagi `mashinali oqitish` papkasi. EasyOCR manba tiliga qarab tanlanadi, tayyor cache `MANGA_EASYOCR_MODEL_DIR` bilan berilishi mumkin. Mavjud bo'lmagan model vaziyatlari zaxira OCRga o'tishi mumkin; `/health` model yuklanganligi va paket mavjudligini alohida ko'rsatadi. `auto` endi yaponcha deb majburan belgilanmaydi.

OCR matnni o'qiy olmagan yoki tarjima sig'magan hududda asl rasm saqlanadi; uzun transkript yangi sahifalarga bo'linadi. Haqiqiy tillardagi sifatni real sahifalar bilan tekshirish kerak.

## Endpointlar

- `GET /health`: konfiguratsiya, provayder, model/paket holati va PDF imkoniyati. HTTP 200 model allaqachon yuklangan degani emas.
- `POST /translate-manga`, `POST /ocr-manga`, `POST /detect-and-translate`: rasm va OCR amallari.
- `POST /translate-pdf`: `file`, `document_mode=book|comic`, `source_lang`, `target_lang`, `output_mode=translated|bilingual`.
- `GET /pdf-jobs/{job_id}`: progress.
- `POST /pdf-jobs/{job_id}/cancel`, `/resume`: bekor qilish/davom ettirish.
- `GET /pdf-jobs/{job_id}/review/{page_index}`: OCR va tarjima bloklari.
- `POST /pdf-jobs/{job_id}/review`: `corrections` ro'yxati bilan tahrirlash.
- `GET /pdf-jobs/{job_id}/download`: tayyor PDF.

Himoyalangan so'rovlar `X-API-Key` oladi. PDF vazifasi yaratilganda qaytadigan `job_token` keyingi so'rovlarda `X-PDF-Job-Token` sifatida yuboriladi. Tokenni ayni vazifa yaratilgan serverdan boshqa manzilga yubormang.

## Limitlar va saqlash

Rasm: 10 MiB / 25 megapiksel. PDF: 50 MiB / 500 bet. `MANGA_MAX_UPLOAD_BYTES`, `MANGA_MAX_IMAGE_PIXELS`, `MANGA_MAX_PDF_UPLOAD_BYTES`, `MANGA_MAX_PDF_PAGES` bilan o'zgartiriladi. OCR render DPI va piksel limiti ham mavjud. `MANGA_PDF_MAX_PENDING_JOBS` standart 8; navbat to'lsa 429 + Retry-After qaytadi. `MANGA_TRANSLATION_TIMEOUT_SECONDS` standart 8 soniya, chegaralangan.

PDF va oraliq natijalar `MANGA_PDF_JOB_DIR`da turadi. Tugagan/xatoli/bekor qilingan vazifalar `MANGA_PDF_JOB_RETENTION_HOURS` (standart 48 soat)dan keyin startup yoki keyingi so'rovlar vaqtida tozalanadi. Bu mustaqil taymer emas. Kitob chiqishi qayta sahifalanadi; asl layout aynan ko'chirilmaydi.

## Docker va cloud

[Joylashtirish yo'riqnomasi](DEPLOY_GCP.md) bo'yicha `prepare_deploy.py`dan paket tayyorlang. Server papkasini to'liq upload qilmang. Paketga faqat aniq ro'yxatdagi kod va modellar olinadi; hujjatlar/feedback/datasetlar kirmaydi.

Cloud Run'da hozir PDF o'chiriladi: mahalliy navbatni doimiy cloud queue/storage o'rnida ishlatish mumkin emas. Lokal yoki doimiy diskli VM'da bitta server worker bilan PDF ishlaydi. Hech bir setup/test bu xizmatni avtomatik deploy qilmaydi.

## Tekshiruv

Server testlari: `venv\Scripts\python.exe -m unittest discover -s tests -v` (Linux'da `venv/bin/python`). Testlarda vaqtinchalik job katalogi va tashqi xizmatlar o'rniga nazorat javoblari ishlatiladi. Deploy paket testi: `python -m unittest discover -s deployment_tests -v`. Bu testlar real OCR yoki tarjima provayderining aniqligini o'lchamaydi.

PDF backend PyMuPDF ishlatadi. Tarqatishdan oldin uning [litsenziya shartlarini](https://pymupdf.readthedocs.io/en/latest/about.html) loyihaning foydalanish shakliga mos tekshiring.
