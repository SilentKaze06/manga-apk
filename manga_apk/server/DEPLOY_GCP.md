# Serverni ishga tushirish va joylashtirish

## Lokal PDF tarjimasi

`run_server.bat` (Windows) yoki `run_server.sh` (Linux) bitta to'liq server virtual muhitidan foydalanadi. Boshqa, faqat training uchun yaratilgan virtual muhit avtomatik tanlanmaydi. Standart tinglash manzili `127.0.0.1:8000`; Android emulatorda `http://10.0.2.2:8000` ishlatiladi.

Telefonni ishonchli Wi-Fi tarmog'ida ulash uchun `MANGA_BIND_HOST=0.0.0.0` va tasodifiy `MANGA_SERVER_API_KEY` belgilang. Ilovaga ayni kalitni kiriting. Debug APK faqat aniq ko'rsatilgan mahalliy HTTP manziliga ruxsat beradi; release uchun HTTPS kerak. Kalitni kodga, APKga yoki build paketiga yozmang.

PDF fayllari `MANGA_PDF_JOB_DIR` ichida saqlanadi. Lokal diskda bitta server worker ishlating. Standart navbat chegarasi 8 ta yakunlanmagan vazifa; ortiqcha yuklama 429 qaytaradi.

## Maxfiy ma'lumotlarsiz Docker paketi

Server papkasidan:

```powershell
python prepare_deploy.py
```

Buyruq yangi build katalogi manzilini chiqaradi. Paketga faqat server kodi, inference uchun kerakli ML modullari, vocabulary va uchta checkpoint olinadi. Mavjud EasyOCR cache'dan faqat ma'lum model fayllari qo'shiladi. PDFlar, tarjima tarixi, feedback, datasetlar, `.env`, oldingi checkpointlar va API kalitlar olinmaydi. `bundle-manifest.json` har bir tanlangan fayl hajmi va SHA-256 xeshini beradi.

Docker image'ni faqat shu katalogdan quring:

```powershell
docker build -t manga-translator "TAYYOR_PAKET_MANZILI"
```

`server` papkasining o'zini endi Docker build konteksti sifatida ishlatmang: ML paketini xavfsiz tarzda `prepare_deploy.py` qo'shadi. Har kod o'zgarishidan so'ng yangi paket tayyorlang. Mavjud output ustiga yozilmaydi.

PDF uchun doimiy diskli VM yoki lokal Docker'da host katalogini `/var/lib/manga/pdf_jobs`ga mount qiling. Konteyner foydalanuvchisi UID 10001; disk yozish huquqi shu foydalanuvchiga tegishli bo'lishi kerak. Tashqi HTTPS reverse proxy va `MANGA_SERVER_API_KEY` kerak. Bir disk uchun bitta worker/instance ishlating.

## Google Cloud Run chegarasi

Hozirgi PDF navbati bitta jarayon va lokal fayl tizimiga tayanadi. Cloud Run'da instance almashtirilishi fayllarni yo'qotishi, javobdan keyingi CPU cheklovi esa fon ishini to'xtatishi mumkin. Shu sabab Cloud Run'da PDF standart ravishda o'chiriladi; OCR/rasm endpointlari ishlashi mumkin. Faqat `MANGA_PDF_ENABLED=true`ni qo'yish doimiy saqlash muammosini hal qilmaydi.

PDFni Cloud Run'ga chiqarish uchun alohida doimiy job-storage, tashqi queue va ko'p instance orasida ishni faqat bir marta bajarish tartibi kerak. Bu resurslar avtomatik yaratilmaydi. Hozir to'liq PDF funksiyasi uchun doimiy diskli bitta VM/lokal serverdan foydalaning. [Cloud Run fon ishlari](https://docs.cloud.google.com/run/docs/tips/general).

`deploy_cloud_run.bat` xavfsiz paketni tayyorlaydi va PDFni o'chirgan holda OCR serverini joylashtiradi. Ishlatishdan oldin:

1. Google Cloud CLI'da kerakli akkaunt va loyihani tanlang.
2. Secret Manager'da server API kalitini saqlang, xizmat akkauntiga shu secret'ni o'qish huquqini bering.
3. `MANGA_SERVER_SECRET`ni secret nomi va versiyasi qilib belgilang, masalan `manga-server-key:latest`.
4. `deploy_cloud_run.bat`ni o'zingiz ishga tushiring. U tarmoqda xizmat yaratadi va pulli resurslardan foydalanishi mumkin.

Script `--allow-unauthenticated`ni Cloud Run IAM qatlami uchun ishlatadi; ilova darajasidagi `X-API-Key` majburiy bo'lib qoladi. Tayyorlash yoki test buyruqlari deploymentni o'z-o'zidan bajarmaydi.

Model fayllari mavjudligi til bo'yicha sifat kafolati emas. `/health` qaysi paketlar o'rnatilgani, qaysi modellar yuklangani, haqiqiy tarjima provayderi va PDF yoqilganligini ko'rsatadi. Kalitsiz ommaviy server tarjima so'rovlarini qabul qilmaydi.
