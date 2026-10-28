# AI Manga Tarjimon — APK va loyiha tahlili

Hisobot sanasi: 26-oktabr 2026. Tekshiruvlar 25–26-oktabrda bajarildi.

**Asosiy baho:** loyiha yig‘iladigan va asosiy qismlari mavjud bo‘lgan prototip. PDF yuklash, tarjima vazifasini kuzatish, tahrirlash va eksport qilish yo‘li yozilgan. Biroq foydalanuvchi jarayonini to‘sadigan xatolar, serverni joylashtirishdagi maxfiylik muammosi va AI sifatini ishonchli baholashdagi kamchiliklar bor. Hozirgi holatini keng tarqatishga tayyor mahsulot deb baholamayman.

Bu hisobot kod, APK tuzilmasi, mavjud testlar va alohida lokal tekshiruvlarga asoslanadi. Telefon/emulator ulanmagan; qurilmadagi to‘liq ishlash, barcha tillardagi tarjima sifati va haqiqiy tarmoq tezligi tasdiqlanmagan. Dastur kodi o‘zgartirilmadi.

## 1. Tahlil doirasi

- [Asosiy APK](<C:/Users/Dosjan/Desktop/manga apk/manga_translator_debug.apk>).
- [Android loyiha](<C:/Users/Dosjan/Desktop/manga apk/manga_apk>): Kotlin, Jetpack Compose, DataStore, OkHttp/Retrofit, Google ML Kit.
- [Python server](<C:/Users/Dosjan/Desktop/manga apk/manga_apk/server>): FastAPI, PDF vazifalari, OCR va tarjima.
- [Mashinaviy o‘qitish loyihasi](<C:/Users/Dosjan/Desktop/manga apk/mashinali oqitish>): PyTorch modellar, datasetlar, o‘qitish va qayta o‘qitish.

Inventarizatsiya: Android’da 41 ta Kotlin fayli, 6 907 satr; serverning ikki asosiy Python faylida 2 097 satr; ML papkasining yuqori darajasidagi 22 ta Python faylida 7 989 satr bor. Oxirgi songa tajriba skriptlari ham kiradi. Bu sonlar test qamrovi yoki kod sifati bahosi emas.

## 2. APK haqida aniq ma’lumot

- Ilova nomi: **AI Manga Tarjimon**.
- Paket: `com.screentranslator.app.debug`.
- Versiya: `1.0.0`, versionCode `1`.
- Hajm: **64 197 354 bayt**, ya’ni 64,20 MB yoki 61,22 MiB.
- Minimal Android: API 24, Android 7.0.
- Target va compile SDK: 35.
- Arxitekturalar: `arm64-v8a`, `armeabi-v7a`, `x86`, `x86_64`.
- APK ichida 609 ta yozuv, 14 ta DEX va 12 ta native kutubxona bor.
- ZIP yaxlitlik tekshiruvi muvaffaqiyatli; APK v2 imzosi tasdiqlandi.
- Imzolovchi: **Android Debug**; `debuggable` yoqilgan. Bu sinov yig‘ilmasi.
- 16 KB ZIP joylashuv tekshiruvi o‘tdi. 64 bit ARM va x86 native kutubxonalarining LOAD alignment qiymatlari 16 KB. Bu natija qurilmada ishga tushirish sinovini almashtirmaydi.

SHA-256: `76fd0e4e6f9be420e4ed7755971dbef67a0c84b9b1186fbaed3ef2d4c5f6e861`.

Asosiy APK va loyihadagi mavjud `app-debug.apk` nusxasining SHA-256 qiymatlari bir xil. Joriy Gradle tekshiruvi ham debug yig‘ilmani dolzarb deb topdi; bu clean rebuild emas, mavjud natijalardan foydalanilgan yig‘ish tekshiruvidir.

APK hajmining katta qismi to‘rtta arxitektura uchun native kutubxonalar: 41 093 952 bayt, taxminan 64%. APK ichida ML Kit OCR modellari mavjud. Alohida ML loyihasidagi `best_manga_*.pt` modellar APK ichida topilmadi; ularni ishlatadigan yo‘l Python tomonda.

## 3. Hozir nima ishlashga ulangan?

Hozirgi bosh oyna PDF tarjimasidir. Foydalanuvchi PDF, kitob/komiks rejimi, manba tili, tarjima tili va ikki tilli chiqishni tanlaydi. Android faylni serverga yuklaydi, vazifa tokenini saqlaydi va holatni davriy so‘raydi. Server matnli PDFdan matn ajratadi yoki skan/komiksni OCRdan o‘tkazadi, matnni tarjima qiladi va yangi PDF yaratadi. Tayyor sahifadagi OCR va tarjima matnini tahrirlash imkoniyati ham bor.

Ma’lumot oqimi: **Android’dagi PDF → sozlangan Python server → OCR → Google tarjima yo‘li → serverda PDF yaratish → telefonga yuklab olish**. Matnli kitob chiqishi qayta sahifalanadi; asl maket aynan saqlanishi kutilmasligi kerak. Komiks yo‘lida sahifa san’atini saqlash va tarjimani joylashtirish mexanizmlari bor.

PDF uchun dastlabki limitlar: 50 MiB va 500 bet. Rasm endpointlarida 10 MiB va 25 megapiksel cheklovi mavjud. Serverda bu qiymatlar muhit sozlamalari orqali o‘zgartiriladi.

Ekran tasvirini olish, suzuvchi tugma, ekrandagi tarjima va mahalliy ML Kit kodi mavjud. Ammo hozirgi MainActivity navigatsiyasida ularni ishga tushiruvchi yo‘l yo‘q. Shuning uchun ularni ushbu versiyaning foydalanuvchiga ochiq, ishlayotgan funksiyasi deb ko‘rsatish to‘g‘ri emas. PDFga yo‘nalish ataylab tanlangan bo‘lishi mumkin; bunday holatda eski sozlamalar, ruxsatlar va tavsiflar ham moslashtirilishi kerak. Manba: [SettingsScreen.kt:210](<C:/Users/Dosjan/Desktop/manga apk/manga_apk/app/src/main/java/com/screentranslator/app/ui/settings/SettingsScreen.kt:210>).

## 4. Birinchi navbatda tuzatiladigan muammolar

**A. Server paketiga foydalanuvchi hujjatlari qo‘shilib ketadi — yuqori ustuvorlik.**

Dockerfile butun server papkasini `COPY . .` bilan ko‘chiradi. `.dockerignore` va `.gcloudignore` esa `pdf_job_data`ni chiqarib tashlamaydi. Ushbu papkada 40 ta fayl, jami 41 715 295 bayt ma’lumot mavjud. Demak, hozirgi papkadan Docker/cloud build qilinsa, saqlangan PDFlar, oraliq rasmlar va tarjimalar build kontekstiga, Docker image’ga ham tushadi. Bu tashqariga allaqachon sizib chiqqan degani emas: muammo joylashtirish jarayonida yuz beradi.

Tuzatish: ikkala ignore fayliga hujjatlar papkasini kiritish, Docker’da faqat zarur kodni aniq ko‘chirish, ishchi fayllarni build kontekstidan tashqarida saqlash. Manba: [Dockerfile:20](<C:/Users/Dosjan/Desktop/manga apk/manga_apk/server/Dockerfile:20>).

**B. Xato yoki bekor qilingan PDFdan keyin yangi fayl tanlash yopilib qoladi — foydalanuvchini to‘sadi.**

PDF tanlash oynasi faqat `currentJob == null` bo‘lganda chiqadi. “Boshqa PDF tanlash” esa faqat `completed` holatida berilgan. `failed` va `canceled` uchun faqat “Davom ettirish” bor. Agar ayni PDF doim xatoga uchrasa, foydalanuvchi o‘sha vazifaga qayta-qayta qaytadi. Vazifa doimiy xotirada saqlangani sabab oddiy qayta ochish ham muammoni hal qilmaydi.

Tuzatish: yakunlangan, xato va bekor qilingan holatlarning barchasida vazifa havolasini tozalash va yangi PDF tanlashga ruxsat berish. Manba: [PdfTranslationScreen.kt:446](<C:/Users/Dosjan/Desktop/manga apk/manga_apk/app/src/main/java/com/screentranslator/app/ui/pdf/PdfTranslationScreen.kt:446>).

**C. Vaqtinchalik internet xatosidan keyin progress yangilanmay qoladi.**

Holatni kuzatish sikli birinchi tarmoq istisnosida tugaydi. Qayta urinish yo‘q; kuzatuv effekti vazifa IDsi, token yoki server sozlamasi o‘zgarmaguncha qayta boshlanmaydi. Server ishni tugatsa ham ilovada eski foiz ko‘rinishi mumkin. Oynadan chiqib qaytish tiklashi mumkin, ammo bu foydalanuvchiga ko‘rsatilmagan.

Tuzatish: cheklangan va oraliq bilan takrorlash, oxirgi yangilanish vaqtini ko‘rsatish, qo‘lda “Yangilash” tugmasi. Manba: [PdfTranslationScreen.kt:169](<C:/Users/Dosjan/Desktop/manga apk/manga_apk/app/src/main/java/com/screentranslator/app/ui/pdf/PdfTranslationScreen.kt:169>).

**D. Sozlamadagi tarjimon tanlovi PDFga ta’sir qilmaydi.**

API oynasi MyMemory, DeepL va LibreTranslate tanlash imkonini beradi. Ammo PDF so‘rovida engine tanlovi va shu xizmatlar kalitlari yuborilmaydi. PDF backend tarjima funksiyasi Google yo‘lidan, xatoda `deep-translator` zaxirasidan foydalanadi. Foydalanuvchi DeepL tanlab, PDF ham DeepL orqali ishlanadi deb o‘ylashi mumkin; bu natija va maxfiylik bo‘yicha noto‘g‘ri kutish yaratadi.

Tuzatish: PDF uchun provayderni aniq ko‘rsatish yoki tanlovni servergacha ulash. Manbalar: [PdfTranslationClient.kt:44](<C:/Users/Dosjan/Desktop/manga apk/manga_apk/app/src/main/java/com/screentranslator/app/data/network/PdfTranslationClient.kt:44>), [main.py:687](<C:/Users/Dosjan/Desktop/manga apk/manga_apk/server/main.py:687>).

**E. ML komiks yo‘lida “Avtomatik” til yaponchaga majburlanadi.**

Asosiy ML yo‘li muvaffaqiyatli ishlasa, `source_lang=auto` uchun `detected_source_lang` to‘g‘ridan-to‘g‘ri `ja` qilinadi. Lokal nazorat misolida inglizcha OCR matni qaytarilganida ham natija `ja` bo‘ldi. Bu til keyingi tarjimaga va davom ettiriladigan sahifalarga ta’sir qiladi. Shu bilan birga pipeline’dagi EasyOCR qat’iy `['ja', 'en']` bilan yaratiladi; foydalanuvchining koreys/xitoy/rus tanlovi ushbu OCR obyektini almashtirmaydi.

Tuzatish: manba tilini pipeline’ga uzatish, tilga mos OCR tanlash va avtomatik holatda matndan tilni aniqlash yoki tarjima provayderiga `auto` yuborish. Manbalar: [main.py:514](<C:/Users/Dosjan/Desktop/manga apk/manga_apk/server/main.py:514>), [manga_pipeline.py:359](<C:/Users/Dosjan/Desktop/manga apk/mashinali oqitish/manga_pipeline.py:359>).

**F. Uzun tarjima eksport PDFdan jim yo‘qolishi mumkin.**

Komiks transkriptida matn avval kichik, keyin kattaroq katakka joylashtiriladi. Ikkinchi urinishning muvaffaqiyati tekshirilmaydi. Lokal tekshiruvda 4 900 belgili tarjima kiritilganda eksportda sarlavha bor, tarjima paragrafi esa yo‘q bo‘ldi. Bu hajm tahrirlash API’sining 5 000 belgili chegarasiga sig‘adi.

Tuzatish: matnni qator va sahifalarga bo‘lish, barcha joylashtirish natijalarini tekshirish, sig‘magan matnni yo‘qotish o‘rniga yangi sahifaga chiqarish. Manba: [pdf_jobs.py:427](<C:/Users/Dosjan/Desktop/manga apk/manga_apk/server/pdf_jobs.py:427>).

**G. Docker’dagi server asosiy ML papkasini topa olmaydi.**

Server `mashinali oqitish`ni o‘z papkasidan ikki daraja yuqoridagi qo‘shni katalogdan yuklaydi. Dokumentatsiyadagi Docker/cloud build esa faqat server papkasini kontekst sifatida oladi. Natijada ML kodi, vocabulary va checkpointlar konteynerga kirmaydi; komiks so‘rovi zaxira OCR yo‘liga o‘tadi. Lokal muhitdagi natija bilan joylashtirilgan muhitdagi natija bir xil bo‘lishi kafolatlanmagan.

Tuzatish: ML qismini o‘rnatiladigan modulga ajratish yoki build kontekstini to‘g‘ri tuzish; faol OCR/pipeline turini health javobi va foydalanuvchi holatida ko‘rsatish. Manbalar: [main.py:68](<C:/Users/Dosjan/Desktop/manga apk/manga_apk/server/main.py:68>), [Dockerfile](<C:/Users/Dosjan/Desktop/manga apk/manga_apk/server/Dockerfile>).

**H. OCR o‘qiy olmagan matn ham rasmdan o‘chirilishi mumkin — yuqori ustuvorlik.**

ML pipeline bo‘sh transcription uchun ham dialog hududini ro‘yxatga qo‘shib, matnni o‘chiruvchi maska yaratadi. Keyin bu hudud inpainting orqali tozalanadi. Server esa bo‘sh transcription’ni tarjima bloklariga qo‘shmaydi. PDF eksporti tozalangan rasmni ishlatsa, asl matn o‘chgan, o‘rniga tarjima yozilmagan joy qolishi mumkin. Bu koddagi ketma-ketlik bilan tasdiqlangan shartli nuqson; ushbu auditda haqiqiy sahifa ustida alohida vizual tajriba bajarilmadi.

Tuzatish: matn ishonchli o‘qilib, tarjima va joylashtirish muvaffaqiyatli bo‘lganidan keyingina asl hududni almashtirish; qolgan hududlarda asl rasmni saqlash. Manbalar: [manga_pipeline.py:653](<C:/Users/Dosjan/Desktop/manga apk/mashinali oqitish/manga_pipeline.py:653>), [main.py:497](<C:/Users/Dosjan/Desktop/manga apk/manga_apk/server/main.py:497>), [pdf_jobs.py:460](<C:/Users/Dosjan/Desktop/manga apk/manga_apk/server/pdf_jobs.py:460>).

**I. Tarjima zaxirasining cheksiz kutishi butun PDF navbatini to‘xtatishi mumkin.**

Birinchi Google so‘rovida 8 soniyali timeout bor. Ammo o‘rnatilgan `deep-translator` zaxirasi ichidagi `requests.get()` chaqiruvida timeout berilmagan. Tarmoq javob bermay qolsa, bitta PDF worker band bo‘lib qoladi va keyingi ishlar boshlanmaydi. Bu kod va o‘rnatilgan kutubxonadan aniqlangan; cheksiz kutadigan tarmoq tajribasi ataylab bajarilmadi.

Tuzatish: barcha tashqi chaqiruvlarga timeout, umumiy vazifa muddati va cheklangan retry qo‘shish. Manbalar: [main.py:716](<C:/Users/Dosjan/Desktop/manga apk/manga_apk/server/main.py:716>), [google.py:67](<C:/Users/Dosjan/Desktop/manga apk/manga_apk/server/venv/Lib/site-packages/deep_translator/google.py:67>), [pdf_jobs.py:678](<C:/Users/Dosjan/Desktop/manga apk/manga_apk/server/pdf_jobs.py:678>).

**J. Server almashtirilganda eski vazifa tokeni yangi serverga yuboriladi.**

Saqlanadigan vazifa havolasida faqat ID va token bor, server manzili yo‘q. Foydalanuvchi API sozlamasida serverni almashtirganda tiklash joriy manzilga eski vazifa tokenini yuboradi. Vazifa topilmaydi; boshqa serverga tegishli maxfiy token ham yangi endpointga uzatiladi. Vazifani yaratgan server manzili reference bilan saqlanishi kerak. Manba: [PdfJobReference.kt:3](<C:/Users/Dosjan/Desktop/manga apk/manga_apk/app/src/main/java/com/screentranslator/app/data/model/PdfJobReference.kt:3>).

Kichikroq interfeys kamchiligi: PDFning foydalanuvchiga ko‘rinadigan asl nomi o‘rniga upload’da vaqtinchalik fayl nomi yuboriladi. Asl `displayName` multipart filename sifatida alohida uzatilishi kerak.

## 5. Ma’lumotlar xavfsizligi va saqlash

PDF to‘liq serverga yuboriladi. Ajratilgan matn keyin tashqi tarjima xizmatiga ketadi. Server README bu oqimni tushuntiradi, ammo ilova ekranidagi “server orqali tarjima” izohi foydalanuvchiga aniq provayderni bildirmaydi.

Manzilga `https://` yozilmasa, PDF klient `http://`ni avtomatik qo‘shadi. Manifest cleartext trafikni butun ilova uchun ruxsat etadi. HTTP tanlanganida hujjat, API kaliti va vazifa tokeni tarmoqda shifrlanmaydi. HTTPS’ni standart qilish, HTTP’ni faqat aniq mahalliy sinov holatida ruxsat etish kerak. Manbalar: [PdfTranslationClient.kt:29](<C:/Users/Dosjan/Desktop/manga apk/manga_apk/app/src/main/java/com/screentranslator/app/data/network/PdfTranslationClient.kt:29>), [AndroidManifest.xml:32](<C:/Users/Dosjan/Desktop/manga apk/manga_apk/app/src/main/AndroidManifest.xml:32>).

Server API kaliti muhitda berilsa tekshiriladi; berilmasa server ogohlantirish bilan autentifikatsiyasiz ishlaydi. Bu lokal sinovga qulay, lekin ommaviy xizmat uchun standart holat sifatida mos emas. Har bir PDFga alohida tasodifiy token berilishi, serverda faqat uning SHA-256 xeshi saqlanishi va doimiy vaqtli solishtirish ishlatilishi yaxshi yechim.

Android PDF tokenini Keystore kaliti bilan AES-GCM orqali shifrlaydi. Boshqa API kalitlari ilovaning xususiy DataStore faylida oddiy qiymat sifatida saqlanadi; bu joy boshqa oddiy ilovalardan sandbox bilan ajratilgan, lekin token vault kabi alohida shifrlanmagan. DataStore backup’dan chiqarilgan.

Eski ekran tarixi rasmlari `files/history_images`ga yoziladi, backup qoidalarida esa faqat `datastore` chiqarilgan. Bunday rasmlar mavjud bo‘lsa, Android backup qoidalariga ko‘ra zaxiraga kirishi mumkin. Hozirgi UI ekran tarjimasini ishga tushirmagani sabab bu eski/dormant funksiyaga tegishli. Manbalar: [TranslationHistoryRepository.kt:42](<C:/Users/Dosjan/Desktop/manga apk/manga_apk/app/src/main/java/com/screentranslator/app/data/preferences/TranslationHistoryRepository.kt:42>), [Android Auto Backup](https://developer.android.com/identity/data/autobackup?hl=en).

Server tugagan, xato yoki bekor qilingan vazifalarni standart 48 soatdan keyin startup yoki keyingi so‘rovlar vaqtida tozalaydi. Bu aniq 48 soatda ishlaydigan mustaqil taymer emas. Lokal fayllar va bitta background worker prototip uchun sodda, ammo ko‘p foydalanuvchili xizmat uchun doimiy saqlash, navbat chegarasi va foydalanuvchi bo‘yicha kvotalar kerak.

Cloud Run bo‘yicha yana bir shart: joriy PDF worker javob qaytarilgandan keyin fonda ishlaydi. Request-based billing holatida bunday davrda CPU o‘chirilishi yoki keskin cheklanishi mumkin. Shuning uchun mavjud joylashtirish namunasi uzoq PDF vazifalari uchun yetarli deb qabul qilinmasligi kerak; ishchi jarayon va saqlash arxitekturasi alohida hal qilinadi. Bu kod bilan rasmiy platforma xulqini solishtirishdan chiqarilgan xulosa; jonli cloud xizmat tekshirilmadi. [Google Cloud tavsiyasi](https://docs.cloud.google.com/run/docs/tips/general).

## 6. AI va datasetlar

Loyihada pufak aniqlovchi model, OCR modeli va fonni tiklovchi inpainting modeli mavjud. OCR modelida alohida belgi va ketma-ket matn bilan ishlash yo‘llari bor; inpainting U-Net uslubida. EasyOCR natijalari va maxsus modellar birgalikda ishlatiladi. Bu modullar mavjudligi tayyor mahsulotdagi aniqlik foizini o‘z-o‘zidan tasdiqlamaydi.

Joriy metadata tekshiruvidagi sonlar:

- Detection: 6 515 yozuv; train 5 212, validation 651, test 652.
- OCR: 2 299 crop; train 1 839, validation 229, test 231. Shundan 671 tasi haqiqiy sahifadan, 1 628 tasi sintetik.
- Inpainting: 1 700 yozuv; train 1 360, validation 170, test 170.

Belgilar arxivi metadata bo‘yicha yana 210 227 rasm va 62 sinfga ega; taqsimot 168 186 / 21 065 / 20 976. Bu arxiv to‘liq qayta dekodlanmagan. Feedback’da 4 938 OCR namunasidan 466 tasi, 33 detector namunasidan 32 tasi va 355 inpainting namunasidan 349 tasi `unverified` holatida. Bu belgi namuna albatta xato ekanini bildirmaydi, lekin inson tasdig‘i bilan to‘liq ajratilmaganligini ko‘rsatadi.

Detection ma’lumotlari uchun kutilgan haqiqiy rasmlar mavjud bo‘lmaganda loader annotation koordinatalari asosida sintetik ellipslar chizadi. Tekshirilgan detection to‘plami shu yo‘ldan foydalanadi. Bu dastur ishlashini sinashga yordam beradi, lekin sintetik validation natijasini haqiqiy manga sahifalaridagi aniqlik sifatida talqin qilib bo‘lmaydi. Manba: [dataset_loader.py:196](<C:/Users/Dosjan/Desktop/manga apk/mashinali oqitish/dataset_loader.py:196>).

**Qayta o‘qitishda sifat pasayishini to‘suvchi tekshiruv yo‘q.** OCR fine-tuning belgi klassifikatsiyasi bo‘yicha o‘qitadi, umumiy model parametrlarini optimizatsiya qiladi va oxirida ishlatilayotgan `best_manga_ocr.pt`ni almashtiradi. Oldingi nusxa saqlanadi, ammo yangi checkpointni alohida validation/test natijasi yaxshilangandagina qabul qiladigan shart yo‘q. Belgilarni yaxshiroq ajratish butun satr OCRini ham yaxshilaganini isbotlamaydi. Bu xavf koddan kelib chiqadi; ushbu tahlilda qayta o‘qitish bajarilmadi va model sifati pasaydi degan tajriba xulosasi chiqarilmadi. Manba: [self_learning_trainer.py:401](<C:/Users/Dosjan/Desktop/manga apk/mashinali oqitish/self_learning_trainer.py:401>).

Keyingi sifat bahosi uchun tillar bo‘yicha alohida haqiqiy sahifalar kerak: OCR uchun CER/WER, pufaklarni topish uchun precision/recall, tayyor PDF uchun matn yo‘qolmasligi va joylashuv tekshiruvi. Datasetni manga nomi yoki sahifa bo‘yicha ajratish bir sahifadan kelgan crop’larning train va testga aralashib ketishini kamaytiradi. Ushbu audit yangi to‘liq benchmark bermaydi.

Modelning o‘zi yaratgan tozalangan rasmlar keyingi o‘qitish siklida `ground_truth` sifatida ishlatilishi mumkin; alohida tekshiruv bo‘lmasa, oldingi xatolar qayta o‘rganiladi. Inpainting va detector checkpointlari ham o‘xshash tarzda validation shartisiz almashtiriladi. Qo‘shimcha sifat cheklovi: OCR loader yorliqni 24 belgigacha kesadi, mavjud to‘liq crop esa qoladi; joriy 15 ta yozuv shu chegaradan uzun. Kichik inference urinishining yakuniy natijasi olinmagan, shuning uchun yangi aniqlik yoki CER raqami keltirilmadi.

## 7. Kuchli tomonlar

- Android tarmoq va fayl amallari IO kontekstiga chiqarilgan; Compose va StateFlow asosidagi holat boshqaruvi bor.
- PDF yuklashda hajm va `%PDF-` sarlavha tekshiriladi; vaqtinchalik nusxa yuborish yakunida tozalanadi.
- Vazifa havolasi va shifrlangan token ilova qayta ochilganda tiklanadi.
- Server vazifa yo‘llarini UUID bilan tekshiradi, oraliq sahifa natijalarini saqlaydi va tugagan sahifalarni davom ettirishda qayta ishlatadi.
- OCR/rasm hajmi cheklovlari, job token, API kaliti va retention mexanizmlari mavjud.
- OCR va tarjimani qo‘lda tahrirlash imkoniyati noto‘g‘ri tanilgan matnni tuzatish uchun foydali.
- Release konfiguratsiyasida minify/resource shrink yoqilgan; signing ma’lumotlari tashqi sozlamalardan olinadi. Bu tekshirilgan debug APKning release bo‘lib qolganini anglatmaydi.
- Server va ML qismlarida ishlaydigan avtomatik testlar bor.

## 8. Tekshiruv natijalari

- APK ZIP/CRC: o‘tdi.
- APK v2 imzosi: o‘tdi, Android Debug sertifikati.
- APK va mavjud debug build xeshlari: teng.
- `assembleDebug`: muvaffaqiyatli, mavjud yig‘ilma dolzarb.
- `lintDebug`: muvaffaqiyatli; **0 xato, 29 ogohlantirish**, XMLda yana 1 informational yozuv.
- `testDebugUnitTest`: **NO-SOURCE**. Android unit testlari mavjud emas; buni “testlar o‘tdi” deb hisoblash mumkin emas.
- Server: **18/18 test o‘tdi**.
- ML: **20/20 test o‘tdi**.
- Lokal tekshiruvlar: `auto → ja` xatosi va 4 900 belgili transkript matnining eksportdan yo‘qolishi qayta hosil qilindi.

Lint ogohlantirishlari asosan dependency/target yangiligi, ishlatilmagan resurslar, ortiqcha SDK tekshiruvlari va ikonka tafsilotlari bilan bog‘liq. StaticFieldLeak ogohlantirishidagi singleton `applicationContext` bilan yaratilgan; uni qo‘shimcha dalilsiz tasdiqlangan Activity memory leak deb hisoblamadim. Dependency yangilanishi mavjudligi ham o‘zi bilan xavfsizlik zaifligi isboti emas.

Server testlaridagi muhim cheklov: tekshirilgan server virtual muhitida `torchvision` topilmadi. Uchta komiks testi asosiy ML importi muvaffaqiyatsiz bo‘lib, testdagi zaxira OCR javoblari bilan o‘tdi. Demak, 18/18 natija asosiy ML integratsiyasini tasdiqlamaydi. Python testlarida tashqi tarjima servisining real sifati ham to‘liq tekshirilmagan.

Birinchi offline lint urinishida cache’da bitta dependency yetishmagani sabab tekshiruv to‘xtadi. Dependency olingach, yakuniy yig‘ish va lint muvaffaqiyatli bajarildi. Saqlangan dalillar: [Gradle log](<C:/Users/Dosjan/Desktop/manga apk/analysis-gradle.log>), [lint hisoboti](<C:/Users/Dosjan/Desktop/manga apk/manga_apk/app/build/reports/lint-results-debug.html>).

## 9. Amaliy tuzatish tartibi

1. PDF ma’lumotlarini Docker/cloud build’dan chiqarish; HTTPS va ommaviy server autentifikatsiyasini qat’iy belgilash.
2. Xato/bekor qilingan vazifadan chiqish va tarmoq uzilgach progressni tiklashni tuzatish.
3. PDFda haqiqatan qaysi tarjimon ishlayotganini interfeys va serverda bir xil qilish.
4. Komiks manba tili, uzun matnni sahifalash va ML Docker integratsiyasini tuzatish.
5. Android’da shu holatlar uchun testlar; serverda real asosiy ML yo‘lini qamraydigan integratsiya testi qo‘shish.
6. Haqiqiy datasetlarda OCR/detection sifatini o‘lchash; yangi checkpoint faqat oldingi modeldan yomon bo‘lmasa qabul qilinadigan tartibni o‘rnatish.
7. Mahsulot PDF tarjimoni bo‘lib qolsa, ekran tarjimonining ulanmagan UI, ruxsat va dependency qismlarini saralash; aks holda uni ishga tushirish oqimini qayta ulash.
8. Haqiqiy qurilmada kichik/katta PDF, uzilgan internet, ilovani qayta ochish, jarayonni bekor qilish, uzun tahrir, turli manba tillari va eksport PDFni ko‘rib chiqish sinovlarini bajarish; so‘ng release yig‘ilmasini tekshirish.

Tahlil davomida source kod tuzatilmagan, model o‘qitilmagan va hech qanday server joylashtirilmagan. Hisobot topilgan holatni va keyingi ish tartibini beradi.
