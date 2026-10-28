# PDF tarjima: loyiha rejasi

## Maqsad

ScreenTranslator Android ilovasida PDF kitoblarni, romanlarni, manga/manhwa/manhua va boshqa komikslarni tanlash, sahifalarini to‘liq tarjima qilish, jarayonni kuzatish va natijani yangi PDF sifatida olish imkonini yaratish.

Mahsulot ikki alohida o‘qish usulini qo‘llaydi:

1. **Kitob/roman rejimi** — matnni hujjatdan ajratadi yoki skan bo‘lsa OCR qiladi; tarjima qilingan, qayta oqimlangan PDF yaratadi. Asosiy saqlanadigan tuzilma: boblar, sarlavhalar, paragraflar, ro‘yxatlar va sahifa raqamlari.
2. **Komiks rejimi** — sahifani rasm sifatida ko‘radi, yozuvlar va dialog pufaklarini aniqlaydi, tarjima qiladi va yakunda komiks ko‘rinishidagi PDF chiqaradi. Rasm ichidagi asl yozuvni sifatli o‘chirish va tarjimani mos shrift bilan joylash alohida, eng murakkab bosqichdir.

## Hozirgi loyihadagi boshlang‘ich holat

- Android ilovasida tarjima rejimlari, til tanlash va API sozlamalari mavjud; `MainActivity` asosiy Compose sozlamalar oynasini ochadi.
- Python FastAPI serveri bitta rasmni OCR qilish va butun rasm sahifasini tarjima qilish endpointlariga ega.
- PDF kitob/komiks uchun PyMuPDF asosidagi job API, sahifa keshi, OCR ishlovi va Android fayl tanlash/progress oynasi ulandi.
- Hozir kitob matnli, skan va aralash PDFlarini OCR qilib qayta oqimlangan hujjatga chiqaradi. Komiks rejimi asl betlarni saqlab, konturi ishonchli topilgan och rangli pufak shakliga tarjimani joylaydi; kontur topilmasa yoki tarjima sig‘masa, asl rasmga tegmaydi va tarjimani transkriptga chiqaradi. Ilova qayta ochilganda vazifa DataStore’dagi ID/token bilan qayta olinadi; token Android Keystore yordamida shifrlanadi va yangi vazifada token xotirada darhol ishlatiladi.
- Kitob matnida ikki ustun aniq ko‘rinsa chap ustun tugaguncha, keyin o‘ng ustunga o‘tadi; ko‘p sahifada takrorlangan yuqori/pastki kolontitul va alohida sahifa raqamlarini tarjimadan chiqaradi. Tayyor ishda OCR matni va tarjimani sahifama-sahifa tahrirlash, so‘ng PDFni qayta yaratish mumkin; OCR ishonchi past bloklar ko‘rsatiladi.
- Mavjud OCR va tarjima qismlari qayta ishlatiladi; PDF importi, hujjat modeli, vazifa navbati va eksport alohida qatlamlarda ishlaydi.

## Mahsulot qarorlari

- PDF tanlash Android Storage Access Framework (`OpenDocument`) orqali amalga oshadi; keng xotira ruxsatlarini so‘ramaydi.
- PDF asl nusxasi o‘zgartirilmaydi. Tarjima alohida yangi fayl bo‘lib saqlanadi.
- **Kitoblar:** birinchi ishonchli natija tarjima matni qayta oqimlangan PDF. Asl sahifa dizaynini pikselma-piksel saqlash shart emas; mazmun va bob/paragraf tartibi saqlanadi.
- **Komikslar:** avval tarjima matni va sahifa tartibini ishonchli aniqlash; keyin asl rasmni saqlab, pufak ichiga tarjima joylash. Tozalash mumkin bo‘lmagan effekt matnlari uchun foydalanuvchiga yonma-yon tarjima/transkript ko‘rsatish zaxira usuli bo‘ladi.
- Aralash PDF sahifa-sahifa tekshiriladi: selectable text bor sahifa matn ekstraksiyasidan, rasmga o‘xshash sahifa OCRdan o‘tadi.
- Uzoq ish bitta HTTP ulanishiga bog‘lanib qolmaydi: server vazifa identifikatori va progress beradi, natijani keyin yuklab olish mumkin bo‘ladi.
- Kitob tarjimasida atamalar va ism-shariflar barqaror chiqishi uchun keyinchalik foydalanuvchi glossariysi va tahrirlash qo‘shiladi.

## Bosqichlar

### 0-bosqich — talab va texnik poydevor

- [x] Mavjud Android, server va tarjima oqimini tekshirish.
- [x] PDF qo‘llab-quvvatlashi uchun bosqichlar, cheklovlar va yakun mezonlarini yozish.
- [x] PDF, OCR, tarjima va eksport uchun alohida modullar chegarasini belgilash.
- [x] PDF hajmi/sahifa soni, job papkasi va natija saqlash muddatini sozlanadigan qilish.

**Chiqish mezoni:** reja va API/UI chegaralari yozilgan; hozirgi rasm tarjima oqimi buzilmagan.

### 1-bosqich — oddiy (matni tanlanadigan) kitob va roman PDFlari

**Android**

- Bosh sahifaga aniq **“PDF tarjima”** amali qo‘shish.
- `OpenDocument` bilan `.pdf` tanlash; fayl nomi, hajmi, bet soni va manba tilini ko‘rsatish.
- Maqsad tilini mavjud til sozlamasidan olish; PDF tarjimasiga kerak bo‘lsa alohida manba-til/avtomatik aniqlash tanlovi.
- Vazifa progressi, sahifa sonog‘i, xato bo‘lsa qayta urinish va tarjima PDFni Android Downloads/Documents ichida saqlash.

**Python server**

- PyMuPDF bilan xavfsiz ochish, metadata/sahifa sonini olish va sahifadan selectable textni ajratish.
- Matn bloklarini o‘qish tartibida saralash; sodda ikki ustunli tartib va takroriy header/footer/sahifa raqami filtri bor. Murakkab layout, footnote va URLni ajratish hali kengaytiriladi.
- Juda uzun matnni jumla/paragraf chegarasida bo‘laklash; API cheklovlariga mos tarjima, takror matnlar uchun cache, retries va rate limit.
- Tarjimani bob/sarlavha/paragraf tuzilishini saqlaydigan PDFga qayta oqimlash; kerakli Unicode shriftlarni ichiga joylash.
- Ish davomida xom PDFni va maxfiy kalitlarni logga yozmaslik.

**Chiqish mezoni:** selectable-text PDF tanlanadi; tarjima qilingan PDF yuklanadi, ochiladi va paragraf tartibi/maqsad tilidagi belgilar to‘g‘ri saqlanadi.

### 2-bosqich — skan qilingan va aralash kitob PDFlari

- Har sahifaning selectable text miqdorini tekshirish; bo‘sh yoki juda kam matnli sahifani skan sifatida belgilash.
- Sahifani sifatli bitmapga render qilish; OCR uchun til/skript modelini tanlash.
- OCR natijasini satr emas, paragraf/blok qilib guruhlash; ustunlar, ro‘yxatlar va footnote tartibini saqlash.
- OCR ishonchi past bo‘lgan betlarni natijada belgilash; butun hujjatni to‘xtatmasdan foydalanuvchiga ko‘rsatish.
- Matnli va skan sahifalarni bir natijaga birlashtirish.

**Chiqish mezoni:** matnli, skan va aralash sahifali kitob bitta vazifada tugaydi; OCR topilmagan sahifalar raqami ko‘rsatiladi.

### 3-bosqich — manga, manhwa, manhua va komiks PDFlari

**Bajarilgan dastlabki qism:** Android’da Kitob/Komiks rejimi; PDF sahifasini asl vektor/rasm san’ati bilan nusxalash; mavjud PaddleOCR, yaponcha MangaOCR va tarjima servisiga ulash; OpenCV yordamida pufak ichidagi oq hududni topib, ishonchli kontur bo‘ylab yozuvni yopish va shriftni sig‘dirish; kontur ishonchsiz yoki tarjima sig‘maganida rasmga tegmasdan transkriptga o‘tish; ikki tilli transkript beti; OCR va tarjima matni uchun sahifama-sahifa tahrir hamda qayta eksport. Murakkab rangli/ochiq pufak, effekt yozuvi, pufakni sahifa previewida qo‘lda tuzatish hali kerak.

- O‘qish tartibini aniqlash (o‘ngdan-chapga manga, chapdan-o‘ngga komiks/manhwa), sahifa paneli va bubble detection.
- Mavjud PaddleOCR/MangaOCR oqimini PDF sahifalariga ulash; tilni sahifa yoki hujjat bo‘yicha aniqlash.
- Dialog, narratsiya, SFX, sarlavha va fon yozuvini tur sifatida ajratish; tarjimada nima ko‘rsatilishini sozlash.
- OCR natijasi, tarjima, koordinata va ishonchni bir sahifa modeli ichida saqlash; pufaklar orasidagi o‘qish tartibini saqlash.
- Asl yozuvni bubble mask/inpainting orqali tozalab, tarjimani shrift o‘lchami va satrga moslab chizish.
- Matnni tozalash xavfli yoki sifatsiz bo‘lsa, asl rasmni saqlab, sahifa ostida tarjima transkriptini berish.
- Matnli tahrir oynasida OCR natijasi va tarjimani qo‘lda tuzatish, saqlab PDFni qayta yaratish mumkin; sahifa rasmi va pufak koordinatasini previewda ko‘rsatish keyingi qism.

**Chiqish mezoni:** bir nechta sahifali komiksda panellar tartibi saqlanadi, tarjima pufaklardan tashqariga chiqmaydi, SFX tarjimasini yoqib/o‘chirish mumkin, asl rasm buzilganda transkript varianti bor.

### 4-bosqich — katta hujjat va vazifani davom ettirish

- Serverda vazifalarni navbatga qo‘yish; bir nechta foydalanuvchi/katta fayl RAMni tugatib qo‘ymasligi. Hozircha bitta worker bor, production navbati va concurrency himoyasi hali kerak.
- `queued / processing / completed / failed / canceled` holatlari, foiz va hozirgi betni qaytarish.
- Vazifa holatini va oraliq bet natijalarini job papkasidagi JSON keshi orqali saqlash; lokal server qayta ishga tushsa navbatga qaytadi. Production uchun SQLite va Cloud Run/GCS kabi barqaror obyekt saqlash kerak.
- To‘xtatish, qayta davom ettirish, failed betni qayta ishlash va muddati o‘tgan fayllarni avtomatik tozalash.
- Fayl baytlari, bet soni, render DPI, parallel ishlash va disk kvotasiga qat’iy limitlar.

**Chiqish mezoni:** server yoki ilova yopilib qayta ochilganda tugallanmagan katta ishni topish/ davom ettirish mumkin; noto‘g‘ri PDF serverni yiqitmaydi.

### 5-bosqich — tarjima sifati va qulaylik

- Atamalar glossariysi: qahramon ismlari, joy nomlari, magic/system terminlari va foydalanuvchi afzalligi.
- Kitobda gap kontekstini paragraf/bob bo‘yicha tarjimaga berish; kontekst limitidan oshmaslik.
- OCR matni va tarjimani sahifama-sahifa ko‘rib tahrirlash; tahrirni saqlaganda natija PDFni qayta yaratish (dastlabki UI/API kesimi kiritildi).
- Ikki tilli PDF, faqat tarjima PDF, komiks ustiga yozilgan PDF va komiks+transkript variantlari.
- Sahifa keshi, tarjima cache, vaqtinchalik faylni saqlash muddati va tarmoq uzilgandagi davom ettirish.

**Chiqish mezoni:** bitta loyiha ichidagi atamalar bir xil tarjima qilinadi; foydalanuvchi OCR/tarjima xatosini tuzatib, qayta eksport qila oladi.

### 6-bosqich — nashrga tayyorlash

- Kichik real sinov to‘plami: selectable text roman, skan roman, aralash PDF, manga, manhwa, ko‘p ustunli sahifa, vertikal yozuv va OCR xatosi ko‘p sahifa.
- Turli Android versiyalari, katta PDF, tarmoq uzilishi, noto‘g‘ri PDF, parolli PDF va server qayta ishga tushishi holatlarini qamrash.
- Maxfiylik matni: qaysi fayl qurilmada va qaysi serverda qayta ishlanishi, tashqi tarjima provayderiga nimalar yuborilishi.
- Xarajat/kvota, server konfiguratsiyasi, APK signing va foydalanuvchiga o‘rnatish qo‘llanmasi.
- PDF kutubxonasi va shriftlarning tarqatish litsenziyasini release oldidan huquqiy/mahsulot talablari bilan solishtirish. PyMuPDF AGPL yoki tijoriy litsenziya bilan tarqatiladi; yopiq kodli tarqatish uchun litsenziya yo‘lini oldindan aniqlash kerak ([rasmiy litsenziya sahifasi](https://pymupdf.readthedocs.io/en/latest/about.html)).

**Chiqish mezoni:** qo‘llab-quvvatlanadigan PDF turlari aniq e’lon qilingan; noaniq/unsupported holatlar tushunarli xabar bilan tugaydi; release rejasi va maxfiylik xabari tayyor.

## Dastlabki texnik tuzilma

```text
Android PDF tanlash
       │ multipart upload + fayl metama’lumoti
       ▼
FastAPI PDF job API ── job_id / progress / cancel / download
       │
       ├── PDF tekshirish va sahifa turi (text / scan / comic) aniqlash
       ├── Matn ekstraksiyasi yoki render → OCR
       ├── Matnni kontekstli bo‘laklash → tarjima provider adapteri
       ├── Kitob uchun qayta oqimlash / komiks uchun bubble overlay
       └── Natijani PDF qilib saqlash → Androidga yuklash
```

## Xavflar va javoblar

- **Tarjima servisining limit/xarajati:** sahifalarni bo‘laklash, kvota va taxminiy hajmni oldindan ko‘rsatish, provider adapterlari va kesh.
- **Skan OCR sifati:** xom OCRni foydalanuvchiga ko‘rsatish, ishonch past sahifalarni belgilash, sahifani qayta ishlash imkoniyati.
- **Komiksda asl chizma buzilishi:** pufakdan tashqarida o‘chirish qilmaslik; xavfsiz bo‘lmasa transliteratsiya/transkript rejimiga o‘tish.
- **Katta PDF serverni band qilishi:** worker/navbat, sahifa/fayl limitlari, RAM va disk nazorati, job progress va cancel.
- **PDF parser xavfsizligi:** vaqtinchalik papkada ishlash, MIME kengaytmasini tekshirish, parolli/malformed PDFni aniq rad etish, shell orqali fayl nomini ishlatmaslik.
- **Shrift va o‘zbekcha belgi:** output PDFga Unicode font joylash; lotin/kirill va o‘ngdan-chapga tilni alohida ko‘rish.

## Hozir boshlangan ish

1-bosqichning dastlabki kesimi (ikki ustun va takroriy kolontitul filtri bilan), 2-bosqich OCR yo‘li va 3-bosqich komiks OCR/maskalash/transkript oqimi kiritildi; Android vazifa tokenini Keystore bilan shifrlab DataStore’da saqlaydi, yangi job tokenini xotirada darhol ishlatadi va qayta ochilganda holatni yangilaydi. Synthetic PDF bilan 14 server regressiya sinovi o‘tdi. Android debug APK yig‘ildi (`assembleDebug`). Uch paragrafli skan-kitob namunasi haqiqiy PaddleOCR va Google tarjima yo‘li bilan oxirigacha ishlab, ikki tilli PDFga chiqarildi; manga namunasi ham ko‘zdan kechirildi. OCR va tarjima sifati hali ayrim sahifalarda xato qilishi mumkin, ammo tahrirlash oynasi shu bloklarni tuzatadi. Keyingi ustuvor ishlar: tahrir oynasini ko‘p bet va yirik hujjatlarda sinash, so‘zlashuv konteksti va glossariy, tarjima sifatini yaxshilash, sahifa preview/pufak tahriri, rangli/ochiq pufaklar hamda panel tartibi, Android fon rejimida kuzatuv, production uchun doimiy job ombori, ko‘proq real roman/komiks sinovlari, worker kvota/qayta urinish va foydalanuvchiga tushunarli xatolar.
