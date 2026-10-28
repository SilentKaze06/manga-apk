# Doimiy belgilar bazasi: 0–9, A–Z, a–z

`characters.zip` — OCR uchun kerakli asl rasm bazasi. Uni o'chirmang yoki
vaqtinchalik fayl deb hisoblamang. Rasmlar arxivdan bevosita o'qiladi;
ularni alohida papkalarga chiqarish shart emas.

- Jami: **210 227 ta PNG rasm, 62 ta belgi**.
- Raqamlar: 34 730 ta; katta harflar: 90 298 ta; kichik harflar: 85 199 ta.
- O'qitish: 168 186 ta; validatsiya: 21 065 ta; test: 20 976 ta.
- Manba: `C:\Users\Dosjan\Downloads\archive (1).zip`.
- Loyihaga qaytarilgan sana: 2026-10-21. Manbadagi fayl ham saqlangan.

Arxivning SHA-256 nazorat summasi:
`d655813ab16e95a298195690d61004195d5273e817ed3996934aaa778beac716`.
Sinf va bo'linma sonlari `dataset_info.json` faylida saqlanadi.

## Qanday ishlatiladi

Ilovaning o'qitish panelidagi qayta o'qitish tugmasi ushbu bazani avtomatik
topadi. OCR uchun faqat o'quv bo'linmasi olinadi. Yig'ilgan tuzatishlar ham
mavjud bo'lsa, namunalar ikki manbadan teng ehtimollik bilan tanlanadi.
Shu tariqa katta baza foydalanuvchi tuzatishlarini siqib chiqarmaydi.

Bu doimiy baza `self_learning_data/ocr/feedback_ocr.jsonl` fayliga
qo'shilmaydi va undagi **5 000 ta namuna chekloviga kirmaydi**. Ilovada
ko'rsatiladigan yig'ilgan namunalar hisoblagichi ushbu doimiy bazadan alohida.

Har bir rasmning bo'linmasi arxiv ichidagi nomining SHA-256 qiymati orqali
aniqlanadi: 10 ga bo'lgandagi qoldiq 0–7 bo'lsa o'qitish, 8 bo'lsa
validatsiya, 9 bo'lsa test. Validatsiya va test rasmlari qayta o'qitishga
berilmaydi. Belgilar mavjud `ocr_vocab.json` indekslariga moslanadi;
lug'at va saqlangan model og'irliklari import paytida qayta yozilmaydi.

Ilovadagi qisqa qayta o'qitish sikli har safar butun bazani ko'rib chiqmaydi;
u belgilangan miqdordagi tasodifiy guruhlar bilan ishlaydi. Baza yakka
belgilarni tasniflash yo'liga ulangan. `train_and_evaluate.py --task ocr`
esa avvalgidek satr matnlari uchun alohida JSONL bazasidan foydalanadi.

Bazani qaytarishning o'zi modelni o'qitmaydi. Qayta o'qitish ilovadagi
tugma orqali alohida boshlanadi.
