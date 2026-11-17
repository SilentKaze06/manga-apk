# Manga & Manhwa Ultra-Fast Go Server (v2.0.0)

Ushbu server Oracle Cloud va zaif (1GB RAM) VPS serverlarida **Time-out** va **Xotira to'lib qolishi (OOM)** muammolarini butunlay yo'qotish uchun maxsus yaratildi.

---

## ⚡ Asosiy Afzalliklari:
1. **Operativ xotira (RAM) sarfi**: Bor-yo'g'i **~15-25 MB** (PyTorch'dagi 2.5 GB o'rniga!).
2. **CPU zo'riqishi**: 0.0% kutish rejimida, tezkor parallel goroutine'lar.
3. **Kompilyatsiya qilingan binary**: Serverda Python, Pip yoki og'ir kutubxonalarni o'rnatish **umuman shart emas**.
4. **Android APK bilan 100% mos**: Barcha API endpointlar (`/translate-manga`, `/detect-and-translate`, `/health`, `/translate-batch`) aynan bir xil.

---

## 🚀 Oracle Cloud Serverda Ishga Tushirish:

### 1-Usul: Tayyor Binary orqali (Eng tezkor)
Siz uchun ikkala arxitekturaga tayyor Linux binary kompilyatsiya qilib qo'yildi:
* **`manga-server-linux`** — Oracle AMD / Intel (x86_64) uchun.
* **`manga-server-linux-arm64`** — Oracle Ampere A1 (ARM64) uchun.

Ushbu `server_go` papkasini Oracle serveringizga yuklang (SCP, Git yoki SFTP orqali) va quyidagini bajaring:

```bash
cd server_go
chmod +x run.sh
./run.sh
```

Yoki to'g'ridan-to'g'ri:
```bash
chmod +x manga-server-linux
PORT=8000 ./manga-server-linux
```

---

### 2-Usul: Systemd (Fon rejimida doimiy ishlashi uchun)
Server o'chib-yonishida avtomatik turishi uchun `/etc/systemd/system/manga.service` faylini yarating:

```ini
[Unit]
Description=Manga Translation Go Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/server_go
ExecStart=/root/server_go/manga-server-linux
Restart=always
RestartSec=3
Environment=PORT=8000
Environment=MANGA_BIND_HOST=0.0.0.0
Environment=MANGA_ALLOW_UNAUTHENTICATED=true

[Install]
WantedBy=multi-user.target
```

Faollashtirish:
```bash
sudo systemctl daemon-reload
sudo systemctl enable manga
sudo systemctl start manga
sudo systemctl status manga
```

