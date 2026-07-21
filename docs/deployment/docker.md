# Docker ile Sunucuya Kurulum (Deployment)

Bu rehber, RL Trading dashboard + tahmin servisini tek bir VPS/sunucuda
Docker Compose ile yayınlamayı anlatır.

**Senaryo:** Modelleri kendi bilgisayarında eğitiyorsun, sunucuda yalnızca
serving (dashboard + tahmin + günlük karar) çalışıyor. GPU yok, torch CPU-only.

---

## 1. Mimari Özet

- **Tek container** (`rltrading`): FastAPI + Dash aynı process'te, `/dash/`
  altında mount. Tek port (8000).
- **Kalıcı veri (volume):** `models/`, `data/`, `results/`, `logs/` host'ta
  tutulur ve container'a bind-mount edilir. Container silinse bile veri kalır.
- **Gizli bilgiler:** `EVDS_API_KEY` gibi değerler `.env` üzerinden runtime'da
  geçirilir; **image'a gömülmez**.
- **Self-healing:** `restart: unless-stopped` — çökerse veya sunucu yeniden
  başlarsa container otomatik kalkar.

---

## 2. Sunucu Ön Hazırlığı

Docker + Compose plugin kurulumu (Ubuntu/Debian):

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER    # oturumu kapat/aç (docker'ı sudo'suz kullanmak için)
docker --version
docker compose version
```

Uygulama için bir dizin oluştur:

```bash
sudo mkdir -p /opt/rltrading
sudo chown $USER:$USER /opt/rltrading
cd /opt/rltrading
```

---

## 3. Kaynak Kodu Sunucuya Al

Git ile (önerilen):

```bash
git clone https://github.com/softdevcan/RL-Trading-Project.git .
```

---

## 4. Eğitilmiş Modelleri Sunucuya Kopyala

Modeller image'a gömülmez; host'taki `models/` klasörüne kopyalanır ve
volume ile container'a bağlanır. Kendi bilgisayarından:

```bash
# RL modelleri (.zip) + prediction ensemble modelleri
scp -r ./models/*        user@SUNUCU_IP:/opt/rltrading/models/
# (opsiyonel) önceden çekilmiş veri / tahmin geçmişi
scp -r ./data/*          user@SUNUCU_IP:/opt/rltrading/data/
```

> `models/prediction/` altındaki tüm dosyalar (xgboost/lightgbm/catboost/
> bilstm/tft + ensemble meta) birlikte kopyalanmalı; ensemble bunların
> hepsini yükler.

---

## 5. Ortam Değişkenleri (.env)

```bash
cp .env.production.example .env
nano .env
```

Doldurulması gerekenler:

- `EVDS_API_KEY` — TCMB EVDS anahtarın (makro veri için).
- `CORS_ORIGINS` — **`*` bırakma.** Dashboard'ı açacağın domain(ler):
  `CORS_ORIGINS=["https://trading.example.com"]`
- `DEBUG=False` — production'da kapalı kalsın.

`.env` dosyası `.gitignore` ve `.dockerignore` içinde; repoya veya image'a
girmez.

---

## 6. Başlat

```bash
docker compose up -d --build
```

İlk build birkaç dakika sürer (torch + bilimsel paketler). Sonrasında:

```bash
docker compose ps          # durum
docker compose logs -f     # loglar
curl http://localhost:8000/health   # {"status":"healthy",...}
```

Dashboard: `http://SUNUCU_IP:8000/dash/`

`make` kuruluysa kısayollar: `make up`, `make logs`, `make health`,
`make down`, `make shell`.

---

## 7. Güncelleme (yeni kod / yeni model)

Kod güncellemesi:

```bash
git pull
docker compose up -d --build
```

Sadece yeni model eklediyseniz (kod değişmediyse): modelleri `models/`'e
kopyalayın — uygulama model cache'ini dosya `mtime`'ına göre otomatik
tazeler, **yeniden başlatma gerekmez**. (Bkz. `prediction_service.py` ve
`trading.py` cache mantığı.)

---

## 8. Yedekleme

Kalıcı durum tamamen host'taki bu klasörlerde:

```bash
tar czf rltrading-backup-$(date +%F).tar.gz models data results
```

---

## 9. HTTPS / Domain (dışa açık yayın)

Dashboard'ı internete açacaksan container'ı doğrudan 8000'de yayınlamak
yerine önüne bir reverse proxy (nginx veya Caddy) koy ve TLS'i orada
sonlandır. İki adım:

1. `docker-compose.yml`'de portu iç ağa çek:
   `ports: ["127.0.0.1:8000:8000"]`
2. nginx/Caddy'yi 80/443'te dinlet, `proxy_pass http://127.0.0.1:8000;`
   yap, Let's Encrypt ile sertifika al.

Caddy ile en kısa yol (otomatik HTTPS) — host'ta `/etc/caddy/Caddyfile`:

```
trading.example.com {
    reverse_proxy 127.0.0.1:8000
}
```

> Bu proxy katmanını compose'a ikinci bir servis olarak eklemek istersen
> söyle, hazırlarım.

---

## 10. Sık Karşılaşılan Sorunlar

| Belirti | Neden / Çözüm |
|---|---|
| `/health` 200 dönmüyor, container restart ediyor | Logları oku (`docker compose logs`); genelde eksik `.env` veya import hatası. |
| Dashboard boş / "backend ulaşılamıyor" | Normalde tek container olduğu için olmaz; `docker compose logs` bak. |
| Tahmin "eğitilmiş model yok" diyor | `models/prediction/` sunucuya kopyalanmamış veya eksik dosya. |
| Makro veri gelmiyor | `EVDS_API_KEY` boş/yanlış. |
| RAM yetmiyor | `mem_limit`'i düşür veya sunucuyu büyüt; serving tek worker ile ~2-4 GB kullanır. |

---

## Neden Kubernetes değil?

Tek makine + tek kullanıcı + tek stateful servis için Kubernetes'in
sağladığı otomatik ölçekleme / rolling update / self-healing değeri,
getirdiği işletme karmaşıklığını karşılamaz. Compose ile self-healing
(`restart`), kolay log/backup ve tek komutla deploy zaten var. İleride
çok sayıda eşzamanlı kullanıcı veya yatay ölçekleme gerekirse, o zaman
K8s'e geçiş değerlendirilebilir (o durumda in-process cache'ler yerine
paylaşımlı bir cache — örn. Redis — gerekir).
