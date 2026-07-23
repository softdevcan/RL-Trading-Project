# Faz 7 — Kimlik Doğrulama, Yetkilendirme ve Kullanıcı Bazlı Çalışma

Sunucuya açılmadan önce panonun tamamı herkese açıktı: `/dash/*`, `/api/*` ve
`/docs` hiçbir kontrol olmadan servis ediliyordu. Faz 7 bunu kapatır ve sistemi
çok kullanıcılı hale getirir.

**Seçilen model:** hibrit izolasyon + SQLite kullanıcı deposu + yalnızca
yöneticinin hesap açması (self-signup yok).

---

## 1. Mimari

```
Tarayıcı
   │  (HttpOnly çerez: rlt_session / rlt_refresh / rlt_csrf)
   ▼
┌──────────────────────────────────────────────────────────┐
│ CORSMiddleware                     (en dış katman)        │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ AuthGateMiddleware                                    │ │
│ │  · public yol mu?         → geç                       │ │
│ │  · access token geçerli?  → kullanıcıyı ContextVar'a  │ │
│ │  · süresi dolmuş?         → refresh ile SESSİZ yenile │ │
│ │  · kimlik yok?            → HTML ise /login, API 401  │ │
│ │  · /api/* yazma?          → CSRF double-submit        │ │
│ │ ┌──────────────────────────────────────────────────┐ │ │
│ │ │ FastAPI rotaları  +  Dash (WSGI mount /dash)      │ │ │
│ │ └──────────────────────────────────────────────────┘ │ │
│ └──────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

**Neden çerez, neden middleware?**
Dash `WSGIMiddleware` ile `/dash` altına mount edilmiş durumda. Tarayıcının
callback POST'ları `Authorization` başlığı taşıyamaz, dolayısıyla oturum çerez
tabanlı olmak zorunda. Aynı nedenle FastAPI `Depends` zinciri Dash isteklerine
uygulanamaz; kapı ASGI seviyesinde, middleware'de duruyor.

### Dosya haritası

| Dosya | Görev |
|---|---|
| `app/auth/models.py` | `User`, `SessionToken` (refresh kaydı), `AuditLog` |
| `app/auth/db.py` | SQLAlchemy engine, WAL pragma'ları, `init_db()` |
| `app/auth/security.py` | bcrypt hash, JWT üretimi/çözümü, CSRF, parola politikası |
| `app/auth/service.py` | Kullanıcı CRUD, `authenticate()`, oturum rotasyonu, audit |
| `app/auth/deps.py` | `CurrentUser`, `RequireAdmin`, `RequireWriter` |
| `app/auth/middleware.py` | Oturum kapısı + sessiz yenileme + CSRF |
| `app/auth/routes.py` | `/login`, `/change-password`, `/auth/*` |
| `app/auth/workspace.py` | Kullanıcı bazlı dizin çözümleyici |
| `app/auth/bootstrap.py` | İlk admin |
| `app/api/routes/admin.py` | `/api/admin/users`, `/api/admin/audit` |
| `dashboard/pages/users.py` | Kullanıcı yönetimi arayüzü (admin) |
| `scripts/create_admin.py` | Sunucuda elle admin açma/parola sıfırlama |

---

## 2. Oturum akışı

1. `POST /auth/login` — e-posta + parola. Başarılıysa:
   - `rlt_session`: access JWT, HttpOnly, 30 dk (varsayılan)
   - `rlt_refresh`: refresh JWT, HttpOnly; `jti`'si DB'de tutulur → iptal edilebilir
   - `rlt_csrf`: HttpOnly **değil**; istemci `X-CSRF-Token` başlığına kopyalar
2. Access token dolduğunda middleware refresh çerezi ile **sessizce** yeniler;
   kullanıcı form doldururken oturumdan düşmez.
3. Refresh rotasyonludur: her kullanımda eskisi iptal edilir. İptal edilmiş bir
   token yeniden kullanılırsa (30 sn'lik yarış penceresi dışında) token çalınmış
   sayılır ve kullanıcının **tüm** oturumları kapatılır.

> Dash bir sayfada onlarca callback'i paralel atar. 30 saniyelik grace penceresi
> tam da bunun için var: aynı anda yenilemeye giren istekler "çalınmış token"
> olarak işaretlenmez.

**Parola:** bcrypt (cost 12). 72 baytlık bcrypt sınırını aşmamak için parola
önce SHA-256 + base64'e indirgenir. Politika: en az 10 karakter, büyük + küçük
harf + rakam. 5 hatalı denemede hesap 15 dakika kilitlenir; ayrıca IP başına
5 dakikada 15 deneme sınırı vardır.

---

## 3. Roller

| Rol | Yetki |
|---|---|
| `viewer` | Yalnızca okuma. Eğitim, tahmin üretme, karar uygulama, veri güncelleme yok. |
| `user` | Kendi çalışma alanında tam yetki: eğitim, tahmin, günlük karar. |
| `admin` | + kullanıcı yönetimi, denetim kaydı, `/docs` ve `/openapi.json`. |

Kod tarafında: `RequireWriter` (viewer'ı bloklar), `RequireAdmin`. Korunan uçlar:
`/trading/train`, `/trading/data/generate|update`, `/trading/daily-decision`,
`/trading/apply-decision`, `DELETE /trading/models/{name}`, `/prediction/train`,
`/prediction/predict`, `/prediction/evaluate`, `/prediction/train-ensemble`,
`/prediction/cross-validate`, `/prediction/optimize`, tüm `/admin/*`.

---

## 4. Hibrit izolasyon — hangi veri ortak, hangisi kullanıcıya ait

**ORTAK** (kullanıcı başına kopyalanmaz — piyasa verisi herkes için aynıdır,
N kullanıcı için N kez yfinance'e gitmek hem yavaş hem rate-limit riski):

```
data/bist/  data/macro/  data/fundamental/  data/gold/
data/raw_stock_data.csv  data/stock_data_with_indicators.csv
```

**KULLANICI BAZLI** — `workspaces/<user_id>/` altında:

```
models/             RL modelleri (.zip)
models/prediction/  ensemble/tahmin modelleri
results/            metrik JSON'ları
results/hyperparameter_studies/
data/predictions/   üretilen tahminler
data/live_trading/  trade_decisions.json, portfolio_history.csv
logs/               TensorBoard
```

Çözümleme `app/auth/workspace.py` üzerinden:

```python
from app.auth.workspace import models_dir, find_file, use_workspace

models_dir()                      # yazma hedefi (aktif kullanıcı)
find_file("results", "x.json")    # okuma: önce kullanıcı, sonra ortak eski dizin
with use_workspace(user_id):      # arka plan görevleri için (ContextVar taşınmaz)
    ...
```

**Geriye dönük uyumluluk:** `WORKSPACE_SHOW_LEGACY=True` iken kullanıcı sistemi
öncesi eğitilmiş `models/` ve `results/` içeriği herkese **salt-okunur** görünür.
Yeni her şey kullanıcının kendi alanına yazılır; kimse ortak dizindeki bir modeli
silemez (403).

`AUTH_ENABLED=False` veya istek bağlamı yoksa (script, test) çözümleyici eski
global dizinlere düşer — mevcut davranış birebir korunur.

### Arka plan görevleri

`BackgroundTasks` istek bağlamını (ContextVar) devralmaz. Bu yüzden kullanıcı
kimliği açıkça taşınır:

```python
background_tasks.add_task(run_training, request, ws.current_user_id())
# run_training içinde:  with ws.use_workspace(user_id): ...
```

### Eşzamanlılık

Eğitim durumu artık kullanıcı başına tutulur:
`trading.py::_training_states[user_id]` ve
`prediction_service.py::_training_state[(user_id, symbol, horizon, source)]`.
Önceden tek global sözlük vardı; bir kullanıcının eğitimi diğerlerini
"Training already in progress" ile blokluyor ve ilerleme çubuklarını
karıştırıyordu.

---

## 5. Kurulum

### Sunucuda ilk açılış

```bash
cp .env.production.example .env
python -c "import secrets; print(secrets.token_urlsafe(64))"   # JWT_SECRET_KEY
# .env: JWT_SECRET_KEY, COOKIE_SECURE=True, AUTH_ENABLED=True, DEBUG=False

docker compose up -d --build
docker compose exec rltrading python scripts/create_admin.py
```

Alternatif: `.env` içine `BOOTSTRAP_ADMIN_EMAIL` + `BOOTSTRAP_ADMIN_PASSWORD`
yazın; ilk açılışta hesap otomatik oluşur. **Hesap açıldıktan sonra bu iki satırı
silin.**

### Yerel geliştirme

`DEBUG=True` iken `JWT_SECRET_KEY` boş bırakılabilir: anahtar üretilip
`data/auth/.jwt_secret` dosyasına yazılır ve `admin@localhost` hesabı rastgele
parolayla açılır — parola **bir kez** log'a düşer.

> `AUTH_ENABLED=False` yalnızca yerel geliştirme içindir. Sunucuda kapatmak,
> panoyu ve tüm API'yi internete açık bırakmak demektir.

### Ters vekil (nginx/Caddy)

- HTTPS zorunlu (`COOKIE_SECURE=True` bunu varsayar).
- `X-Forwarded-For` yalnızca vekil tarafından set edilmeli; audit kaydındaki IP
  ve giriş hız sınırı bu başlığı okur.

---

## 6. Denetim kaydı

`auth_audit_log` tablosu; `/api/admin/audit` ve panodaki Kullanıcılar sayfasından
görüntülenir. Kaydedilen olaylar: `login` (başarılı/başarısız + neden),
`logout`, `password.change`, `user.create`, `user.update`, `user.delete`,
`user.password_reset`, `user.revoke_sessions`, `session.reuse_detected`.

---

## 7. Testler

```bash
python tests/test_auth.py                  # 28 kontrol — uçtan uca oturum akışı
python tests/test_workspace_isolation.py   # 18 kontrol — izolasyon + RBAC + Dash render
```

Her ikisi de geçici SQLite DB ve geçici `workspaces/` dizini kullanır; proje
verisine dokunmaz.

---

## 8. Bilinçli olarak yapılmayanlar

- **Self-signup / e-posta doğrulama:** finansal panelde saldırı yüzeyini
  gereksiz büyütür. Hesapları admin açar.
- **2FA / TOTP:** sonraki adım için doğal aday; şu an kapsam dışı.
- **Kullanıcı silinince çalışma alanı silme:** eğitilmiş modellerin kaybı geri
  alınamaz. Hesap silinir, `workspaces/<id>/` diskte kalır; temizlik operatörde.
- **Çok worker'lı uvicorn:** eğitim durumu hâlâ süreç içi bellekte. Birden fazla
  worker gerekirse Redis'e taşınmalı (auth'un kendisi çok worker'lı çalışır).
