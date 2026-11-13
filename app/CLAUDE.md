<!-- Kok CLAUDE.md'den tasindi (2026-11-14). Bu dizindeki bir dosyaya dokununca yuklenir. -->

# CLAUDE.md — app/ (API, kimlik, servisler)

## Auth & kullanici bazli calisma (Faz 7)
- Giris: `/login` (cerez tabanli oturum — Dash WSGI header tasiyamaz)
- Kapi: `app/auth/middleware.py` — `/dash/*`, `/api/*`, `/docs` korumali;
  acik yollar: `/login`, `/auth/login`, `/auth/refresh`, `/health`, `/static/*`
- Roller: `viewer` (okuma) / `user` (kendi alaninda yazma) / `admin` (+ kullanici yonetimi)
- Kayit yok: hesaplari admin acar (`/dash/users` veya `python scripts/create_admin.py`)
- Dizin cozumleme: `app/auth/workspace.py` → `models_dir()`, `results_dir()`,
  `live_trading_dir()`, `training_runs_dir()`, `find_file(kind, name)`, `use_workspace(user_id)`
- Faz 6 manifest de kullanici bazli: `prediction/manifest.py` → `runs_dir()`/`find_manifest()`;
  `train_batch(user_id=)` arka plan gorevinde calisma alanini sarmalar (thread'e de tasinir)
- Detay: `docs/development/phase-7-auth.md`, `docs/development/phase-6-backend-performance.md`

## Yapit silme (Faz 8/I)
- Model: `DELETE /api/trading/models/{name}` (RequireWriter). Panoda: Modeller
  sayfasinda cok secimli liste + "Secilenleri sil".
- **Ortak (kullanici oncesi) model iki katmanli**: `user` silemez (403, mesaj
  yoneticiyi isaret eder), `admin` siler + `model.delete_shared` denetim kaydi.
  Faz 7'de "kimse silemez" idi; o kural deneme modellerini panodan
  temizlemenin hicbir yolunu birakmiyordu.
- Optimizasyon: `DELETE /api/hyperopt/studies/{id}` **kaydi kalici siler**;
  iptal ayri uctadir (`POST /studies/{id}/cancel`). Calisan kosum silmede 409.
- **`OPTUNA_STORAGE` calisma alanina gore COZULMUYOR** — depo kokune sabit
  bagli, yani optimizasyon calismalari tum kullanicilar arasinda ORTAK.
  Bilinen gedik, ayri is olarak ele alinmali (bkz. Faz 8/I.3).

## Hesap ve profil (Faz 8/F)
- Sayfa: `/dash/account` ("Hesabim") — kenar cubugu altindaki **avatar satiri**
  buraya gider (`dashboard/components/sidebar.py::_account_link`). Her rol erisir.
- Uclar: `app/api/routes/account.py` → `GET /api/account/me`,
  `PATCH /api/account/profile`, `GET /api/account/sessions`,
  `POST /api/account/sessions/revoke-others`, `GET /api/account/activity`,
  `GET /api/account/notifications`. Hepsi `CurrentUser`.
- **Neden `/api/*`, `/auth/*` degil:** pano callback'leri `api_client` uzerinden
  cagiriyor; `/api/*` altinda CSRF + RBAC middleware'den bedava geliyor.
  `/auth/*` tarayicinin dogrudan cagirdigi yuzey (giris formu, tema anahtari) —
  orada CSRF ucun kendi isi.
- **Hedef her zaman oturumdaki kullanici**; govdeden kullanici kimligi ALINMAZ.
  `role`/`is_active`/`email` semada yok → kendi rolunu yukseltme yolu kapali.
- **Kasitli oturum iptali kaydi SILER**, `revoked_at` ile isaretlemez: iptal
  edilmis jti 30 sn'lik grace penceresinde (`REFRESH_REUSE_GRACE_SEC`) yeniden
  kullanilirsa `rotate_session` yeni oturum veriyor; isaretleme birakmak
  "diger oturumlari kapat"i atlatilabilir kilardi (bkz.
  `service.revoke_other_sessions` docstring'i).
- **Kendi etkinligi yalnizca `user_id` ile filtrelenir**, `target` ile DEGIL:
  yoneticinin bu hesap uzerindeki islemi o satirda YONETICIYI tasir; gostermek
  admin kimligini ve IP'sini yonetici olmayan bir yuzeye sizdirirdi
  (`service.list_audit_for_user`).
- `dbc.NavLink`'e `title` VERME — kabul etmedigi prop tum Dash agacini
  render edilemez yapar, `/dash/` 500 doner. Ipucunu sarmalayan Div'e koy.
- Detay: `docs/development/phase-8-ui-theming.md` → "Faz F — Profil sayfasi"

## Egitim suresi tahmini (ETA)
- Servis: `app/services/training_eta.py` — `estimate()` (on tahmin), `live_eta()` (canli), `record_run()`
- Uclar: `GET /api/trading/train/estimate?algorithm=&phase=&total_timesteps=` ve
  `GET /api/trading/train/status` (eta_seconds/eta_text/finish_at/steps_per_sec/phase_name)
- Model: `toplam = hazirlik + total_timesteps x adim_maliyeti(algoritma, sembol) + degerlendirme`
- Ogrenme: her tamamlanan kosum `results/training_eta_history.json`'a yazilir (kullanici bazli);
  sonraki tahmin medyanini kullanir. Gecmis yoksa yerlesik katsayi (`confidence='default'`)
- **`n_symbols` `get_symbols(phase)`'ten ALINMAZ** — egitim rotasi o listeyi yalnizca veri
  cekerken kullanir, egitim yuklenen CSV'nin tamamiyla yapilir (faz 1 secilse de 30 sembol)
- Varsayilan katsayilari yeniden olcmek icin:
  `python scripts/benchmarking/measure_training_eta_defaults.py --steps 10000,25000`
  (kucuk N'de olcum ~2x oynar; egim buyuk N'de kararli)

## Kagit portfoy ve kar/zarar (ONEMLI)
- **`summary.daily_return_pct` KAR/ZARAR DEGILDIR.** `portfolio_before` ve
  `portfolio_after` ayni gunun ayni fiyatlariyla hesaplanir; alim-satim
  nakit<->hisse takasi oldugu icin portfoy degeri degismez, geriye yalnizca
  komisyon kalir. Metrik tanim geregi ~0 veya negatif cikar (olculdu:
  100.000,00 -> 99.998,99 = "-0,0010%", komisyon 1,01 TL). Kar/zarar icin
  `portfolio.value_portfolio()` kullan — pozisyonu BASKA BIR GUNUN fiyatiyla
  degerler.
- Tek gercek kaynak `workspaces/<uid>/data/live_trading/portfolio.json`
  (`app/services/portfolio.py`). `daily_trading.py` kararin NASIL uretildigini,
  `portfolio.py` SONUCUNU bilir.
- **Nakit tanimlari `interpret_actions_with_risk` ile ayni olmali:**
  `BUY -> cost = adet*fiyat*(1+komisyon)` (komisyon DAHIL),
  `SELL -> revenue = adet*fiyat*(1-komisyon)` (DUSULMUS). `avg_cost` bu yuzden
  komisyon dahil tutulur; aksi halde gerceklesmis kar komisyon kadar sisik cikar.
- **Ayni tarih iki kez uygulanmaz** (`applied_dates`) — "Uygula"ya iki kez
  basmak pozisyonu iki katina cikarirdi.
- `/daily-decision` icin `balance`/`shares` artik OPSIYONEL: verilmezse
  portfoyden yuklenir. Panoda "Portfoyu elle gir" anahtari acikken gonderilir
  ve o karar UYGULANAMAZ (uydurma bir baslangictan ilerletirdi).
- Fiyati cekilemeyen sembol ortalama maliyetiyle degerlenir ve
  `missing_prices` ile raporlanir — 0 saymak portfoyu kucuk, sessiz kalmak
  kar/zarari sahte gosterirdi.
- `POST /portfolio/reset` gecmis dosyalarina DOKUNMAZ (trade_decisions.json,
  portfolio_history.csv); silmek geri alinamaz, ayri bir istek olmali.
- Test: `python tests/test_paper_portfolio.py` (43 kontrol)

## Panel tazeligi ve yfinance taslak satiri (ONEMLI)
- **Tazelik olcutu takvim gunu DEGIL, kapanmis olmasi beklenen son SEANS.**
  `cached_last < end_date.date()` sorusu 30 Agustos Pazar hedefinde CSV 28
  Agustos Cuma'ya kadar dolu olsa bile dogru cikiyordu; tek karar istegi 30 CSV
  okumasi + **30 yfinance indirmesi** yapiyordu, sonuc hic degismeden.
  `daily_trading.last_expected_session()` hafta sonunu geriye sarar. Resmi BIST
  tatilleri bilinmiyor; o gunlerde tek bosuna tur atilir, TTL onbellegi keser.
- **yfinance seans kapanmadan once OHLC'si NaN, volume'u DOLU taslak satir
  dondurur.** Bu satir diske yazilirsa iki zarar birden: panelde tamamen bos
  bir gun kalir **ve** `fetch_incremental`'in `min_last_date`'i o gune kayar,
  `fetch_from` ertesi gun olur, o seansin gercek verisi bir daha HIC cekilmez.
  Filtre `fetch_incremental`'da **yalnizca yeni veriye** uygulanir — `existing`
  ham panel ve 2005 oncesi negatif fiyat artefaktlarini tasiyor (asagiya bak);
  oraya `close > 0` uygulamak gecmisi sessizce degistirirdi.
- **`actual_date` sembol kapsamina bakar** (`MIN_SESSION_COVERAGE`, %90).
  Panelin bir kismi tazelenip kalani tazelenmeyince (28.08.2026: 30 sembolden
  3'u) en son tarihte yalnizca o 3 sembolun kapanisi oluyordu; kalan 27 sembol
  tek tek bir onceki gune dusuyor, durum vektoru **iki gunun karisimini**
  tasiyordu. Esigi gecemeyen gun atlanir.
- Panel `(semboller, hedef, lookback, CSV mtime)` anahtariyla 15 dk onbellege
  alinir. Veri sayfasi yeni veri indirince mtime degisir, onbellek TTL
  beklemeden duser. Testte `dt.clear_panel_cache()` cagir — yoksa ikinci
  cagriyi onbellek karsilar ve test sahte yesil verir.
- Test: `python tests/test_panel_freshness.py` (21 kontrol)

## Modelin sembol evreni model ADINDAN cozulmez (ONEMLI)
- Egitim rotasi `get_symbols(phase)` listesini YALNIZCA veri cekerken kullanir;
  egitimi yuklenen panelin tamamiyla yapar. Yani `ppo_phase1_...` adli bir model
  pekala 30 sembolle egitilmistir (obs=331, PHASE1_SYMBOLS ise 5 -> obs=56).
  `/trading/daily-decision` evreni adindan tahmin edince durum vektoru 56
  cikiyordu ve SB3 `predict` asamasinda patliyordu:
  `Unexpected observation shape (56,) ... please use (331,)`.
- Tek gercek kaynak MODELIN KENDISI: `action_space.shape[0]` = sembol sayisi.
  `daily_trading.resolve_trade_universe()` adaylari bu sayiya gore eler:
  yan dosya -> egitim paneli (yeniden uretilir) -> ad tabanli sabitler.
- **Yeni modeller kendi evrenini tasir**: egitim `model.save()` yanina
  `<model>.meta.json` yazar (`write_model_meta`). Eski modeller icin:
  `python scripts/backfill_model_meta.py --write` (once `--write`siz deneme kosumu).
- **Sembol SIRASI onemli** — durum vektoru sembolleri env'deki sirayla diziyor
  (`TradingEnv.symbols` = panelin gorunme sirasi). `sorted()` uygulamak state'i
  sessizce bozar; cozumleyici sirayi oldugu gibi tasir.
- Model silinince yan dosya da silinir (`delete_model`); `list_models` yalnizca
  `.zip` suzdugu icin meta dosyasi model gibi listelenmez.
- Evren dogru sayida cikip durum yine de uymazsa (faz 2 modeli: hisse basina 17
  ozellik + 6 makro; `build_live_state` 10 uretir) rota SB3'a girmeden 400 ile
  nedeni soyler.
