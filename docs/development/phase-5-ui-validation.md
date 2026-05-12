# Faz 5 — Sidebar 8 Sayfa UI Doğrulama + Kritik Bug Fix

**Durum:** Planlama
**Önceki faz:** Faz 4 (modüler feature sistemi, prediction sayfası altın standart)
**Sonraki faz:** Faz 6 (RL pipeline'a feature_groups entegrasyonu, ATR/Kelly UI bağlama, ensemble→daily_trading enjeksiyonu)

---

## Context

Faz 1-3 (RL temel + ensemble tahmin + risk yönetimi) ve Faz 4 (modüler feature sistemi) tamamlandı. Faz 4'te [dashboard/pages/prediction.py](../../dashboard/pages/prediction.py) "altın standart" haline geldi: interaktif fiyat grafiği (`/prediction/price-history`), feature_groups checkbox paneli, target_type radioitem (log_return / abs_price), dinamik source seçimi, polling-based eğitim takibi.

**Bu fazın hedefi:** Sidebar'daki diğer 7 sayfayı (Dashboard, Egitim, Veri, Modeller, Trading, Akademik, HiperParam) sayfa-bazlı doğrulamadan geçirmek; her button/dropdown/callback'in beklenen işi yaptığını teyit etmek; kritik bug'ları gidermek; Faz 4 standardına yükseltme için zemini hazırlamak.

**Kapsam dışı (Faz 6'ya bırakılan):** RL pipeline'a feature_groups entegrasyonu, ATR/Kelly UI bağlama, ensemble tahminin daily_trading kararına enjeksiyonu. Bu faz **sadece UI doğrulama + kritik bug fix** odaklı.

**Kullanıcı kararları:**
- Eski `models/` ve `results/` deneme amaçlıydı → silinebilir, geriye uyumluluk zorunluluğu yok
- Backend `/config/*` endpoint'leri eklenecek (hardcoded listeler kalkacak)
- Smoke test: manuel checklist + standalone `tests/test_smoke.py`

---

## Faz A — Kritik Bug Fix'ler (Quick Wins, paralel uygulanabilir)

### A.1 — `generate_report` POST/GET uyuşmazlığı
- **Dosya:** [dashboard/api_client.py:115-116](../../dashboard/api_client.py#L115-L116)
- **Sorun:** `_get("/trading/analysis/generate-report")` çağırıyor, backend `@router.post(...)` bekliyor. Akademik sayfasında "Rapor Oluştur" butonu sessiz fail.
- **Düzeltme:** `_post("/trading/analysis/generate-report", json={}) or {}`
- **Doğrulama:** Akademik sayfada butona bas → 202 Accepted, status panelinde polling başlamalı.

### A.2 — `hyperopt` reward_type backend'e geçirilmiyor
- **Dosya:** [app/api/routes/hyperopt.py](../../app/api/routes/hyperopt.py)
- **Sorun:** `OptimizationStartRequest.reward_type` field'ı request'te var, ama `optimize(...)` çağrısına geçirilmiyor (TODO yorum satırı dosyada).
- **Düzeltme:** `optimize(...)` çağrısına `reward_type=request.reward_type` parametresi ekle. Gerekirse `hyperparameter_optimization/optimizer.py` içinde kanal aç.
- **Doğrulama:** UI'dan "Sortino" seç, study başlat, study detail'de reward_type'ın doğru kayıtlı olduğunu gör.

### A.3 — `model-comparison` response_model eksik
- **Dosya:** [app/api/routes/trading.py](../../app/api/routes/trading.py) — `/analysis/model-comparison`
- **Düzeltme:** [app/schemas/trading.py](../../app/schemas/trading.py) içine `ModelComparisonResponse(BaseModel)` ekle (`models: Dict[str, Any]`, `count: int`); endpoint dekoratörüne `response_model=ModelComparisonResponse` ver.
- **Doğrulama:** `/docs` (Swagger) altında schema görünmeli.

### A.4 — `home.py` sessiz exception
- **Dosya:** [dashboard/pages/home.py](../../dashboard/pages/home.py)
- **Sorun:** `get_prediction_models()` try/except'te yutuluyor; backend kapalıyken kullanıcı hata göremiyor.
- **Düzeltme:** Exception logla ve UI'da `dbc.Alert("Tahmin modelleri yüklenemedi", color="warning", dismissable=True)` göster.

### A.5 — Blocking I/O — `/data/earliest` 90s timeout
- **Dosya:** [app/api/routes/trading.py](../../app/api/routes/trading.py) — `/data/earliest`
- **Sorun:** `period='max'` sorgusu uzun sürüyor, UI bloke oluyor.
- **Düzeltme:** Sonucu basit in-memory cache'e koy (TTL 24h). Cache key: `(symbol, source)`.

### A.6 — `hyperopt.py` `_ALL` import konumu
- **Dosya:** [dashboard/pages/hyperopt.py](../../dashboard/pages/hyperopt.py) (~satır 247)
- **Düzeltme:** Pattern-matching callback'te kullanılan `ALL` (veya `MATCH`) import'unu dosya başına taşı.

**Paralelleştirme:** A.1 / A.2 / A.3 / A.4 / A.6 tamamen bağımsız. A.5 ayrı PR.

---

## Faz B — Backend Config Endpoint

Sayfa yükseltmelerinden önce yapılır; sonraki adımları besler.

- **Yeni dosya:** [app/api/routes/config.py](../../app/api/routes/config.py)
- **Endpoint'ler:**
  - `GET /config/algorithms` → `["A2C", "PPO", "TD3", "SAC"]`
  - `GET /config/phases` → `[1, 2]` (faz tanımları + label)
  - `GET /config/reward-types` → `["sharpe", "sortino", "total_return", "risk_adjusted", "psr"]`
  - `GET /config/feature-groups` → [prediction/feature_groups.py](../../prediction/feature_groups.py) `get_registry()` çıktısı (id, label, category, default, requires)
- **app/main.py:** router include et
- **dashboard/api_client.py:** `get_config_algorithms()`, `get_config_phases()`, `get_config_reward_types()`, `get_config_feature_groups()` wrapper'ları ekle
- **Tüketim:** training.py, hyperopt.py, daily_trading.py, data.py hardcoded listeleri kaldırıp callback ile bu endpoint'lerden yükle

---

## Faz C — Sayfa Bazlı Doğrulama Checklist'leri

Her sayfa için: çalıştır → her component'i tıkla → callback'in döndüğünü teyit et → bug bulursan ayrı issue aç.

### C.1 — `home.py` (Dashboard)
**Kritik dosya:** [dashboard/pages/home.py](../../dashboard/pages/home.py)
- [ ] Health badge 30s interval ile yeşil/kırmızı geçişi yapıyor mu?
- [ ] 5 KPI kartı (getiri, sharpe, drawdown, portfoy, işlem) — backend kapalıyken "—" gösteriyor mu?
- [ ] Portfoy performans grafiği — boş history için "Portfoy gecmisi yok" fallback'i var mı?
- [ ] Algoritma karşılaştırma grafiği render ediyor mu?
- [ ] RL modeller listesi + Tahmin modelleri listesi — empty state mesajları doğru mu?
- [ ] (A.4 sonrası) backend kapalıyken tahmin modelleri için Alert gösteriliyor mu?

### C.2 — `training.py` (Egitim)
**Kritik dosya:** [dashboard/pages/training.py](../../dashboard/pages/training.py)
- [ ] Algoritma dropdown — Faz B sonrası `/config/algorithms`'tan yüklensin
- [ ] Faz radioitem — `/config/phases`'tan yüklensin
- [ ] Hyperopt study dropdown algoritma seçimine göre filtreleniyor mu?
- [ ] "Eğitimi Başlat" butonu disabled state'i (form invalid iken) doğru mu?
- [ ] Polling — eğitim bitince Interval `disabled=True` oluyor mu?
- [ ] Form reset — bitiş sonrası "Yeni Eğitim" butonu çalışıyor mu?

### C.3 — `data.py` (Veri)
**Kritik dosya:** [dashboard/pages/data.py](../../dashboard/pages/data.py)
- [ ] Kaynak switch (BIST/Gold/Macro/Fundamental) collapse panelleri açılıp kapanıyor mu?
- [ ] BIST chip-list — "Tümü Sec/Kaldır", quick select butonları
- [ ] Gold kaynak değişince metrik checkbox'ları enable/disable
- [ ] Tarih hızlı seçim (1Y/3Y/5Y/En Eski) — En Eski butonu (A.5 sonrası) hızlı dönmeli
- [ ] Incremental update — sadece eksik günler indiriliyor, success alert
- [ ] Full update — overwrite, success alert
- [ ] Veri özet panel — last_date, missing_days, symbol count doğru mu?
- [ ] Dosya listesi — `/data/list` endpoint'inden CSV'ler

### C.4 — `models.py` (Modeller)
**Kritik dosya:** [dashboard/pages/models.py](../../dashboard/pages/models.py)
- [ ] Model checklist + Tümü Sec/Kaldır/Yenile
- [ ] DataTable — 9 metrik kolonu, native sort, page_size=10
- [ ] Sharpe/Sortino/Calmar bar chart
- [ ] Detail modal — portfolio chart, activity chart, trades table
- [ ] Modal'daki equity curve veri kaynağı: `metrics.get("portfolio_history")` doğru mu doluyor?
- [ ] Modal "Trades" tablosu boş gelirse fallback mesajı var mı?

### C.5 — `daily_trading.py` (Trading)
**Kritik dosya:** [dashboard/pages/daily_trading.py](../../dashboard/pages/daily_trading.py)
- [ ] Model dropdown — sayfa yüklenince options doluyor mu? (`load_models` callback)
- [ ] Risk modu radioitem (conservative/moderate/aggressive)
- [ ] 5 hisse slot — dropdown + quantity input
- [ ] "Karar Al" → decision table + summary cards + portfolio chart
- [ ] "Uygula" → backend'e POST + portfolio history yenileniyor
- [ ] "JSON Export" → `dcc.Download` çalışıyor

### C.6 — `prediction.py` (Tahmin — REGRESYON)
**Kritik dosya:** [dashboard/pages/prediction.py](../../dashboard/pages/prediction.py)
- Sadece kırılma yok mu kontrolü: feature_groups checkbox, target_type toggle, fiyat grafiği, polling, train/predict tab'ları çalışmaya devam etmeli.

### C.7 — `academic.py` (Akademik)
**Kritik dosya:** [dashboard/pages/academic.py](../../dashboard/pages/academic.py)
- [ ] (A.1 sonrası) "Rapor Oluştur" butonu → 202 Accepted, polling başlıyor
- [ ] 4 best model card (Sharpe/Return/DD/WinRate)
- [ ] Comparison table doluyor
- [ ] 4 grafik (portfolio overlay / risk-return scatter / metrics bar / winrate bar)
- [ ] Polling 6 ticks sonra otomatik duruyor mu?

### C.8 — `hyperopt.py` (HiperParam)
**Kritik dosya:** [dashboard/pages/hyperopt.py](../../dashboard/pages/hyperopt.py)
- [ ] (Faz B sonrası) Algoritma + reward_type dropdown'ları `/config`'tan yükleniyor
- [ ] Search space hint algoritma değişince güncelleniyor
- [ ] (A.2 sonrası) reward_type seçimi backend'e geçiyor — study detail'de doğrulanabiliyor
- [ ] "Optimizasyon Başlat" → background task + study card oluşuyor
- [ ] Studies grid — pattern-matching callback ile detay modal
- [ ] Modal — info table, best_params JSON, trials table

---

## Faz D — Smoke Test Script'i

**Yeni dosya:** [tests/test_smoke.py](../../tests/test_smoke.py) (standalone, `python tests/test_smoke.py` ile çalışır — pytest yok, mevcut tests/ pattern'ine uygun)

İçerik:
- `/health` — 200
- `/trading/models`, `/trading/data/status`, `/trading/portfolio-history` — 200
- `/prediction/symbols`, `/prediction/models` — 200
- `/hyperopt/studies`, `/hyperopt/search-spaces/PPO` — 200
- `/config/algorithms`, `/config/phases`, `/config/reward-types`, `/config/feature-groups` — Faz B sonrası 200
- `/trading/analysis/generate-report` POST — A.1 sonrası 202

Output: text rapor (PASS/FAIL satır satır). Hata varsa exit code != 0.

---

## Uygulama Sırası ve Bağımlılıklar

```
Adım 1: Faz A (kritik bug fix) — paralel 6 alt-task, hepsi bağımsız
Adım 2: Faz B (config endpoint + api_client wrapper) — Faz C'yi besler
Adım 3: Faz C (sayfa-bazlı doğrulama) — C.1-C.8 paralel, her sayfa için bulgular ayrı liste
Adım 4: Faz D (smoke test script) — Faz A ve B sonrası yazılabilir
Adım 5: Eski `models/` ve `results/` içeriği temizleme (kullanıcı kararı)
```

---

## Critical Files

- [dashboard/api_client.py](../../dashboard/api_client.py) — A.1 (POST/GET fix), Faz B (config wrapper'ları)
- [app/api/routes/trading.py](../../app/api/routes/trading.py) — A.3, A.5
- [app/api/routes/hyperopt.py](../../app/api/routes/hyperopt.py) — A.2
- [app/api/routes/config.py](../../app/api/routes/config.py) — Faz B (yeni dosya)
- [app/main.py](../../app/main.py) — Faz B router include
- [app/schemas/trading.py](../../app/schemas/trading.py) — A.3 ModelComparisonResponse
- [dashboard/pages/home.py](../../dashboard/pages/home.py) — A.4
- [dashboard/pages/training.py](../../dashboard/pages/training.py) — Faz B tüketim, C.2
- [dashboard/pages/hyperopt.py](../../dashboard/pages/hyperopt.py) — A.6, Faz B tüketim, C.8
- [dashboard/pages/academic.py](../../dashboard/pages/academic.py) — C.7 (A.1 sonrası test)
- [prediction/feature_groups.py](../../prediction/feature_groups.py) — Faz B `/config/feature-groups` kaynağı
- [tests/test_smoke.py](../../tests/test_smoke.py) — Faz D (yeni dosya)

---

## Reused Components (Kod Tekrarı Engelleme)

- [dashboard/components/metric_card.py](../../dashboard/components/metric_card.py) — KPI render
- [dashboard/components/sidebar.py](../../dashboard/components/sidebar.py) — Nav (değiştirilmeyecek)
- [dashboard/api_client.py](../../dashboard/api_client.py) `_get`, `_post` helper'ları
- [dashboard/theme.py](../../dashboard/theme.py) — renk paleti, `apply_dark_template`, `empty_figure` fallback
- [prediction/feature_groups.py](../../prediction/feature_groups.py) `get_registry()`, `default_groups()`, `groups_by_category()` — `/config/feature-groups` endpoint'i bunları kullanır

---

## Verification (Faz Sonu)

1. **Backend**: `python run_server.py` → `python tests/test_smoke.py` tüm satırlar PASS
2. **Frontend**: `http://localhost:8000/dash/` → Sidebar'daki 8 sayfayı sırayla aç
3. **Sayfa-sayfa**: Faz C checklist'lerini elle çalıştır; her sayfada bulguları faz notuna kaydet
4. **Akademik**: "Rapor Oluştur" → 202 → polling → success card grubu doluyor
5. **Hyperopt**: PPO + Sortino seç, başlat → study detail'de reward_type=sortino kayıtlı
6. **Eski `models/` ve `results/` temizliği** (kullanıcı isterse `git rm -r models/ results/` veya manuel sil)

Faz bitiş çıktısı: temiz çalışan UI + smoke test PASS + sayfa-sayfa "doğrulandı / iyileştirme adayı" listesi (Faz 6 için RL Phase 4 entegrasyon backlog'u).

---

## Belge Güncelleme Notu

Faz 5 tamamlandığında: bu dosyaya gerçek bulgular eklenecek (her checklist'in PASS/FAIL durumu, açılan issue'lar, Faz 6'ya devredilen iyileştirmeler). [docs/development/roadmap.md](roadmap.md) güncellenecek (Faz 5 ✅). [docs/README.md](../README.md) içindeki development bölümüne link eklenecek.
