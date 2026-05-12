# RL Stabilite & Portföy Yönetimi Hazırlık Analizi

> **Durum:** Salt analiz — bu doküman hazırlanırken kod değiştirilmedi. Mevcut yapının ne kadarının çalıştığını, neyin kırılgan olduğunu ve portföy yönetimi için neyin eksik olduğunu haritalar.
> **Tarih:** 2026-05-12
> **Kapsam:** RL eğitim hattı (`env/`, `scripts/`, `app/api/routes/trading.py`), arayüz akışı (`dashboard/`), günlük işlem/portföy katmanı (`app/services/daily_trading.py`), tahmin entegrasyonu (`prediction/`).

---

## 0. Çalıştırma Kapsamı — Stabilite Çalışması (anlaşılan iş)

> Bu bölüm uygulanacak işi tanımlar. Aşağıdaki §1–§5 arka plan analizidir; geniş "düzeltme listesi" (§4) ileri çalışma için referanstır, **bu turda yapılmayacak**.

**Amaç:** Hocaya gösterilmek üzere RL eğitiminin **stabil çalıştığını** kanıtlamak. Üretim değil — demo amaçlı stabilite çalışması. Tahmini süre ~yarım gün + kısa eğitim koşumu.

**Yapılacaklar:**
1. **Reward NaN/Inf koruması** (`env/reward_functions.py`): `_calculate_differential_sharpe_ratio` içinde `sqrt(B − A²)` yerine `sqrt(max(B − A², eps))`; final reward'ı sonlu aralığa clip et; `env/trading_env.py` `step()` içinde NaN/Inf yok assert'i. — Tek "demo ortasında hard-crash" riski; öncelikli.
2. **`/train` veri-tazeliği ön kontrolü** (`app/api/routes/trading.py`): mevcut `get_source_status` (`data/data_fetcher.py`) ile bayat/eksik veride uyar veya engelle — yanlışlıkla boş/eski veriyle eğitim yapıp düz eğri elde etmeyi önler.
3. **Tam arayüz yolunu smoke-test et:** `python run_server.py` → data sayfası güncelle → kısa PPO eğitim koşumu → models sayfası → daily-decision sayfası; kırılan neyse düzelt.
4. *(opsiyonel)* DSR ölçeğini clamp'le ki reward eğrisi temiz görünsün; eğitim arayüzünde varsayılanı PPO + bilinen-iyi timestep sayısı yap.

**Açıkça kapsam dışı:** ortam checkpoint'i save/load refaktörü; hiperparametre birleştirme (scriptler arası); ModelAnalyzer'ı dashboard'a getirme; tüm Tier-3 portföy-yönetimi işi (rebalancing, tahminleri canlı kararlara bağlama, portföy risk limitleri).

**Başlamadan önce teyit edilecek açık soru:** Demo daily-decision/portföy sayfasını içeriyor mu, yoksa sadece RL eğitimi + metrikler mi? Daily-decision dahilse, `build_live_state` (`app/services/daily_trading.py`) bugünün yfinance verisine karşı temiz çalışıyor mu da doğrula — sessizce yanlış tarihe fallback yapabilir.

---

## 1. "Veri indir → eğit → analiz et" akışı arayüzden çalışıyor mu?

**Büyük ölçüde evet, eksiklerle.**

| Aşama | Arayüzden bağlı mı? | Nerede |
|---|---|---|
| Veri indirme | ✅ Evet | `dashboard/pages/data.py` → `POST /api/trading/data/update` → `data/data_fetcher.py` (+ gold/macro/fundamental fetcher'ları) → `data/` altında CSV'ler |
| Ön işleme (temizle + indikatör) | ✅ Evet | `run_training()` içinde (`app/api/routes/trading.py`), `stock_data_with_indicators.csv` yoksa otomatik üretiliyor |
| Eğitim | ✅ Evet | `dashboard/pages/training.py` → `POST /api/trading/train` → arka plan görevi `run_training()` → `models/{algo}_{phase}_{ts}.zip` + `results/{name}_metrics.json` |
| Eğitim ilerlemesi | ✅ Evet | `GET /api/trading/train/status`, 3 sn'de bir polling |
| Analiz | ⚠️ Kısmi | `dashboard/pages/models.py` backtest metriklerini (`Sharpe/Sortino/Calmar/MDD/win rate`) `results/*.json`'dan gösterir. `app/services/model_analysis.py` (`ModelAnalyzer`) zengin karşılaştırma/grafik/LaTeX üretir ama yalnızca subprocess endpoint'inden (`/analysis/generate-report`) erişilebilir; dashboard'a bağlı değil. `academic` sayfası bu çıktıya bağlanmamış. |

**Akıştaki boşluklar:**
- `/train` veri tazeliğini **kontrol etmiyor** — sessizce bayat veriyle eğitim yapılabilir. `/data/status` `missing_days` raporluyor ama eğitim bunu hiç kullanmıyor.
- Faz 2 eğitimi fundamental + makro CSV'lerine ihtiyaç duyar; eğitim başlamadan bunların varlığı kontrol edilmiyor.
- "Eğitimi iptal et" endpoint'i yok.
- Aynı isim tekrar kullanılırsa modeller birbirinin üzerine yazabilir (timestamp script yolunda var ama her yerde zorunlu değil).
- "Apply decision" sonrası portföy grafiği otomatik yenilenmiyor.
- `model_analysis.py` yetenekleri etkin biçimde gizli — en büyük "analiz" boşluğu.

---

## 2. Eğitim "stabil ve sorunsuz" mu? — Bulunan riskler

### A. Reward fonksiyonu (`env/reward_functions.py`) — sayısal kırılganlık
- **Differential Sharpe Ratio** (`_calculate_differential_sharpe_ratio`, ~L143–179): varyans terimi `B_t − A_t²` float hatasından hafif negatife düşebilir → `sqrt(nan)` → reward `NaN` olur → SB3 eğitimi diverge eder / patlar. Tek koruma `1e-9` tabanı; negatif değerleri kapsamıyor. **Reward'da hiçbir yerde açık NaN/Inf clamp'i yok.**
- **Ölçek uyumsuzluğu:** portföy getirisi yüzde (~±1), DSR `tanh(...)·100` (~±100), MDD cezası kuadratik `mdd·|mdd|/10`, trade-frekans bonusu ±5. Ağırlıklarla (0.50/0.30/0.10/0.05/0.05) toplanıyor ama ham büyüklükler ~100× farklı; DSR domine eder, küçük ağırlık değişiklikleri davranışı öngörülemez kılar.
- **Kuadratik MDD + kuadratik volatilite cezaları**, ajanı "hiçbir şey yapmama"ya itecek kadar agresif (düz equity eğrisi iyi puan alıyor).
- **Olası çift komisyon cezası:** `trading_env.py` içindeki `_execute_trade()` komisyonu zaten bakiyeden düşüyor (bu da sonraki portföy getirisini azaltıyor) ve reward ayrıca bir komisyon ceza terimi çıkarıyor (~L126). Teyit gerekiyor ama çift sayım gibi görünüyor.

### B. Ortam (`env/trading_env.py`) — sağlamlık
- **Gözlem sert clip [-10, 10]** (~L559–561): herhangi bir feature sıçraması (makro şoklar, düşük fiyatlı + küçük std hisse) sessizce kesiliyor — bilgi kaybı, uyarı yok. `[-10,10]` kutusu "gradyan stabilitesi için" seçilmiş (makul) ama clip normalizasyon sorunlarını düzeltmek yerine maskeliyor.
- **Price-stats fallback** (~L181–193): sembol bazlı fiyat mean/std eksikse `{mean:50, std:50}` varsayılıyor, `std<1e-8 → 1.0`. Düşük volatiliteli semboller kötü ölçeklenmiş z-skor alıyor.
- **Eksik satırda forward-fill** (~L498–507): veri boşluklarında *son sembolün* feature bloğunu tekrar ediyor gibi görünüyor (sembol-bazlı fill yerine) — çok günlük boşluklarda eski/yanlış state.
- **ATR sizing** (`_atr_position_size`, `_get_atr`, ~L565–627): yalnızca `atr_period+1` satır besleniyor; <2 satırda 0 dönüyor → fixed-lot'a fallback, yani "adaptif" sizing çoğu zaman aktif değil. Pozisyon formülü obfuske bir biçimde yazılı (`position_value / (price·atr_stop/price)`).
- **Kelly criterion** (`_kelly_position_size`, ~L629–679): simetrik kazanç/kayıp varsayıyor (`a = atr/price`); `win_prob ≤ 0` veya `≥ 1` ise fixed-lot'a fallback; negatif Kelly 0'a clip ediliyor → düşük güvenli günlerde **tüm işlemleri bastırıyor**. Tek güvenlik payı çeyrek-Kelly (0.25); tahmin hatası için pay yok.
- `use_atr_sizing` / `use_kelly` varsayılan `False` (doğru — mevcut eğitimli modelleri korur), bu yüzden yukarıdakilerin çoğu yalnızca opt-in edersen devreye girer.
- **Modelle birlikte ortam checkpoint'i kaydedilmiyor.** `model.save()` yalnızca SB3 ağırlık/hiperparametrelerini yazıyor. Price stats, sembol listesi, faz, reward tipi, indikatör lookback'leri **kalıcı değil**. Inference (`build_live_state`) normalizasyonu bağımsız yeniden türetiyor, bu yüzden eğitim/inference normalizasyonu sessizce kayabilir — yalnız stabilite değil doğruluk riski.
- Slippage / piyasa etkisi / likidite modellemesi yok — POC için olur, "portföy yönetimi"ne dokunduğunda önemli.

### C. Eğitim scriptleri / hiperparametreler
- Hiperparametreler giriş noktaları arasında tutarsız: `scripts/train_with_gold.py` çıplak `PPO('MlpPolicy', env)` (tüm varsayılanlar), `scripts/training/train_a2c_phase1.py` ayarlı bir A2C config, `tests/test_ppo.py` başka bir PPO config. Tek doğru kaynak yok; arayüzden eğittiğin "model" ayarladığınla aynı olmayabilir.
- A2C ve TD3, `tests/test_all_algorithms.py` içinde PPO'dan daha az stabil olarak işaretli. PPO güvenli varsayılan.
- Vektörlü ortam aksiyon şekli `(1, n_stocks)` vs `(n_stocks,)` ele alınmış (iyi — açık flatten+validate var) ve `DummyVecEnv` `done` dizi indeksleme ele alınmış. Bunlar gerçek bug'lardı, artık yamalı.
- Yalnızca tek koşum değerlendirmesi; seed/tekrar/anlamlılık yok — yüksek varyanslı sonuçlar, kendini kandırmak kolay.

### D. Eşzamanlılık
- `_training_lock` process-başına. Sunucu çoklu worker ile çalışırsa iki eğitim aynı anda başlayıp `training_state`'i ve model dosyalarını ezebilir.

---

## 3. Bu eğitim portföy yönetiminde kullanılabilir mi?

**Kısmen — tesisatın ~%60'ı var; henüz portföy yöneticisi değil.**

Bugün çalışan kısım (`app/services/daily_trading.py` + `POST /api/trading/daily-decision` + `dashboard/pages/daily_trading.py`):
- Eğitilmiş model yükle, yfinance ile son OHLCV+indikatör çek, gözlemi yeniden kur (`build_live_state`), `model.predict(state, deterministic=True)`.
- `interpret_actions_with_risk()` ham aksiyonu AL/SAT/BEKLE'ye çeviriyor: min sinyal eşiği, max pozisyon % (risk moduna göre 20–40), günlük işlem sayısı sınırı (2–5), komisyon modellemesi.
- Kararları `data/live_trading/trade_decisions.json`'a yazıyor, `portfolio_history.csv`'ye ekliyor; "Apply" history'yi güncelliyor; dashboard karar tablosu + portföy grafiği + export gösteriyor.

Gerçek portföy yönetimi için eksik olanlar:
- **Yalnızca tek-adım, tek-varlık sinyalleri.** Rebalancing takvimi yok (haftalık/aylık), hisse + altın + döviz arası hedef ağırlık / varlık dağılımı yok.
- **Hisseler arası korelasyon** pozisyon boyutlandırmada yok — her isim bağımsız boyutlandırılıyor, korele isimlere aşırı yoğunlaşabilir.
- **Inference anında portföy seviyesi risk limitleri yok** (max drawdown stop, volatilite hedefi, leverage kontrolü, stop-loss / take-profit). Risk "modları" sadece sonradan uygulanan clamp'ler, öğrenilmiş ya da portföy kısıdı olarak zorlanmış değil.
- **Ensemble tahminleri canlı kararlara bağlı değil.** `prediction/` + `prediction_service.predict()` `{predicted_return, predicted_direction, confidence, ensemble_agreement}` üretiyor; `trading_env.py` `prediction_features` (4/hisse) *alabiliyor* ve Kelly sizing güven skorunu *kullanabiliyor* — ama `daily-decision` `PredictionService`'i hiç çağırmıyor, `build_live_state`'e `prediction_data` geçmiyor, eğitim scriptleri de tahmin yüklemiyor. Yani Faz 2 "RL + tahmin" mimaride var ama uykuda.
- **Inference'ta ortam-state yüklenmesi yok** (§2.B'deki checkpoint boşluğu) — canlı kararlara güveni zayıflatıyor.
- **Performans geri besleme döngüsü yok** (gerçekleşen vs tahmin) ve canlı equity izleme yok; history CSV manuel append.
- `models.py` analizi backtest metriklerini gösteriyor ama tahmin doğruluğu, feature/tahmin katkısı veya zamana karşı exposure'ı göstermiyor.

**Özet:** Arayüzden eğitilmiş bir modelden günlük al/sat öneri listesi alabilir ve kâğıt portföy takip edebilirsin. Buna "portföy yönetimi" demek için şunları eklemek gerekir: rebalancing kadansı, hedef-ağırlık/dağılım mantığı, portföy seviyesi risk kısıtları, korelasyon-farkında boyutlandırma ve (Faz 2'nin değerini istiyorsan) ensemble tahminlerini hem eğitime hem canlı inference'a fiilen besleme.

---

## 4. Önceliklendirilmiş düzeltme listesi (kod yazılmadı — karar senin)

**Stabilite (önce bunlar):**
1. Reward'ı clamp'le/NaN-koru (`reward_functions.py`): `sqrt(B − A²)` yerine `sqrt(max(B − A², eps))`, final reward'ı sonlu aralığa clip et, `step()`'te NaN/Inf yok assert'i.
2. Reward terim ölçeklerini uzlaştır (DSR'ı ~±1'e normalize et veya ağırlıkları yeniden dengele) ve çift komisyon cezasını teyit et / kaldır.
3. `model.save()` ile birlikte ortam checkpoint'i kaydet (price stats, sembol listesi, faz, reward tipi, indikatör config) ve `build_live_state`'te yükle ki eğitim/inference normalizasyonu uyuşsun.
4. `/train`'e veri-tazeliği ön kontrolü ekle (mevcut `get_source_status` ile) ve Faz 2 veri-varlığı kontrolü; uyar veya engelle.
5. Hiperparametreleri tek bir config'te birleştir ve `run_training`, scriptler ve testlerde kullan; varsayılan algoritma = PPO.

**Sorunsuzluk / UX:**
6. `ModelAnalyzer` çıktısını dashboard'a getir (`academic`/`models` sayfasını subprocess-only yol yerine `/analysis/...`'a bağla).
7. "Apply" sonrası portföy grafiğini otomatik yenile; uygulamadan önce onay dialog'u ekle.
8. "Eğitimi iptal et" endpoint'i ekle; timestamp'li model isimlerini zorla.

**Portföy yönetimi (daha geniş kapsam, ayrı çalışma):**
9. `PredictionService.predict()`'i `/daily-decision`'a bağla ve `prediction_data`'yı `build_live_state`'e geçir; isteğe bağlı olarak eğitim ortamı kurarken tahminleri yükle.
10. Hisse-başı sinyallerin üstüne rebalancing scheduler + hedef-ağırlık dağıtım katmanı ekle.
11. Portföy seviyesi risk limitleri (drawdown stop, vol hedefi, exposure tavanları) ve korelasyon-farkında boyutlandırma ekle.
12. Gerçekleşen-vs-tahmin geri besleme log'u ve canlı equity izleme ekle.

---

## 5. İlgili dosyalar
- `env/trading_env.py` — gözlem/aksiyon uzayı, step, `_execute_trade`, `_get_observation`, `_atr_position_size`, `_kelly_position_size`, `get_metrics`
- `env/reward_functions.py` — `RewardCalculator` (PSR), `SimpleRewardCalculator`
- `app/api/routes/trading.py` — `/data/update`, `/data/status`, `/train`, `/train/status`, `/models`, `/daily-decision`, `/apply-decision`, `/portfolio-history`, `run_training()`, `TrainingStateCallback`
- `app/services/daily_trading.py` — `build_live_state`, `interpret_actions_with_risk`, `get_risk_parameters`, portföy kalıcılığı
- `app/services/model_analysis.py` — `ModelAnalyzer` (yeterince açığa çıkarılmamış)
- `app/services/prediction_service.py` — ensemble predict (canlı işleme bağlı değil)
- `dashboard/pages/{data,training,daily_trading,models,academic}.py`
- `scripts/train_with_gold.py`, `scripts/training/train_a2c_phase1.py`
- `tests/test_env.py`, `tests/test_ppo.py`, `tests/test_all_algorithms.py`

## Doğrulama (değişiklik yapıldığında)
- `python tests/test_env.py` — gözlem şekli/sınırları, işlemler yürütülüyor.
- `python tests/test_ppo.py` — kısa PPO eğitimi + değerlendirme, loglarda NaN reward'a dikkat.
- `python tests/test_all_algorithms.py` — A2C/PPO/TD3 karşılaştırması.
- `python run_server.py` ardından dashboard: data sayfası → güncelle; training sayfası → eğit; models/academic sayfası → analiz; daily_trading sayfası → karar + apply + grafik.
