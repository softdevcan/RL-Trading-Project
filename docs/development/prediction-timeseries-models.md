# Ensemble'a Klasik Zaman Serisi Modelleri Eklemek — Ölçüm ve Değerlendirme

> **Durum:** Salt analiz — bu doküman hazırlanırken tahmin hattında kod değiştirilmedi. Tek eklenen dosya ölçümü tekrarlanabilir kılan `scripts/analysis/probe_timeseries_structure.py`.
> **Tarih:** 2026-08-30
> **Soru:** Stacking ensemble'a ARIMA / SARIMAX gibi zaman serisi modelleri eklenebilir mi? Mevsimselliği veya yönü yakalamak için daha güçlü bir yapı kurulabilir mi?
> **Kapsam:** `prediction/models/ensemble.py`, `prediction/models/base.py`, `prediction/feature_engineer.py`, `data/bist/raw_stock_data.csv` (30 sembol, 76.058 gözlem).

---

## 0. Özet

**ARIMA/SARIMAX teknik olarak eklenebilir ama beklenen karşılığı vermez.** Nedeni modelin kendisi değil, hedef seri: bir model ancak veride var olan yapıyı yakalayabilir ve ölçüm, ARIMA'nın bakacağı yerde yapı olmadığını gösteriyor.

Aynı ölçüm iki alternatifi işaret ediyor:

| Yön | Ölçülen dayanak | Değerlendirme |
|---|---|---|
| ARIMA / SARIMAX | Getiri ρ₁² = **%0,33**; 30 sembolden 18'inde Ljung-Box yapı bulamıyor | Baktığı üç bilgi kaynağı da zaten özellik matrisinde — ensemble'a çeşitlilik katmaz |
| **GARCH / EGARCH** | \|Getiri\| ρ₁² = **%4,55**; **30/30** sembolde yapı var | Mevcut hiçbir model ileriye dönük varyans üretmiyor — **yeni bilgi** |
| **Yön meta-learner'ı** | Meta-learner regresyon; yön hiçbir yerde doğrudan optimize edilmiyor | Altyapı (`_predict_direction_raw`) hazır ve kullanılmıyor — en az dokunuşla en çok kazanç |

---

## 1. Yöntem

Ölçüm, tahmin hattının hedefiyle **aynı tanımı** kullanır (`feature_engineer.py::_build_targets`, `target_type='log_return'`):

```
target = log(P_{t+1} / P_t)          # günlük ufuk
```

Üç büyüklük hesaplanır:

1. **Getiri otokorelasyonu** (ρ₁…ρ₅ + Ljung-Box Q(10)) — ARIMA'nın AR/MA yapısının yakalayabileceği her şey buradadır.
2. **|Getiri| otokorelasyonu** — volatilite kümelenmesi, yani GARCH ailesinin alanı.
3. **Haftanın günü / ay etkisi** (tek yönlü ANOVA + açıklanan varyans) — SARIMA'nın mevsimsel `S` bileşeninin ve mevcut takvim özelliklerinin hedeflediği yapı.

Ljung-Box hipotezi: **H₀ = ilk h gecikmenin tamamı sıfır** (seri beyaz gürültü). `p > 0,05` ⇒ yapı bulunamadı.

`statsmodels` **gerekmez** — otokorelasyon ve Ljung-Box elle hesaplanır, yalnızca `scipy.stats` (ki-kare kuyruğu, F-testi) kullanılır. Karar vermek için önce bağımlılık eklemek gerekmesin diye böyle yazıldı.

```bash
python scripts/analysis/probe_timeseries_structure.py
python scripts/analysis/probe_timeseries_structure.py --lags 20 --json out.json
```

> Not: `scripts/` `.dockerignore`'da hariç tutulur (sunucu imajı ince kalsın diye; yalnızca `create_admin.py` istisna). Konteynerde koşturmak için dosyayı `docker cp` ile taşıyın.

---

## 2. Bulgu 1 — Getiride yapı yok, volatilitede çok var

| Seri | Ortalama ρ₁ | ρ₁² (açıklanan varyans) | Ljung-Box Q(10) |
|---|---|---|---|
| **Getiri** (hedef) | +0,0325 | **%0,33** | 30 sembolden **18'inde yapı YOK** (p > 0,05) |
| **\|Getiri\|** (volatilite) | +0,2071 | **%4,55** | **30/30 sembolde yapı VAR** (p < 0,0001) |

Sembol düzeyinde ilk sekiz satır:

```
sembol          n     rho1     LB p   |   |r| rho1     LB p
AKBNK.IS     2535   0.0709   0.0017   |     0.1632   0.0000
ARCLK.IS     2535   0.0116   0.4948   |     0.2185   0.0000
ASELS.IS     2535  -0.0381   0.0150   |     0.2508   0.0000
BIMAS.IS     2538   0.0215   0.0815   |     0.1399   0.0000
EKGYO.IS     2535   0.0084   0.1755   |     0.1979   0.0000
ENKAI.IS     2535  -0.0756   0.0006   |     0.2597   0.0000
EREGL.IS     2536  -0.0275   0.9155   |     0.1446   0.0000
FROTO.IS     2536   0.0168   0.6584   |     0.2384   0.0000
```

**Okuma:** Getiri serisinin kendi geçmişiyle açıklanabilen kısmı %0,33'tür — ARIMA'nın AR/MA yapısının **tavanı** budur, gerçekleşen katkı değil. Volatilitede aynı ölçü 14 kat daha büyük ve tek bir sembolde bile istisnasız.

Bu, finansal serilerin bilinen iki olgusuyla birebir uyumlu: getiriler yaklaşık ilişkisiz, oynaklık ise kümelenir.

---

## 3. Bulgu 2 — Mevsimsellik istatistiksel olarak var, ekonomik olarak yok

```
Haftanin gunu ANOVA: F=35,52  p<0,0001   -> aciklanan varyans %0,186
Ay etkisi     ANOVA: F=20,32  p<0,0001   -> aciklanan varyans %0,293

Pzt +0,2591%   Sal +0,0403%   Car -0,0267%   Per +0,2450%   Cum +0,1071%
Toplam gozlem: 76.058
```

**p değerleri burada yanıltıcıdır.** n = 76.058'de neredeyse her fark anlamlı çıkar; asıl soru etki büyüklüğüdür ve o **%0,19–0,29** seviyesinde.

En büyük gün farkı ≈ 0,29 puan. Tek yön komisyon %0,1, gidiş-dönüş %0,2 (`interpret_actions_with_risk`, `commission_rate=0.001`). Marjin işlem maliyetinde büyük ölçüde erir — mevsimsellik **tek başına işlem sinyali üretmez**.

**Dikkat edilmesi gereken bir karışıklık:** Pazartesi'nin en yüksek pozitif ortalamayı vermesi, literatürdeki klasik *negatif* Pazartesi etkisinin tersidir. Örneklem 2018–2026 arası yüksek nominal TL getirisi dönemini kapsıyor ve **her gün pozitif ortalamaya sahip**. Yani burada görülen büyük olasılıkla mevsimsel bir alfa değil, genel yukarı **drift**'tir. Reel getiri veya piyasa-nötr bazda tekrar ölçülmeden sezonalite iddiası kurulmamalı.

---

## 4. ARIMA/SARIMAX neden ensemble'a çeşitlilik katmaz

Ensemble çeşitliliği, yeni modelin **farklı bilgi** ya da **farklı hata yapısı** getirmesiyle artar. SARIMAX'ın baktığı üç bilgi kaynağının üçü de özellik matrisinde zaten var:

| SARIMAX bileşeni | Projede karşılığı |
|---|---|
| AR / MA (gecikmeli getiri) | `return_1d`, `return_5d`, …, `close_lag_*` — `feature_engineer.py:219-228` |
| S (mevsimsel periyot) | `day_of_week_sin/cos`, `month_sin/cos`, `quarter`, `week_of_year`, `is_month_end` — `_add_calendar_features`, `feature_engineer.py:338-355`; ayrıca `_add_seasonality_features` |
| X (exogenous) | Makro grubu: politika faizi, TÜFE, USD/TRY, BIST100, VIX, US10Y, DXY |

Dahası SARIMAX bu ilişkileri **doğrusal ve sabit periyotlu** biçimde modeller; XGBoost/LightGBM/CatBoost aynı sütunları doğrusal olmayan ve etkileşimli biçimde kullanabiliyor. Yani eklenen model, mevcut modellerin gördüğü bilginin **kısıtlanmış bir alt kümesini** işler.

### Mimari sürtünme (ikincil ama gerçek)

- **Arayüz tabular.** `BasePredictionModel._fit(X_train, y_train, X_val, y_val)` bir `(n_samples, n_features)` matrisi alır; `_predict_raw(X)` satır bazlı tahmin üretir. ARIMA `X` kullanmaz — kesintisiz ve sıralı `y` geçmişi ister. Bir adaptör (model kendi `y` geçmişini saklar) yazılabilir, ancak sözleşmeyi zorlar.
- **Yeniden fit maliyeti.** Walk-forward'da ARIMA her adımda yeniden fit edilmek ister. Faz 6'da eğitim koşumu 533,9 s → 157,6 s'ye (−%70,5) indirildi; bu kazanç geri verilir.
- **3'lü kronolojik split** (`ensemble.py`, %60/%20/%20) sıralılığı koruduğu için bu tarafta engel yok — sorun sözleşme ve maliyet.

**Sonuç:** Eklenebilir; ama ölçüm, katkısının gürültü seviyesinde kalacağını söylüyor.

---

## 5. Ölçümün işaret ettiği iki fırsat

### 5.1 GARCH / EGARCH — koşullu volatilite

Ölçümdeki en güçlü sinyal burada: %4,55'e karşı %0,33, ve 30/30 sembolde tutarlı.

Projede hâlihazırda **gerçekleşmiş** volatilite özellikleri var — `realized_vol_{5,10,20}`, `vol_ratio_5_20`, `garman_klass_vol_20`, `std_{5,20}` (`_add_volatility_features`, `feature_engineer.py:240-268`). Hepsi **geçmişe** bakar. Eksik olan **ileriye dönük koşullu varyans** — GARCH'ın ürettiği şey tam olarak budur ve mevcut modellerin hiçbiri bunu üretmiyor. Bu yüzden ARIMA'nın aksine **yeni bilgi** sayılır.

İki doğal bağlantı noktası:

1. **Risk yönetimi (Faz 3.3):** ATR tabanlı pozisyon boyutlandırma ve Kelly criterion (`use_atr_sizing`, `use_kelly`) doğrudan volatilite tahminine dayanır; şu an geçmiş ATR kullanılıyor.
2. **Güven kalibrasyonu:** Yüksek volatilite rejiminde yön güveninin kısılması. `interpret_actions_with_risk` içindeki `min_signal_threshold` sabit; rejime duyarlı hale gelebilir.

### 5.2 Yön için ayrı meta-learner

"Yönü daha güçlü yakalamak" sorusunun asıl cevabı bu.

Mevcut durum: meta-learner bir **regresör** (Ridge veya XGBoost), hedefi `target_price` (= log getiri). Yön, bu regresyon çıktısının **işaretinden** türetiliyor. Yani sistem yönü hiçbir yerde doğrudan optimize etmiyor — düşük genlikli ama doğru işaretli bir tahmin, yüksek genlikli yanlış işaretli bir tahminle aynı kayıp fonksiyonunda yarışıyor.

Öneri: base modellerin tahminleri **artı** BiLSTM/TFT'nin `_predict_direction_raw()` çıktıları girdi olan bir **sınıflandırıcı meta-learner**, üstüne olasılık kalibrasyonu (Platt / isotonic). Altyapı hazır: `base.py:120` `_predict_direction_raw` tanımlı ve `ensemble.py:500`'de toplanıyor, ama yalnızca güven hesabında kullanılıyor.

Yön doğruluğunu artırmanın yolu çoğu zaman daha iyi bir yön modeli değil, **ne zaman işlem yapmayacağını bilmektir**; kalibre edilmiş güven bunu verir.

---

## 6. Sınırlar ve karşı-kanıt yolu

Bu doküman bir dizi ölçüme dayanıyor ve ölçümlerin kapsamı sınırlı:

- Ölçüm **ham getiri serisi** üzerinde yapıldı; özellik mühendisliği, ICEEMDAN gürültü filtresi ve model hattından geçmedi. ICEEMDAN sonrası artık seride otokorelasyon yapısı değişebilir.
- ARIMA'nın **kesin olarak faydasız** olduğu gösterilmedi; gösterilen şey **bakacağı yerde bilgi olmadığıdır**. Doğrusal olmayan ya da rejime bağlı bir yapı bu testlerden kaçabilir.
- Ljung-Box ve ANOVA, sabit varyans varsayımına duyarlıdır; volatilite kümelenmesi güçlü olduğu için getiri testlerinin gücü olduğundan düşük tahmin edilmiş olabilir.
- Örneklem 2018–2026 tek bir rejimi (yüksek nominal getiri) ağırlıklı içeriyor; §3'teki drift uyarısı bunun sonucudur.
- Sonuçlar BIST-30 paneline özgüdür; başka piyasada tekrarlanması gerekir.

**Kesin cevap için deney:** `prediction/models/` altına bir `ArimaModel` (veya `SarimaxModel`) ekleyip `MODEL_REGISTRY`'ye kaydetmek, `trainer.py`'nin walk-forward hattında MAPE / direction accuracy / IC olarak ensemble'ın kalanıyla karşılaştırmak. Beklenen getirisi düşük ama maliyeti de sınırlı bir deney; §4'teki iddiayı doğrudan sınar.

---

## 7. Öneri sırası

1. **GARCH tabanlı koşullu volatilite** — önce bir özellik olarak (`prediction/feature_engineer.py` yeni grup), ardından risk yönetimine bağlantı. En yüksek ölçülen dayanak.
2. **Yön meta-learner'ı + olasılık kalibrasyonu** — en az mimari dokunuşla en yüksek beklenen kazanç; altyapı zaten duruyor.
3. *(opsiyonel)* **ARIMA/SARIMAX deneyi** — §6'daki kurulumla, iddiayı yanlışlamak için.

Mevsimsellik için ayrı bir iş önerilmiyor: ölçülen etki büyüklüğü (%0,19–0,29) işlem maliyetinin altında ve ilgili bilgi zaten Fourier terimleri olarak özellik matrisinde.

---

## İlgili dokümanlar

- [prediction-system.md](prediction-system.md) — tahmin sistemi mimarisi (ensemble, feature engineering)
- [phase3-implementation.md](phase3-implementation.md) — ICEEMDAN, TATS, ATR/Kelly, SHAP
- [phase-6-backend-performance.md](phase-6-backend-performance.md) — eğitim throughput'u (§4'teki refit maliyeti tartışması)
