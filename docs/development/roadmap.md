# Geliştirme Yol Haritası

**Proje:** BIST-30 DRL Trading System
**Ana Referans (Faz 1-3):** Ansari et al. (2024) — *"A Multifaceted Approach to Stock Market Trading Using Reinforcement Learning"*
**Tez Vizyonu:** Çoklu-ajan RL + İletişim + Tahmin entegrasyonu (detay için: [thesis/vision-and-scope.md](../thesis/vision-and-scope.md))

Bu belge projenin **geçmişini (tamamlanan fazlar)** ve **gelecek yol haritasını (tez milestone'ları)** özetler. Detaylı durum/mimari için:
- Tahmin sistemi: [prediction-system.md](prediction-system.md)
- Faz 3 uygulama detayları: [phase3-implementation.md](phase3-implementation.md)
- Tez vizyon ve kapsam: [../thesis/vision-and-scope.md](../thesis/vision-and-scope.md)

---

## Güncel Durum (2026-04-21)

Üç faz tamamlandı. Tek-ajanlı temel altyapı, ensemble tahmin sistemi ve risk yönetimi çalışır durumda. Sonraki adım **tez için çoklu-ajan mimariye** geçiş (Milestone 0-4).

| Katman | Durum | Konum |
|---|---|---|
| Veri pipeline (OHLCV + makro + fundamental + altın/döviz + VIX/US10Y/DXY) | ✅ | `data/` |
| Teknik indikatörler (MACD, RSI, CCI, ADX, Turbulence) | ✅ | `data/technical_indicators.py` |
| RL environment (tek-ajan, PSR reward, ATR/Kelly) | ✅ | `env/` |
| Ensemble tahmin (XGB + LGBM + CatBoost + BiLSTM + TFT + Ridge/XGB meta) | ✅ | `prediction/` |
| ICEEMDAN gürültü filtresi + TATS trend düzeltici | ✅ | `prediction/iceemdan_processor.py`, `prediction/tats.py` |
| SHAP açıklanabilirlik | ✅ | `prediction/explainability.py` |
| FastAPI backend + Dash dashboard (`/dash/` mount) | ✅ | `app/`, `dashboard/` |
| Hiperparametre optimizasyonu (Optuna, RL + Prediction ayrı) | ✅ | `hyperparameter_optimization/`, `prediction/hyperopt.py` |

---

## Tamamlanan Fazlar

### ✅ Faz 1 — Proof of Concept

5 hisse (AKBNK, THYAO, TUPRS, BIMAS, ASELS) üzerinde A2C/PPO/TD3 ile temel RL altyapısı.

- Gymnasium tabanlı multi-stock trading environment (56 feature state space)
- Stable-Baselines3 entegrasyonu (A2C/PPO/TD3)
- 5 teknik indikatör (Ansari et al. seti)
- FastAPI backend + web UI (ilk sürüm)
- Tensorboard logging, metrik hesaplama (Sharpe, Return, Drawdown)

**Ana çıktı:** Çalışır tek-ajan RL baseline + dashboard.

### ✅ Faz 2 — Multifaceted Prediction System

Ansari et al. metodolojisinin tam uygulaması + tahmin katmanı.

- Fundamental veri entegrasyonu (ROE, ROA, P/E, P/B, D/E, profit margin, ...)
- Makro veri pipeline (TCMB EVDS faiz/enflasyon + yfinance döviz/BIST100)
- Altın/döviz pipeline (borsapy + yfinance)
- Feature engineering v2 (10 grup özellik; en az 1 gün gecikmeli — leakage yok)
- Feature selector (mutual information + permutation importance, 3 aşamalı)
- Multi-model tahmin: XGBoost + LightGBM + CatBoost + BiLSTM (PyTorch/CUDA) + TFT
- Stacking ensemble (Ridge / XGBoost meta-learner)
- Optuna HPO (TPE sampler + median pruner, TimeSeriesSplit CV)
- Walk-forward eğitim (purge gap 5 gün, embargo 3 gün)
- RL entegrasyonu: observation space'e tahmin özellikleri eklendi (+4×N: return/direction/confidence/agreement)
- PSR reward function (Ansari Eq. 1) — `env/reward_functions.py`
- Dash dashboard (8 sayfa, `/dash/` altında mount)

**Ana çıktı:** Tahmin-destekli tek-ajan RL sistemi + tam dashboard.

### ✅ Faz 3 — Production Improvements

Bug fix'ler + tahmin kalitesi + risk yönetimi + açıklanabilirlik.

**3.1 Bug Fixes:**
- PSR reward `total_trades` sayım hatası giderildi
- Meta-learner data leakage 3-way chronological split ile çözüldü (60/20/20, OOF)
- Embargo `prev_test_end` takibi ile her fold'da doğru uygulanıyor
- TFT VSN 50-değişken sınırı kaldırıldı (feature selector zaten 80 ile sınırlar)
- Direction head: BiLSTM/TFT sigmoid çıktısı confidence hesabında kullanılıyor
- Permutation importance feature_selector'a entegre edildi

**3.2 Tahmin Kalitesi:**
- ICEEMDAN gürültü filtreleme (`prediction/iceemdan_processor.py`)
- TATS trend-adjusted düzeltici (`prediction/tats.py`)
- Global makro göstergeler: VIX, US10Y, DXY (`macro_fetcher.py` + `feature_engineer.py`)

**3.3 Risk Yönetimi:**
- ATR tabanlı dinamik pozisyon boyutlandırma (`use_atr_sizing=True`)
- Kelly Criterion pozisyon boyutlandırma (`use_kelly=True`, quarter-Kelly)

**3.4 Explainability & Monitoring:**
- SHAP explainability (`prediction/explainability.py`, `/prediction/explain/{symbol}` API)
- Sortino, Calmar, Deflated Sharpe Ratio, Turnover metrikleri

**Ana çıktı:** Faz 2 sisteminin "production-grade" kalite için düzeltilmiş ve zenginleştirilmiş sürümü.

---

## Gelecek — Tez Yol Haritası

Tezin ana vizyonu: **tek-ajanlı sistemi çoklu-ajan + iletişim paradigmasına evirmek**, tahmin sistemini bu iletişime entegre etmek. Detaylı tasarım seçenekleri ve akademik konumlama için [thesis/vision-and-scope.md](../thesis/vision-and-scope.md) belgesine bakılmalıdır.

**Tez araştırma sorusu (kısaca):** BIST-30 portföy yönetiminde sektör-bazlı çoklu-ajan RL + tahmin-destekli iletişim, tek-ajanlı alternatiflere göre ne düzeyde üstünlük sağlar?

Milestone haritası:

### Milestone 0 — Baseline Dondurma (0-3 hafta)
Mevcut Faz 1-3 sistemini "kanonik baseline" olarak dondurmak, reprodüktibilite altyapısı kurmak.
- Veri snapshot (parquet), seed + hyperparam kilitleme
- Full BIST-30 (30 hisse) üzerinde baseline eğitimi
- `results/baseline/` versiyonlama
- Ensemble DT vs DTF ablation
- **Çıktı:** Tez Bölüm 4 taslağı + Makale 2 (Ensemble+RL) ham materyali

### Milestone 1 — Literatür + Tez Önerisi (3-5 hafta)
Akademik çerçeve formalize + ilk makale taslağı.
- Genişletilmiş literatür taraması (60+ makale)
- Danışman revizyonu
- Makale 2 ilk submission-ready taslak
- **Çıktı:** Tez Bölüm 1-3 taslağı

### Milestone 2 — MARL Framework (5-11 hafta)
SB3 → PettingZoo + RLlib geçişi + no-comm baseline + attention communication.
- Sektör-bazlı (~8 ajan) çevre tasarımı
- IPPO no-comm baseline
- Attention-based communication (TarMAC-benzeri)
- Tek-ajan vs no-comm vs attention karşılaştırması
- **Çıktı:** Tez Bölüm 5 taslağı + Makale 1 çekirdek deneyler

### Milestone 3 — Prediction-Augmented Communication (11-16 hafta)
Tezin özgün katkısı.
- GNN-based communication (PyTorch Geometric + GAT)
- Prediction confidence → message weight entegrasyonu
- Meta-learner regime signal broadcast
- 4-yol karşılaştırma: no-comm / attention / GNN / prediction-augmented
- Rejim-bazlı ablation (boğa/ayı/yatay)
- Attention ağırlıkları nitel yorumlama
- **Çıktı:** Tez Bölüm 6-8 taslağı + Makale 1 & 3 ham materyali

### Milestone 4 — Yazım, Submit, Savunma (16-20 hafta)
Paketleme.
- Tez tüm bölümler tamamlandı
- Makale 1 submit
- Makale 3 taslak
- Makale 4 (Türkçe, ulusal)
- LaTeX + figure paketi final
- Savunma slaytları + demo
- **Çıktı:** Tez teslim + makale submissionlar

---

## Kapsam Dışı (Tez Bağlamında)

Tezin sınırlarını netleştirmek için **yapılmayacaklar listesi** (detay için vision-and-scope.md §4.2):
- Intraday / high-frequency trading
- Türev ürünler (opsiyon, future)
- Short selling
- Canlı para ile işlem
- Başka borsalar (BIST dışı)
- LLM-tabanlı haber/sentiment entegrasyonu
- Learned discrete communication protocols (DIAL/RIAL) — risk yüksek
- Derin hiyerarşik MARL (feudal RL) — karmaşıklık yüksek

---

## Belge Güncelleme Notu

Bu belge her milestone sonunda revize edilir. Milestone durumları burada güncellenir; tasarım seçenekleri ve kararlar `thesis/vision-and-scope.md` içinde belgelenir.

### Versiyon Geçmişi
- **2026-04-21:** Faz 1-3 durum dökümü + tez milestone haritası eklendi. Önceki (Kasım 2025) taslak "Faz 2 TODO" versiyonu arşivlendi (git history'den erişilebilir).
