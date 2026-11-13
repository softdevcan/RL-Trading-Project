# RL Trading System — BIST-30

BIST-30 hisseleri için derin pekiştirmeli öğrenme (DRL) tabanlı algoritmik işlem sistemi ve tahmin
destekli gözlem uzayı. Ansari vd. (2024) *A Multifaceted Approach to Stock Market Trading Using
Reinforcement Learning* metodolojisinin BIST-30'a uyarlanmasıyla başladı; bir yüksek lisans tezi ve
makale çalışmasının deney altyapısıdır.

> **Durum (v1.9.0, 2026-11):** Faz 9 — araştırma geçerliliği sürüyor. RL ve tahmin sonuçları
> **ön sonuçtur**: bilinen ve kayıt altındaki metodolojik açıklar kapatılıyor (aşağıda). Yatırım
> tavsiyesi değildir.

## Özellikler

- **RL ortamı** (Gymnasium): çok hisseli işlem, A2C / PPO / TD3 / SAC (Stable-Baselines3); MACD, RSI,
  CCI, ADX, türbülans; opsiyonel ATR pozisyon boyutlandırma ve Kelly
- **Değerlendirme protokolü:** sabit tarihli walk-forward bölme (`wf-v1`, 6 kat), doğrulamada kontrol
  noktası seçimi, ortak HPO yolu (Optuna); deneme kaydı ve protokol etiketleri
- **Tahmin sistemi:** XGBoost, LightGBM, CatBoost, BiLSTM, TFT + stacking meta-öğrenici; walk-forward CV
  (purge/embargo, kat içi özellik seçimi); TATS trend düzeltici (Kazemdehbashi 2026 uyarlaması);
  ICEEMDAN; SHAP
- **Veri:** yfinance (OHLCV), TCMB EVDS (politika faizi, TÜFE/ÜFE — nokta-zamanlı, açıklanma gecikmeli),
  risksiz faiz (TCMB ağırlıklı ortalama fonlama), VIX / US10Y / DXY, altın/döviz; veri bütünlüğü kapısı
- **İstatistik** (`evaluation/`): PSR/DSR, Diebold-Mariano, Ledoit-Wolf Sharpe farkı, MCS, rliable,
  PBO/CSCV, BH-FDR; tohum, veri anlık görüntüsü (SHA-256), ortam kaydı
- **Uygulama:** FastAPI backend + Dash pano (`/dash/`); çerez tabanlı oturum, roller
  (admin / user / viewer), kullanıcı başına çalışma alanı; kâğıt portföy ve günlük karar; açık/koyu tema

## Kurulum

```bash
git clone https://github.com/softdevcan/RL-Trading-Project.git
cd RL-Trading-Project
python -m venv venv
# Windows: .\venv\Scripts\activate   |   Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # en az EVDS_API_KEY (TCMB EVDS makro verisi); üretimde .env.production.example
```

GPU'lu PyTorch için CUDA sürümünüze uygun `torch` paketini ayrıca kurun.

## Çalıştırma

```bash
python scripts/create_admin.py --email admin@ornek.com   # ilk yönetici hesabı (kayıt yok; hesapları admin açar)
python run_server.py                                      # http://localhost:8000
```

- Pano: **http://localhost:8000/dash/** (giriş `/login`)
- API belgesi: `/docs` (yalnız admin)
- Docker: `make up` (ayrıntı `make help`; yönetici `make admin`)

Kimlik doğrulama varsayılan olarak açıktır; `AUTH_ENABLED=False` ile sunucu herkese açık hâle gelir,
yalnız yerel denemede kullanın.

## Testler

Testler bağımsız betiklerdir (pytest değil): `python tests/<ad>.py`. Hangi testin neyi denetlediği ve
beklenen kontrol sayıları: [tests/CLAUDE.md](tests/CLAUDE.md). CPU'da koşmak için
`CUDA_VISIBLE_DEVICES=-1`.

Tahmin golden testi (`tests/test_prediction_regression.py`) DL hizalama düzeltmesinden (B11) beri
**bilinen nedenle kırmızıdır**; Milestone 0' ile bir kez yenilenecektir.

## Proje yapısı

| Dizin | İçerik |
|---|---|
| `env/` | RL ortamı (Gymnasium) ve ödül — *sanal ortam `venv/` ile karıştırmayın* |
| `evaluation/` | İstatistiksel testler, walk-forward bölme, deneme kaydı, tekrar üretilebilirlik |
| `prediction/` | Özellik mühendisliği, modeller, ensemble, TATS, CV, HPO |
| `data/` | Veri çekme (yfinance, EVDS), göstergeler, veri bütünlüğü kapısı |
| `app/` | FastAPI: rotalar, kimlik/yetki, çalışma alanı, servisler |
| `dashboard/` | Dash pano (sayfalar, bileşenler, tema) |
| `hyperparameter_optimization/` | RL HPO (Optuna) |
| `scripts/` | Ölçüm, analiz ve bakım betikleri |
| `docs/` | Plan, ölçümler, kılavuzlar, tez — indeks [docs/README.md](docs/README.md) |

`models/`, `results/`, `logs/`, `workspaces/` git dışındadır (eğitilmiş modeller ve kullanıcı yapıtları).

## Araştırma geçerliliği

Sonuçların hangi koşulda savunulabildiği planlarda ve ölçüm JSON'larında izlenir:

- [Faz 9 planı](docs/development/phase-9-research-validity-plan.md) — iş paketleri ve §9 uygulama
  durumu (her paketin ölçümü, hakem paneli, açık kalanlar)
- [Faz 10 planı](docs/development/phase-10-marl-plan.md) — MARL (çok ajanlı) yol haritası
- [Bilinen metodolojik açıklar](docs/development/roadmap.md) — "Bilinen Metodolojik Açıklar" bölümü
- Ölçümler: [docs/development/measurements/](docs/development/measurements/) (panel SHA-256 ve commit içerir)

Akademik rapor betiği (`scripts/analysis/generate_academic_report.py`) baseline ve raporlama paketi (D5)
tamamlanana kadar bilerek durur (çıkış 2).

## Sürümleme

Tek kaynak `app/__init__.py` (`__version__`, SemVer). Her birleştirme sonrası tag ve GitHub release;
tez/makale sonuçları adlandırılmış tag alır. Ayrıntı: [docs/development/versioning.md](docs/development/versioning.md).

## Dokümantasyon

- Kılavuzlar: [algoritmalar](docs/guides/ALGORITHMS.md) · [akademik analiz](docs/guides/ACADEMIC_GUIDE.md) ·
  [HPO](docs/guides/API_HYPEROPT_GUIDE.md) · [GPU performansı](docs/guides/GPU_PERFORMANCE_GUIDE.md)
- Geliştirme: [yol haritası](docs/development/roadmap.md) ·
  [tahmin sistemi](docs/development/prediction-system.md) · [hakem paneli](docs/development/review-panel.md)
- Referans: [tez (final)](docs/reference/tez-final.pdf) · arşiv [docs/archive/](docs/archive/)

## Referanslar

- Ansari vd. (2024). *A Multifaceted Approach to Stock Market Trading Using Reinforcement Learning.* IEEE Access.
- Kazemdehbashi (2026). *Trend-Adjusted Time Series Models with an Application to Gold Price Forecasting.*
  arXiv:2601.12706 — TATS uyarlamasının kaynağı (yöntem farkları: Faz 9 planı §9.17).
- Tezin doğrulanmış kaynakçası: [docs/thesis/](docs/thesis/).

## Uyarı

Bu proje akademik ve eğitim amaçlıdır. Gerçek parayla işlem için kullanılmamalıdır; **yatırım tavsiyesi
değildir**.

## Lisans

Lisans dosyası henüz eklenmedi.
