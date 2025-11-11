# Proje: DRL ile Algoritmik Ticaret - Teknik Yol Haritası

Bu belge, "Deep Reinforcement Learning ile Otomatik Hisse Senedi Alım-Satım Stratejisi Geliştirme" projesinin teknik mimarisini ve kodlama görevlerini özetlemektedir.

## 1. Teknik Hedef

[cite_start]**`Ansari et al.`**  [cite_start]makalesinde sunulan "çok yönlü" (multifaceted) veri yaklaşımını ve **`Proje Önerisi`** [cite: 2059-2144] belgesinde belirtilen algoritmaları (A2C, PPO, TD3) kullanarak BIST-30 endeksi için bir alım-satım ajanı (agent) geliştirmek.

* **Ara Rapor (Demo) Hedefi:** BIST-30 verisiyle çalışan özel bir `TradingEnv` (Gym Ortamı) oluşturmak ve bir DRL ajanı (örn. A2C) ile çökmeden çalıştığını (proof-of-concept) kanıtlamak.
* [cite_start]**Final Proje Hedefi:** `Ansari`'nin  PSR ödül fonksiyonunu ve temel analiz verilerini entegre etmek, birden fazla modeli eğitmek ve bunları bir API üzerinden sunmaktır.

## 2. Ana Teknolojiler (Stack)

* **Programlama Dili:** Python 3.10+
* **Veri Analizi:** Pandas, NumPy
* **Veri Çekme:** `yfinance` (veya BIST-30 için alternatif API'ler)
* **DRL Ortamı:** `OpenAI Gym` (yeni `gymnasium`)
* [cite_start]**DRL Algoritmaları:** `Stable-Baselines3` [cite: 2089] [cite_start](A2C, PPO, DDPG/TD3 [cite: 2082-2084] için)
* [cite_start]**Finans Kütüphanesi:** `FinRL` (opsiyonel, ilham almak için) [cite: 2088]
* **API Sunucusu (Final):** `FastAPI`, `Uvicorn`
* [cite_start]**Görselleştirme:** `Matplotlib` [cite: 2092]

## 3. Proje Mimarisi (Final Hedefi)

Proje, iki ana hattan oluşacaktır: **Eğitim** ve **Çıkarım (Inference)**.

```text
1. EĞİTİM (Offline)
   [BIST-30 Verisi (CSV)] -> [data_pipeline.py] -> [trading_env.py] -> [train.py] -> [Kayıtlı Model (model.zip)]

2. ÇIKARIM (Canlı - API ile)
   [Canlı Borsa Verisi] -> [İSTEMCİ (Bot)] -> POST /predict -> [FastAPI Sunucusu (main.py) + model.zip] -> Dönen JSON {eylem} -> [İSTEMCİ] -> [Emir İletimi]