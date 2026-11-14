<!-- Kok CLAUDE.md'den tasindi (2026-11-14). Bu dizindeki bir dosyaya dokununca yuklenir. -->

# CLAUDE.md — data/ (veri hatti)

**RL veri kapisi, ham panel, tam yeniden indirme ve fiyat olcegi kurallari `env/CLAUDE.md` -> "Veri butunlugu"** (RL tarafinda tek kaynak).

## Data pipeline
```
# RL pipeline
yfinance → data_fetcher.py → technical_indicators.py → trading_env.py → SB3 model

# Tahmin pipeline
yfinance       → data_fetcher.py       ─┐
TCMB EVDS      → macro_fetcher.py      ─┤→ feature_engineer.py → feature_selector.py
yfinance       → fundamental_fetcher.py─┤
borsapy/yf     → gold_fetcher.py       ─┘
                                         → trainer.py (walk-forward + purge/embargo)
                                         → models/ (XGB + LGBM + CatBoost + BiLSTM + TFT)
                                         → ensemble.py (stacking meta-learner)
                                         → evaluator.py → trading_env.py (RL obs space)
```

## Sembol evreni (Faz 9 E0, Y11/K7)
- Tek kaynak `data/universe/bist30_panel.json` (RL panelinin 30 hissesi, panel sirasiyla; GERCEK
  BIST-30 uyeligi DEGIL). `BIST30_SYMBOLS`, `STOCK_INFO`, pano listeleri ve egitim rotasinin varsayilan
  cekimi buradan. Elle liste yazma; evren degisirse `scripts/freeze_universe.py` YENI surum adiyla.

## data_fetcher.py özellikleri
- `fetch_stock_data()`: Retry (exponential backoff) + class-level cache + coverage check (%80)
- `fetch_incremental()`: Sadece eksik günleri çeker, mevcut CSV'ye append eder
- `get_source_status()`: Son tarih ve eksik gün sayısını raporlar
- `clean_data(leading='bfill')`: ffill → bas NaN satirlar ('bfill' varsayilan: gelecek degerle
  doldur | 'drop': at; B7) → negatif fiyat temizleme, sembol bazli. Ham panelde bas NaN yok

## macro_fetcher.py özellikleri
- EVDS (Faz 9 B2, 2026-11-06'da EVDS kataloğundan doğrulandı):
  `policy_rate` = TP.BISPOLFAIZ.TUR (aylık), TÜFE = TP.TUKFIY2025.GENEL, ÜFE = TP.TUFE1YI.T1
  (ikisi de aylık, yıllık % değişim). **Eski kodlar (TP.YSSK.A01, TP.FG.G0) EVDS'de YOK**;
  TP.FG.J0 (2003=100) 2026-01'de bitiyor. Bu yüzden Temmuz 2026'ya kadarki her çekim faizde
  sabit 50'ye, TÜFE/ÜFE'de 0'a düşüyordu — 50 hiçbir zaman EVDS değeri değildi.
- **EVDS tuzakları:** sütun adı alt çizgili döner (`TP_FG_J0`); aylık `Tarih` 'YYYY-M';
  tek istekte en fazla 1000 satır, fazlası aralığın BAŞINDAN kesilir (`_evds_series` parçalar).
- **Nokta-zamanlı (B3):** ay m TÜFE/ÜFE'si ay m+1'in 6'sından, ay m politika faizi ay m+1'in
  1'inden itibaren geçerli (tasarım varsayımı). **`bfill` YOK (B1)**; baş boşluk atılır,
  bitiş tarihinden sonrası yazılmaz. `.bfill()` geri getirme — `tests/test_macro_pit.py`
  ilk satırın değerini DOĞRUDAN denetler (kesme değişmezliği bfill'i YAKALAMAZ).
- Olay zamanlı, **vintage değil**: BIS aylık faizi ~5 hafta gecikmeli; geçmişte ay m−1, canlıda
  m−2 görünür (açık iş, Faz 9 §9.4 #4).
- Artımlı çekim EVDS'i 120 gün geriye bakarak çeker; yedek değerli delta dosyaya YAZILMAZ.
  Kalite bayraklı mevcut dosya artımlı değil TAM çekimle yenilenir.
- **Risksiz faiz ayrı dosyada:** `data/macro/risk_free.csv` (`fetch_risk_free()`; TP.APIFON4
  TCMB ağırlıklı ortalama fonlama maliyeti, iş günü, 2011'den). Veri sayfası makro güncellemesi
  bunu da yeniler. Yedeği yok.
- yfinance: usd_try, eur_try, bist100_index, **vix**, **us10y**, **dxy** (Faz 3.2'de eklendi)
- **Makro 2018'den başlıyor** (varsayılan `start_date`); RL penceresi 2013-06. Faz 2 RL ortamı
  2013–2017'de makro gözlemi 0 görür — bilinen açık.
