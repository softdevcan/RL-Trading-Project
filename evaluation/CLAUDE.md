<!-- Kok CLAUDE.md'den tasindi (2026-11-14). Bu dizindeki bir dosyaya dokununca yuklenir. -->

# CLAUDE.md — evaluation/ (istatistik, deneme kaydi, protokol)

## Degerlendirme ve deneme kaydi (Faz 9 A + C)
- **Degerlendirme icin `evaluation/` paketini kullan**, `prediction/evaluator.py`'deki eski
  DSR/DM degil (onlar yonlendirildi). DSR icin N ve V deneme kaydindan
  (`evaluation/trial_registry.py`, `results_dir()/trial_registry.jsonl`) gelir — N=1'de DSR yok.
- **Deneme kaydi protokol tasir**: bilinen acik madde (test olcegi Y13, gosterge bfill Y15,
  ADX K18) varsa kayit `protocol_open` etiketi alir ve `load_trials` onu varsayilan DISLAR.
  Budanan/basarisiz denemeler de kaydedilir (N'ye sayilir). Kind/split/pencere karistirma.
  R1 maddeleri: `eval_window_unexposed` (R1 oncesi basarimla okunmus pencere,
  `EXPOSED_EVAL_WINDOWS`: 2022-07-18..2026-08-28, olculdu -> okunmamis tek kat 2021) ve
  `fundamentals_pit` (faz 2 tek anlik temel oran). Yeni bir pencereyi basarimla okuyan olcum
  yaparsan o pencereyi listeye ekle.
- **RL sonuclari on sonuctur**: eski ortamin her kolunda `obs_health` acik (turbulans; K26 = a, kalici);
  'stats' kolunda ayrica `price_in_support`/`features_in_support`. R1 (§9.11) tek bolmeyi,
  dogrulamasiz secimi ve gostergesiz HPO'yu kapatti. Kol karsilastirmasi Milestone 0'da.
- **Risksiz faiz** yalnizca `evaluation/risk_free.py`'den (varsayilan `risk_free.csv`
  `funding_rate`); dosya yoksa ya da seri yedek degerliyse acik hata — sabite dusurme.
  Eski 0,15 / 0,02 / 0 sabitleri BILEREK baglanmadi (odul = davranis dondurma; pano/rapor D'de).
- **RL tohumu**: `TrainingRequest.seed` (yoksa uretilir); meta + manifest + deneme kaydina yazilir.
- **Yon metrigi**: `BasePredictionModel.compute_metrics`'in `direction_accuracy`'si fark tabanli
  (`np.diff`) ve makaledeki TDA DEGIL; TATS karsilastirmasinda `ensemble_test_metrics['tats'][*]['tda']`
  kullan (Faz 9 §9.17).
- **Rejim etiketi** yalniz `evaluation/regimes.py`'den (B6): XU100 200g trend x 60g vol GENISLEYEN
  tercil, <= t (B0 testli). Gerceklesen getiriye kosullu "rejim" YAZMA (Y3; eski fonksiyon kaldirildi).
  **Birincil = `load_primary_labels()` = USD bazli XU100 (XU100 / USDTRY_{t-1}), genisleyen tercil**
  (kullanici karari 2026-11-15, M8 on kaydi); nominal TL ve 5y kayan tercil duyarlilik kollari. Nominal
  trend enflasyonla 'bull'a kayar (2022'de hic bear yok). Seriler `data/macro/xu100.csv`, `usdtry.csv`
  (`python -m data.index_fetcher`). `episodes` bagimsiz donem DEGIL; `episodes_min20` + blok bootstrap.
- **Kalibrasyon sifati**: `informative` kapisi (T sinirda degil, BSS > 0, AUC GA'si 0,5'i dislar) gecilmeden
  "kalibre" yazma; "sicaklik olcekli, ayirt edici degil" (F, I2). GPU'da DL tek kosu sayilari oynak (C1).
- **Baseline ve birimler (D1/D3/D4):** `evaluation/baselines.py` (1/N aylik, al-tut, XU030 fiyat endeksi — temettusuz,
  maliyetsiz; sentetik yedek YOK), `evaluation/units.py` (nominal / TUFE-reel basamak / USD = USDTRY_{t-1}). Maliyet
  `COST_GRID_BPS` gidis-donus, ana 20; tek yon bps/2 x devir. Birincil Sharpe nominal TL A8 fazla getiri; reel/USD `sharpe_rf0`.
  Eski 0,02 sabitleri kaldirildi (`ModelAnalyzer(return_dates=)`; tarihsiz -> `risk_free_source='rf0 ...'`). Odulun 0,15'i
  eski ortamda BEYAN (Ansari sadakati), G2 yeni ortamda. Test `tests/test_baselines.py` (19).
- Test: `python tests/test_eval_stats.py` (51), `python tests/test_rl_reproducibility.py` (43)
