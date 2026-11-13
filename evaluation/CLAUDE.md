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
- Test: `python tests/test_eval_stats.py` (51), `python tests/test_rl_reproducibility.py` (39)
