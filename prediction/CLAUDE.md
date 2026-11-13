<!-- Kok CLAUDE.md'den tasindi (2026-11-14). Bu dizindeki bir dosyaya dokununca yuklenir. -->

# CLAUDE.md — prediction/ (tahmin hatti)

## ensemble.py özellikleri (Faz 3)
- 3-way chronological split (60/20/20): base train / OOF meta-train / final test.
  **"Leakage yok" DEGIL:** dagitim oncesi %80 yeniden uyumda meta-ogrenici orneklem ICI
  tahminlerle yeniden uyduruluyor (ensemble.py:345–384; Faz 10 §8 #7 → K19)
- Meta-learner: Ridge (default) veya XGBoost (`meta_learner_type='xgboost'`)
- TATS düzeltici: `use_tats=True` ile trend-adjusted output
- Direction head: BiLSTM/TFT'nin `_predict_direction_raw()` çıktısı confidence hesabında kullanılır
- TATS (Faz 9 B10 + K21): uyum girdileri HAM `df['close']`'tan (`_tats_inputs`; ICEEMDAN kolunda
  `feat_df['close']` filtreli); test metriginde `ensemble_test_metrics['tats']` (TATS'li/TATS'siz);
  `<model>_<ufuk>_tats.json` ile kaydedilir/yuklenir. Duzeltme ISARETI KORUR (makaledeki TDA degismez; `direction_accuracy` fark tabanli, TDA DEGIL,
  degisebilir; TATS blogunda `tda` ayri) — makaledeki gibi yonu
  cevirmez (§9.17 yontem farklari); 'TATS' kollari bu uyarlamayi olcer
- BiLSTM/TFT egitim penceresi satir i'yi icerir (B11, Y22 kapandi); yon basi etiketi ham hedeften,
  model `target_type` ozniteligini ensemble/trainer'dan alir

## evaluator.py metrikleri (Faz 3)
- Sortino Ratio, Calmar Ratio, Deflated Sharpe Ratio (Bailey & Lopez de Prado), Turnover

## Davranis dondurma ve golden
- Golden: `tests/test_prediction_regression.py`. **B11'den beri bilinen nedenle kirmizi** (Faz 9 §9.18,
  K1 secenek 1); Milestone 0' ile bir kez yenilenir — o zamana kadar `--update` yapma, golden'i kanit sayma.
- **DL perf knob'lari (Faz 6) varsayilan KAPALI (opt-in)**: tam hatta RNG sirasini kaydirip golden'i
  degistiriyorlar (GPU-preload, TFT gruplanmis-VSN, AMP olculup reddedildi). Sifirdan yeniden egitimde
  acilir, golden o donanimda yenilenir. Ayrinti `docs/development/phase-6-backend-performance.md`.
