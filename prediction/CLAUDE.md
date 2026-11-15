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

## Yon olasiligi ve kalibrasyon (Faz 9 F1-F4)
- `StackingEnsemble(confidence_mode='legacy'|'calibrated')`, varsayilan 'legacy' (davranis dondurma):
  yon fiyat isaretinden, guven DL yon baslarinin ortalamasindan max(p,1-p) — Y4: ikisi ZIT yonu
  gosterebilir (catisma orani test kalibrasyonunda `head_vs_price_conflict_rate`).
- 'calibrated': p(UP) meta duzeyde lojistik (girdi: taban getiri tahminleri + DL yon baslari), OOF meta
  bolumunun ILK yarisinda; sicaklik T IKINCI yarida; yarilar arasinda ve sonda h arindirma; test yalniz
  degerlendirme. predict_next `calibration`: `p_up_raw`, `p_up_cal` (F4), `direction_price`,
  `direction_conflict`; yon = 1[p_cal>=0,5]. Fiyat tahmini iki modda AYNI. Mod model meta'sindan yuklenir.
- Her egitimde `ensemble_test_metrics['calibration']` (iki yol da): ECE 15 kutu, Brier, guvenilirlik
  (`evaluation/calibration.py`). `compute_calibration` (5 kutu, tracker) bununla KARISTIRILMAZ.
- Sinir: yon modeli %60 modellerinin OOF'unda uyar, dagitilan modeller %80 (K19/Y26) — P'de cozulur.
  Sicaklik sembol basina; plandaki havuzlu T (Faz 10 §3.1.2) P'de (batch). Test: `tests/test_calibration.py`.

## evaluator.py metrikleri (Faz 3)
- Sortino Ratio, Calmar Ratio, Deflated Sharpe Ratio (Bailey & Lopez de Prado), Turnover

## Davranis dondurma ve golden
- Golden: `tests/test_prediction_regression.py`. **B11'den beri bilinen nedenle kirmizi** (Faz 9 §9.18,
  K1 secenek 1); Milestone 0' ile bir kez yenilenir — o zamana kadar `--update` yapma, golden'i kanit sayma.
- **DL perf knob'lari (Faz 6) varsayilan KAPALI (opt-in)**: tam hatta RNG sirasini kaydirip golden'i
  degistiriyorlar (GPU-preload, TFT gruplanmis-VSN, AMP olculup reddedildi). Sifirdan yeniden egitimde
  acilir, golden o donanimda yenilenir. Ayrinti `docs/development/phase-6-backend-performance.md`.

## Canli tahmin = son satir (Y28 kapandi, Faz 9 §9.19)
- Cikarim yolu (`predict_next`, SHAP ucu) `build_inference_features` kullanir: hedefi NaN olan son h
  satir TUTULUR, son ozellik satiri = df'in son tarihi. Egitim/CV/HPO `build_features` (dropna'li) — yeni
  bir cikarim yolu yazarken `build_features` KULLANMA, bir ufuk bayat olur. Test: `test_tats_labels` (Y28).
- `predict_next` ciktisi ve tracker kaydi `feature_date` + `stale_features` tasir; `feature_date`'siz kayit Y28 oncesi,
  `stale_features=True` kayit son gun ozelligi NaN (ikisi de bayat, canli dogruluga girmez).
- **Y30 ACIK:** SHAP ucu `predict_next`'ten farkli matris kuruyor (makro/capraz/temel/feature_groups yok).
- **Y31 ACIK:** tracker `GOLD_GRAM_TRY` gerceklesen kapanisini `GC=F`'ten (ons, USD) cekiyor; tahmin gram/TRY ->
  o sembolde hata ve yon dogrulugu anlamsiz (Faz 9 §9.20). Duzeltme: tahmini ureten veri hattindan cek.
- Tracker kurallari tek yerde: `prediction/tracker_rules.py` (Y29 kapandi, §9.20). `prediction_date` =
  `feature_date` + h is gunu (tahmini; BIST tatilleri bilinmiyor). Gerceklesen kapanis = `feature_date`'ten
  SONRAKI h. GERCEK seans (`actual_date` kaydedilir; NaN taslak satir seans sayilmaz). Gecersiz kayit
  (`invalid_reason`) degerlendirilmez ve `get_performance_history`'den (kayan metrik, ozet, grafik) duser.
  `compute_calibration` filtrelenmez (F3'te egitim tahminleri de kullanir).
