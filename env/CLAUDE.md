<!-- Kok CLAUDE.md'den tasindi (2026-11-14). Bu dizindeki bir dosyaya dokununca yuklenir. -->

# CLAUDE.md — env/ (RL ortami, bolme, veri butunlugu)

Rota (`app/api/routes/trading.py`), HPO (`hyperparameter_optimization/`) ve `evaluation/` ayni kurallara tabi. `env/trading_env.py` degisince env bekci testleri hook ile otomatik kosar.

## State space
- **Phase 1 (RL)**: 56 features — balance(1) + shares_owned(5) + OHLCV(25) + technicals(25)
- **Phase 2 (RL + Prediction)**: +4×N features — predicted_return, predicted_direction, prediction_confidence, ensemble_agreement (N sembol başına)
- **Phase 3 (RL)**: ATR tabanlı dinamik pozisyon boyutlandırma + Kelly Criterion (opt-in, `use_atr_sizing`, `use_kelly`)

## Sembol evreni ve train/test bolmesi (ONEMLI)
- Gozlem uzayi sembol sayisina bagli: `1 + n + 5n + 5n`. 5 sembol -> 56,
  30 sembol -> 331 ozellik.
- **RL bolmesi `evaluation/splits.py`'den (Faz 9 R1):** rota ve HPO katlari `WF_V1`'den
  (`get_spec`, `fold_frames`) ve `evaluation/rl_folds.build_fold` ile alir; oranli
  `DataFetcher.split_data()` (panel uzadikca kayan sinirlar) RL'de kullanilmaz.
  Tanim DEGISMEZ (ozet testte sabit): uzatmak yeni surumdur (wf-v2). Secim yalniz
  dogrulama blogunda (10 kontrol noktasi, fazla getiri Sharpe'i); test secilen modelle
  bir kez okunur. Purge 0 kat sinirinda B0 ile olculu; odul ufku buyurse (K12)
  `tests/test_rl_walk_forward.py` kirilir. Ayrinti: Faz 9 §9.11.
- Sembollerin gecmisleri esit degilse bloklarin sembol sayisi FARKLI cikar (eskiden
  56 ozellikle egitilen model 331 ile degerlendirilip SB3 `predict`'te patliyordu).
  `build_fold` val/test'i **egitim blogunun sembol evrenine hizalar** ve dusen sembolu
  raporlar; rota `training_state["warnings"]` ile panoya tasir. Sessiz hizalama YAPMA:
  kullanici 30 sembol sandigi modeli 5 sembolle egitmis olur.
- Kalici cozum veride: eksik sembollerin tam gecmisini indir (Veri sayfasi ->
  "Yeniden Indir (tam)").

## Veri butunlugu (ONEMLI)
- **`raw_stock_data.csv` HAM veridir — TradingEnv'e vermeden once mutlaka
  `DataFetcher.clean_data()` cagir.** 2026-07-27 indirmesinde 8.155 negatif fiyat satiri
  vardi (2005 TL sadelestirmesi oncesi artefaktlar, 2000-05-10..2004-08-30); 2026-11-11 tam
  indirmesi 2005-01-03'te basliyor ve 0 satir tasiyor, ama kural kaynaktan bagimsiz.
  Cagirmazsan: negatif fiyat -> BUY'da `cost < 0` -> bakiye artar (yoktan para),
  SELL'de bakiye sinirsiz duser -> gozlemdeki `log()` NaN -> SB3 "Normal(loc: nan)" cokusu.
- `TradingEnv` artik `fiyat <= 0` veya NaN olan satirlarda **islem yapmaz** ve
  `balance_norm` tabanlidir — bozuk veri egitimi cokertemez, ama sessizce de kabul edilmez
  (kurulumda tek seferlik `GECERSIZ FIYAT` raporu basilir).
- Eksik `(sembol, tarih)` cifti yapisaldir (sembol borsaya sonradan girmis olabilir);
  uyari **sembol basina bir kez** verilir, toplam kapsam kurulumda ozetlenir.
- **RL veri kapisi (Faz 10 / V, K17): `data/integrity.py::prepare_rl_panel`.** Egitim
  rotasi ve HPO paneli buradan gecirir:
  - pencere `RL_DATA_START = 2013-06-01` (`app/core/config.py`): panelde en gec baslayan
    sembol ODAS (2013-05-21); 2000-2005 artefaktlari pencere disinda kalir
  - pencere ICINDE fiyat artefakti varsa `DataIntegrityError` — sessizce temizlenmez
  - hayalet gun (aktif satirlarin >= %90'i duz: volume=0 ve O=H=L=C) atilir; tek sembolun
    duz satiri (bayat hucre) atilir, hucre EKSIK sayilir. "= onceki kapanis" sarti ARANMAZ:
    yfinance kapali gunde duzeltilmis fiyati farkli yazabiliyor (2013-10-15 boyle kaciyordu)
  - olcum (2026-11-13, panel 2005-01-03..2026-11-09): tam panelde 117 hayalet gun, 10 artefakt
    satiri (hepsi pencere disi); pencerede 0 artefakt, 67 hayalet gun + 39 bayat hucre atiliyor,
    eksik hucre %0,115 (`docs/development/measurements/phase9-r1/folds.json` -> `gate`)
  - gosterge isinma `bfill`'i: `prepare_rl_panel(drop_indicator_warmup=True)` bfill imzali bas
    blogu atar (rota + HPO; B7, Faz 9 §9.7). Rotada 'wilder' ADX'in 27 satirlik NaN isinmasi da
    atilir (gercek panelde ODAS 17, PGSUS 24 satir; K18, §9.9). Tahmin hatti `warmup='nan'` +
    `adx='wilder'`'a Milestone 0' golden yenilemesiyle gecer (K1), oncesinde degil.
- **Tam yeniden indirme GECMISI degistirir** (yfinance auto_adjust: yeni temettu tum gecmisi
  olcekler, gecmis temettu carpanlari revize edilebilir; 2026-11-11'de SKBNK). Bir sonucu
  yeniden uretirken veriyi yeniden INDIRME; panel SHA-256'si ve anlik goruntu (C3) ile calis.
  Yedekler `data/bist/backup_<tarih>/` (git disi). Paneli tazelemenin yolu Veri sayfasi
  "Yeniden Indir (tam)". Legacy "veri uret" (`/data/generate`) panelin USTUNE 5 sembollu
  2018 paneli yaziyordu; dugme + uc kaldirildi (Faz 9 §9.13). Geri getirme.
- **Tarayici surekli yuksek ama sicramayan seviyeyi YAKALAMAZ** (TUPRS 2000-2003 > 10.000 TL
  satirlari "level_jump" degil). Bu satirlar pencere disinda; pencereyi geri cekme.
- **`TradingEnv(missing_data_mode=...)`**: `'legacy'` (varsayilan, egitilmis modeller) eksik
  hucrede ONCEKI SEMBOLUN blogunu kopyalar ve fiyati 0 sayar — bilinen hata, bit-es dondurulmus.
  `'own_last'` sembolu kendi son gozlemiyle tasir, degerlemeyi kendi son kapanisiyla yapar,
  o hucrede islem yapmaz. Yeni egitim ve HPO `own_last` kullanir.
  Test: `python tests/test_data_integrity.py` (27), `python tests/test_env_missing_mode.py` (17)
- **Val/test/degerlendirme ortami EGITIM olcegini alir (Faz 9 B9, Faz 10 Y13):**
  `TradingEnv(price_stats=None)` fiyat z-skorunu ortamin KENDI df'inden kurar — yalnizca
  egitim ortami icin dogru. Held-out pencerede kurulan her env'e
  `price_stats=<egitim env>.get_price_stats()` (ya da modelin meta `price_stats`'i) VER;
  vermezsen t gozlemi t sonrasi fiyatlara baglanir. Eksik sembol / bozuk deger `ValueError`
  (sessiz yedek yok). Meta'sinda olcek olmayan eski model icin olcek UYDURMA (K13).
  - Protokol beyan degil olcum: `evaluation.trial_registry.rl_eval_protocol(eval_env, TRAIN_ENV)`
    (rota + HPO ortak). Olcek SOZLUGUNU degil egitim ORTAMINI ver: etiket olcegin kokenini
    olcer (egitim env'i kendi penceresinden mi, pencere degerlendirmeden once mi bitiyor,
    degerler esit mi). Sabit `'train_window'` YAZMA.
  - Bedeli: eski ortamda TL fiyatlari egitim olceginin cok ustunde -> test fiyat degerlerinin
    %74,7'si +-10'da kirpik, %98,9'u egitim destegi disinda (Faz 9 §9.5). `price_in_support`
    protokol maddesi bunu makine tarafindan uygular: `price_norm='stats'` kolunun kaydi ACIK kalir.
    Olcum `evaluation/observation_health.py`; rota manifest/kayit/`_metrics.json`'a yazar.
  - **K25 (b) nedensel kol:** `TradingEnv(price_norm='causal', price_context=<onceki satirlar>)`
    — O/H/L/C sembolun <= t son 252 kapanisinin olcegiyle (MACD varsayilan eski tanimda;
    rota `TrainingRequest.price_norm='causal'` ile egitim env'ine pencere ONCESI son 252 seansi
    `price_norm_macd=True` opt-in, karar K26); rota egitim env'ine pencere ONCESI son 252 gecerli
    satiri (`integrity.rl_context_before`), test env'ine baglam + egitim + dogrulamayi verir. Baglam
    env'in ilk tarihinden once bitmeli (yoksa ValueError). Etiket `'causal'` temizdir; iki env farkli
    tanimla (mod, W, macd) -> `'mixed_norm'`, pencere dolmadan satir (seri basi haric) ->
    `'causal_short_context'`. Nedensel z'nin destek icinde kalmasi GARANTI DEGIL; olculur.
  - **Gozlem sagligi kapilari (K18):** `obs_health` (kapili bloklarda kirpma <= %1, sabit boyut
    yok, ozellik basina doygunluk `sat_frac` <= %90; egitim + degerlendirme) ve
    `features_in_support` (fiyat DISI her ozellik, havuzlu) ve `dims_in_support` (HER gozlem
    boyutu = sembol x ozellik <= %1; havuz tek sembolu gizler). Eski ortamda `tanh(turbulence/2)`
    %99,7 doygun: HER kolun kaydi `obs_health` ile acik. K26 = (a): eski ortamda DUZELTILMEZ,
    beyan edilir; olceksiz ozellikler (MACD, turbulans, hacim) M2 `obs_spec`'inde tasarlanir.
  - Sarmalayicilarda (`create_env`, `create_env_with_weights`, HPO `create_env`) `price_stats`
    ZORUNLU anahtar; egitim cagrisi `price_stats=None`'i acikca yazar.
  - AST bekcisi held-out adli df (`test_df`, `splits.test`, `data['test']`, `df.loc[test_mask]`)
    ile kurulan env cagrisinda (`make_env`, `TradingEnv`, adinda "env" gecen sarmalayicilar)
    `price_stats` arar ve `None` sabitini reddeder; bilerek eski olcekle olcum yapan satira
    `# B9-legacy` yaz. Test: `python tests/test_price_stats_window.py`
  - **HPO rotanin yolunu kullanir (R1; Y21 kapandi):** panel, kat ve ortam `evaluation.rl_folds`'tan;
    pencere istekten degil `evaluation/splits.py` katindan (varsayilan ilk kat 2021: dogrulamasi
    butun testlerden once biter). Rota HPO soyunu `hpo_lineage` ile olcer (pencere, tohum, evren,
    bolme, gozlem). Optuna sampler'i hala tohumsuz (acik, §9.11).
  - `scripts/analysis/generate_academic_report.py` D5'e kadar DURUR (cikis 2; `--allow-invalid`).
  - Olcum betigi: `scripts/analysis/measure_price_scale.py`; JSON'lar
    `docs/development/measurements/phase9-b9/` (panel SHA-256 + commit iceriyor).
  - Testte CPU'ya zorlamak icin `CUDA_VISIBLE_DEVICES=-1`: Windows'ta bos dize degiskeni SILER.
- **ADX (K18):** `stock_data_with_indicators.csv`'deki ADX legacy hesabin sabit 0'idir (indeks
  hizalama hatasi; eski modeller boyle egitildi). Rota ADX'i `with_adx(df, 'wilder')` ile
  yeniden hesaplar ve meta'ya `indicator_spec` yazar. CSV'yi yeniden URETME: eski modeller ve
  canli karar ayni dosyayi okuyor. Canli karar (`build_live_state`) hala legacy ADX ve 50/50
  fiyat olcegiyle durum kuruyor — meta'daki `indicator_spec`/`price_norm`'u okumasi K13 isi.
