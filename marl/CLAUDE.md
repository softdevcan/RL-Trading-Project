# CLAUDE.md — marl/ (Faz 10 M2 cok ajanli ortam)

Tek gercek kaynak: `docs/development/phase-10-m2-design.md`. Muhasebe, projeksiyon, gozlem ya da odulu
degistirirsen o belgeyi AYNI commit'te guncelle; belge ile kod ayrisirsa kod hatalidir.

## Iki katman (K9)
- Saf cekirdek (`weights`, `accounting`, `orders`, `obs_spec`, `panel`, `selection`, `rule_baselines`): numpy/pandas. Ana `venv/` (canli
  karar) ve `venv-marl/` (egitim) ikisi de import eder. Buraya pettingzoo/torchrl/benchmarl import'u KOYMA.
- `env.py`: PettingZoo. YALNIZ `venv-marl/`. `app/` ve `dashboard/` bunu import etmez.
- `comm_models.py` (M4, M5): attn/gat/self politika modelleri + S-msg guven terimi; YALNIZ `venv-marl/`. gat = attn + sabit graf maskesi, self = kendine dikkat (ayni parametre); `conf_mode='none'` M4a koşularindaki modelle bit-es KALMALI. `conf_temp` (GLM) yalniz guven terimini bozar; gozlem/mesaj icerigi degismez. `set_sender_mask` (M7a) yalniz DEGERLENDIRME araci: kalici degil, durum sozlugune yazilmaz; None iken ileri gecis bit-es. Tasarim `docs/development/phase-10-m4-design.md`, `phase-10-m5-design.md` §3.
- `benchmarl_task.py` (M3): BenchMARL gorevi; YALNIZ `venv-marl/`. Tohum bit-esligi icin `OMP_NUM_THREADS=1`
  `MKL_NUM_THREADS=1` torch import'undan ONCE (`torch.set_num_threads` surec ortasinda yetmiyor).
- `batched_env.py` (Faz 10 §3.2.8): B tane `MarlPortfolioEnv`'i tek surecte adimlayan TorchRL ortami; YALNIZ `venv-marl/`. Cekirdek AYNI fonksiyonlar, parite bit-es (`test_marl_batched_env`). Tohum duzeni `PettingZooWrapper._set_seed` ile ayni (kurulumda reset(seed + i), sonra tohumsuz). Gorevde `batched=True`, egitici `--batched-env`; varsayilan kapali.
- `panel_fix.py` (M8, tasarim §6.1): panel veri duzeltmesi `m8-v1`; YALNIZ ana venv (data/ import eder), ortam ve
  canli karar import ETMEZ. Yalniz `prepare_panel_m2.py --data-fix m8-v1` cagirir; varsayilan `none` M2 paneliyle
  bit-es KALMALI. Ikinci kaynak onbellegi `data/marl/sources/isyatirim/` (git disi; `cross_source_audit_m8.py --fetch`).
- `sarl.py` (M8 tasarimi A2, K15): SARL kollari; YALNIZ venv-marl. `SarlPortfolioEnv` ayni cekirdek ustunde tek-ajan
  gorunumu (ajan `agent_0`, grup 'agent'; odul = ajan odullerinin ortalamasi); `RowSharedPolicy` (esmimari) MARL ajan
  MLP'siyle ayni katmanlar. Gorevde `single_agent=True`, egitici `--sarl {agirlik, esmimari}`; varsayilan MARL (bit-es).
- `marl/__init__.py` alt modul import ETMEZ (ana venv'de `import marl.orders` pettingzoo istemesin).
- `frozen.py` (Faz 11 S0): dondurulmus M8 politika arsivi `data/marl/frozen/<ad>/` (git disi, ortak, salt okunur) ve
  `verify_archive` (dosya SHA + `params_sha` + on kayit listesi). ANA venv (torch tembel import; BenchMARL yok). `state_digest`
  TEK tanim burada; `m3_runner` ithal eder. Arsiv ustune yazilmaz; gunluk is (S4) her calismadan once dogrular.
- Eski ortam `env/trading_env.py` ve state space'i DEGISMEZ; yeni ortam oraya konmadi (tasarim §1).

## Kurallar
- **Ortam ve canli karar AYNI fonksiyonu cagirir**: gozlem `obs_spec.build_obs_at` (K2), hedef
  `weights.target_weights` + `weights.project` (K6), emir `orders.rebalance_orders` (K5). Kopya mantik
  yazma; parite testleri bit-eslik ister.
- **`norm_stats` zorunlu**: `'fit'` yalniz `mode='train'`; held-out ortama egitim ortaminin `norm_stats`'i
  verilir (B9/Y13 dersi). None ya da eval'de 'fit' -> ValueError. Sessiz varsayilan ekleme.
- **rf zorunlu** (`arrays['rf']`, A8 `evaluation.risk_free`): yoksa hata; 0 ya da 0,15 sabitine DUSME.
- **Maliyet TEK YON** `c = bps_gidis_donus / 2 / 1e4` (`evaluation/baselines.py` D4 ile ayni); devir yalniz
  hisseler, nakit devre girmez; maliyet bir kez sayilir.
- **Logit siniri `5 + ln N`** (kullanici karari 2026-11-16); nakit logiti sabit 0. Sabit 5'e geri donme.
- **Eksik hucre atilmaz**: `fill_prices` tasir, hisse o gun dondurulur. Gozleme t+1 gecerliligi KONMAZ
  (t+1 acilisinda bilinir).
- **Birincil odul `'excess'`** (kullanici karari 2026-11-16): alpha onelemesi yalniz burada; `'differential'` yalniz alpha=0
  ablasyonu, kodda zorlanir. Gevsetme.
- **Listeleme kilidi** `LISTING_LOCKOUT=20` seans (`fill_prices`, ortam `listing_lockout`); canli karar da ayni
  `tradable` maskesini kullanir. Fiyatlar yfinance duzeltilmis: tam sayi modunda lot/fiyat seviyesi iddiasi yok.
- **Taban acilista satis engeli, DONEM BAGIMLI marj** (tasarim §2.5 ON KAYIT v0.7; hakem-2 H1): `limit_down_gaps(dates)`
  -> esik marj - 0,005; marj BISTECH oncesi %20 (DOGRULANMADI), 2015-11-30 %20, 2020-03-13'ten bugune %10 (Borsa
  Istanbul 2020/20; 2021+ yeni giris yok, §2.5 2026-11-20). Kapsam `PRICE_MARGIN_VERIFIED_UNTIL` (2026-11-09) sonrasi
  HATA: veri yenilenince kaynagi yeniden denetle, `scripts/marl/price_limit_audit_m8.py`'yi kos, sonra tarihi uzat.
  Ortam varsayilani `limit_down_gap='schedule'`; sabit `LIMIT_DOWN_GAP_FIXED=0.095` yalniz duyarlilik/sentetik test.
  `project(no_sell)` ve `rebalance_orders(no_sell)` ayni maskeyi alir. Canli karar `no_sell=None`. Tavan alim engeli YOK.
- **Sembol sirasi panelin gorunme sirasi**; `sorted()` uygulama.
- Legacy ADX (hep 0) `panel.build_arrays`'te reddedilir; `with_adx(df, 'wilder')` kullan.
- 2021 (m0p-v2 dogrulayici penceresi) ve sonrasi M2 gelistirmesinde basarimla OKUNMAZ.

## Test
- **M8 kural portfoyleri** (`rule_baselines.py`): `m3_runner.rule_series` buradan yeniden ihrac edilir; ew/bh/cash davranisi
  `test_marl_m3` baseline testiyle sabit. minvar/hrp yalniz <= t satirlarini okur (kesilmis panel testi).
- **M8 kat degerlendirmesi** (M8 tasarimi A1): `train_m3.py --checkpoints K` yalniz politika agirligini yazar (egitim
  kontrol noktasiz kosuyla bit-es KALMALI); `eval_fold_m8.py` testi YALNIZ secilen noktayla BIR KEZ okur, cikti varsa
  durur. Secim `selection.py` = `evaluation/rl_folds.py` (parite testi); tanim degisirse ikisi birlikte degisir.
- `python tests/test_marl_core.py` (63, ana venv)
- `python tests/test_marl_frozen.py` (3, ana venv; Faz 11 S0 arsivi: kur/dogrula, bozulma yakalanir, gercek arsiv 72/72)
- `python tests/test_marl_baselines.py` (7, ana venv; M8 A4 kural portfoyleri: LW/SLSQP paritesi, HRP, ileriye bakmama (e), ay basi dengeleme)
- `python scripts/marl/analyze_m8.py --self-test` (11, ana venv; M8 okuma + `--status` saglik kaydi)
- `venv-marl\Scripts\python tests/test_marl_env.py` (36, yalniz venv-marl)
- `venv-marl\Scripts\python tests/test_marl_m3.py` (20, yalniz venv-marl; BenchMARL gorevi, bolum sonu bootstrap, tohum, kaydet/yukle, deterministik eval, baseline, R7, M8 kontrol noktalari)
- `venv-marl\Scripts\python tests/test_marl_sarl.py` (4, yalniz venv-marl; SARL muhasebe paritesi + takim odulu, toplu yol, esmimari esitligi, kisa MAPPO + tohum + kaydet/yukle)
- `venv-marl\Scripts\python tests/test_marl_m4.py` (20, yalniz venv-marl; `comm_models` attn/gat/self: graf maskesi, esit butce, gat'te komsu disi etkisiz, self iletisimsiz, tohum, kaydet/yukle)
- `venv-marl\Scripts\python tests/test_marl_batched_env.py` (4, yalniz venv-marl; toplu ortam paritesi)
- `venv-marl\Scripts\python tests/test_marl_m5_msg.py` (20, yalniz venv-marl; S-msg conf_mode, beta, gecit, karistirilmis guven, GLM gonderici sicakligi; 'none' M4a ile bit-es)
