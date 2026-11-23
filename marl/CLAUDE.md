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
- `forward_views.py` (Faz 11 S5): pano veri katmani (pencere 'forward' | 'test2026'; kol ort. +- std; urun portfoyu tam sayi
  lot). Karsilastirma YALNIZ ayni pencere ve maliyet. 2026 onbellegi `data/marl/forward/m8/cache/` (arsiv SHA anahtarli).
- `orders.step_integer` (Faz 11 S5): tam sayi modu adimi; ortam (`_step_integer`) ve defter (`LedgerState(integer=True)`) AYNI
  fonksiyonu cagirir. Kesirli yol (`step_fractional`) ayri ve S3 paritesiyle sabit.
- `forward_job.py` (Faz 11 S4): gunluk is `run_daily` (arsiv dogrula -> S1 panel -> marj kapsami -> defter); ANA venv. Defter deposu
  `data/marl/forward/m8/ledger/` (state.json atomik + rows.csv yalniz eklenir, `%.17g` yaz / `round_trip` oku). Depo YALNIZ ilk
  ileri gun (2026-11-23) panelde olunca kurulur (S6 eki §3). Idempotent ve bosluk doldurma bit-es KALMALI (test_marl_forward_job).
- `margin_coverage.py` (Faz 11 S4): ileri marj kapsami = min(panel son gunu, son kaynak denetimi + 31 gun); denetim kaydi
  `margin_source_checks.json` (git'te, ELLE eklenir; aylik). `limit_down_gaps(verified_until=None)` varsayilani sabit (ortam/M8).
- `sim_ledger.py` (Faz 11 S3): sanal para defteri, ANA venv. Ortamin AYNI cekirdegi (`step_fractional`, `fill_prices` +
  `limit_down_gaps`, `project`, A8 rf); politika `FrozenPolicy` TEK gozlemle, kural `rule_baselines` ile ayni karar. 2026
  penceresi 72/72 M8 serisiyle bit-es (`scripts/marl/s3_parity.py`). `env.py`/`rule_series` muhasebesi degisirse parite
  yeniden kosulur. `LedgerState` JSON gidis-donusu kayipsiz; kaldigi yerden devam tam kosuyla bit-es KALMALI (S4).
- `forward_panel.py` (Faz 11 S1): M8 ileri paneli, ANA venv (data/ import eder). Dondurulmus kaynak `data/marl/frozen/m8-source/`
  (`data/bist` CSV'si OKUNMAZ; uygulama onu gunceller). Panel her seferinde bastan kurulur ve onceki panelle bit-es denetlenir;
  ileri satirlar append-only (`data/marl/forward/m8/rows.csv`; YALNIZ `forward_panel.read_rows` ile oku: varsayilan `read_csv` 1 ULP saptirir, prefix bekcisi reddeder). Kapanis capraz denetimi NEDENSEL getiri kurali (m8-v1 merkezi
  medyani ileriye bakar, ileri veride KULLANMA). rf ileri deponun kendi kopyasindan; ortak risk_free.csv'ye yazma.
  Guncelleme `scripts/marl/forward_panel_m8.py --update` (gunluk is bunu cagirir). Ileri gun marj kapsami `margin_coverage` (S4).
- `frozen_policy.py` (Faz 11 S2): dondurulmus politikanin sade ileri gecisi, ANA venv (duz torch; BenchMARL/TorchRL YOK).
  `FrozenPolicy(dizin).act(obs[N, d])` = egitimdeki deterministik eylem, 72/72 bit-es (`scripts/marl/s2_parity.py`). TEK
  gozlemle cagir: toplu cagri float32 toplama sirasini degistirir (S-agr ~5e-6). Ag yapisi egitim kodundan kopya oldugu
  icin `comm_models`/`sarl`/BenchMARL MLP degisirse parite yeniden kosulur. venv-marl'da politikalari tek surecte farkli
  kollardan yukleme: `MarlTask.PORTFOLIO` yapilandirmasi surec ici tasir (single_agent).
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
- `python tests/test_marl_forward_job.py` (4, ana venv; S4: marj kapsami, her gun = bosluk sonrasi = tek seferde bit-es, idempotent, cokme kurtarma + bekciler)
- `python tests/test_marl_sim_ledger.py` (5, ana venv; S3 defter: M8/rule_series bit-es, JSON durumla devam bit-es, mutasyon, ileri panelde marj kapsami hatasi, S5 tam sayi mod)
- `python tests/test_marl_forward_panel.py` (5, ana venv; S1 ileri panel: sifir satirla bit-es, zincir, gun kapisi, rows.csv yaz/oku round_trip bit-es, append + B0 + mutasyon)
- `python tests/test_marl_frozen.py` (4, ana venv; Faz 11 S0 arsivi: kur/dogrula, bozulma yakalanir, gercek arsiv 72/72; S2 sade ileri gecis referansla bit-es)
- `python tests/test_marl_baselines.py` (7, ana venv; M8 A4 kural portfoyleri: LW/SLSQP paritesi, HRP, ileriye bakmama (e), ay basi dengeleme)
- `python scripts/marl/analyze_forward_m8.py --self-test` (9, ana venv; S6 ileri veri okumasi; kural ek `phase-11-m8-forward-annex.md`, koda ustun)
- `python scripts/marl/analyze_m8.py --self-test` (11, ana venv; M8 okuma + `--status` saglik kaydi)
- `venv-marl\Scripts\python tests/test_marl_env.py` (36, yalniz venv-marl)
- `venv-marl\Scripts\python tests/test_marl_m3.py` (20, yalniz venv-marl; BenchMARL gorevi, bolum sonu bootstrap, tohum, kaydet/yukle, deterministik eval, baseline, R7, M8 kontrol noktalari)
- `venv-marl\Scripts\python tests/test_marl_sarl.py` (4, yalniz venv-marl; SARL muhasebe paritesi + takim odulu, toplu yol, esmimari esitligi, kisa MAPPO + tohum + kaydet/yukle)
- `venv-marl\Scripts\python tests/test_marl_m4.py` (20, yalniz venv-marl; `comm_models` attn/gat/self: graf maskesi, esit butce, gat'te komsu disi etkisiz, self iletisimsiz, tohum, kaydet/yukle)
- `venv-marl\Scripts\python tests/test_marl_batched_env.py` (4, yalniz venv-marl; toplu ortam paritesi)
- `venv-marl\Scripts\python tests/test_marl_m5_msg.py` (20, yalniz venv-marl; S-msg conf_mode, beta, gecit, karistirilmis guven, GLM gonderici sicakligi; 'none' M4a ile bit-es)
