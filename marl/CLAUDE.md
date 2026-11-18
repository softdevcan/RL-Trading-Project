# CLAUDE.md — marl/ (Faz 10 M2 cok ajanli ortam)

Tek gercek kaynak: `docs/development/phase-10-m2-design.md`. Muhasebe, projeksiyon, gozlem ya da odulu
degistirirsen o belgeyi AYNI commit'te guncelle; belge ile kod ayrisirsa kod hatalidir.

## Iki katman (K9)
- Saf cekirdek (`weights`, `accounting`, `orders`, `obs_spec`, `panel`): numpy/pandas. Ana `venv/` (canli
  karar) ve `venv-marl/` (egitim) ikisi de import eder. Buraya pettingzoo/torchrl/benchmarl import'u KOYMA.
- `env.py`: PettingZoo. YALNIZ `venv-marl/`. `app/` ve `dashboard/` bunu import etmez.
- `benchmarl_task.py` (M3): BenchMARL gorevi; YALNIZ `venv-marl/`. Tohum bit-esligi icin `OMP_NUM_THREADS=1`
  `MKL_NUM_THREADS=1` torch import'undan ONCE (`torch.set_num_threads` surec ortasinda yetmiyor).
- `marl/__init__.py` alt modul import ETMEZ (ana venv'de `import marl.orders` pettingzoo istemesin).
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
  -> esik marj - 0,005; marj BISTECH oncesi %20 (DOGRULANMADI), 2015-11-30 %20, 2020-03-13 %10. Kapsam
  `PRICE_MARGIN_VERIFIED_UNTIL` (2020-12-31) sonrasi HATA: 2021+ icin once marj gecmisini kaynakla dogrula, takvimi uzat.
  Ortam varsayilani `limit_down_gap='schedule'`; sabit `LIMIT_DOWN_GAP_FIXED=0.095` yalniz duyarlilik/sentetik test.
  `project(no_sell)` ve `rebalance_orders(no_sell)` ayni maskeyi alir. Canli karar `no_sell=None`. Tavan alim engeli YOK.
- **Sembol sirasi panelin gorunme sirasi**; `sorted()` uygulama.
- Legacy ADX (hep 0) `panel.build_arrays`'te reddedilir; `with_adx(df, 'wilder')` kullan.
- 2021 (m0p-v2 dogrulayici penceresi) ve sonrasi M2 gelistirmesinde basarimla OKUNMAZ.

## Test
- `python tests/test_marl_core.py` (61, ana venv)
- `venv-marl\Scripts\python tests/test_marl_env.py` (36, yalniz venv-marl)
- `venv-marl\Scripts\python tests/test_marl_m3.py` (18, yalniz venv-marl; BenchMARL gorevi, bolum sonu bootstrap, tohum, kaydet/yukle, deterministik eval, baseline, R7)
