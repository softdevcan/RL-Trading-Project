"""
Faz 9 / D1, D3, D4 + A8 sabitleri (`evaluation/baselines.py`, `evaluation/units.py`).

Kontroller:
  1. D1 1/N aylik: 2 sembol, 2 ay elle hesap — gunluk getiri, ay basi devir, kayan agirlik.
  2. D1 al-tut: devir yalniz ilk gun; son deger = fiyat oranlarinin ortalamasi (esit ilk agirlik).
  3. Gec listelenen sembol: 1/N'ye bir sonraki ay basinda girer, al-tut'a hic girmez; eksik hucre
     getiri 0 ile tasinir.
  4. D4: net = brut - bps/2/1e4 * devir; ilk gun 20 bps -> -0,001; izgara monoton.
  5. D3: reel = nominal / ayin TUFE'si (basamak); USD = nominal / USDTRY_{t-1}; ayni gun kur
     mutasyonu farkli sonuc verir; eksik deflator UnitUnavailable.
  6. A8: model_analysis / eski benchmark / rapor betiklerinde 0,02 sabiti yok; return_dates verilince
     fazla getiri A8 serisinden; tarihsiz rf0 etiketli.
  7. Gercek veri (dosyalar varsa): 2021 penceresi, uc birim, XU030, maliyet arttikca Sharpe duser.

Ag yok. ~10 sn.
    python tests/test_baselines.py
"""

import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from evaluation import baselines as B  # noqa: E402
from evaluation import units as U  # noqa: E402

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [OK]   {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


print("\n[1] 1/N aylik (elle)")
d = pd.DatetimeIndex(['2021-01-28', '2021-01-29', '2021-02-01', '2021-02-02'])
close = pd.DataFrame({'A': [10.0, 11.0, 11.0, 12.1], 'B': [20.0, 20.0, 18.0, 18.0]}, index=d)
ew = B.equal_weight(close, '2021-01-01', rebalance='monthly')
# gun1: rA=0.1, rB=0 -> rp=0.05; w=(0.55,0.5)/1.05 = (0.5238, 0.4762)
# gun2 (Subat ilk seans): rA=0, rB=-0.1 -> rp = -0.04762; kayan w=(0.55,0.45)/1.05/0.95238...
w1 = np.array([0.55, 0.5]) / 1.05
rp2 = float(w1 @ np.array([0.0, -0.1]))
w2 = w1 * np.array([1.0, 0.9]) / (1 + rp2)
turn2 = float(np.abs(np.array([0.5, 0.5]) - w2).sum())
check("gunluk brut getiri elle hesapla ayni",
      np.allclose(ew['gross'].to_numpy(), [0.0, 0.05, rp2, 0.5 * 0.1]), str(ew['gross'].tolist()))
check("devir: ilk gun 1, Subat ilk seansi |1/N - kayan w|, digerleri 0",
      np.allclose(ew['turnover'].to_numpy(), [1.0, 0.0, turn2, 0.0]), str(ew['turnover'].tolist()))

print("\n[2] al-tut")
bh = B.equal_weight(close, '2021-01-01', rebalance='none')
v = (1 + bh['gross']).prod()
check("devir yalniz ilk gun", np.allclose(bh['turnover'].to_numpy(), [1, 0, 0, 0]))
check("son deger = fiyat oranlarinin ortalamasi", abs(v - np.mean([12.1 / 10, 18 / 20])) < 1e-12, str(v))

print("\n[3] gec listelenen sembol + eksik hucre")
d3 = pd.bdate_range('2021-01-27', '2021-02-03')
c3 = pd.DataFrame({'A': [10, 10.5, 11, np.nan, 11.5, 12], 'C': [np.nan, np.nan, 5.0, 5.5, 5.5, 6.0]},
                  index=d3, dtype=float)
panel = c3.stack().rename('close').rename_axis(['date', 'symbol']).to_frame()
cm = B.close_matrix(panel)
e3 = B.equal_weight(cm, '2021-01-01', rebalance='monthly')
b3 = B.equal_weight(cm, '2021-01-01', rebalance='none')
check("eksik hucre (A, 02-01) son kapanisla tasinir -> getiri 0; C (+%10) henuz agirliksiz -> brut 0",
      cm.loc['2021-02-01', 'A'] == 11.0 and abs(e3['gross'].iloc[3]) < 1e-15
      and abs(e3['gross'].iloc[4] - 0.5 * (11.5 / 11 - 1)) < 1e-12,
      str(e3['gross'].tolist()))
check("C listelenmeden once 1/N'de yok; Subat ilk seansinda girer (devir > 0); al-tut'ta hic yok",
      e3['turnover'].iloc[3] > 0 and abs(b3['gross'].iloc[3]) < 1e-12 and abs(b3['gross'].iloc[4] - (11.5 / 11 - 1)) < 1e-12,
      f"{e3['turnover'].tolist()} {b3['gross'].tolist()}")

print("\n[4] maliyet izgarasi")
n20 = B.net_returns(ew['gross'], ew['turnover'], 20)
check("ilk gun 20 bps gidis-donus -> -0,001 (tek yon 10 bps x devir 1)", abs(n20.iloc[0] + 0.001) < 1e-15)
tot = [float((1 + B.net_returns(ew['gross'], ew['turnover'], b)).prod()) for b in B.COST_GRID_BPS]
check("izgara 0/10/20/25/50 ve son deger monoton azalan", list(B.COST_GRID_BPS) == [0, 10, 20, 25, 50]
      and all(x > y for x, y in zip(tot, tot[1:])), str(tot))

print("\n[5] birimler")
dd = pd.DatetimeIndex(['2021-01-28', '2021-01-29', '2021-02-01', '2021-02-02'])
vals = pd.Series([100.0, 100.0, 100.0, 100.0], index=dd)
cpi = pd.Series([50.0, 100.0], index=pd.DatetimeIndex(['2021-01-01', '2021-02-01']))
rv = U.convert_values(vals, 'real_try', cpi=cpi)
check("reel: sabit nominal, TUFE Subat'ta 2 kat -> reel deger ay basinda yariya (basamak)",
      np.allclose(rv.to_numpy(), [2, 2, 1, 1]))
fx = pd.Series([5.0, 6.0, 7.0, 8.0], index=dd)
uv = U.convert_values(vals, 'usd', fx=pd.concat([pd.Series([4.0], index=[pd.Timestamp('2021-01-27')]), fx]))
check("USD: V_t / USDTRY_{t-1}", np.allclose(uv.to_numpy(), [100 / 4, 100 / 5, 100 / 6, 100 / 7]))
same_day = vals / fx
check("mutasyon: ayni gun kur farkli sonuc verir", not np.allclose(uv.to_numpy(), same_day.to_numpy()))
try:
    U.convert_values(vals, 'real_try', cpi=cpi.iloc[:1])
    check("eksik TUFE ayi -> UnitUnavailable", False)
except U.UnitUnavailable:
    check("eksik TUFE ayi -> UnitUnavailable", True)
try:
    U.convert_values(vals, 'usd', fx=fx)
    check("ilk gunden once kur yok -> UnitUnavailable", False)
except U.UnitUnavailable:
    check("ilk gunden once kur yok -> UnitUnavailable", True)

print("\n[6] A8 sabitleri")
files = ['app/services/model_analysis.py', 'scripts/benchmarking/benchmark_strategies.py',
         'scripts/analysis/generate_academic_report.py', 'scripts/analysis/extract_comparison_table.py']
hits = {f: re.findall(r'risk_free_rate\s*[=:][^\n]*0\.02|0\.02\s*/\s*252|risk_free_rate=0\.02',
                      open(f, encoding='utf-8').read()) for f in files}
check("0,02 risksiz faiz sabiti dort dosyada yok", not any(hits.values()), str(hits))
rf_path = os.path.join('data', 'macro', 'risk_free.csv')
from app.services.model_analysis import ModelAnalyzer  # noqa: E402
import tempfile  # noqa: E402
ma = ModelAnalyzer(results_dir=tempfile.mkdtemp(prefix='ma_'))
rng = np.random.default_rng(0)
r = rng.normal(0.001, 0.01, 60)
pv = 100 * np.r_[1, np.cumprod(1 + r)]
m0 = ma.calculate_advanced_metrics(pv, r, [])
check("tarihsiz -> rf0 etiketli", m0['risk_free_source'] == 'rf0 (tarih yok)')
if os.path.exists(rf_path):
    from evaluation.risk_free import risk_free_series  # noqa: E402
    rd = pd.bdate_range('2021-03-01', periods=60)
    m1 = ma.calculate_advanced_metrics(pv, r, [], return_dates=rd)
    rf = np.nan_to_num(risk_free_series(rd).to_numpy(), nan=0.0)
    ex = r - rf
    exp = ex.mean() / (ex.std() + 1e-10) * np.sqrt(252)
    check("return_dates -> A8 serisi; Sharpe elle ile ayni ve rf0'dan farkli",
          m1['risk_free_source'].startswith('A8') and abs(m1['sharpe_ratio'] - exp) < 1e-9
          and m1['sharpe_ratio'] < m0['sharpe_ratio'])
else:
    print("    [ATLANDI] risk_free.csv yok")

print("\n[7] gercek veri")
need = [os.path.join('data', 'bist', 'stock_data_with_indicators.csv'), U.DEFAULT_CPI_CSV,
        B.DEFAULT_XU030_CSV, rf_path, os.path.join('data', 'macro', 'usdtry.csv')]
if all(os.path.exists(p) for p in need):
    import logging  # noqa: E402
    logging.disable(logging.WARNING)
    from data.data_fetcher import DataFetcher  # noqa: E402
    from data.integrity import prepare_rl_panel  # noqa: E402
    pnl, _ = prepare_rl_panel(DataFetcher().load_data('stock_data_with_indicators.csv'))
    tab = B.baseline_table(B.close_matrix(pnl), '2021-01-01', '2022-01-01', xu030=B.load_xu030())
    st = tab['strategies']
    sh = [st['ew_monthly']['costs'][str(b)]['nominal_try']['sharpe_excess'] for b in B.COST_GRID_BPS]
    check("2021: 1/N, al-tut, XU030 x 3 birim dolu", all(
        set(st[k]['costs'][c]) == set(U.UNITS) for k in st for c in st[k]['costs']) and 'xu030' in st)
    check("1/N fazla getiri Sharpe'i maliyetle monoton azalir", all(x > y for x, y in zip(sh, sh[1:])), str(sh))
    ew_ = st['ew_monthly']['costs']['20']
    check("2021 USD ve reel yillik getiri nominalden dusuk (TL degerkaybi, enflasyon)",
          ew_['usd']['ann_return'] < ew_['nominal_try']['ann_return']
          and ew_['real_try']['ann_return'] < ew_['nominal_try']['ann_return'], str(ew_))
else:
    print("    [ATLANDI] veri dosyalari eksik")

print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
