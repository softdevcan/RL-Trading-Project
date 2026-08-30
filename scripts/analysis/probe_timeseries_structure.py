"""
Zaman serisi yapisi olcumu — ARIMA/SARIMAX/GARCH kararina veri saglar.

Soru: tahmin ensemble'ina klasik zaman serisi modelleri (ARIMA, SARIMAX)
eklemek ne kazandirir? Cevap modelin kendisinde degil, HEDEF SERIDE saklidir:
bir model ancak veride var olan yapiyi yakalayabilir.

Bu betik uc seyi olcer:

  1. GETIRI otokorelasyonu  -> ARIMA'nin AR/MA yapisinin tavani
  2. |GETIRI| otokorelasyonu -> volatilite kumelenmesi (GARCH'in alani)
  3. Haftanin gunu / ay etkisi -> SARIMA'nin S kismi ile mevcut takvim
     ozelliklerinin (feature_engineer._add_calendar_features) hedefledigi yapi

Hedef tanimi `prediction/feature_engineer.py::_build_targets` ile ayni tutulur:
gunluk ufukta `log(P_{t+1} / P_t)`.

statsmodels GEREKTIRMEZ — otokorelasyon ve Ljung-Box elle hesaplanir, yalnizca
scipy.stats.chi2 (ki-kare kuyruk olasiligi) ve F-testi kullanilir. Boylece
karar vermek icin once bagimlilik eklemek gerekmez.

Kullanim:
    python scripts/analysis/probe_timeseries_structure.py
    python scripts/analysis/probe_timeseries_structure.py --lags 20 --json out.json

Bulgular ve yorumu: docs/development/prediction-timeseries-models.md
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from data.data_fetcher import DataFetcher  # noqa: E402

MIN_OBS = 500
DAY_NAMES = ['Pzt', 'Sal', 'Car', 'Per', 'Cum']


def acf(x, lags):
    """Ornek otokorelasyon katsayilari (rho_1..rho_lags)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom == 0:
        return [0.0] * lags
    return [float(np.dot(x[:-k], x[k:]) / denom) for k in range(1, lags + 1)]


def ljung_box(x, h=10):
    """Ljung-Box Q istatistigi ve p-degeri.

    H0: ilk h gecikmenin tamami sifir (seri beyaz gurultu).
    p > 0.05  -> yapi bulunamadi, ARIMA'nin yakalayacagi sey yok.
    """
    n = len(x)
    r = acf(x, h)
    q = n * (n + 2) * sum(rk ** 2 / (n - k - 1) for k, rk in enumerate(r))
    return float(q), float(stats.chi2.sf(q, h))


def load_returns(filename='raw_stock_data.csv'):
    """Sembol -> gunluk log getiri serisi (tarih indeksli)."""
    fetcher = DataFetcher()
    df = fetcher.clean_data(fetcher.load_data(filename))
    out = {}
    for sym in df.index.get_level_values('symbol').unique():
        close = df.xs(sym, level='symbol')['close'].dropna()
        close = close[close > 0]
        if len(close) < MIN_OBS:
            continue
        r = np.log(close / close.shift(1)).dropna()
        r = r[np.isfinite(r)]
        if len(r) >= MIN_OBS:
            out[sym] = r
    return out


def probe_autocorrelation(returns, lags):
    rows = []
    for sym, r in returns.items():
        v = r.values
        rho_r = acf(v, 5)
        rho_a = acf(np.abs(v), 5)
        _, p_r = ljung_box(v, lags)
        _, p_a = ljung_box(np.abs(v), lags)
        rows.append({
            'symbol': sym, 'n': len(v),
            'ret_rho1': rho_r[0], 'ret_lb_p': p_r,
            'abs_rho1': rho_a[0], 'abs_lb_p': p_a,
        })
    return rows


def probe_seasonality(returns):
    r = np.concatenate([s.values for s in returns.values()])
    dow = np.concatenate([s.index.dayofweek.values for s in returns.values()])
    mon = np.concatenate([s.index.month.values for s in returns.values()])
    ok = np.isfinite(r)
    r, dow, mon = r[ok], dow[ok], mon[ok]

    dow_groups = [r[dow == d] for d in range(5)]
    mon_groups = [r[mon == m] for m in range(1, 13)]
    f_d, p_d = stats.f_oneway(*[g for g in dow_groups if len(g) > 10])
    f_m, p_m = stats.f_oneway(*[g for g in mon_groups if len(g) > 10])

    # Etki buyuklugu: p-degeri n=76k'da her seyi anlamli gosterir, asil soru
    # gunun/ayin toplam varyansin ne kadarini acikladigi.
    ss_tot = float(((r - r.mean()) ** 2).sum())
    ss_dow = float(sum(len(g) * (g.mean() - r.mean()) ** 2
                       for g in dow_groups if len(g) > 10))
    ss_mon = float(sum(len(g) * (g.mean() - r.mean()) ** 2
                       for g in mon_groups if len(g) > 10))
    return {
        'n': int(len(r)),
        'dow_f': float(f_d), 'dow_p': float(p_d),
        'mon_f': float(f_m), 'mon_p': float(p_m),
        'dow_var_explained_pct': ss_dow / ss_tot * 100,
        'mon_var_explained_pct': ss_mon / ss_tot * 100,
        'dow_means': {DAY_NAMES[d]: float(g.mean())
                      for d, g in enumerate(dow_groups) if len(g) > 10},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lags', type=int, default=10,
                    help='Ljung-Box gecikme sayisi (varsayilan 10)')
    ap.add_argument('--json', help='Sonuclari JSON olarak bu dosyaya yaz')
    ap.add_argument('--head', type=int, default=8,
                    help='Tabloda gosterilecek sembol sayisi')
    args = ap.parse_args()

    returns = load_returns()
    if not returns:
        print('Yeterli gecmise sahip sembol bulunamadi.')
        return 1

    rows = probe_autocorrelation(returns, args.lags)
    seas = probe_seasonality(returns)

    ret_rho1 = [r['ret_rho1'] for r in rows]
    abs_rho1 = [r['abs_rho1'] for r in rows]
    ret_flat = sum(1 for r in rows if r['ret_lb_p'] > 0.05)
    abs_struct = sum(1 for r in rows if r['abs_lb_p'] < 0.05)

    line = '=' * 78
    print(line)
    print("1) GETIRI OTOKORELASYONU  (ARIMA'nin AR/MA yapisinin alani)")
    print(line)
    print(f"{'sembol':<11}{'n':>6}{'rho1':>9}{'LB p':>9}   |  "
          f"{'|r| rho1':>9}{'LB p':>9}")
    for r in rows[:args.head]:
        print(f"{r['symbol']:<11}{r['n']:>6}{r['ret_rho1']:>9.4f}"
              f"{r['ret_lb_p']:>9.4f}   |  {r['abs_rho1']:>9.4f}"
              f"{r['abs_lb_p']:>9.4f}")
    if len(rows) > args.head:
        print(f"{'...':<11}({len(rows) - args.head} sembol daha)")

    print()
    print(f"  getiri   ortalama rho1 = {np.mean(ret_rho1):+.4f}")
    print(f"  Ljung-Box p>0.05 (yapi YOK) : {ret_flat}/{len(rows)} sembol")
    print(f"  |getiri| ortalama rho1 = {np.mean(abs_rho1):+.4f}")
    print(f"  Ljung-Box p<0.05 (yapi VAR) : {abs_struct}/{len(rows)} sembol")
    print()
    print('  rho1^2 = serinin kendi gecmisiyle aciklanabilen varyans orani:')
    print(f"    getiri   : {np.mean([x ** 2 for x in ret_rho1]) * 100:.2f}%")
    print(f"    |getiri| : {np.mean([x ** 2 for x in abs_rho1]) * 100:.2f}%")

    print()
    print(line)
    print("2) MEVSIMSELLIK  (SARIMA'nin S kismi / takvim ozellikleri)")
    print(line)
    print(f"  Haftanin gunu ANOVA: F={seas['dow_f']:.2f}  p={seas['dow_p']:.4f}")
    for name, mean in seas['dow_means'].items():
        print(f"    {name}: ort {mean * 100:+.4f}%")
    print(f"  Ay etkisi     ANOVA: F={seas['mon_f']:.2f}  p={seas['mon_p']:.4f}")
    print()
    print(f"  Aciklanan varyans  haftanin gunu: "
          f"{seas['dow_var_explained_pct']:.3f}%")
    print(f"  Aciklanan varyans  ay           : "
          f"{seas['mon_var_explained_pct']:.3f}%")
    print(f"  Toplam gozlem: {seas['n']:,}")

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as fh:
            json.dump({'per_symbol': rows, 'seasonality': seas},
                      fh, ensure_ascii=False, indent=2)
        print(f"\nJSON yazildi: {args.json}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
