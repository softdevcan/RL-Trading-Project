"""Faz 9 §9.38 (a): PC060 pozitif kontrolunun RL'siz ust siniri (on kayit: plan §9.38, commit fe65b0b).

Sinyal BILEREK gelecekten (pc_auc060: y_up = 1[kapanis_{t+1} > kapanis_t]); bu bir ust sinir olcumudur,
strateji onerisi degildir. Kurallar on kayittan; bu betik yeni karar eklemez.

  Yurutme: t kapanisinda karar ve yurutme (eski ortamla ayni); getiri t -> t+1 kapanis, t+1'e yazilir.
  Maliyet: tek yon 0,001 * sum|w_t - w~_{t-1}| (w~ fiyatla kaymis onceki agirlik); her kat nakitten baslar.
  S_yon (birincil): p_up_cal >= 0,5 ve meta_ok sembollere esit agirlik (yoksa 1/N). S_top10: en yuksek 10.
  1/N: 30 sembol esit agirlik, gunluk yeniden dengeleme. DT: M0' DT kolu tohum ortalamasi (PPO, A2C).
  Okuma: 2021'de dSR(S_yon - 1/N) < 0,43 -> PC060 bu gucte gecerli pozitif kontrol degil; >= 0,43 -> sinyalin
  ulasilabilir degeri MDE'yi asiyor (boru hatti/ogrenme aciklamasini destekler, kanit degil).

    python scripts/m0_pc_upper_bound.py --out docs/development/measurements/phase10-m0/pc_upper_bound.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PROTOCOL = os.path.join('docs', 'development', 'milestone0', 'protocol-v2.json')
COST = 0.001
THRESHOLD_2021 = 0.43   # §9.37 PC060 havuzlu MDE (on kayit)
THRESHOLD_ALL = 0.17    # 6 kat (kesifsel)
ANN = np.sqrt(252.0)


def _am():
    spec = importlib.util.spec_from_file_location('am', os.path.join(ROOT, 'scripts', 'analyze_milestone0.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_strategy(weights_fn, close: pd.DataFrame, dates: pd.DatetimeIndex, gross: bool = False) -> pd.Series:
    """Kat penceresi `dates` icin gunluk getiri. weights_fn(t) -> agirlik Series (t'deki karar)."""
    cal = close.index
    pos = cal.get_indexer(dates)
    if (pos < 1).any():
        raise SystemExit('pencere tarihi panel takviminde yok ya da ilk gun')
    out, w_prev_drift = [], None
    for p in pos:
        t, t1 = cal[p - 1], cal[p]
        w = weights_fn(t)
        prev = (w_prev_drift.reindex(w.index).fillna(0.0) if w_prev_drift is not None
                else pd.Series(0.0, index=w.index))   # kat nakitten baslar
        cost = 0.0 if gross else COST * float((w - prev).abs().sum())
        g = (close.loc[t1, w.index] / close.loc[t, w.index]).fillna(1.0)
        growth = float((w * g).sum())
        out.append((t1, growth * (1.0 - cost) - 1.0))
        w_prev_drift = (w * g) / growth
    return pd.Series(dict(out)).sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    am = _am()
    from data.data_fetcher import DataFetcher
    from data.integrity import prepare_rl_panel
    from evaluation.risk_free import risk_free_series
    from evaluation.sharpe_tests import sharpe_diff_hac
    from prediction.pit_table import load_table

    p = json.load(open(PROTOCOL, encoding='utf-8'))
    lvl = p['prediction_levels']['PC060']
    table, tinfo = load_table(lvl['table'])
    if tinfo['sha256'] != lvl['table_sha256']:
        raise SystemExit(f"PC060 tablo SHA {tinfo['sha256']} != protokol {lvl['table_sha256']}")
    table = table.copy()
    table['date'] = pd.to_datetime(table['date']).dt.tz_localize(None).dt.normalize()

    panel, _ = prepare_rl_panel(DataFetcher().load_data('stock_data_with_indicators.csv'))
    close = panel['close'].unstack('symbol').sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    symbols = list(close.columns)

    # M0' DT kolu ve test pencereleri (analiz betiginin kendi fonksiyonlariyla)
    mroot = os.path.join('results', 'milestone0', p['protocol_id'])
    _, cells, _ = am.read_ledger(os.path.join(mroot, 'ledger.jsonl'))
    _, allrec = am.count_n(am.registry_dirs(mroot, 'results'))
    by_dir = {}
    for t in allrec.values():
        by_dir.setdefault(am.norm_dir(t['_dir']), []).append(t)
    pred_sha = {k: v.get('table_sha256') for k, v in p['prediction_levels'].items()}
    rf_fn = lambda idx: risk_free_series(idx).reindex(idx)
    _, per_fold = am.series_table(cells, by_dir, rf_fn, pred_sha)
    algos = p['factors']['algorithm']
    folds = list(p['stages']['main']['folds'])

    piv_p = table.pivot_table(index='date', columns='symbol', values='p_up_cal', aggfunc='first')
    piv_ok = table.pivot_table(index='date', columns='symbol', values='meta_ok', aggfunc='first')
    ew = pd.Series(1.0 / len(symbols), index=symbols)

    def w_dir(t):
        pr = piv_p.loc[t].reindex(symbols) if t in piv_p.index else pd.Series(np.nan, index=symbols)
        ok = (piv_ok.loc[t].reindex(symbols) if t in piv_ok.index else pd.Series(False, index=symbols))
        ok = ok.fillna(False).astype(bool) & close.loc[t].notna()
        sel = ok & (pr >= 0.5)
        if not sel.any():
            return ew
        return sel.astype(float) / float(sel.sum())

    def w_top(t, k=10):
        pr = piv_p.loc[t].reindex(symbols) if t in piv_p.index else pd.Series(np.nan, index=symbols)
        ok = (piv_ok.loc[t].reindex(symbols) if t in piv_ok.index else pd.Series(False, index=symbols))
        pr = pr.where(ok.fillna(False).astype(bool) & close.loc[t].notna())
        if pr.notna().sum() == 0:
            return ew
        top = pr.nlargest(k).index
        return pd.Series(0.0, index=symbols).add(pd.Series(1.0 / len(top), index=top), fill_value=0.0)

    def w_ew(t):
        live = close.loc[t].notna()
        return live.astype(float) / float(live.sum())

    def window(fs):
        """Kat listesi icin birlestirilmis seriler (fazla getiri)."""
        ser = {k: [] for k in ('S_yon', 'S_top10', 'EW', 'S_yon_brut', 'EW_brut')}
        dts = {al: [] for al in algos}
        for f in fs:
            dt = {al: am.seed_mean(per_fold, al, 'DT', [f]) for al in algos}
            dates = dt[algos[0]].index
            rf = rf_fn(dates)
            for k, fn, g in (('S_yon', w_dir, False), ('S_top10', w_top, False), ('EW', w_ew, False),
                             ('S_yon_brut', w_dir, True), ('EW_brut', w_ew, True)):
                r = run_strategy(fn, close, dates, gross=g)
                ser[k].append(r - rf.reindex(r.index))
            for al in algos:
                dts[al].append(dt[al])
        out = {k: pd.concat(v).sort_index() for k, v in ser.items()}
        out.update({f'DT_{al}': pd.concat(v).sort_index() for al, v in dts.items()})
        return pd.DataFrame(out)

    def summarize(df: pd.DataFrame, threshold: float) -> dict:
        df = df.dropna()
        sr = {c: float(df[c].mean() / df[c].std(ddof=1) * ANN) for c in df.columns}
        tests = {}
        for a_, b_ in (('S_yon', 'EW'), ('S_top10', 'EW'), ('S_yon_brut', 'EW_brut'),
                       *[('S_yon', f'DT_{al}') for al in algos], *[('EW', f'DT_{al}') for al in algos]):
            r = sharpe_diff_hac(df[a_].to_numpy(), df[b_].to_numpy())
            tests[f'{a_} - {b_}'] = {'delta_sharpe_annual': r['delta'] * ANN, 'se_annual': r['se'] * ANN,
                                     'p_value': r['p_value']}
        prim = tests['S_yon - EW']['delta_sharpe_annual']
        return {'n_days': int(len(df)), 'sharpe_annual': sr, 'tests': tests, 'threshold': threshold,
                'primary_delta': prim, 'primary_exceeds_threshold': bool(prim >= threshold)}

    conf = summarize(window(['2021']), THRESHOLD_2021)
    allf = summarize(window(folds), THRESHOLD_ALL)
    if conf['primary_exceeds_threshold']:
        reading = ("dSR(S_yon - 1/N) >= 0,43: sinyalin ulasilabilir degeri MDE'yi asiyor; RL'nin bunu almamasi boru "
                   "hatti ya da ogrenme aciklamasini DESTEKLER (kanit degil; neden (b) ile sinanir).")
    else:
        reading = ("dSR(S_yon - 1/N) < 0,43: PC060 bu gucte gecerli bir pozitif kontrol DEGIL; AUC 0,60'lik sinyal RL'siz "
                   "kural portfoyunde bile tespit edilebilir fark uretmiyor -> §9.37'deki bos sonuc boru hatti hakkinda "
                   "bilgi vermez.")
    rec = {'preregistration': 'plan §9.38, commit fe65b0b (sonuctan once)', 'table': tinfo['run_id'],
           'table_sha256': tinfo['sha256'], 'cost_one_way': COST, 'execution': 't kapanis (eski ortamla ayni)',
           'confirmatory_2021': conf, 'exploratory_all_folds': allf, 'reading_2021': reading,
           'limits': ['sinyal bilerek gelecekten; ust sinir, strateji onerisi degil',
                      'eski ortamin hisse adedi eylemi ve max_shares_per_trade siniri burada yok'],
           'warning': 'tez sonucu degil; RL piyasayi yener / tahmin katki vermez cumlesine donusturulmez'}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
    print(json.dumps({'2021': {k: conf[k] for k in ('n_days', 'sharpe_annual', 'primary_delta')},
                      '2021_tests': conf['tests'], 'all': {k: allf[k] for k in ('n_days', 'primary_delta')},
                      'all_tests': {k: allf['tests'][k] for k in ('S_yon - EW', 'S_top10 - EW')},
                      'reading': reading}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
