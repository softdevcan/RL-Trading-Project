"""Faz 10 M8 engel 1b: panel kaynaginin (yfinance, duzeltilmis) ikinci kaynakla karsilastirmasi (M2 tasarimi §6).

Neden: (i) SKBNK'nin iki bedelli sermaye artirimi yfinance'te YANLIS GUNDEN itibaren duzeltilmis (duzeltme siniri hak
kullanim gununden once; sinirdan oncesi ham kalmis -> sahte dusus). (ii) yfinance gun cubugu bazi gunlerde kesik
(2022-12-16 kismi gun; 2023-01-05 seans sonundaki dusus yok): kapanis resmi kapanistan farkli. Kucuk farklar fiyat
siniri denetimine takilmaz. Ikinci kaynak: Is Yatirim HisseTekil (HG_KAPANIS ham, HGDG_KAPANIS duzeltilmis, HG_MIN,
HG_MAX, SERMAYE). Normal gunlerde iki kaynagin duzeltilmis kapanisi 4 basamakta esit (or. THYAO 2022-12-14 133,8331).

Olculenler (sembol basina, ortak gunler, 2013-06+): q_t = bizim kapanis / IsY duzeltilmis kapanis.
  * sapma_t = log q_t - log q'nun 11 seanslik merkezi medyani (sabit olcek farkini duser): tek gunluk / kisa sapma.
    Gun ve yil bazinda |sapma| > %1, > %2, > %5 hucre sayilari; 2021+ hucre listesi (|sapma| > %2).
  * IsY SERMAYE degisimi (sermaye artirimi) gunu E: +-10 seansta |dlog q| > %2 basamak yoksa 'ok'; E'de varsa
    'duzeltilmemis'; E'den farkli bir B gununde varsa 'yanlis_gun' (B, basamak, ham kapanis orani raporlanir).
  * gun kumesi farki: bizde olup IsY'de olmayan (or. iptal edilen seans) ve tersi.
Basarim okunmaz; yalniz fiyat serisi tutarliligi.

    python scripts/marl/cross_source_audit_m8.py --fetch     # ham yanitlar data/marl/sources/isyatirim/ (git disi)
    python scripts/marl/cross_source_audit_m8.py --out docs/development/measurements/phase10-m8/cross_source_audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

SRC = os.path.join('data', 'bist', 'stock_data_with_indicators.csv')
CACHE = os.path.join('data', 'marl', 'sources', 'isyatirim')
URL = ('https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil'
       '?hisse={s}&startdate=01-06-2013&enddate=09-11-2026')
START = '2013-06-01'
BIG = 0.02            # olay basamagi esigi (|dlog q|)
EVENT_WIN = 10        # sermaye degisimi gunu +- bu kadar ortak seans
MED_WIN = 11          # sapma icin merkezi medyan penceresi


def provenance() -> dict:
    git = lambda *c: subprocess.run(['git', *c], capture_output=True, text=True).stdout.strip()   # noqa: E731
    with open(os.path.abspath(__file__), 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return {'commit': git('rev-parse', 'HEAD'), 'git_status_porcelain': git('status', '--porcelain'), 'script_sha256': sha}


def _sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def symbols():
    with open(os.path.join('data', 'universe', 'bist30_panel.json'), encoding='utf-8') as fh:
        return json.load(fh)['symbols']


def fetch():
    os.makedirs(CACHE, exist_ok=True)
    for s in symbols():
        code = s.replace('.IS', '')
        req = urllib.request.Request(URL.format(s=code), headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
        v = json.loads(body.decode('utf-8'))
        if not v.get('ok') or not v.get('value'):
            raise RuntimeError(f"{code}: bos ya da hatali yanit")
        with open(os.path.join(CACHE, f'{code}.json'), 'wb') as fh:
            fh.write(body)
        print(code, len(v['value']), v['value'][0]['HGDG_TARIH'], v['value'][-1]['HGDG_TARIH'])
        time.sleep(1.0)


def load_isy(code: str) -> pd.DataFrame:
    with open(os.path.join(CACHE, f'{code}.json'), encoding='utf-8') as fh:
        d = pd.DataFrame(json.load(fh)['value'])
    d.index = pd.to_datetime(d['HGDG_TARIH'], format='%d-%m-%Y').dt.strftime('%Y-%m-%d')
    return d[['HG_KAPANIS', 'HGDG_KAPANIS', 'HG_MIN', 'HG_MAX', 'SERMAYE']].astype(float)


def analyze_symbol(ours: pd.Series, isy: pd.DataFrame):
    j = pd.concat([ours.rename('close'), isy], axis=1, join='inner')
    j = j[(j['close'] > 0) & (j['HGDG_KAPANIS'] > 0)]
    lq = np.log(j['close'] / j['HGDG_KAPANIS'])
    dev = lq - lq.rolling(MED_WIN, center=True, min_periods=MED_WIN // 2 + 1).median()
    dl = lq.diff().fillna(0.0).to_numpy()
    dates = list(j.index)
    events = []
    cap = j['SERMAYE'].to_numpy()
    for i in np.flatnonzero(np.diff(cap) != 0) + 1:
        lo, hi = max(i - EVENT_WIN, 1), min(i + EVENT_WIN, len(dates) - 1)
        near = [k for k in range(lo, hi + 1) if abs(dl[k]) > BIG]
        e = {'date': dates[i], 'capital_from': float(cap[i - 1]), 'capital_to': float(cap[i]),
             'raw_close_ratio': round(float(j['HG_KAPANIS'].iloc[i] / j['HG_KAPANIS'].iloc[i - 1]), 4),
             'isy_adj_over_raw_before': round(float(j['HGDG_KAPANIS'].iloc[i - 1] / j['HG_KAPANIS'].iloc[i - 1]), 4),
             'isy_adj_over_raw_after': round(float(j['HGDG_KAPANIS'].iloc[i] / j['HG_KAPANIS'].iloc[i]), 4),
             'big_q_steps_near': [{'date': dates[k], 'dlog': round(float(dl[k]), 4)} for k in near]}
        e['status'] = 'ok' if not near else ('duzeltilmemis' if near == [i] else 'yanlis_gun')
        events.append(e)
    return dev, events, set(ours.index), set(isy.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fetch', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.fetch:
        fetch()
        return
    if not a.out:
        ap.error('--out gerekli (ya da --fetch)')
    df = pd.read_csv(SRC, usecols=['symbol', 'date', 'close'])
    df['date'] = df['date'].astype(str).str[:10]
    df = df[df['date'] >= START]
    out = {'source_csv': SRC.replace(os.sep, '/'), 'source_sha256': _sha(SRC), 'window_start': START,
           'second_source': {'name': 'Is Yatirim HisseTekil (HG_KAPANIS ham, HGDG_KAPANIS duzeltilmis, HG_MIN, HG_MAX, SERMAYE)',
                             'url_template': URL, 'cache_dir': CACHE.replace(os.sep, '/'), 'files_sha256': {}},
           'params': {'big_step': BIG, 'event_window': EVENT_WIN, 'median_window': MED_WIN}}
    devs, cap_events, ours_only, isy_only = {}, [], {}, {}
    for s in symbols():
        code = s.replace('.IS', '')
        out['second_source']['files_sha256'][code] = _sha(os.path.join(CACHE, f'{code}.json'))
        isy = load_isy(code)
        isy = isy[isy.index >= START]
        dev, ev, od, idd = analyze_symbol(df[df['symbol'] == s].set_index('date')['close'], isy)
        devs[s] = dev
        cap_events += [{'symbol': s, **e} for e in ev]
        ours_only[s], isy_only[s] = sorted(od - idd), sorted(idd - od)
    A = pd.DataFrame(devs).abs()
    out['deviation'] = {
        'cells': int(A.notna().sum().sum()),
        'n_gt': {k: int((A > v).sum().sum()) for k, v in (('0.01', 0.01), ('0.02', 0.02), ('0.05', 0.05))},
        'by_year_gt_0.01': {y: int(v) for y, v in (A > 0.01).groupby(A.index.str[:4]).sum().sum(axis=1).items()},
        'by_year_gt_0.02': {y: int(v) for y, v in (A > 0.02).groupby(A.index.str[:4]).sum().sum(axis=1).items()},
        'days_ge5_symbols_gt_0.01': {d: int(n) for d, n in (A > 0.01).sum(axis=1).loc[lambda x: x >= 5]
                                     .sort_values(ascending=False).items()},
        'cells_2021plus_gt_0.02': [{'date': d, 'symbol': s, 'dev': round(float(D[s]), 4)}
                                   for d, D in pd.DataFrame(devs).loc['2021-01-01':].iterrows()
                                   for s in D.index if abs(D[s]) > 0.02]}
    out['capital_events'] = cap_events
    out['capital_events_not_ok'] = [e for e in cap_events if e['status'] != 'ok']
    common_only = sorted(set.intersection(*[set(v) for v in ours_only.values()]))
    out['dates_ours_only_all_symbols'] = common_only
    out['dates_isy_only'] = {s: v for s, v in isy_only.items() if v}
    out['provenance'] = provenance()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)

    dv = out['deviation']
    print('hucre', dv['cells'], '| |sapma| >', dv['n_gt'])
    print('yil >%1:', dv['by_year_gt_0.01'])
    print('yil >%2:', dv['by_year_gt_0.02'])
    print('>=5 hissede >%1 sapma olan gun sayisi:', len(dv['days_ge5_symbols_gt_0.01']),
          '| ilk 10:', dict(list(dv['days_ge5_symbols_gt_0.01'].items())[:10]))
    print('2021+ |sapma| > %2 hucre:', len(dv['cells_2021plus_gt_0.02']),
          '| gunler:', sorted({c['date'] for c in dv['cells_2021plus_gt_0.02']}))
    print('sermaye olayi:', len(cap_events), '| ok olmayan:')
    for e in out['capital_events_not_ok']:
        print('  ', e['symbol'], e['date'], e['status'], 'ham oran', e['raw_close_ratio'], 'IsY duz/ham',
              e['isy_adj_over_raw_before'], '->', e['isy_adj_over_raw_after'], e['big_q_steps_near'])
    print('yalniz bizde olan gun (30 hissede):', common_only)
    print('yalniz IsY de olan gun:', {s: len(v) for s, v in out['dates_isy_only'].items()})


if __name__ == '__main__':
    main()
