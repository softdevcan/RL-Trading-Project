"""Faz 10 M8 engel 1a: 2021+ fiyat marji dogrulamasinin VERI tarafi (M2 tasarimi §2.5, 2026-11-20).

Belge kaniti tasarim §2.5'te: Borsa Istanbul 2020/20 sayili duyuru (onay 12/03/2020; 13/03/2020'den "ikinci bir
duyuruya kadar" Yildiz + Ana Pazar paylarinda marj %10) bugun de yururlukte; Alt Pazar/YIP/POIP genel kural %10.
Belge zinciri, ara donemde baslayip biten gecici bir marj duyurusunu tek basina dislayamaz; bu betik ayni kurali
veride sinar. Marj m iken X_t / C_{t-1} (X = acilis, yuksek, dusuk, kapanis; C_{t-1} sembolun son gecerli kapanisi,
duzeltilmis fiyat) +-(m + 0,005) bandini asmamali ve hareketler +-m'de yigilmali. Donem ve yil bazinda sayar;
+-%20 donemi (2015-11-30..2020-03-12) karsilastirmadir.

Ozel denetimler:
  * 2023-02-15: 8 Subat 2023 islemlerinin tamami iptal edildi (Borsa Istanbul KAP, 08/02/2023), baz fiyat
    7 Subat kapanisi. Acilis/kapanis 2023-02-07 kapanisina gore oranlanir; verideki 2023-02-08 satiri ayrica betimlenir.
  * 2022-12-16: komsu 10 seansa gore goreli hacim (kismi gun kaydi adayi) ve 2022-12-15 -> 12-19 iki gunluk oran.
Basarim OKUNMAZ (2021+ dogrulayici/test penceresi); yalniz fiyat siniri ve hacim.

    python scripts/marl/price_limit_audit_m8.py --out docs/development/measurements/phase10-m8/price_limit_audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

SRC = os.path.join('data', 'bist', 'stock_data_with_indicators.csv')   # panel kaynagi (prepare_panel_m2 ile ayni)
TOL = 0.005
PERIODS = [('bistech_20', '2015-11-30', '2020-03-12', 0.20),
           ('margin_10_2020', '2020-03-13', '2020-12-31', 0.10),
           ('y2021', '2021-01-01', '2021-12-31', 0.10), ('y2022', '2022-01-01', '2022-12-31', 0.10),
           ('y2023', '2023-01-01', '2023-12-31', 0.10), ('y2024', '2024-01-01', '2024-12-31', 0.10),
           ('y2025', '2025-01-01', '2025-12-31', 0.10), ('y2026', '2026-01-01', '2026-12-31', 0.10)]
FIELDS = ('open', 'high', 'low', 'close')


def provenance() -> dict:
    git = lambda *c: subprocess.run(['git', *c], capture_output=True, text=True).stdout.strip()   # noqa: E731
    with open(os.path.abspath(__file__), 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return {'commit': git('rev-parse', 'HEAD'), 'git_status_porcelain': git('status', '--porcelain'), 'script_sha256': sha}


def _sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from marl import accounting as A

    df = pd.read_csv(SRC, usecols=['symbol', 'date', *FIELDS, 'volume'])
    df['date'] = df['date'].astype(str).str[:10]
    P = {k: df.pivot(index='date', columns='symbol', values=k).sort_index() for k in (*FIELDS, 'volume')}
    dates = P['close'].index.to_numpy()
    sym = P['close'].columns.to_numpy()
    o, c = P['open'].to_numpy(dtype=float), P['close'].to_numpy(dtype=float)
    valid = np.isfinite(o) & (o > 0) & np.isfinite(c) & (c > 0)
    prev = pd.DataFrame(np.where(valid, c, np.nan)).ffill().shift(1).to_numpy()   # son gecerli kapanis, t-1'e kadar
    with np.errstate(invalid='ignore', divide='ignore'):
        R = {k: P[k].to_numpy(dtype=float) / prev for k in FIELDS}

    out = {'source_csv': SRC.replace(os.sep, '/'), 'source_sha256': _sha(SRC), 'last_date': str(dates[-1]),
           'symbols': [str(s) for s in sym], 'tolerance': TOL, 'periods': {}}
    for name, lo, hi, m in PERIODS:
        ok = valid & np.isfinite(prev) & ((dates >= lo) & (dates <= hi))[:, None]
        r = {'from': lo, 'to': hi, 'margin_by_source': m, 'cells': int(ok.sum())}
        for k in FIELDS:
            x = np.where(ok, R[k], np.nan)
            dev = np.abs(x - 1.0)
            r[k] = {'max': float(np.nanmax(x)), 'min': float(np.nanmin(x)),
                    'n_beyond_0.105': int(np.nansum(dev > 0.105)),
                    'n_in_0.105_0.205': int(np.nansum((dev > 0.105) & (dev <= 0.205))),
                    'n_near_0.10 (0.095-0.105)': int(np.nansum((dev >= 0.095) & (dev <= 0.105))),
                    'n_beyond_own_band': int(np.nansum(dev > m + TOL))}
            if k == 'open':   # ortamin satis engeli esigi (%10 doneminde 1 - (0,10 - 0,005))
                r[k]['n_le_0.905'] = int(np.nansum(x <= 0.905))
        out['periods'][name] = r

    # 2021+ bant disi hucreler (marj %10), alan bazinda; ozel denetimlerle siniflanir
    ok21 = valid & np.isfinite(prev) & (dates >= '2021-01-01')[:, None]
    cells = {}
    for k in FIELDS:
        for t, i in np.argwhere(ok21 & (np.abs(R[k] - 1.0) > 0.10 + TOL)):
            cells.setdefault((str(dates[t]), str(sym[i])), {})[k] = round(float(R[k][t, i]), 4)

    # 2023-02-15: baz 2023-02-07 (8 Subat islemleri iptal)
    d0, dx, d1 = '2023-02-07', '2023-02-08', '2023-02-15'
    rc = P['close'].loc[d1] / P['close'].loc[d0]
    ro = P['open'].loc[d1] / P['close'].loc[d0]
    row = df[df['date'] == dx]
    nb = P['volume'].iloc[[j for j in range(list(dates).index(dx) - 5, list(dates).index(dx) + 6) if dates[j] != dx]]
    flat = np.isclose(row['open'], row['close']) & np.isclose(row['high'], row['close']) & np.isclose(row['low'], row['close'])
    out['cancelled_session_2023_02_08'] = {
        'base_date': d0, 'reopen_date': d1,
        'reopen_close_over_base': {'max': float(rc.max()), 'min': float(rc.min()), 'n': int(rc.notna().sum())},
        'reopen_open_over_base': {'max': float(ro.max()), 'min': float(ro.min())},
        'row_2023_02_08': {'n_rows': int(len(row)), 'n_ohlc_flat': int(flat.sum()),
                           'n_volume_zero': int((row['volume'] == 0).sum()),
                           'volume_over_neighbor_median_max': float((P['volume'].loc[dx] / nb.median()).max()),
                           'close_over_prev_close': {'min': float((P['close'].loc[dx] / P['close'].loc[d0]).min()),
                                                     'max': float((P['close'].loc[dx] / P['close'].loc[d0]).max())}}}
    # 2022-12-16: kismi gun kaydi adayi
    dp, dn, dq = '2022-12-16', '2022-12-19', '2022-12-15'
    ip = list(dates).index(dp)
    nbp = P['volume'].iloc[[j for j in range(ip - 5, ip + 6) if j != ip]]
    rel = P['volume'].loc[dp] / nbp.median()
    two = P['close'].loc[dn] / P['close'].loc[dq]
    out['partial_day_2022_12_16'] = {
        'volume_over_neighbor_median': {'median': float(rel.median()), 'min': float(rel.min()), 'max': float(rel.max())},
        'two_day_close_ratio_2022_12_19_over_2022_12_15_max': float(two.max()), 'bound_1.10_squared': 1.21}

    cls = {'reopen_after_cancelled_2023_02_08': [], 'after_partial_day_2022_12_16': [], 'unexplained': []}
    for (d, s), v in sorted(cells.items()):
        key = ('reopen_after_cancelled_2023_02_08' if d == d1 else
               'after_partial_day_2022_12_16' if d == dn else 'unexplained')
        cls[key].append({'date': d, 'symbol': s, **v})
    out['beyond_band_2021plus'] = {k: {'n_cells': len(v), 'cells': v} for k, v in cls.items()}
    out['schedule_in_code_at_run'] = A.margin_schedule_meta()
    out['provenance'] = provenance()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)

    print('son tarih', out['last_date'], '| kaynak', out['source_csv'], out['source_sha256'][:12])
    print(f"{'donem':15s} {'hucre':>6s} | {'alan':5s} {'max':>7s} {'min':>7s} {'>0,105':>7s} {'0,105-0,205':>11s} {'~0,10':>6s}")
    for name, r in out['periods'].items():
        for k in FIELDS:
            q = r[k]
            print(f"{name:15s} {r['cells']:6d} | {k:5s} {q['max']:7.4f} {q['min']:7.4f} {q['n_beyond_0.105']:7d} "
                  f"{q['n_in_0.105_0.205']:11d} {q['n_near_0.10 (0.095-0.105)']:6d}")
    print('2023-02-15 / 2023-02-07:', out['cancelled_session_2023_02_08']['reopen_close_over_base'],
          '| 02-08 satiri:', out['cancelled_session_2023_02_08']['row_2023_02_08'])
    print('2022-12-16:', out['partial_day_2022_12_16'])
    print('2021+ bant disi:', {k: v['n_cells'] for k, v in out['beyond_band_2021plus'].items()})
    for c_ in out['beyond_band_2021plus']['unexplained']['cells']:
        print('  aciklanmamis:', c_)


if __name__ == '__main__':
    main()
