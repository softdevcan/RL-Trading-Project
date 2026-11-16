"""Faz 10 M2: acilis boslugu denetimi, BIST fiyat marji donemine gore (hakem-2 M2 kapanisi #1, #6).

Donemler (kaynakla dogrulanan; tasarim §2.5):
  pre_bistech   < 2015-11-30 : marj DOGRULANMADI; baz fiyat onceki seansin agirlikli ortalamasi (AOF)
                               (Borsa Istanbul, "BISTECH sistemiyle pay piyasasi yeni yapidaki islevsel degisiklikler" §2.1.1)
  bistech_20    2015-11-30 .. 2020-03-12 : paylar icin marj %20, baz = onceki kapanis (ayni belge §2.1.2)
  margin_10     >= 2020-03-13 : marj %10 (bianet 13.03.2020; Borsa Istanbul Genel Mudurlugu duyurusu)
Bosluk = O_t / Cf_{t-1} (duzeltilmis fiyat). Bant disi = |bosluk - 1| > marj + 0,005 (muhtemel veri hatasi
ya da duzeltme etkisi). Panel salt okunur (`data/marl/panel_m2.npz`); 2021+ yok.

    python scripts/marl/open_gap_audit_m2.py --out docs/development/measurements/phase10-m2/open_gap_audit.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

REGIMES = [('pre_bistech', '0000-00-00', '2015-11-29', None),
           ('bistech_20', '2015-11-30', '2020-03-12', 0.20),
           ('margin_10', '2020-03-13', '9999-12-31', 0.10)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from marl import accounting as A
    from marl import panel as PN
    arr = PN.load_npz(os.path.join('data', 'marl', 'panel_m2.npz'))
    d, sy = arr['dates'], arr['symbols']
    px = A.fill_prices(arr['open'], arr['close'], A.LISTING_LOCKOUT, A.LIMIT_DOWN_GAP)
    with np.errstate(invalid='ignore', divide='ignore'):
        gap = arr['open'][1:] / px['close'][:-1]
        gday = arr['close'][1:] / arr['open'][1:]
    v = px['valid'][1:] & px['listed'][:-1]
    rd = d[1:]
    out = {'panel_sha256': arr['sha256'], 'regimes': {}, 'down_0.095_cells': []}
    for name, lo, hi, band in REGIMES:
        m = v & ((rd >= lo) & (rd <= hi))[:, None]
        g = gap[m]
        r = {'from': lo, 'to': hi, 'margin': band, 'valid_cells': int(m.sum()),
             'down_le_0.905': int((g <= 0.905).sum()), 'up_ge_1.095': int((g >= 1.095).sum()),
             'down_le_0.805': int((g <= 0.805).sum()), 'up_ge_1.195': int((g >= 1.195).sum())}
        if band is not None:
            bad = m & (np.abs(gap - 1.0) > band + 0.005)
            r['beyond_band'] = [{'date': str(rd[t]), 'symbol': str(sy[i]), 'gap': float(gap[t, i])}
                                for t, i in np.argwhere(bad)]
        out['regimes'][name] = r
    for t, i in np.argwhere(v & (gap <= 0.905)):
        reg = next(n for n, lo, hi, _ in REGIMES if lo <= rd[t] <= hi)
        out['down_0.095_cells'].append({'date': str(rd[t]), 'symbol': str(sy[i]), 'regime': reg,
                                        'gap': float(gap[t, i]), 'open_to_close': float(gday[t, i])})
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    for n, r in out['regimes'].items():
        print(n, {k: (len(v_) if isinstance(v_, list) else v_) for k, v_ in r.items()})


if __name__ == '__main__':
    main()
