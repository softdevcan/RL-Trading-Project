"""
Milestone 0' K4 olcek esitlemesi (hakem-2 kapanis #6; kullanici karari 2026-11-15: tek seviyeli esitleme).

Kural: k = std(ema_level terimi) / std(differential terimi), referans = esit agirlikli TAM yatirimli
portfoyun gunluk getirisi, pencere 2013-06-01..2019-12-31 (her wf-v1 dogrulama/test penceresinden ONCE:
en erken dogrulama 2020). Terimler odul modulunun kendisinden (`RewardCalculator.calculate_psr_reward`,
varsayilan agirliklar ve r_f), ilk 30 adim haric (isinma). Differential kolunda w2 = 0,30 * k; N artmaz.

    python scripts/analysis/measure_k4_scale.py
"""

import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PANEL = os.path.join('data', 'bist', 'stock_data_with_indicators.csv')
OUT = os.path.join('docs', 'development', 'measurements', 'phase9-m0', 'k4_scale.json')
WINDOW = ('2013-06-01', '2019-12-31')
WARMUP = 30


def term_series(returns, sharpe_term):
    from env.reward_functions import RewardCalculator
    rc = RewardCalculator(sharpe_term=sharpe_term)
    rc.reset()
    pv, rets, out = [1_000_000.0], [], []
    for i, r in enumerate(returns):
        pv.append(pv[-1] * (1 + r))
        rets.append(r)
        _, comp = rc.calculate_psr_reward(pv, rets, current_step=i + 1, trades_executed=0,
                                          commission_cost=0.0, initial_balance=1_000_000.0)
        out.append(comp['sharpe_ratio'])
    return np.asarray(out[WARMUP:], dtype=float), rc.get_reward_spec()


def main():
    import hashlib
    import pandas as pd
    with open(PANEL, 'rb') as fh:
        psha = hashlib.sha256(fh.read()).hexdigest()
    df = pd.read_csv(PANEL, parse_dates=['date'])
    close = df.pivot_table(index='date', columns='symbol', values='close').sort_index()
    ret = close.pct_change(fill_method=None).mean(axis=1, skipna=True)
    ret = ret[(ret.index >= WINDOW[0]) & (ret.index <= WINDOW[1])].dropna()
    ema, spec = term_series(ret.to_numpy(), 'ema_level')
    dif, _ = term_series(ret.to_numpy(), 'differential')
    k = float(ema.std(ddof=1) / dif.std(ddof=1))
    out = {'commit': subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip(),
           'panel': PANEL, 'panel_sha256': psha, 'window': list(WINDOW), 'n': int(len(ema)),
           'reference': 'esit agirlikli tam yatirim, gunluk getiri satir ortalamasi (pct_change fill_method=None)',
           'reward_spec_default': spec,
           'ema_level': {'std': float(ema.std(ddof=1)), 'mean': float(ema.mean())},
           'differential': {'std': float(dif.std(ddof=1)), 'mean': float(dif.mean())},
           'k_std_ratio': k, 'w2_default': 0.30, 'w2_differential_scaled': round(0.30 * k, 6),
           'rule': 'differential kolunda w2 = 0,30 * k (k bu dosyadan; dondurulur)'}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps({k_: out[k_] for k_ in ('n', 'ema_level', 'differential', 'k_std_ratio',
                                              'w2_differential_scaled')}, indent=1))


if __name__ == '__main__':
    main()
