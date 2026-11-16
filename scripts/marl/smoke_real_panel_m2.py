r"""Faz 10 M2: gercek panelde ortam duman kosusu (hakem-2 M2 kapanisi #10). venv-marl; performans sayisi DEGIL.

Egitim ortami 2013-08-27..2018-12-31 norm_stats=fit; dogrulama ortami 2019..2020 egitim istatistigiyle, kesirli ve
tam sayi modunda rastgele eylemle bir bolum. Kontrol: adimlar sonlu, gozlem boyutu, satis engeli tetiklenir.

    venv-marl\Scripts\python scripts/marl/smoke_real_panel_m2.py
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
from marl import panel as PN, obs_spec as OS
from marl.env import MarlPortfolioEnv
A = PN.load_npz('data/marl/panel_m2.npz'); d = A['dates']
te = int(np.flatnonzero(d <= '2018-12-31')[-1]); ve = len(d) - 1
tr = MarlPortfolioEnv(A, OS.WARMUP_ROWS, te, norm_stats='fit', mode='train')
rng = np.random.default_rng(0)
for integer in (False, True):
    ev = MarlPortfolioEnv(A, te, ve, norm_stats=tr.norm_stats, mode='eval', integer_shares=integer, initial_capital=1e7)
    ev.reset(seed=0); t0 = time.time(); blocked = 0; Rs = []
    while ev.agents:
        ev.step({a: rng.uniform(-ev.bound, ev.bound, 1).astype(np.float32) for a in ev.agents})
        blocked += ev.last_info['n_sell_blocked']; Rs.append(ev.last_info['R'])
    print('tam sayi' if integer else 'kesirli', 'adim', len(Rs), 'sn %.1f' % (time.time() - t0),
          'n_sell_blocked toplam', blocked, 'sonlu R', np.all(np.isfinite(Rs)), 'obs dim', ev.spec.local_dim)
print('meta limit', tr.model_meta()['limit_down_no_sell']['rule'], 'train_range', tr.model_meta()['train_range'])
