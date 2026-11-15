"""
Faz 9 / D1 + D3 + D4 olcumu: 1/N (aylik), al-tut ve XU030; uc birim; maliyet izgarasi.

Pencereler: wf-v1 test yillari (2021..2026) ve tam RL penceresi (2013-06-01..2026-11-01).
Baseline'larda secim yok; test pencerelerini okumak bir secime girmiyor (yine de kayitli).

    python scripts/analysis/measure_baselines.py
"""

import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT = os.path.join('docs', 'development', 'measurements', 'phase9-d')


def sha(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def plot(full, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from evaluation.baselines import COST_GRID_BPS

    surface, ink, muted, grid = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e1'
    colors = {'ew_monthly': '#2a78d6', 'buy_hold': '#eb6834'}
    names = {'ew_monthly': '1/N aylik', 'buy_hold': 'Al-tut (1/N baslangic)'}
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    st = full['strategies']
    for k in ('ew_monthly', 'buy_hold'):
        y = [st[k]['costs'][str(b)]['nominal_try']['sharpe_excess'] for b in COST_GRID_BPS]
        ax.plot(COST_GRID_BPS, y, color=colors[k], lw=2, marker='o', ms=6,
                markeredgecolor=surface, markeredgewidth=2, label=names[k])
        ax.annotate(names[k], (COST_GRID_BPS[-1], y[-1]), xytext=(6, 0), textcoords='offset points',
                    va='center', fontsize=8, color=ink)
    xs = st['xu030']['costs']['0']['nominal_try']['sharpe_excess']
    ax.axhline(xs, color=muted, lw=1, ls='--', label='XU030 fiyat endeksi (maliyetsiz)')
    ax.axvline(20, color=grid, lw=1, zorder=0)
    ax.set_xticks(list(COST_GRID_BPS))
    ax.set_xlabel('Islem maliyeti, gidis-donus (bps; ana = 20)', color=muted, fontsize=9)
    ax.set_ylabel('Yillik fazla getiri Sharpe (nominal TL, A8)', color=muted, fontsize=9)
    ax.set_title(f"Maliyet duyarliligi, {full['window'][0]}..{full['window'][1]}",
                 color=ink, fontsize=10, loc='left')
    ax.grid(axis='y', color=grid, lw=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc='center left')
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main():
    import logging
    logging.disable(logging.WARNING)
    from data.data_fetcher import DataFetcher
    from data.integrity import prepare_rl_panel
    from evaluation import baselines as B
    from evaluation import units as U
    from evaluation.splits import WF_V1

    panel_path = os.path.join('data', 'bist', 'stock_data_with_indicators.csv')
    pnl, rep = prepare_rl_panel(DataFetcher().load_data('stock_data_with_indicators.csv'))
    close = B.close_matrix(pnl)
    xu = B.load_xu030()
    cpi, fx = U.load_cpi(), U.load_usdtry()
    windows = {f.name: (f.test_start, f.test_end) for f in WF_V1.folds}
    windows['rl_full'] = ('2013-06-01', WF_V1.folds[-1].test_end)
    res = {k: B.baseline_table(close, a, b, xu030=xu, cpi=cpi, fx=fx) for k, (a, b) in windows.items()}
    out = {
        'commit': subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip(),
        'inputs': {p: sha(p) for p in (panel_path, U.DEFAULT_CPI_CSV, B.DEFAULT_XU030_CSV,
                                       os.path.join('data', 'macro', 'risk_free.csv'),
                                       os.path.join('data', 'macro', 'usdtry.csv'))},
        'gate': {k: rep.get(k) for k in ('window_start', 'n_ghost_days', 'n_stale_cells_dropped',
                                          'missing_cell_frac')},
        'cost_grid_bps': list(B.COST_GRID_BPS), 'main_cost_bps': B.MAIN_COST_BPS,
        'design': {'ew_monthly': 'ayin ilk seansi kapanisinda 1/N (o gun fiyati olan semboller)',
                   'buy_hold': 'ilk seans 1/N, yeniden dengeleme yok',
                   'xu030': 'yfinance XU030.IS fiyat endeksi, temettu yok, maliyetsiz',
                   'cost': 'tek yon bps/2 x devir (sum |dw|); ilk alim devri 1',
                   'units': 'reel: ayin TUFE duzeyi (basamak); USD: USDTRY_{t-1}; '
                            'reel/USD Sharpe rf=0, AYLIK getiriden (sharpe_rf0_monthly)',
                   'universe': ('HAYATTA KALAN dondurulmus panel 30 (E0), tarihsel BIST 30 uyeligi degil '
                                '(E1 acik); XU030 ayni evren degil + temettusuz -> ustunluk cikarimi YOK'),
                   'not_tested': '1/N - al-tut farki test edilmedi',
                   'cost': 'yalniz komisyon; spread/lot/limit yok; al-tutta cikis maliyeti yok'},
        'windows': res,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'baselines.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, default=float)
    plot(res['rl_full'], os.path.join(OUT, 'cost_curve.png'))
    for k, r in res.items():
        st = r['strategies']
        print(k, r['window'], 'ew20', round(st['ew_monthly']['costs']['20']['nominal_try']['sharpe_excess'], 3),
              'bh20', round(st['buy_hold']['costs']['20']['nominal_try']['sharpe_excess'], 3),
              'xu', round(st['xu030']['costs']['0']['nominal_try']['sharpe_excess'], 3))
    print('->', OUT)


if __name__ == '__main__':
    main()
