"""Faz 10 M6a olcumu: HMM ikinci rejim tanimi gercek seride (ana venv, ag yok). Performans sayisi YOK.

Girdi birincil: XU100 / USDTRY_{t-1} gunluk log getirisi x 100 (`evaluation.regimes_hmm.load_primary_returns`).
Ilk capa = MARL panelinin ilk seansi (olasiligin gerektigi ilk gun; M6 tasarimi §3 M6a, K-M6-2); sonra her 63 seansta
genisleyen yeniden uyum. Durum sayisi ilk capada {2, 3} BIC ile. Kayit: BIC'ler, capa basina kanonik (sigma artan)
ortalama/sigma ve 'ortalama sigma ile ters sirali mi' olcumu, yakinsama; gelistirme pencerelerinde (eval_m3 ile ayni)
iki hucreli HMM etiketi (bull_like / bear_like) gun / donem / >= 20 g donem sayisi ve kural tabanli birincil egilim
etiketiyle gun bazinda uyusma. `--test-years`: wf-v1 test yillarinda yalniz etiket sayimi (K-M6-4; basarim okunmaz).

    python scripts/marl/hmm_regimes_m6.py --out docs/development/measurements/phase10-m6/hmm_regimes.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'marl'))
os.chdir(ROOT)

from evaluation import regimes as R  # noqa: E402
from evaluation import regimes_hmm as H  # noqa: E402
from regime_windows_m6 import PANEL, _git, _sha, summarize, test_year_dates, window_dates  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--test-years', action='store_true')
    a = ap.parse_args()
    z = np.load(PANEL, allow_pickle=True)
    pdates = pd.DatetimeIndex(pd.to_datetime(z['dates'].astype(str)))
    first_anchor = str(pdates[0].date())
    win = test_year_dates() if a.test_years else window_dates(pdates)
    end = None if a.test_years else '2021-01-01'
    r = H.load_primary_returns()
    t0 = time.time()
    h = H.filtered_regimes(r, first_anchor, end=end)
    secs = time.time() - t0
    lab = H.hmm_labels(h)
    rule = R.load_primary_labels()
    res = {'note': 'M6a HMM ikinci tanim olcumu (betimsel tasarim girdisi); performans sayisi YOK'
                   + ('; wf-v1 test yillari, YALNIZ etiket sayimi (K-M6-4)' if a.test_years else ''),
           'first_anchor': first_anchor, 'refit_every': H.REFIT_EVERY, 'k': h.k,
           'bic_first': {str(k): v for k, v in h.bic_first.items()}, 'n_anchors': len(h.fit_log),
           'all_converged': all(f['converged'] for f in h.fit_log),
           'mean_desc_with_sd_all': all(f['mean_desc_with_sd'] for f in h.fit_log),
           'mean_desc_with_sd_count': int(sum(f['mean_desc_with_sd'] for f in h.fit_log)),
           'fit_log': h.fit_log, 'seconds': round(secs, 1),
           'windows': {k: [str(v[0].date()), str(v[-1].date()), int(len(v))] for k, v in win.items()}, 'by_window': {}}
    for w, d in win.items():
        two = R.asof_labels(lab, d, 'two_cell')
        state = R.asof_labels(lab, d, 'state')
        trend = R.asof_labels(rule, d, 'trend')
        both = two.notna() & trend.notna()
        mapped = two[both].map({'bull_like': 'bull', 'bear_like': 'bear'})
        res['by_window'][w] = {
            'two_cell': summarize(two),
            'state': summarize(state.map(lambda v: f's{int(v)}' if pd.notna(v) else np.nan)),
            'agreement_with_rule_trend': float((mapped == trend[both]).mean()) if both.any() else None,
            'mean_p_high_vol': float(R.asof_labels(lab, d, 'p_high_vol').astype(float).mean())}
    from data.index_fetcher import DEFAULT_INDEX_CSV, DEFAULT_USDTRY_CSV
    res['provenance'] = {'commit': _git('rev-parse', 'HEAD'), 'git_status_porcelain': _git('status', '--porcelain'),
                         'script_sha256': _sha(os.path.abspath(__file__)),
                         'module_sha256': _sha(os.path.join(ROOT, 'evaluation', 'regimes_hmm.py')),
                         'panel_sha256': _sha(PANEL), 'xu100_sha256': _sha(DEFAULT_INDEX_CSV),
                         'usdtry_sha256': _sha(DEFAULT_USDTRY_CSV)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print('k', h.k, 'bic', res['bic_first'], 'capa', len(h.fit_log), 'yakinsama', res['all_converged'],
          'ortalama sirasi', res['mean_desc_with_sd_count'], '/', len(h.fit_log), 'sn', res['seconds'])
    for w, v in res['by_window'].items():
        print(w, {k: (x['days'], x['episodes_min20']) for k, x in v['two_cell'].items() if isinstance(x, dict)},
              'uyusma', None if v['agreement_with_rule_trend'] is None else round(v['agreement_with_rule_trend'], 3))
    return 0


if __name__ == '__main__':
    sys.exit(main())
