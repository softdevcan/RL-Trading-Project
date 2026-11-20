"""Faz 10 M6 on olcumu: MARL degerlendirme pencerelerinde rejim basina gun ve donem sayisi (ana venv, ag yok).

Soru: H4 (iletisim katkisi rejime gore farkli) gelistirme pencerelerinde (egitim WARMUP..2018-12-31, dogrulama
2019..2020-12-31) sinanabilir mi? Her rejim hucresinde kac seans ve kac bagimsiz sayilabilecek donem var?
Etiket `evaluation/regimes.py` (B6, nedensel, <= t). Kollar (M8 on kaydi, Faz 9 plani M8 "Rejim etiketi"):
  birincil  : XU100 / USDTRY_{t-1}, genisleyen tercil (`load_primary_labels`)
  nominal   : XU100 TL, genisleyen tercil
  rolling5y : XU100 / USDTRY_{t-1}, 5 yillik (1260) kayan tercil
Tarihler MARL panelinin seanslari (`data/marl/panel_m2.npz`), pencere satirlari `eval_m3.window_rows` ile ayni.
Kollar arasi uyusma (ayni gun ayni hucre orani) betimsel. Sonuc yok; yalniz tasarim girdisi.

    python scripts/marl/regime_windows_m6.py --out docs/development/measurements/phase10-m6/regime_windows.json
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

from evaluation import regimes as R  # noqa: E402
from marl.obs_spec import WARMUP_ROWS  # noqa: E402  (saf cekirdek; ana venv'de de calisir)

PANEL = os.path.join('data', 'marl', 'panel_m2.npz')
TRAIN_END, VAL_END = '2018-12-31', '2020-12-31'   # eval_m3.py ile ayni


def window_dates(dates: pd.DatetimeIndex) -> dict:
    train = dates[WARMUP_ROWS:][dates[WARMUP_ROWS:] <= pd.Timestamp(TRAIN_END)]
    val = dates[(dates > pd.Timestamp(TRAIN_END)) & (dates <= pd.Timestamp(VAL_END))]
    return {'train': train, 'val': val}


def arm_labels() -> dict:
    from data.index_fetcher import DEFAULT_INDEX_CSV, load_index_close
    return {
        'birincil': R.load_primary_labels(),
        'nominal': R.regime_labels(load_index_close(DEFAULT_INDEX_CSV)),
        'rolling5y': R.load_xu100_usd_labels(tercile='rolling', tercile_window=1260),
    }


def summarize(lab_on_dates: pd.Series) -> dict:
    """Seans dizisi uzerinde (panel seanslari) hucre basina gun, ardisik blok ve >= 20 gunluk blok."""
    known = lab_on_dates.dropna()
    run_id = (known != known.shift()).cumsum()
    run_len = run_id.map(run_id.value_counts())
    out = {}
    for name, grp in known.groupby(known):
        ids = run_id[grp.index]
        out[str(name)] = {'days': int(len(grp)), 'episodes': int(ids.nunique()),
                          'episodes_min20': int(ids[run_len[grp.index] >= R.MIN_EPISODE_DAYS].nunique())}
    out['_unlabeled_days'] = int(lab_on_dates.isna().sum())
    return out


def _sha(path: str) -> str:
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _git(*args) -> str:
    return subprocess.run(['git', *args], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    z = np.load(PANEL, allow_pickle=True)
    dates = pd.DatetimeIndex(pd.to_datetime(z['dates'].astype(str)))
    win = window_dates(dates)
    arms = arm_labels()
    res = {'note': 'M6 on olcumu (betimsel tasarim girdisi); performans sayisi YOK',
           'windows': {k: [str(v[0].date()), str(v[-1].date()), int(len(v))] for k, v in win.items()},
           'min_episode_days': R.MIN_EPISODE_DAYS, 'arms': {}, 'agreement': {}}
    on = {}
    for arm, lab in arms.items():
        res['arms'][arm] = {}
        for w, d in win.items():
            cell = R.asof_labels(lab, d, 'regime')
            trend = R.asof_labels(lab, d, 'trend')
            on[(arm, w)] = cell
            res['arms'][arm][w] = {'regime': summarize(cell), 'trend': summarize(trend)}
    for w in win:
        base = on[('birincil', w)]
        for arm in ('nominal', 'rolling5y'):
            other = on[(arm, w)]
            both = base.notna() & other.notna()
            res['agreement'][f'birincil_vs_{arm}.{w}'] = float((base[both] == other[both]).mean()) if both.any() else None
    from data.index_fetcher import DEFAULT_INDEX_CSV, DEFAULT_USDTRY_CSV
    res['provenance'] = {'commit': _git('rev-parse', 'HEAD'), 'git_status_porcelain': _git('status', '--porcelain'),
                         'script_sha256': _sha(os.path.abspath(__file__)), 'panel_sha256': _sha(PANEL),
                         'xu100_sha256': _sha(DEFAULT_INDEX_CSV), 'usdtry_sha256': _sha(DEFAULT_USDTRY_CSV)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    for arm in res['arms']:
        for w in win:
            r = res['arms'][arm][w]['regime']
            print(arm, w, {k: (v['days'], v['episodes_min20']) if isinstance(v, dict) else v for k, v in r.items()})
    print('uyusma', res['agreement'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
