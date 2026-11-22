"""Faz 11 S1: M8 ileri panelini guncelle (ANA venv; ag: yfinance, Is Yatirim HisseTekil, EVDS).

Kural ve kapi `marl/forward_panel.py`; plan `docs/development/phase-11-integration-plan.md` §3 S1, §3.3.

    python scripts/marl/forward_panel_m8.py --freeze-source            # bir kez: dondurulmus kaynak kopyasi
    python scripts/marl/forward_panel_m8.py --update [--as-of 2026-11-22T20:00] [--out <olcum.json>]

--update: son yazili gunden sonraki seanslari sirayla dener; ilk engelde durur (sebep log.jsonl'a). Kabul edilen gunler
rows.csv'ye EKLENIR (var olan satir degismez), panel bastan kurulur, onceki panelin tum satirlari bit-es denetlenir,
`data/marl/forward/m8/panel_m8_fwd.npz` yazilir. Ham cekimler `fetch/<zaman>/` altinda SHA'lariyla saklanir.
Ciktilarda yerel yol YAZILMAZ.
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
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from marl import forward_panel as FP  # noqa: E402
from marl import panel as PN  # noqa: E402

TRT = ZoneInfo('Europe/Istanbul')
ISY_URL = ('https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil'
           '?hisse={s}&startdate={a}&enddate={b}')


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch_yf(symbols, start: str, end: str, out_dir: str) -> tuple[dict, dict]:
    import yfinance as yf
    data, shas = {}, {}
    for s in symbols:
        h = yf.Ticker(s).history(start=start, end=end, auto_adjust=True)
        if h.empty:
            continue
        h.index = pd.DatetimeIndex(h.index).tz_convert(TRT).strftime('%Y-%m-%d')
        h = h.rename(columns=str.lower)[['open', 'high', 'low', 'close', 'volume']]
        p = os.path.join(out_dir, f'yf_{s}.csv')
        h.to_csv(p)
        with open(p, 'rb') as fh:
            shas[s] = _sha_bytes(fh.read())
        data[s] = h
    return data, shas


def fetch_isy(symbols, start: str, end: str, out_dir: str) -> tuple[dict, dict]:
    from marl import panel_fix as PF
    a = pd.Timestamp(start).strftime('%d-%m-%Y')
    b = pd.Timestamp(end).strftime('%d-%m-%Y')
    shas = {}
    for s in symbols:
        code = s.replace('.IS', '')
        req = urllib.request.Request(ISY_URL.format(s=code, a=a, b=b), headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
        v = json.loads(body.decode('utf-8'))
        if not v.get('ok'):
            raise RuntimeError(f'Is Yatirim {code}: hatali yanit')
        with open(os.path.join(out_dir, f'{code}.json'), 'wb') as fh:
            fh.write(body)
        shas[s] = _sha_bytes(body)
        time.sleep(1.0)
    return {s: PF.load_isy(s, cache=out_dir) for s in symbols if shas.get(s)}, shas


def fetch_rf(path: str) -> str:
    from data.macro_fetcher import MacroDataFetcher
    df = MacroDataFetcher().fetch_risk_free(save=False)
    df.to_csv(path)
    with open(path, 'rb') as fh:
        return _sha_bytes(fh.read())


def git_head() -> str:
    return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()


def update(as_of: datetime) -> dict:
    root = FP.FWD_ROOT
    os.makedirs(root, exist_ok=True)
    src_man = FP.verify_source()
    panel_path = os.path.join(root, FP.PANEL_FWD)
    rows_path = os.path.join(root, FP.ROWS)
    ref = PN.load_npz(panel_path if os.path.exists(panel_path) else FP.PANEL_M8)
    old_rows = pd.read_csv(rows_path, dtype={'date': str}) if os.path.exists(rows_path) else None
    symbols = [str(s) for s in ref['symbols']]
    anc = FP.anchors(ref)
    d_prev = str(ref['dates'][-1])
    stamp = as_of.strftime('%Y%m%dT%H%M%S')
    fdir = os.path.join(root, 'fetch', stamp)
    os.makedirs(fdir, exist_ok=True)
    start = (pd.Timestamp(min(v[0] for v in anc.values())) - timedelta(days=10)).strftime('%Y-%m-%d')
    end = (as_of + timedelta(days=1)).strftime('%Y-%m-%d')
    yf, yf_sha = fetch_yf(symbols, start, end, fdir)
    isy, isy_sha = fetch_isy(symbols, start, as_of.strftime('%Y-%m-%d'), fdir)
    rf_path = os.path.join(fdir, FP.RF_FILE)
    rf_sha = fetch_rf(rf_path)
    from evaluation.risk_free import load_annual_rate
    rate = load_annual_rate(rf_path)

    cands = sorted({d for y in yf.values() for d in y.index if d > d_prev})
    days, new_rows = [], []
    for d in cands:
        rows, rep = FP.chain_day(d, anc, yf, isy)
        g = FP.day_gate(d, d_prev, rows, rep, len(symbols), rate, as_of)
        g['symbols_report'] = rep
        days.append(g)
        if g['skip']:
            continue
        if not g['accept']:
            break
        new_rows += rows
        for r in rows:
            anc[r['symbol']] = (d, r['close'])
        d_prev = d
    out = {'as_of': as_of.isoformat(timespec='minutes'), 'fetch': stamp, 'commit': git_head(),
           'source_manifest_csv_sha256': src_man['csv_sha256'], 'candidates': cands,
           'days': [{k: v for k, v in g.items() if k != 'symbols_report'} for g in days],
           'fetch_sha256': {'yf': yf_sha, 'isy': isy_sha, 'rf': rf_sha}}
    if new_rows:
        add = pd.DataFrame(new_rows)
        allrows = add if old_rows is None else pd.concat([old_rows, add], ignore_index=True)
        src, _ = FP.load_source()
        arrays = FP.build_arrays(src, allrows, rf_csv=rf_path)
        out['prefix'] = FP.check_prefix(arrays, ref)
        T = len(ref['dates'])
        written = list(arrays['dates'][T:])
        accepted = [g['date'] for g in days if g['accept']]
        if written != accepted:
            raise ValueError(f'kapidan gecen gunler panelde ayni degil: {accepted} vs {written}')
        allrows.to_csv(rows_path, index=False)
        meta = {'kind': 'm8_forward', 'source': src_man, 'rows_sha256': FP._sha(rows_path), 'rf_sha256': rf_sha,
                'last_fetch': stamp, 'commit': out['commit'], 'base_panel_sha256': src_man['panel_sha256']}
        info = PN.save_npz(arrays, panel_path, meta=meta)
        info['path'] = os.path.relpath(panel_path, ROOT).replace(os.sep, '/')
        out['panel'] = info
        out['new_dates'] = written
        out['n_patched'] = int(add['patched'].sum())
        out['max_abs_isy_dev'] = float(add['isy_dev'].abs().max())
    with open(os.path.join(root, FP.LOG), 'a', encoding='utf-8') as fh:
        fh.write(json.dumps({**out, 'days_detail': days}, ensure_ascii=False, default=str) + '\n')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze-source', action='store_true')
    ap.add_argument('--update', action='store_true')
    ap.add_argument('--as-of', default=None, help='ISO zaman (TRT); varsayilan simdi')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    res = {}
    if a.freeze_source:
        res['freeze'] = {k: v for k, v in FP.freeze_source().items() if k != 'isy_sha256'}
        print('kaynak donduruldu:', res['freeze'])
    if a.update:
        as_of = datetime.fromisoformat(a.as_of).replace(tzinfo=TRT) if a.as_of else datetime.now(TRT)
        res['update'] = update(as_of)
        u = res['update']
        print(json.dumps({k: u[k] for k in u if k not in ('fetch_sha256',)}, indent=1, ensure_ascii=False, default=str))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False, default=str)


if __name__ == '__main__':
    main()
