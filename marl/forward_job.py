"""Faz 11 S4: M8 ileri veri gunluk isi (plan phase-11-integration-plan.md §3 S4; ek phase-11-m8-forward-annex.md). ANA venv.

Sira (`run_daily`): arsiv dogrulama -> ileri panel guncelleme (S1, ag) -> marj kapsami (`margin_coverage`) -> defterler
(S3 `sim_ledger`, 72 dondurulmus politika + 4 kural portfoyu) kaldigi yerden kapsamin son gunune kadar ilerler.

Defter deposu `data/marl/forward/m8/ledger/` (git disi; ortak, salt okunur arastirma alani, K-F11-3):
  state.json  {'meta': {...}, 'strategies': {sid: LedgerState}}   (atomik yazim: gecici dosya + os.replace)
  rows.csv    sid + sim_ledger.ROW_KEYS + written_at                (yalniz eklenir)
Gunluk `data/marl/forward/m8/log.jsonl` (S1 panel guncellemeleriyle ayni dosya; kayit turu 'ledger').

Ozellikler (test_marl_forward_job):
  - IDEMPOTENT: yeni gun yoksa hicbir dosya degismez.
  - Bosluk doldurma: k gun sonra tek calisma = her gun calisma (bit-es). Panel her gun bastan kurulsa da onceki satirlar
    bit-es (S1) ve ozellikler nedensel; defter durumdan devam eder (S3 testi 2).
  - Cokme kurtarma: rows.csv'de durumdan fazla satir varsa (rows yazildi, state yazilmadi) fazlasi atilir.
  - Depo YALNIZ ilk ileri gun (FORWARD_START) panelde olunca kurulur (ek §3: ilk defter kaydi dondurmadan sonra).
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from marl import forward_panel as FP
from marl import frozen as FZ
from marl import margin_coverage as MC
from marl import panel as PN
from marl import sim_ledger as SL

FORWARD_START = '2026-11-23'          # ek §2: on kayit dondurma commit'inden sonraki ilk seans
READ_SESSIONS = 126                   # ek §2: tam 126 yurutme seansi
ARCHIVE = os.path.join(FZ.FROZEN_ROOT, 'm8-2026')
READING = os.path.join('docs', 'development', 'measurements', 'phase10-m8', 'm8_reading.json')
LEDGER_DIR = os.path.join(FP.FWD_ROOT, 'ledger')
STATE = 'state.json'
ROWS = 'rows.csv'
LOCK = '.job.lock'
LOCK_STALE_SEC = 2 * 3600
RULES = ('ew', 'bh', 'minvar', 'hrp')
ROW_COLS = ('sid',) + SL.ROW_KEYS + ('written_at',)


# --- stratejiler ----------------------------------------------------------------------------------------------------
def manifest(archive: str = ARCHIVE) -> dict:
    with open(os.path.join(archive, 'manifest.json'), encoding='utf-8') as fh:
        return json.load(fh)


def strategy_ids(archive: str = ARCHIVE, rules=RULES) -> list:
    man = manifest(archive)
    return [f"policy/{p['arm']}/{p['seed']}" for p in man['policies']] + [f'rule/{k}' for k in rules]


def make_strategy(sid: str, archive: str = ARCHIVE):
    kind, *rest = sid.split('/')
    if kind == 'rule':
        return SL.RuleStrategy(rest[0])
    arm, seed = rest
    p = {(q['arm'], q['seed']): q for q in manifest(archive)['policies']}[(arm, seed)]
    return SL.PolicyStrategy(os.path.join(archive, arm, seed), p['ckpt'], sid=sid)


# --- panel ---------------------------------------------------------------------------------------------------------
def truncate(arrays: dict, until: str) -> dict:
    """Satirlari tarih <= until olan panel (zaman eksenli tum diziler kesilir). Nedensel hat: onceki satirlar ayni."""
    dates = np.asarray(arrays['dates'])
    T = len(dates)
    keep = int(np.searchsorted(dates, str(until)[:10], side='right'))
    if keep == T:
        return arrays
    return {k: (v[:keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T else v)
            for k, v in arrays.items()}


def start_row(dates, start_date: str = FORWARD_START):
    """Ilk karar satiri (eval_fold_m8.decision_rows kurali): start_date'ten onceki son satir; ileri gun yoksa None."""
    dates = np.asarray(dates)
    if not (dates >= start_date).any():
        return None
    before = np.flatnonzero(dates < start_date)
    if before.size == 0:
        raise ValueError(f"{start_date} oncesi satir yok")
    return int(before[-1])


# --- depo ----------------------------------------------------------------------------------------------------------
def load_store(store_dir: str) -> tuple[dict, dict, pd.DataFrame]:
    sp, rp = os.path.join(store_dir, STATE), os.path.join(store_dir, ROWS)
    if not os.path.exists(sp):
        return {}, {}, pd.DataFrame(columns=list(ROW_COLS))
    with open(sp, encoding='utf-8') as fh:
        raw = json.load(fh)
    states = {sid: SL.LedgerState.from_json(json.dumps(d)) for sid, d in raw['strategies'].items()}
    rows = pd.read_csv(rp, dtype={'sid': str, 'date': str}, float_precision='round_trip') if os.path.exists(rp) else pd.DataFrame(columns=list(ROW_COLS))
    # Cokme kurtarma: rows yazildi, state yazilmadiysa fazlasi atilir; eksik satir ise bozulmadir
    keep = np.ones(len(rows), dtype=bool)
    for sid, st in states.items():
        m = (rows['sid'] == sid).to_numpy()
        n = int(m.sum())
        if n < st.n_steps:
            raise ValueError(f"defter bozuk: {sid} satir {n} < durum {st.n_steps}")
        if n > st.n_steps:
            keep[np.flatnonzero(m)[st.n_steps:]] = False
    return raw.get('meta', {}), states, rows[keep].reset_index(drop=True)


def _atomic_json(path: str, obj) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def advance(arrays: dict, store_dir: str, sids: list, start_date: str = FORWARD_START,
            verified_until: str | None = None, capital: float = SL.INITIAL_CAPITAL, archive: str = ARCHIVE,
            now: str | None = None, meta_extra: dict | None = None) -> dict:
    """Defterleri panelin son satirina kadar ilerletir. Yeni gun yoksa dosyaya dokunmaz."""
    r0 = start_row(arrays['dates'], start_date)
    if r0 is None:
        return {'status': 'baslamadi', 'start_date': start_date, 'panel_last': str(arrays['dates'][-1])}
    meta, states, rows = load_store(store_dir)
    if states and set(states) != set(sids):
        raise ValueError(f"defterdeki strateji kumesi farkli: {sorted(set(states) ^ set(sids))[:5]}")
    end = len(arrays['dates']) - 1
    todo = [s for s in sids if s not in states or states[s].t < end]
    for s, st in states.items():
        if st.start_row != r0:
            raise ValueError(f"{s}: baslangic satiri degisti ({st.start_row} -> {r0})")
        if st.t > end:
            raise ValueError(f"{s}: defter panelden ileride ({st.t} > {end}); panel geri gitti")
    if not todo:
        return {'status': 'degisiklik_yok', 'last_date': str(arrays['dates'][end]), 'n_sessions': _sessions(states)}
    mk = SL.Market(arrays, verified_until=verified_until)
    stamp = now or datetime.now().isoformat(timespec='seconds')
    new = []
    for sid in sids:
        st = states.get(sid) or SL.LedgerState.start(r0, mk.N, capital, meta={'sid': sid})
        if st.t < end:
            _, out = SL.run(mk, make_strategy(sid, archive), st, end)
            new += [{'sid': sid, **{k: r[k] for k in SL.ROW_KEYS}, 'written_at': stamp} for r in out]
        states[sid] = st
    os.makedirs(store_dir, exist_ok=True)
    add = pd.DataFrame(new, columns=list(ROW_COLS))
    rows_path = os.path.join(store_dir, ROWS)
    full = add if rows.empty else pd.concat([rows, add], ignore_index=True)
    full.to_csv(rows_path + '.tmp', index=False, float_format='%.17g')
    os.replace(rows_path + '.tmp', rows_path)
    meta = {**meta, 'start_date': start_date, 'start_row': r0, 'archive': os.path.basename(archive),
            'capital': capital, 'updated_at': stamp, 'panel_sha256': arrays.get('sha256'),
            'verified_until': verified_until, **(meta_extra or {})}
    meta.setdefault('created_at', stamp)
    _atomic_json(os.path.join(store_dir, STATE), {'meta': meta, 'strategies': {
        s: json.loads(states[s].to_json()) for s in sids}})
    dates = sorted(set(add['date']))
    n = _sessions(states)
    return {'status': 'yazildi', 'new_dates': dates, 'n_new_rows': len(add), 'n_sessions': n,
            'last_date': str(arrays['dates'][end]), 'read_session_reached': n >= READ_SESSIONS}


def _sessions(states: dict) -> int:
    ns = {st.n_steps for st in states.values()}
    if len(ns) > 1:
        raise ValueError(f"stratejilerin seans sayilari farkli: {sorted(ns)}")
    return ns.pop() if ns else 0


def read_rows(store_dir: str = LEDGER_DIR) -> pd.DataFrame:
    return load_store(store_dir)[2]


def session_dates(store_dir: str = LEDGER_DIR) -> list:
    rows = read_rows(store_dir)
    if rows.empty:
        return []
    return sorted(set(rows['date']))


# --- gunluk is -----------------------------------------------------------------------------------------------------
class JobLock:
    def __init__(self, path: str):
        self.path = path

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if os.path.exists(self.path) and time.time() - os.path.getmtime(self.path) > LOCK_STALE_SEC:
            os.remove(self.path)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(f"ileri veri isi zaten calisiyor (kilit {self.path})") from None
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return self

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass


def _log(entry: dict) -> None:
    os.makedirs(FP.FWD_ROOT, exist_ok=True)
    with open(os.path.join(FP.FWD_ROOT, FP.LOG), 'a', encoding='utf-8') as fh:
        fh.write(json.dumps({'kind': 'ledger', **entry}, ensure_ascii=False, default=str) + '\n')


WRITER = 'WRITER'                     # ileri deponun tek yazari (plan §3.12): dosyadaki kimlik = MARL_FORWARD_WRITER_ID
WRITER_ENV = 'MARL_FORWARD_WRITER_ID'


def writer_check(root: str | None = None) -> dict:
    """Depoda WRITER dosyasi varsa yalniz ayni kimlikle calisan surec yazar (iki makine iki ayri defter uretmesin).
    Dosya yoksa (devirden once, testler) serbest."""
    p = os.path.join(root or FP.FWD_ROOT, WRITER)
    if not os.path.exists(p):
        return {'ok': True, 'writer': None}
    with open(p, encoding='utf-8') as fh:
        want = fh.read().strip()
    have = os.environ.get(WRITER_ENV, '').strip()
    return {'ok': have == want, 'writer': want, 'this': have or None}


def run_daily(as_of: datetime | None = None, update_panel: bool = True, store_dir: str = LEDGER_DIR) -> dict:
    """Panel -> karar -> defter. as_of TRT (None = simdi). Ag: yalniz update_panel=True iken (S1 cekimi)."""
    from zoneinfo import ZoneInfo
    as_of = as_of or datetime.now(ZoneInfo('Europe/Istanbul'))
    wc = writer_check()
    if not wc['ok']:   # baska makinenin deposu: hicbir dosyaya (gunluk dahil) dokunulmaz
        return {'as_of': as_of.isoformat(timespec='minutes'), 'status': 'yazar_degil', 'writer': wc}
    rep = {'as_of': as_of.isoformat(timespec='minutes'), 'started': datetime.now().isoformat(timespec='seconds')}
    with JobLock(os.path.join(FP.FWD_ROOT, LOCK)):
        try:
            with open(READING, encoding='utf-8') as fh:
                fwd = json.load(fh)['forward_policies_2026']
            rep['archive'] = {k: v for k, v in FZ.verify_archive(ARCHIVE, forward=fwd).items() if k != 'files'}
            if update_panel:
                try:
                    rep['panel_update'] = _update_panel(as_of)
                except Exception as e:   # noqa: BLE001 — ag hatasi panelde zaten olan gunleri bekletmez
                    rep['panel_update'] = {'error': f'{type(e).__name__}: {e}'}
            path = os.path.join(FP.FWD_ROOT, FP.PANEL_FWD)
            arrays = PN.load_npz(path if os.path.exists(path) else FP.PANEL_M8)
            cov = MC.coverage_until(str(arrays['dates'][-1]))
            rep['margin_coverage'] = cov
            arrays = truncate(arrays, cov['until'])
            rep['ledger'] = advance(arrays, store_dir, strategy_ids(), verified_until=cov['until'],
                                    meta_extra={'margin_last_check': cov['last_check']})
            rep['status'] = rep['ledger']['status']
        except Exception as e:   # noqa: BLE001 — gunluge yazilir, cagirana da doner
            rep['status'] = 'hata'
            rep['error'] = f'{type(e).__name__}: {e}'
        rep['finished'] = datetime.now().isoformat(timespec='seconds')
        _log(rep)
    return rep


def _update_panel(as_of: datetime) -> dict:
    """S1 panel guncellemesi (scripts/marl/forward_panel_m8.update); ozet doner."""
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location('forward_panel_m8',
                                                  os.path.join(root, 'scripts', 'marl', 'forward_panel_m8.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    u = mod.update(as_of.replace(tzinfo=None) if as_of.tzinfo is None else as_of)
    return {'new_dates': u.get('new_dates', []), 'n_patched': u.get('n_patched', 0),
            'stopped': [{'date': d['date'], 'reason': d.get('reason')} for d in u['days'] if not d['accept'] and not d['skip']]}


def status(store_dir: str = LEDGER_DIR) -> dict:
    """Pano ve API icin ozet (ag yok, salt okuma)."""
    out = {'forward_start': FORWARD_START, 'read_sessions': READ_SESSIONS}
    meta, states, rows = load_store(store_dir)
    out['n_sessions'] = _sessions(states) if states else 0
    out['last_date'] = str(rows['date'].max()) if not rows.empty else None
    out['n_strategies'] = len(states)
    out['meta'] = meta
    logp = os.path.join(FP.FWD_ROOT, FP.LOG)
    last = None
    if os.path.exists(logp):
        with open(logp, encoding='utf-8') as fh:
            for line in fh:
                d = json.loads(line)
                if d.get('kind') == 'ledger':
                    last = d
    out['last_run'] = None if last is None else {k: last.get(k) for k in ('as_of', 'status', 'error', 'finished')} | {
        'stopped': (last.get('panel_update') or {}).get('stopped'), 'margin_coverage': last.get('margin_coverage')}
    return out
