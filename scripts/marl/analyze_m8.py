"""Faz 10 M8 okuma betigi (ana venv) — M8 tasarimi §6 ON KAYIT v1.0; kosudan ONCE yazildi (A4).

Iki mod:
  --status <kat>  Yil sonu SAGLIK kaydi (§6.3 "kismi sonuc okunmaz"): biten/eksik hucre, cokme ve yeniden deneme,
                  sure, git izi, panel/graf SHA, pencereler, yapilandirma, secimi tanimsiz (degenerate) hucre sayisi.
                  Test dosyalarini ACMAZ, basarim sayisi yazmaz. Cikti `<root>/status/<kat>.json` + `<root>/progress.md`.
  (varsayilan)    Nihai okuma, 6 kat bittikten sonra BIR KEZ: §6.4 karar kurali (`m8_stats`), §6.5 betimsel,
                  §6.6 denetimler. Cikti `measurements/phase10-m8/m8_reading.json` (yerel yol yok).

Hucre metrigi (§6.4): SR = marl.selection.excess_sharpe(R, rf = R - X) test yilinda; tanimsizsa 0 (tumuyle nakit) ve
sayilir. Karsitliklar (esleme ayni tohum indeksi, ayni kat):
  H1 M-none <-> S-esm (FARK_VAR_MARL / FARK_VAR_SARL); H2 M-none <-> S-agr, H3 M-gat <-> M-self (Holm, p_max);
  NK M-none <-> bos kol (negatif kontrol; FARK_VAR ise tum etiketler `negatif_kontrol_uyarisi`).
Ekler: `katlar_tutarli`/`yila_bagli` (kat ortalamasi isaret-cevirme, 64 duzen), `secime_dayanikli`/`secime_duyarli`
(test_last ayni isaret ve p_max <= 0,10), `tanimsiz_uyarisi` (kolda tanimsiz > %10 ya da fark > 5 puan; duyarlilik
satiri tanimsizlar dusurulmus), `UYARILI` (bir denetim bozuk). Kolda >= 8 eksik hucre -> o kolun karsitliklari EKSIK.

  python scripts/marl/analyze_m8.py --status 2021
  python scripts/marl/analyze_m8.py --out docs/development/measurements/phase10-m8/m8_reading.json [--register]
  python scripts/marl/analyze_m8.py --self-test
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import m8_stats as ST  # noqa: E402
from marl.selection import excess_sharpe  # noqa: E402

PREREG = 'phase-10-m8-design.md §6 ON KAYIT v1.0'
FOLDS = tuple(str(y) for y in range(2021, 2027))
ARMS = ('M-none', 'S-esm', 'S-agr', 'M-gat', 'M-self', 'null')
MAIN_SEEDS = tuple(range(201, 213))
NULL_SEEDS = tuple(range(213, 225))
ARM_FLAGS = {'M-none': ('none', 'none'), 'S-esm': ('esmimari', 'none'), 'S-agr': ('agirlik', 'none'),
             'M-gat': ('none', 'gat'), 'M-self': ('none', 'self'), 'null': ('none', 'none')}   # (sarl, comm)
EXPECTED_ARGS = {'frames': 600_000, 'checkpoints': 10, 'cost_bps': 20.0, 'lr': 5e-4, 'batched_env': True,
                 'alpha': 0.25, 'n_envs': 10, 'epochs': 10, 'frames_per_batch': 6000, 'init_std': 0.12,
                 'episode_len': 252, 'algo': 'mappo', 'use_state': False, 'pred_path': None, 'extras_path': None,
                 'conf_mode': 'none', 'panel': os.path.join('data', 'marl', 'panel_m8.npz'), 'graph_rule': 'sector+mst'}
CONTRASTS = (('H1', 'M-none', 'S-esm', 'MARL', 'SARL'), ('H2', 'M-none', 'S-agr', 'M-none', 'S-agr'),
             ('H3', 'M-gat', 'M-self', 'M-gat', 'M-self'), ('NK', 'M-none', 'null', 'M-none', 'null'))
RULES = ('argmax', 'last')
COST_GRID = (0, 10, 20, 25, 50)
COST_BPS = 20.0
MISSING_LIMIT = 8                 # §6.6: 72 hucrenin %10'undan fazlasi
UNDEF_RATE, UNDEF_GAP = 0.10, 0.05
LAST_P = 0.10
Z975, Z80 = 1.959963984540054, 0.8416212335729143
PANEL_JSON = os.path.join('docs', 'development', 'measurements', 'phase10-m8', 'panel_m8.json')
GRAPH = os.path.join('docs', 'development', 'measurements', 'phase10-m4', 'graph_rules.json')
GRAPH_RECORD = os.path.join('docs', 'development', 'measurements', 'phase10-m4', 'neighbor_map.json')


def seeds_of(arm: str):
    return NULL_SEEDS if arm == 'null' else MAIN_SEEDS


def fold_dates(fold: str) -> dict:
    """wf-v1 (evaluation/splits.py) kat sinirlari ve run_m8 egitim sonu (dogrulama baslangicindan bir gun once)."""
    from evaluation.splits import get_spec
    f = get_spec('wf-v1').fold(fold)
    y = int(fold)
    return {'train_start': f.train_start, 'val_start': f.val_start, 'test_start': f.test_start,
            'test_end': f.test_end, 'train_end': f'{y - 2}-12-31'}


def provenance() -> dict:
    git = lambda *c: subprocess.run(['git', *c], capture_output=True, text=True, cwd=ROOT).stdout.strip()  # noqa: E731
    with open(os.path.abspath(__file__), 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    with open(os.path.join(HERE, 'm8_stats.py'), 'rb') as fh:
        sha_st = hashlib.sha256(fh.read()).hexdigest()
    return {'commit': git('rev-parse', 'HEAD'), 'git_status_porcelain': git('status', '--porcelain'),
            'script_sha256': sha, 'm8_stats_sha256': sha_st}


def _sha(path: str) -> str:
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _json(path: str):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


# ---------------------------------------------------------------- kayit toplama ve denetimler (§6.6)

def load_record(root: str, fold: str, arm: str, seed: int) -> dict | None:
    d = os.path.join(root, fold, arm, f's{seed}')
    fe, tr = os.path.join(d, 'eval', 'fold_eval.json'), os.path.join(d, 'train.json')
    if not (os.path.exists(fe) and os.path.exists(tr)):
        return None
    return {'fold': fold, 'arm': arm, 'seed': seed, 'dir': d, 'fe': _json(fe), 'train': _json(tr)}


def record_checks(r: dict, ref: dict) -> dict:
    """Bir hucrenin denetimleri; ref = {'panel_sha', 'graph_sha'}. Basarim okumaz."""
    fe, tr, a = r['fe'], r['train'], r['train'].get('args', {})
    fd = fold_dates(r['fold'])
    sarl, comm = ARM_FLAGS[r['arm']]
    cfg = all(a.get(k) == v for k, v in EXPECTED_ARGS.items()) and a.get('seed') == r['seed'] \
        and a.get('sarl', 'none') == sarl and a.get('comm', 'none') == comm and a.get('train_end') == fd['train_end'] \
        and len(fe.get('validation', [])) == EXPECTED_ARGS['checkpoints']
    tw, vw, sw = tr.get('train_window', ['', '']), fe['windows']['val'], fe['windows']['test']
    win = fd['train_start'] <= tw[0] and tw[1] < fd['val_start'] <= vw[0] and vw[1] < fd['test_start'] <= sw[0] \
        and sw[1] < fd['test_end'] and fe.get('fold') == r['fold']
    graph = (tr.get('graph_sha256') == ref['graph_sha']) if comm == 'gat' else tr.get('graph_sha256') is None
    return {'commit_train': tr.get('commit'), 'commit_eval': fe['provenance'].get('commit'),
            'clean': not tr.get('git_status_porcelain') and not fe['provenance'].get('git_status_porcelain'),
            'panel': tr.get('panel_sha256') == ref['panel_sha'] == fe.get('panel_sha256'),
            'graph': graph, 'windows': win, 'config': cfg,
            'degenerate': bool(fe['selection'].get('degenerate')),
            'same_as_argmax_last': bool(fe['tests'].get('last', {}).get('same_as_argmax'))}


def reference_shas() -> dict:
    return {'panel_sha': _json(os.path.join(ROOT, PANEL_JSON))['npz']['sha256'],
            'graph_sha': _json(os.path.join(ROOT, GRAPH_RECORD))['graph_sha256'],
            'graph_file_sha': _sha(os.path.join(ROOT, GRAPH))}


def summarize_checks(checks: list, ref: dict) -> dict:
    commits = sorted({c['commit_train'] for c in checks} | {c['commit_eval'] for c in checks})
    out = {'n': len(checks), 'commits': commits, 'single_commit': len(commits) == 1,
           'all_clean': all(c['clean'] for c in checks), 'panel_sha_ok': all(c['panel'] for c in checks),
           'graph_sha_ok': all(c['graph'] for c in checks) and ref['graph_sha'] == ref['graph_file_sha'],
           'windows_ok': all(c['windows'] for c in checks), 'config_ok': all(c['config'] for c in checks)}
    out['all_ok'] = all(out[k] for k in ('single_commit', 'all_clean', 'panel_sha_ok', 'graph_sha_ok', 'windows_ok',
                                         'config_ok'))
    return out


def _jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as fh:
        return [json.loads(x) for x in fh if x.strip()]


# ---------------------------------------------------------------- yil sonu saglik kaydi

def status(root: str, fold: str, ref: dict | None = None) -> dict:
    """Test dosyasi acilmaz, basarim yazilmaz (§6.3)."""
    ref = ref or reference_shas()
    per_arm, checks, missing = {}, [], []
    for arm in ARMS:
        done = deg = 0
        for s in seeds_of(arm):
            r = load_record(root, fold, arm, s)
            if r is None:
                missing.append(f'{arm}/s{s}')
                continue
            c = record_checks(r, ref)
            checks.append(c)
            done += 1
            deg += c['degenerate']
        per_arm[arm] = {'done': done, 'expected': len(seeds_of(arm)), 'degenerate_selection': deg}
    fails = [f for f in _jsonl(os.path.join(root, 'failures.jsonl')) if str(f.get('fold')) == fold]
    timing = [t for t in _jsonl(os.path.join(root, 'timing.jsonl')) if str(t.get('fold')) == fold]
    wall = None
    if timing:
        starts = [dt.datetime.fromisoformat(t['start']) for t in timing]
        ends = [dt.datetime.fromisoformat(t['start']) + dt.timedelta(seconds=t['sec']) for t in timing]
        wall = round((max(ends) - min(starts)).total_seconds() / 60, 1)
    sec_arm = {arm: round(float(np.mean([t['sec'] for t in timing if t['arm'] == arm])) / 60, 2)
               for arm in ARMS if any(t['arm'] == arm for t in timing)}
    prev = {}
    for f in FOLDS:
        p = os.path.join(root, 'status', f'{f}.json')
        if f != fold and os.path.exists(p):
            prev[f] = _json(p)['checks']['commits']
    summ = summarize_checks(checks, ref) if checks else {'n': 0, 'all_ok': False, 'commits': []}
    same_as_prev = all(c == summ['commits'] for c in prev.values())
    out = {'fold': fold, 'prereg': PREREG, 'written': dt.datetime.now().isoformat(timespec='seconds'),
           'note': 'SAGLIK kaydi; test dosyasi acilmadi, basarim yok (on kayit §6.3)',
           'cells_done': sum(v['done'] for v in per_arm.values()), 'cells_expected': 72, 'per_arm': per_arm,
           'missing': missing, 'failures': fails, 'n_retry_success': sum(1 for t in timing if t.get('attempt') == 2),
           'wall_minutes': wall, 'mean_minutes_per_run_by_arm': sec_arm, 'checks': summ,
           'commit_same_as_previous_folds': same_as_prev, 'previous_folds': sorted(prev)}
    out['ok'] = bool(summ['all_ok'] and not missing and same_as_prev)
    return out


def write_status(root: str, st: dict):
    os.makedirs(os.path.join(root, 'status'), exist_ok=True)
    with open(os.path.join(root, 'status', f"{st['fold']}.json"), 'w', encoding='utf-8') as fh:
        json.dump(st, fh, indent=1, ensure_ascii=False)
    md = os.path.join(root, 'progress.md')
    new = not os.path.exists(md)
    with open(md, 'a', encoding='utf-8') as fh:
        if new:
            fh.write('# M8 kosu ilerlemesi (saglik kaydi; basarim yok, on kayit §6.3)\n\n'
                     '| Kat | Yazildi | Hucre | Eksik | Cokme | Duvar (dk) | Commit | Denetim | Durum |\n'
                     '|---|---|---|---|---|---|---|---|---|\n')
        c = st['checks']
        fh.write(f"| {st['fold']} | {st['written']} | {st['cells_done']}/72 | {len(st['missing'])} | "
                 f"{len(st['failures'])} | {st['wall_minutes']} | {','.join(x[:8] for x in c['commits'])} | "
                 f"{'tamam' if c['all_ok'] else 'BOZUK'} | {'TAMAM' if st['ok'] else 'DIKKAT'} |\n")


# ---------------------------------------------------------------- hucre metrikleri

def load_series(r: dict, rule: str) -> dict:
    t = r['fe']['tests'][rule]
    path = os.path.join(r['dir'], 'eval', t['file'])
    if _sha(path) != t['sha256']:
        raise ValueError(f"test dosyasi SHA tutmuyor: {r['fold']}/{r['arm']}/s{r['seed']} {rule}")
    z = np.load(path, allow_pickle=False)
    return {k[len('policy_'):]: z[k] for k in z.files}


def cell_metrics(ser: dict) -> dict:
    R, X = np.asarray(ser['R'], float), np.asarray(ser['X'], float)
    sr = excess_sharpe(R, rf=R - X)
    cw = np.asarray(ser['cash_weight'], float)
    out = {'sr': 0.0 if sr is None else sr, 'undefined': sr is None, 'mean_excess_ann': float(X.mean() * 252),
           'cash': float(cw.mean()), 'gross': float(1 - cw.mean()), 'turnover': float(np.mean(ser['turnover'])),
           'sell_blocked': int(np.sum(ser.get('n_sell_blocked', 0)))}
    tau, c0 = np.asarray(ser['turnover'], float), COST_BPS / 2 / 1e4
    for bps in COST_GRID:         # §6.5 D4 ex post: agirlik yolu maliyetten bagimsiz (accounting.step_fractional)
        Rc = (1 + R) * (1 - bps / 2 / 1e4 * tau) / (1 - c0 * tau) - 1
        s = excess_sharpe(Rc, rf=R - X)
        out[f'sr_{bps}'] = 0.0 if s is None else s
    return out


def matrices(cells: dict) -> dict:
    """cells[(fold, arm, seed, rule)] -> metrik; donus M[rule][key][arm] (6 x 12), eksik NaN."""
    keys = ('sr', 'undefined', 'mean_excess_ann', 'cash', 'gross', 'turnover') + tuple(f'sr_{b}' for b in COST_GRID)
    M = {rule: {k: {arm: np.full((len(FOLDS), 12), np.nan) for arm in ARMS} for k in keys} for rule in RULES}
    for (fold, arm, seed, rule), m in cells.items():
        i, j = FOLDS.index(fold), seeds_of(arm).index(seed)
        for k in keys:
            M[rule][k][arm][i, j] = float(m[k])
    return M


# ---------------------------------------------------------------- karar kurali (§6.4)

def contrast_block(M: dict, a: str, b: str, la: str, lb: str, R: int, B: int) -> dict:
    d = M['argmax']['sr'][a] - M['argmax']['sr'][b]
    c = ST.contrast(d, la, lb, R=R, B=B)
    fs = ST.fold_signflip(d)
    dl = M['last']['sr'][a] - M['last']['sr'][b]
    cl = ST.contrast(dl, la, lb, R=R, B=B)
    und = {x: M['argmax']['undefined'][x] for x in (a, b)}
    rate = {x: float(np.nansum(u) / max(np.isfinite(u).sum(), 1)) for x, u in und.items()}
    undef_flag = max(rate.values()) > UNDEF_RATE or abs(rate[a] - rate[b]) > UNDEF_GAP
    out = {**c, 'fold_signflip': fs, 'per_fold_delta': fs['fold_means'],
           'suffix_folds': 'katlar_tutarli' if fs['p'] <= ST.ALPHA else 'yila_bagli',
           'last': cl, 'suffix_selection': ('secime_dayanikli' if np.sign(cl['delta_bar']) == np.sign(c['delta_bar'])
                                             and cl['p_max'] <= LAST_P else 'secime_duyarli'),
           'undefined_rate': rate, 'undefined_flag': bool(undef_flag)}
    if undef_flag:
        ds = np.where((und[a] == 1) | (und[b] == 1), np.nan, d)
        out['sensitivity_undefined_dropped'] = ST.contrast(ds, la, lb, R=R, B=B)
    d21 = d[:1]
    out['fold_2021'] = {'delta_bar': ST.delta_bar(d21), 'ci95': ST.bootstrap(d21, B=B)['ci95'],
                        'n_pairs': int(np.isfinite(d21).sum()), 'note': 'etiket yok (§6.7)'}
    s = np.array([np.nanstd(M['argmax']['sr'][x], axis=1, ddof=1) for x in (a, b)])
    sig = np.sqrt(np.nanmean(s ** 2, axis=1))
    out['mde_recomputed'] = {'sigma_seed': sig.tolist(), 'note': 'betimsel, normal yaklasim, iki yonlu %5 / guc %80',
                             'mde80': float((Z975 + Z80) * np.sqrt((sig ** 2).sum()) / np.sqrt(max(c['n_pairs'], 1)))}
    return out


def decide(M: dict, missing: dict, warn_checks: bool, R: int = ST.R_SIGNFLIP, B: int = ST.B_BOOT) -> dict:
    res = {name: contrast_block(M, a, b, la, lb, R, B) for name, a, b, la, lb in CONTRASTS}
    hp = ST.holm([res['H2']['p_max'], res['H3']['p_max']])
    for name, p in zip(('H2', 'H3'), hp):
        r = res[name]
        r['p_holm'] = p
        _, a, b, la, lb = next(c for c in CONTRASTS if c[0] == name)
        r['label'] = f"FARK_VAR_{la if r['delta_bar'] > 0 else lb}" if p <= ST.ALPHA and r['delta_bar'] != 0 \
            else 'FARK_GOZLENMEDI'
    nk_warn = res['NK']['label'].startswith('FARK_VAR')
    for name, a, b, *_ in CONTRASTS:
        r = res[name]
        if missing[a] >= MISSING_LIMIT or missing[b] >= MISSING_LIMIT:
            r['label'] = 'EKSIK'
        tags = [r['suffix_folds'], r['suffix_selection']]
        tags += ['tanimsiz_uyarisi'] if r['undefined_flag'] else []
        tags += ['negatif_kontrol_uyarisi'] if nk_warn and name != 'NK' else []
        tags += ['UYARILI'] if warn_checks else []
        r['tags'] = tags
        r['label_full'] = f"{r['label']} ({', '.join(tags)}; 6 yilda ortalama)"
    return res


# ---------------------------------------------------------------- betimsel (§6.5)

def spearman(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None
    rx, ry = np.argsort(np.argsort(x[ok])), np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def descriptive(M: dict, recs: dict, cells: dict, series: dict, train_days: dict) -> dict:
    out = {'arms': {}, 'per_fold': {}}
    sr = M['argmax']['sr']
    for arm in ARMS:
        m = sr[arm]
        out['arms'][arm] = {
            'mean_sr': float(np.nanmean(m)), 'std_sr': float(np.nanstd(m, ddof=1)),
            'per_fold_mean': np.nanmean(m, axis=1).tolist(), 'per_fold_sd_seed': np.nanstd(m, axis=1, ddof=1).tolist(),
            'sd_ratio_to_mnone': (np.nanstd(m, axis=1, ddof=1) / np.nanstd(sr['M-none'], axis=1, ddof=1)).tolist(),
            'per_fold_mean_excess_ann': np.nanmean(M['argmax']['mean_excess_ann'][arm], axis=1).tolist(),
            'per_fold_cash': np.nanmean(M['argmax']['cash'][arm], axis=1).tolist(),
            'per_fold_gross': np.nanmean(M['argmax']['gross'][arm], axis=1).tolist(),
            'per_fold_turnover': np.nanmean(M['argmax']['turnover'][arm], axis=1).tolist(),
            'n_undefined': int(np.nansum(M['argmax']['undefined'][arm])),
            'last': {'mean_sr': float(np.nanmean(M['last']['sr'][arm]))},
            'share_last_equals_argmax': float(np.mean([c['same_as_argmax_last'] for (f, a, s), c in recs.items()
                                                       if a == arm])) if any(a == arm for _, a, _ in recs) else None,
            'cost_grid_mean_sr': {str(b): float(np.nanmean(M['argmax'][f'sr_{b}'][arm])) for b in COST_GRID}}
    try:
        from evaluation.seed_stats import aggregate_with_ci
        out['rliable'] = aggregate_with_ci({a: np.nan_to_num(sr[a].T) for a in ARMS if a != 'null'}, reps=2000, seed=0)
    except Exception as e:   # noqa: BLE001  (rliable yoksa betimsel satir duser, karar etkilenmez)
        out['rliable'] = f'hesaplanmadi: {type(e).__name__}'
    for i, fold in enumerate(FOLDS):
        val = [(c['val_metric'], c['sr']) for (f, a, s, rule), c in cells.items() if f == fold and rule == 'argmax'
               and c.get('val_metric') is not None]
        r = float(np.corrcoef(*zip(*val))[0, 1]) if len(val) > 2 else None
        out['per_fold'][fold] = {'train_days': train_days.get(fold),
                                 'frames_per_train_day': (EXPECTED_ARGS['frames'] / train_days[fold]
                                                          if train_days.get(fold) else None),
                                 'val_test_corr': r}
    out['trend_spearman'] = {}
    out['cost_grid_delta'] = {}
    for name, a, b, *_ in CONTRASTS:
        dfold = np.nanmean(sr[a] - sr[b], axis=1)
        out['trend_spearman'][name] = spearman(np.arange(len(FOLDS)), dfold)
        out['cost_grid_delta'][name] = {str(bp): float(np.nanmean(M['argmax'][f'sr_{bp}'][a] - M['argmax'][f'sr_{bp}'][b]))
                                        for bp in COST_GRID}
    out['hac_pooled'] = hac_pooled(series)
    out['dsr'] = dsr_block(series)
    return out


def hac_pooled(series: dict) -> dict:
    """K20: kat basina tohum ciftlerinin gunluk fazla getirisi, ortak HAC (betimsel)."""
    from evaluation.sharpe_tests import sharpe_diff_hac_pooled
    out = {}
    for name, a, b, *_ in CONTRASTS:
        out[name] = {}
        for fold in FOLDS:
            pairs = [(series[(fold, a, sa, 'argmax')]['X'], series[(fold, b, sb, 'argmax')]['X'])
                     for sa, sb in zip(seeds_of(a), seeds_of(b))
                     if (fold, a, sa, 'argmax') in series and (fold, b, sb, 'argmax') in series]
            try:
                h = sharpe_diff_hac_pooled(pairs)
                out[name][fold] = {'D_per_period': h.get('delta', h.get('D')), 'p': h.get('p_value', h.get('p')),
                                   'n_pairs': len(pairs)}
            except Exception as e:   # noqa: BLE001
                out[name][fold] = f'hesaplanmadi: {e}'
    return out


def dsr_block(series: dict) -> dict:
    """N = kol x secim kurali (bos kol haric, 10); seri = tohum ortalamasi gunluk fazla getiri, katlar ardisik."""
    from evaluation.sharpe import deflated_sharpe_ratio
    conf = {}
    for arm in ARMS[:-1]:
        for rule in RULES:
            parts = []
            for fold in FOLDS:
                xs = [series[k]['X'] for k in series if k[0] == fold and k[1] == arm and k[3] == rule]
                if xs:
                    n = min(len(x) for x in xs)
                    parts.append(np.mean([x[:n] for x in xs], axis=0))
            if parts:
                conf[f'{arm}|{rule}'] = np.concatenate(parts)
    srs = {k: float(v.mean() / v.std(ddof=1)) for k, v in conf.items() if v.std(ddof=1) > 0}
    if len(srs) < 2:
        return {'note': 'yetersiz yapilandirma'}
    V = float(np.var(list(srs.values()), ddof=1))
    return {'n_trials': len(srs), 'n_is_lower_bound': True, 'var_trials_per_period': V,
            'per_config': {k: deflated_sharpe_ratio(n_trials=len(srs), var_trials=V, returns=conf[k])['dsr']
                           for k in srs}, 'note': 'betimsel; M8 en iyiyi secmez (§6.5)'}


def baselines(panel_path: str, test_rows: dict, test_dates: dict) -> dict:
    """§6.2: ew, bh, minvar, hrp (ayni muhasebe, 20 bps, ayni test pencereleri) + XU030 referans satiri."""
    from marl import panel as PN
    from marl.rule_baselines import rule_series
    arrays = PN.load_npz(panel_path)
    out = {}
    for fold, (ts, te) in test_rows.items():
        out[fold] = {}
        for kind in ('ew', 'bh', 'minvar', 'hrp'):
            s = rule_series(arrays, ts, te, kind, COST_BPS)
            m = cell_metrics({**s, 'n_sell_blocked': 0})
            out[fold][kind] = {k: m[k] for k in ('sr', 'undefined', 'mean_excess_ann', 'cash', 'turnover')}
        try:
            import pandas as pd
            from evaluation.baselines import index_returns, load_xu030
            d = pd.to_datetime(test_dates[fold])
            ir = index_returns(load_xu030(), d).to_numpy()
            rf = arrays['rf'][ts + 1:te + 1]
            out[fold]['xu030_ref'] = {'sr': excess_sharpe(ir, rf=rf),
                                      'note': 'fiyat endeksi, temettusuz, maliyetsiz; kiyas degil (D1)'}
        except Exception as e:   # noqa: BLE001
            out[fold]['xu030_ref'] = f'hesaplanmadi: {type(e).__name__}'
    return out


def register(cells: dict, series: dict, recs: dict, panel_sha: str, directory: str | None = None) -> int:
    """§6.6 C2: her kol x secim kurali x kat x tohum bir deneme. Protokolde yalniz olculen maddeler beyan edilir;
    olculmeyenler acik kalir (kayit N sayimi icindir)."""
    from evaluation.splits import get_spec
    from evaluation.trial_registry import exposed_windows_overlapping, register_trial
    spec = get_spec('wf-v1')
    n = 0
    for (fold, arm, seed, rule), m in sorted(cells.items()):
        fd = fold_dates(fold)
        ser = series[(fold, arm, seed, rule)]
        register_trial('marl_m8', config={'prereg': PREREG, 'arm': arm, 'selection_rule': rule, **EXPECTED_ARGS},
                       seed=seed, metrics={'excess_sharpe': m['sr'], 'undefined': m['undefined']},
                       returns=ser['R'], dates=list(ser['date']), split='test', fold=fold,
                       split_spec={'name': spec.name, 'digest': spec.digest()}, data_sha256=panel_sha,
                       protocol={'price_stats_source': 'train_window', 'split_from_spec': True,
                                 'selection_metric': 'val_excess_sharpe',
                                 'eval_window_unexposed': not exposed_windows_overlapping(fd['test_start'],
                                                                                          fd['test_end']),
                                 'hyperparams_clean': True},
                       directory=directory, tags=['phase10_m8'])
        n += 1
    return n


# ---------------------------------------------------------------- nihai okuma

def read_all(root: str, R: int = ST.R_SIGNFLIP, B: int = ST.B_BOOT, ref: dict | None = None,
             panel_path: str | None = None, panel_dates=None, do_register: bool = False,
             registry_dir: str | None = None) -> dict:
    ref = ref or reference_shas()
    recs, cells, series, checks, missing = {}, {}, {}, [], {a: 0 for a in ARMS}
    test_rows, test_dates, train_days = {}, {}, {}
    for fold in FOLDS:
        for arm in ARMS:
            for s in seeds_of(arm):
                r = load_record(root, fold, arm, s)
                if r is None:
                    missing[arm] += 1
                    continue
                c = record_checks(r, ref)
                checks.append(c)
                recs[(fold, arm, s)] = c
                test_rows.setdefault(fold, tuple(r['fe']['rows']['test']))
                for rule in RULES:
                    ser = load_series(r, rule)
                    series[(fold, arm, s, rule)] = ser
                    m = cell_metrics(ser)
                    m['val_metric'] = r['fe']['selection'].get('metric') if rule == 'argmax' else None
                    cells[(fold, arm, s, rule)] = m
                test_dates.setdefault(fold, list(series[(fold, arm, s, 'argmax')]['date']))
                if fold not in train_days and panel_dates is not None:
                    tw = r['train'].get('train_window')
                    train_days[fold] = int(((panel_dates >= tw[0]) & (panel_dates <= tw[1])).sum())
    summ = summarize_checks(checks, ref)
    M = matrices(cells)
    res = decide(M, missing, warn_checks=not summ['all_ok'], R=R, B=B)
    from evaluation.trial_registry import exposed_windows_overlapping
    exposure = {f: exposed_windows_overlapping(fold_dates(f)['test_start'], fold_dates(f)['test_end']) for f in FOLDS}
    fwd = {arm: {f's{s}': {'ckpt': None} for s in seeds_of(arm)} for arm in ARMS}
    for arm in ARMS:
        for s in seeds_of(arm):
            r = load_record(root, '2026', arm, s)
            if r is not None:
                sel = r['fe']['selection']
                fwd[arm][f's{s}'] = {'ckpt': sel.get('file'), 'params_sha': sel.get('params_sha')}
    out = {'prereg': PREREG, 'checks': summ, 'missing_cells': missing,
           'undefined_cells': {a: int(np.nansum(M['argmax']['undefined'][a])) for a in ARMS},
           'contrasts': res, 'fold_exposure': {f: {'exposed_windows': v, 'unexposed': not v} for f, v in exposure.items()},
           'descriptive': descriptive(M, recs, cells, series, train_days),
           'forward_policies_2026': fwd, 'R': R, 'B': B}
    if panel_path:
        out['baselines'] = baselines(panel_path, test_rows, test_dates)
    if do_register:
        out['registered_trials'] = register(cells, series, recs, ref['panel_sha'], registry_dir)
    out['provenance'] = provenance()
    return out


# ---------------------------------------------------------------- oz-test

def _synth_tree(root, effects, sigma=0.15, seed=0, drop=(), undefined=(), dirty=(), bad_lr=(), bad_window=()):
    """Sentetik kosu agaci. effects[arm] = SR ortalamasina eklenen etki; hucre SR'si tam olarak dikilir."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    ref = {'panel_sha': 'p' * 64, 'graph_sha': 'g' * 64, 'graph_file_sha': 'g' * 64}
    for fold in FOLDS:
        fd = fold_dates(fold)
        dates = [str(x.date()) for x in pd.bdate_range(fd['test_start'], fd['test_end'], inclusive='left')]
        fold_mu = rng.normal(0.5, 0.3)
        for arm in ARMS:
            for j, s in enumerate(seeds_of(arm)):
                if (fold, arm, s) in drop:
                    continue
                d = os.path.join(root, fold, arm, f's{s}')
                os.makedirs(os.path.join(d, 'eval'), exist_ok=True)
                sarl, comm = ARM_FLAGS[arm]
                args = {**EXPECTED_ARGS, 'seed': s, 'sarl': sarl, 'comm': comm, 'train_end': fd['train_end']}
                if (fold, arm, s) in bad_lr:
                    args['lr'] = 1e-3
                dirty_s = ' M x.py' if (fold, arm, s) in dirty else ''
                with open(os.path.join(d, 'train.json'), 'w', encoding='utf-8') as fh:
                    json.dump({'commit': 'c0', 'git_status_porcelain': dirty_s, 'args': args,
                               'panel_sha256': ref['panel_sha'], 'train_window': ['2013-06-03', fd['train_end']],
                               'graph_sha256': ref['graph_sha'] if comm == 'gat' else None}, fh)
                tests = {}
                for rule in RULES:
                    target = fold_mu + effects.get(arm, 0.0) + rng.normal(0, sigma)
                    n = len(dates)
                    z = rng.normal(size=n)
                    z = (z - z.mean()) / z.std(ddof=1)
                    X = 0.01 * (z + target / np.sqrt(252))
                    rf = np.full(n, 0.0004)
                    if (fold, arm, s) in undefined:
                        X = np.zeros(n)
                    R = X + rf
                    p = os.path.join(d, 'eval', f'test_{rule}.npz')
                    np.savez_compressed(p, policy_date=np.array(dates), policy_R=R, policy_X=X,
                                        policy_turnover=np.full(n, 0.05), policy_cash_weight=np.full(n, 0.1),
                                        policy_n_sell_blocked=np.zeros(n, int))
                    tests[rule] = {'file': f'test_{rule}.npz', 'sha256': _sha(p), 'same_as_argmax': rule == 'argmax'}
                vw = [f"{int(fold) - 1}-01-02", f"{int(fold) - 1}-12-31"]
                if (fold, arm, s) in bad_window:
                    vw = [fd['test_start'], fd['test_start']]
                fe = {'fold': fold, 'windows': {'val': vw, 'test': [dates[0], dates[-1]]},
                      'rows': {'val': [0, 1], 'test': [0, 1]}, 'panel_sha256': ref['panel_sha'],
                      'validation': [{'step': k} for k in range(10)],
                      'selection': {'file': 'ckpt_0010.pt', 'params_sha': 'x', 'metric': float(rng.normal()),
                                    'degenerate': False},
                      'tests': tests, 'provenance': {'commit': 'c0', 'git_status_porcelain': ''}}
                with open(os.path.join(d, 'eval', 'fold_eval.json'), 'w', encoding='utf-8') as fh:
                    json.dump(fe, fh)
    return ref


def self_test() -> int:
    ok_all = []

    def check(name, ok):
        ok_all.append(bool(ok))
        print(f"[{'OK' if ok else 'HATA'}] {name}")

    R = B = 2000
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, 'a')
        ref = _synth_tree(root, {'M-none': 0.3, 'null': 0.3, 'M-gat': 0.0, 'M-self': 0.0, 'S-agr': 0.3})
        out = read_all(root, R=R, B=B, ref=ref)
        c = out['contrasts']
        check("1. dikilen etki: H1 FARK_VAR_MARL, H2/H3/NK FARK_GOZLENMEDI, denetimler tamam",
              c['H1']['label'] == 'FARK_VAR_MARL' and c['H2']['label'] == c['H3']['label'] == c['NK']['label']
              == 'FARK_GOZLENMEDI' and out['checks']['all_ok'] and 'UYARILI' not in c['H1']['tags'])
        r0 = load_record(root, '2021', 'M-none', 201)
        m0 = cell_metrics(load_series(r0, 'argmax'))
        check("2. hucre SR'si tam dikilir; D4 20 bps = ana SR; tanimsiz yok",
              abs(m0['sr_20'] - m0['sr']) < 1e-9 and not m0['undefined'] and m0['sr_0'] > m0['sr_50'])
        st = status(root, '2021', ref)
        check("3. saglik kaydi: 72/72, tamam, basarim alani yok",
              st['ok'] and st['cells_done'] == 72 and 'sr' not in json.dumps(st['per_arm']))
        for f in os.listdir(os.path.join(root, '2022', 'S-esm', 's201', 'eval')):
            if f.startswith('test_'):
                os.remove(os.path.join(root, '2022', 'S-esm', 's201', 'eval', f))
        check("4. saglik kaydi test dosyalarini acmaz (silinmis npz ile calisir)", status(root, '2022', ref)['ok'])

        root2 = os.path.join(tmp, 'b')
        ref = _synth_tree(root2, {'S-esm': 0.3, 'S-agr': 0.3}, seed=1)
        check("5. ters etki: H1 FARK_VAR_SARL, H2 FARK_VAR_S-agr (Holm)",
              (lambda o: o['H1']['label'] == 'FARK_VAR_SARL' and o['H2']['label'] == 'FARK_VAR_S-agr'
               and o['H2']['p_holm'] >= o['H2']['p_max'])(read_all(root2, R=R, B=B, ref=ref)['contrasts']))

        root3 = os.path.join(tmp, 'c')
        drop = {(f, 'S-esm', s) for f, s in zip(FOLDS + FOLDS[:2], MAIN_SEEDS[:8])}
        ref = _synth_tree(root3, {'M-none': 0.3, 'null': 0.3}, seed=2, drop=drop,
                          dirty={('2023', 'M-gat', 205)}, undefined={(f, 'S-agr', s) for f in FOLDS[:3]
                                                                     for s in MAIN_SEEDS[:3]})
        o = read_all(root3, R=R, B=B, ref=ref)
        check("6. 8 eksik hucre -> H1 EKSIK; H2 eksik etkilenmez",
              o['missing_cells']['S-esm'] == 8 and o['contrasts']['H1']['label'] == 'EKSIK'
              and o['contrasts']['H2']['label'] != 'EKSIK')
        check("7. kirli agac -> denetim bozuk, tum etiketler UYARILI",
              not o['checks']['all_clean'] and all('UYARILI' in o['contrasts'][n]['tags'] for n in ('H1', 'H2', 'H3')))
        h2 = o['contrasts']['H2']
        check("8. tanimsiz 9/72 (%12,5) -> tanimsiz_uyarisi + duyarlilik satiri",
              o['undefined_cells']['S-agr'] == 9 and h2['undefined_flag'] and 'tanimsiz_uyarisi' in h2['tags']
              and h2['sensitivity_undefined_dropped']['n_pairs'] == 63)

        root4 = os.path.join(tmp, 'd')
        ref = _synth_tree(root4, {}, seed=3, bad_lr={('2024', 'M-self', 207)}, bad_window={('2025', 'S-esm', 203)},
                          drop={('2021', 'M-none', 201)})
        st = status(root4, '2024', ref)
        o = read_all(root4, R=R, B=B, ref=ref)
        check("9. yapilandirma ve pencere sapmasi yakalanir; eksik hucre saglik kaydinda",
              not st['checks']['config_ok'] and not o['checks']['windows_ok'] and not o['checks']['config_ok']
              and status(root4, '2021', ref)['missing'] == ['M-none/s201'])
        check("10. etkisiz agacta H1..NK FARK_GOZLENMEDI; Holm [0,01, 0,04] -> [0,02, 0,04]",
              all(o['contrasts'][n]['label'] == 'FARK_GOZLENMEDI' for n in ('H1', 'H2', 'H3', 'NK'))
              and np.allclose(ST.holm([0.01, 0.04]), [0.02, 0.04]))
        reg = os.path.join(tmp, 'reg')
        n = register({k: v for k, v in list(sorted(_cells_for(root4, ref).items()))[:4]},
                     _series_for(root4, ref), {}, ref['panel_sha'], reg)
        lines = open(os.path.join(reg, 'trial_registry.jsonl'), encoding='utf-8').read().splitlines()
        check("11. deneme kaydi: 4 deneme, kind marl_m8, wf-v1 ozeti, 2021 acik degil",
              n == 4 and len(lines) == 4 and all(json.loads(x)['kind'] == 'marl_m8' for x in lines)
              and json.loads(lines[0])['protocol']['eval_window_unexposed'] is True)
    print(f"\n{sum(ok_all)}/{len(ok_all)} gecti")
    return 0 if all(ok_all) else 1


def _cells_for(root, ref):
    out = {}
    for arm in ARMS:
        for s in seeds_of(arm)[:1]:
            r = load_record(root, '2021', arm, s)
            if r:
                out[('2021', arm, s, 'argmax')] = cell_metrics(load_series(r, 'argmax'))
    return out


def _series_for(root, ref):
    out = {}
    for arm in ARMS:
        for s in seeds_of(arm)[:1]:
            r = load_record(root, '2021', arm, s)
            if r:
                out[('2021', arm, s, 'argmax')] = load_series(r, 'argmax')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join('results', 'marl_m8'))
    ap.add_argument('--status', default=None, help='yil sonu saglik kaydi (kat adi)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--register', action='store_true', help='deneme kaydina yaz (§6.6 C2)')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    os.chdir(ROOT)
    if a.status:
        st = status(a.root, a.status)
        write_status(a.root, st)
        print(json.dumps({k: st[k] for k in ('fold', 'cells_done', 'missing', 'wall_minutes', 'ok')},
                         ensure_ascii=False))
        print('denetim:', {k: v for k, v in st['checks'].items() if k != 'commits'})
        return
    missing = [f for f in FOLDS if not os.path.exists(os.path.join(a.root, 'status', f'{f}.json'))]
    if missing:
        raise SystemExit(f'saglik kaydi olmayan kat var: {missing} (once --status; on kayit §6.3 kismi okuma yok)')
    from marl import panel as PN
    panel = EXPECTED_ARGS['panel']
    out = read_all(a.root, panel_path=panel, panel_dates=PN.load_npz(panel)['dates'], do_register=a.register)
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
    for name, *_ in CONTRASTS:
        r = out['contrasts'][name]
        print(f"{name}: {r['label_full']}  delta {r['delta_bar']:+.3f}  p_isaret {r['p_signflip']:.4f}  "
              f"p_boot {r['p_bootstrap']:.4f}")


if __name__ == '__main__':
    main()
