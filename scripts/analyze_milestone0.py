"""Milestone 0' m0p-v2 analizi: on kayitli dogrulayici test (Q2), pozitif kontrol okumasi (Q3), N, rho ve MDE.

Kurallarin tek kaynagi dondurulmus protokol (`docs/development/milestone0/protocol-v2.json`, etiket
`m0-protocol-v2`); bu betik onlari koda doker, yeni karar eklemez:
  - Kayit: her hucre icin ana defterdeki `trial_id` (argmax kaydi) -> ayni isci kaydinda
    `selection_rule='last_checkpoint'` ve `extra.paired_trial_id == trial_id` olan TEK kayit (`selection_rule`).
  - Birim (K20): yapilandirma x tohum; katlarin orneklem disi gunluk getirileri tarihe gore birlestirilir.
  - Getiri: A8 risksiz faiz ustu (`evaluation.risk_free.risk_free_series`), maliyet ortamda dusulmus.
  - Q2 (dogrulayici, tek test): her algoritmada 5 tohumun gunluk fazla getirisi ortalanir; istatistik iki
    algoritmanin yillik Sharpe farklarinin (DTF - DT) esit agirlikli ortalamasi; SE ortak delta yontemiyle
    HAC (`evaluation.sharpe_tests.sharpe_diff_hac_pooled`; "birlesik seri" = farklarin etki fonksiyonunun gunluk
    ortalamasi). Iki tarafli test + TOST +-0,5 yillik, alpha 0,05. Pencere: `confirmatory_window` (2021 kati;
    on kayit sonrasi ileri veri henuz yok).
  - Q3 (`positive_control_rule`): PC060 - DT ayni pencerede ve 6 katta; okuma cumlesi kurala gore.
  - N (`fixed.registry_analysis`): kok + workers/w*/results deneme kayitlari, 'spike' haric.
  - rho: tohum ortalamali DT ve DTF fazla getiri serilerinin korelasyonu; MDE = (z_0.975 + z_0.8) * SE.

    python scripts/analyze_milestone0.py --data-root <ana depo> --out <json>
    python scripts/analyze_milestone0.py --self-test        # sentetik kayitlarla, gercek sonuca bakmaz
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PROTOCOL = os.path.join('docs', 'development', 'milestone0', 'protocol-v2.json')
ANN = np.sqrt(252.0)


def sha(path: str) -> str:
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def read_ledger(path: str):
    with open(path, encoding='utf-8') as fh:
        lines = [json.loads(x) for x in fh if x.strip()]
    header, cells, dup = lines[0], {}, {}
    for r in lines[1:]:
        if r.get('state') != 'completed':
            continue
        if r['cell'] in cells:
            dup.setdefault(r['cell'], []).append(r.get('trial_id'))
            continue
        cells[r['cell']] = r  # ilk tamamlanan gecerli (protokol fixed.registry_analysis)
    return header, cells, dup


def registry_dirs(mroot: str, results_root: str):
    return [results_root] + sorted(glob.glob(os.path.join(mroot, 'workers', 'w*', 'results')))


def count_n(dirs):
    from evaluation.trial_registry import load_trials
    seen = {}
    for d in dirs:
        for t in load_trials(exclude_tags=('spike',), directory=d):
            seen[t['trial_id']] = t
    return len(seen), seen


def last_checkpoint_record(cell: dict, by_dir: dict) -> dict:
    recs = [t for t in by_dir[cell['results_dir']]
            if (t.get('config') or {}).get('selection_rule') == 'last_checkpoint'
            and (t.get('extra') or {}).get('paired_trial_id') == cell['trial_id']]
    if len(recs) != 1:
        raise SystemExit(f"{cell['cell']}: last_checkpoint kaydi {len(recs)} (tam 1 olmali)")
    return recs[0]


def series_table(cells: dict, by_dir: dict, rf_fn):
    """{(algo, pred, seed): fazla getiri Series (katlar birlestirilmis)}, {(algo,pred,seed,fold): Series}."""
    from evaluation.trial_registry import load_returns
    per_fold = {}
    for c in cells.values():
        t = last_checkpoint_record(c, by_dir)
        r = load_returns(t).astype(float)
        r.index = pd.to_datetime(r.index)
        per_fold[(c['algorithm'], c['prediction'], int(c['seed']), str(c['fold']))] = r - rf_fn(r.index)
    joined = {}
    for (al, pr, sd, fd), s in per_fold.items():
        joined.setdefault((al, pr, sd), []).append(s)
    out = {}
    for k, parts in joined.items():
        s = pd.concat(parts).sort_index()
        if s.index.duplicated().any():
            raise SystemExit(f'{k}: katlarin test tarihleri cakisiyor')
        out[k] = s
    return out, per_fold


def seed_mean(per_fold: dict, algo: str, pred: str, folds) -> pd.Series:
    seeds = sorted({k[2] for k in per_fold if k[0] == algo and k[1] == pred})
    cols = []
    for sd in seeds:
        parts = [per_fold[(algo, pred, sd, f)] for f in folds if (algo, pred, sd, f) in per_fold]
        if len(parts) != len(folds):
            raise SystemExit(f'{algo}/{pred}/tohum {sd}: {len(parts)}/{len(folds)} kat')
        cols.append(pd.concat(parts).sort_index().rename(sd))
    m = pd.concat(cols, axis=1, join='inner')
    if len(m) != len(cols[0]):
        raise SystemExit(f'{algo}/{pred}: tohumlarin test tarihleri farkli')
    return m.mean(axis=1)


def contrast(per_fold, algos, a_pred: str, b_pred: str, folds, margin_annual=None) -> dict:
    """(a_pred - b_pred) Sharpe farki, algoritmalar uzerinden esit agirlikli; yillik birimde."""
    from evaluation.sharpe_tests import sharpe_diff_hac_pooled
    pairs, rhos = [], {}
    for al in algos:
        a = seed_mean(per_fold, al, a_pred, folds)
        b = seed_mean(per_fold, al, b_pred, folds)
        if not a.index.equals(b.index):
            raise SystemExit(f'{al}: {a_pred} ve {b_pred} tarihleri farkli')
        pairs.append((a.to_numpy(), b.to_numpy()))
        rhos[al] = float(np.corrcoef(a, b)[0, 1])
    res = sharpe_diff_hac_pooled(pairs, margin=None if margin_annual is None else margin_annual / ANN)
    se_a = res['se'] * ANN
    out = {
        'contrast': f'{a_pred} - {b_pred}', 'folds': list(folds), 'algorithms': list(algos), 'n_days': res['n'],
        'delta_sharpe_annual': res['delta'] * ANN,
        'per_algorithm_delta_annual': dict(zip(algos, [d * ANN for d in res['deltas']])),
        'se_annual': se_a, 'z': res['stat'], 'p_value': res['p_value'], 'rho': rhos,
        'mde_annual_at_measured_rho': float((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * se_a),
        'method': res['method'],
    }
    if margin_annual is not None:
        out.update(tost_margin_annual=margin_annual, p_tost=res['p_tost'])
    return out


def q3_reading(q2: dict, pc: dict, alpha: float) -> str:
    detected = pc['p_value'] < alpha and pc['delta_sharpe_annual'] > 0
    equiv = q2.get('p_tost', 1.0) < alpha
    if q2['p_value'] < alpha:
        return 'Q2 farki tespit edildi (yon: ' + ('DTF > DT' if q2['delta_sharpe_annual'] > 0 else 'DTF < DT') + ').'
    if equiv and detected:
        return "Q2: esdegerlik (TOST) ve ayni pencerede PC060 farki tespit edildi -> 'tahmin katki vermez' yazilabilir."
    if not detected:
        return ("Q2 bos; PC060 farki da tespit edilmedi -> 'bu boru hatti ve bu gucle AUC 0,60'lik bir sinyal bile "
                "aktarilamadi'. P tablosu hakkinda bir sey soylenmez.")
    return ("Q2 bos ve esdegerlik gosterilemedi; PC060 farki tespit edildi -> 'bu gucle tespit edilmedi (MDE = x)'; "
            "'tahmin katki vermez' YAZILMAZ (TOST basarisiz).")


def analyze(protocol_path: str, mroot: str, results_root: str, rf_fn) -> dict:
    p = json.load(open(protocol_path, encoding='utf-8'))
    header, cells, dup = read_ledger(os.path.join(mroot, 'ledger.jsonl'))
    want = {'protocol_sha256': sha(protocol_path), 'protocol_doc_sha256': sha(p['doc'])}
    for k, v in want.items():
        if header.get(k) != v:
            raise SystemExit(f'{k}: defter {header.get(k)} != dosya {v} (dondurulmus protokol degismis)')
    st = p['stages']['main']
    n_cells = (len(p['factors']['algorithm']) * len(p['factors']['prediction']) * len(st['folds'])
               * len(st['seeds']))
    if len(cells) != n_cells:
        raise SystemExit(f'tamamlanan hucre {len(cells)}/{n_cells}; analiz tum hucreler bitince')
    dirs = registry_dirs(mroot, results_root)
    n_trials, allrec = count_n(dirs)
    by_dir = {}
    for t in allrec.values():
        by_dir.setdefault(t['_dir'], []).append(t)
    for c in cells.values():
        by_dir.setdefault(c['results_dir'], [])
    _, per_fold = series_table(cells, by_dir, rf_fn)
    inf = p['inference']
    algos = p['factors']['algorithm']
    margin = inf['confirmatory_family'][0]['tost_margin_sharpe_annual']
    conf_folds = list(p['unexposed_folds'])
    all_folds = list(st['folds'])
    q2 = contrast(per_fold, algos, 'DTF', 'DT', conf_folds, margin)
    pc_conf = contrast(per_fold, algos, 'PC060', 'DT', conf_folds)
    return {
        'protocol_id': p['protocol_id'], **want, 'ledger_cells': len(cells),
        'duplicate_completed_cells': dup, 'N_trials_registry_union': n_trials, 'registry_dirs': dirs,
        'confirmatory': {'Q2_DTF_vs_DT': q2, 'alpha': inf['alpha'],
                         'decision_two_sided': 'red' if q2['p_value'] < inf['alpha'] else 'red yok',
                         'decision_tost': 'esdeger' if q2['p_tost'] < inf['alpha'] else 'esdegerlik gosterilemedi'},
        'positive_control_same_window': pc_conf,
        'Q3_reading': q3_reading(q2, pc_conf, inf['alpha']),
        'exploratory': {
            'DTF_vs_DT_all_folds': contrast(per_fold, algos, 'DTF', 'DT', all_folds, margin),
            'PC060_vs_DT_all_folds': contrast(per_fold, algos, 'PC060', 'DT', all_folds),
            'note': 'kesifsel; dogrulayici aileye girmez, coklu duzeltme yok',
        },
        'required_footnotes': {'exposure_note': p['exposure_note'], 'table': inf['result_table_footnote']},
        'cannot_claim': inf['cannot_claim'],
    }


# ----------------------------------------------------------------------------
# Sentetik oz-sinama: gercek sonuca bakmadan betigin enjekte edilen farki geri bulmasi
# ----------------------------------------------------------------------------

def _self_test() -> int:
    p = json.load(open(PROTOCOL, encoding='utf-8'))
    st = p['stages']['main']
    tmp = tempfile.mkdtemp(prefix='m0a_selftest_')
    mroot = os.path.join(tmp, 'results', 'milestone0', 'm0p-v2')
    wres = os.path.join(mroot, 'workers', 'w0', 'results')
    os.makedirs(os.path.join(wres, 'trial_returns'), exist_ok=True)
    rng = np.random.default_rng(0)
    years = {f: pd.bdate_range(f'{f}-01-04', periods=250) for f in st['folds']}
    common = {f: rng.normal(4e-4, 0.015, 250) for f in st['folds']}
    shift = {'DT': 0.0, 'DTF': 0.0, 'PC060': 25e-4}  # PC060'a buyuk, DTF'ye sifir etki
    lines, ledger = [], [{'header': True, 'protocol_sha256': sha(PROTOCOL), 'protocol_doc_sha256': sha(p['doc'])}]
    i = 0
    for al in p['factors']['algorithm']:
        for pr in p['factors']['prediction']:
            for f in st['folds']:
                for sd in st['seeds']:
                    i += 1
                    r = common[f] + rng.normal(0, 0.004, 250) + shift[pr]
                    tid, lid = f'arg{i}', f'last{i}'
                    rp = os.path.join('trial_returns', f'{lid}.csv')
                    pd.DataFrame({'ret': r}, index=years[f]).to_csv(os.path.join(wres, rp))
                    lines.append({'trial_id': tid, 'kind': 'rl_train', 'tags': ['protocol_open'],
                                  'config': {'selection_rule': 'argmax_validation'}, 'extra': {}})
                    lines.append({'trial_id': lid, 'kind': 'rl_train', 'tags': ['protocol_open'],
                                  'config': {'selection_rule': 'last_checkpoint'}, 'returns_path': rp,
                                  'extra': {'paired_trial_id': tid}})
                    cell = f'algorithm={al}|prediction={pr}|fold={f}|seed={sd}'
                    ledger.append({'cell': cell, 'algorithm': al, 'prediction': pr, 'fold': f, 'seed': sd,
                                   'state': 'completed', 'trial_id': tid, 'results_dir': wres})
    with open(os.path.join(wres, 'trial_registry.jsonl'), 'w', encoding='utf-8') as fh:
        fh.writelines(json.dumps(x) + '\n' for x in lines)
    with open(os.path.join(mroot, 'ledger.jsonl'), 'w', encoding='utf-8') as fh:
        fh.writelines(json.dumps(x) + '\n' for x in ledger)
    root_res = os.path.join(tmp, 'results')
    out = analyze(PROTOCOL, mroot, root_res, lambda idx: pd.Series(0.0, index=idx))
    q2, pc = out['confirmatory']['Q2_DTF_vs_DT'], out['positive_control_same_window']
    checks = {
        'hucre 180': out['ledger_cells'] == 180,
        'N = 360 (her hucrede iki kayit)': out['N_trials_registry_union'] == 360,
        'Q2 2021 tek kat': q2['folds'] == ['2021'] and q2['n_days'] == 250,
        'Q2 sifir etkide red yok': q2['p_value'] >= 0.05,
        'PC060 buyuk etkide tespit': pc['p_value'] < 0.05 and pc['delta_sharpe_annual'] > 0,
        'rho yuksek (ortak bilesen)': min(q2['rho'].values()) > 0.8,
        'kesifsel 6 kat': out['exploratory']['DTF_vs_DT_all_folds']['n_days'] == 1500,
        # Q2 bos + PC060 tespit: 'katki vermez' cumlesi YALNIZ TOST basariliysa
        'Q3 okumasi kurala uygun': (q2['p_tost'] < 0.05) == ('yazilabilir' in out['Q3_reading'])
                                   and q2['p_value'] >= 0.05,
    }
    # mutasyon: bir hucrenin last_checkpoint kaydi silinirse betik durmali
    with open(os.path.join(wres, 'trial_registry.jsonl'), 'w', encoding='utf-8') as fh:
        fh.writelines(json.dumps(x) + '\n' for x in lines[:-1])
    try:
        analyze(PROTOCOL, mroot, root_res, lambda idx: pd.Series(0.0, index=idx))
        checks['eksik kayitta durur'] = False
    except SystemExit:
        checks['eksik kayitta durur'] = True
    for k, v in checks.items():
        print(('GECTI ' if v else 'KALDI ') + k)
    print('Q3:', out['Q3_reading'])
    ok = sum(checks.values())
    print(f'{ok}/{len(checks)}')
    return 0 if ok == len(checks) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', default=PROTOCOL)
    ap.add_argument('--data-root', default=None, help='results/ ve data/macro iceren depo koku')
    ap.add_argument('--out', default=None)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        os.chdir(ROOT)
        sys.exit(_self_test())
    if not a.out:
        raise SystemExit('--out gerekli')
    protocol_path = os.path.abspath(a.protocol)
    os.chdir(a.data_root or ROOT)
    from evaluation.risk_free import risk_free_series
    from app.auth import workspace as ws
    p = json.load(open(protocol_path, encoding='utf-8'))
    results_root = ws.results_dir()
    mroot = os.path.join(results_root, 'milestone0', p['protocol_id'])
    # p['doc'] goreli yol: data-root'taki (ana depo) dondurulmus belge; SHA defter basligiyla karsilastirilir
    out = analyze(protocol_path, mroot, results_root, lambda idx: risk_free_series(idx).reindex(idx))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
    print(json.dumps({k: out[k] for k in ('confirmatory', 'Q3_reading')}, indent=1, ensure_ascii=False,
                     default=str))


if __name__ == '__main__':
    main()
