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


def norm_dir(d: str) -> str:
    return os.path.normcase(os.path.abspath(d))


def last_checkpoint_record(cell: dict, by_dir: dict) -> dict:
    recs = [t for t in by_dir.get(norm_dir(cell['results_dir']), [])
            if (t.get('config') or {}).get('selection_rule') == 'last_checkpoint'
            and (t.get('extra') or {}).get('paired_trial_id') == cell['trial_id']]
    if len(recs) != 1:
        raise SystemExit(f"{cell['cell']}: last_checkpoint kaydi {len(recs)} (tam 1 olmali)")
    return recs[0]


def series_table(cells: dict, by_dir: dict, rf_fn, pred_sha: dict):
    """{(algo, pred, seed): fazla getiri Series (katlar birlestirilmis)}, {(algo,pred,seed,fold): Series}."""
    from evaluation.trial_registry import load_returns
    per_fold = {}
    for c in cells.values():
        t = last_checkpoint_record(c, by_dir)
        have = ((t.get('config') or {}).get('prediction_table') or {}).get('sha256')
        if have != pred_sha[c['prediction']]:  # hakem-2 sonuc okumasi #10: kol dogru tabloya bagli mi
            raise SystemExit(f"{c['cell']}: P tablosu SHA {have} != protokol {pred_sha[c['prediction']]}")
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


def seed_frame(per_fold: dict, algo: str, pred: str, folds) -> pd.DataFrame:
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
    return m


def seed_mean(per_fold: dict, algo: str, pred: str, folds) -> pd.Series:
    return seed_frame(per_fold, algo, pred, folds).mean(axis=1)


def _ann_sharpe(x: np.ndarray) -> float:
    return float(x.mean() / x.std(ddof=1) * ANN)


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
        ok = np.isfinite(a.to_numpy()) & np.isfinite(b.to_numpy())  # testle ayni maske
        rhos[al] = float(np.corrcoef(a.to_numpy()[ok], b.to_numpy()[ok])[0, 1])
    mg = None if margin_annual is None else margin_annual / ANN
    res = sharpe_diff_hac_pooled(pairs, margin=mg)
    se_a = res['se'] * ANN
    z90 = stats.norm.ppf(0.95)
    mde_k = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
    out = {
        'contrast': f'{a_pred} - {b_pred}', 'folds': list(folds), 'algorithms': list(algos), 'n_days': res['n'],
        'delta_sharpe_annual': res['delta'] * ANN,
        'per_algorithm_delta_annual': dict(zip(algos, [d * ANN for d in res['deltas']])),
        'se_annual': se_a, 'z': res['stat'], 'p_value': res['p_value'], 'rho': rhos,
        'mde_annual_at_measured_rho': float((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * se_a),
        'method': res['method'],
        'ci90_annual': [res['delta'] * ANN - z90 * se_a, res['delta'] * ANN + z90 * se_a],
        'se_annual_no_prewhiten': sharpe_diff_hac_pooled(pairs, prewhiten=False)['se'] * ANN,
    }
    if margin_annual is not None:
        out.update(tost_margin_annual=margin_annual, p_tost=res['p_tost'])
    # hakem-2 sonuc okumasi #3/#9: havuzlu ortalama heterojenligi gizleyebilir -> algoritma bazinda (kesifsel)
    per = {}
    for al, (a, b) in zip(algos, pairs):
        r1 = sharpe_diff_hac_pooled([(a, b)], margin=mg)
        se1 = r1['se'] * ANN
        per[al] = {'delta_sharpe_annual': r1['delta'] * ANN, 'se_annual': se1, 'p_value': r1['p_value'],
                   'ci90_annual': [r1['delta'] * ANN - z90 * se1, r1['delta'] * ANN + z90 * se1],
                   'mde_annual': float(mde_k * se1), **({'p_tost': r1['p_tost']} if mg is not None else {})}
        # hakem-2 #4: tohum basina fark (tek ajan duzeyi; SE'ye girmez)
        fa, fb = seed_frame(per_fold, al, a_pred, folds), seed_frame(per_fold, al, b_pred, folds)
        ok = np.isfinite(fa.to_numpy()).all(1) & np.isfinite(fb.to_numpy()).all(1)
        d = [_ann_sharpe(fa[c].to_numpy()[ok]) - _ann_sharpe(fb[c].to_numpy()[ok]) for c in fa.columns]
        per[al]['per_seed_delta_annual'] = dict(zip([int(c) for c in fa.columns], d))
        per[al]['per_seed_delta_mean_std'] = [float(np.mean(d)), float(np.std(d, ddof=1))]
    out['per_algorithm'] = per
    return out


def q3_reading(q2: dict, pc: dict, alpha: float, equivalence_allowed: bool) -> str:
    """positive_control_rule. 'Tespit' = anlamli VE pozitif (PC060 > DT; yonlu yorum, beyan). Esdegerlik ancak
    equivalence_allowed iken (on kayit sonrasi ileri veri) kullanilir: cannot_claim '2021'de esdegerlik'."""
    detected = pc['p_value'] < alpha and pc['delta_sharpe_annual'] > 0
    equiv = equivalence_allowed and q2.get('p_tost', 1.0) < alpha
    if not equivalence_allowed:
        tost_txt = (f"p_TOST = {q2.get('p_tost', float('nan')):.3f} gozlendi; on kayit geregi bu pencerede "
                    "esdegerlik iddiasi kurulamaz")
    else:
        tost_txt = 'esdegerlik TOST ile gosterildi' if equiv else 'esdegerlik gosterilemedi'
    if q2['p_value'] < alpha:
        return 'Q2 farki tespit edildi (yon: ' + ('DTF > DT' if q2['delta_sharpe_annual'] > 0 else 'DTF < DT') + ').'
    if equiv and detected:
        return "Q2: esdegerlik (TOST) ve ayni pencerede PC060 farki tespit edildi -> 'tahmin katki vermez' yazilabilir."
    if not detected:
        return (f"Q2 bos (fark tespit edilmedi; {tost_txt}); PC060 farki da tespit edilmedi -> 'bu boru hatti ve bu gucle AUC 0,60'lik bir sinyal "
                "bile aktarilamadi'. P tablosu hakkinda bir sey soylenmez; 'tahmin katki vermez' YAZILMAZ "
                "(positive_control_rule'un ikinci kosulu yok).")
    return (f"Q2 bos ({tost_txt}); PC060 farki tespit edildi -> 'bu gucle tespit edilmedi (MDE = "
            f"{q2['mde_annual_at_measured_rho']:.2f})'; 'tahmin katki vermez' YAZILMAZ.")


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
        by_dir.setdefault(norm_dir(t['_dir']), []).append(t)
    pred_sha = {k: v.get('table_sha256') for k, v in p['prediction_levels'].items()}
    _, per_fold = series_table(cells, by_dir, rf_fn, pred_sha)
    inf = p['inference']
    algos = p['factors']['algorithm']
    margin = inf['confirmatory_family'][0]['tost_margin_sharpe_annual']
    conf_folds = list(p['unexposed_folds'])
    all_folds = list(st['folds'])
    q2 = contrast(per_fold, algos, 'DTF', 'DT', conf_folds, margin)
    pc_conf = contrast(per_fold, algos, 'PC060', 'DT', conf_folds)
    # On kayit sonrasi ileri veri henuz yok: dogrulayici pencere yalniz 2021 -> esdegerlik kurulamaz (cannot_claim)
    forward_data = False
    equivalence_allowed = forward_data
    return {
        'protocol_id': p['protocol_id'], **want, 'ledger_cells': len(cells),
        'duplicate_completed_cells': dup, 'N_trials_registry_union': n_trials, 'registry_dirs': dirs,
        'confirmatory': {'Q2_DTF_vs_DT': q2, 'alpha': inf['alpha'],
                         'decision_two_sided': 'red' if q2['p_value'] < inf['alpha'] else 'red yok',
                         'status': 'ARA KARAR: yalniz 2021 kati (on kayit sonrasi ileri veri yok)',
                         'p_tost_observed': q2['p_tost'],
                         'equivalence_claim': ('kurulamaz (2021 penceresi; protocol cannot_claim); ileri veride sinanir'
                                               if not equivalence_allowed else
                                               ('esdeger' if q2['p_tost'] < inf['alpha'] else 'gosterilemedi'))},
        'positive_control_same_window': pc_conf,
        'Q3_reading': q3_reading(q2, pc_conf, inf['alpha'], equivalence_allowed),
        'exploratory': {
            'DTF_vs_DT_all_folds': contrast(per_fold, algos, 'DTF', 'DT', all_folds, margin),
            'PC060_vs_DT_all_folds': contrast(per_fold, algos, 'PC060', 'DT', all_folds),
            'note': 'kesifsel; dogrulayici aileye girmez, coklu duzeltme yok',
        },
        'required_footnotes': {'exposure_note': p['exposure_note'], 'table': inf['result_table_footnote']},
        'cannot_claim': inf['cannot_claim'],
        'notes': ["Tahmin edilen nicelik: her algoritmada 5 tohumun gunluk getirisinin esit agirlikli ortalamasi "
                  "(5 ajanlik topluluk portfoyu); SE yalniz zaman orneklemesini kapsar, egitim (tohum) varyansini "
                  "icermez; tohum basina farklar per_algorithm.per_seed_delta_annual'da. 'Ajan duzeyinde etki' denmez.",
                  "Pozitif kontrolde 'tespit' = anlamli VE pozitif (PC060 > DT); yonlu yorum, protokolde yon yok (beyan).",
                  "On kayitli guc oncülu (mde.json: rho <= 0,95, tohum ortalamasi modellenmedi) karamsar cikti; "
                  "olculen rho ve MDE contrast alanlarinda. Havuzlu MDE tek algoritma MDE'sinden kucuktur.",
                  "Her katin ilk seansinda risksiz faiz tanimsiz (evaluation.risk_free.per_period_rate ilk seans NaN); "
                  "o gun test, rho ve MDE'den ortak maskeyle dusulur (2021: 248 -> 247 gun)."],
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
                    tsha = p['prediction_levels'][pr].get('table_sha256')
                    lines.append({'trial_id': lid, 'kind': 'rl_train', 'tags': ['protocol_open'],
                                  'config': {'selection_rule': 'last_checkpoint',
                                             'prediction_table': {'sha256': tsha} if tsha else None},
                                  'returns_path': rp,
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
        # sentetikte TOST basarili ve PC060 tespit: yine de 2021'de 'katki vermez' YAZILMAZ (hakem-2 sonuc #2)
        'TOST+PC060 tespit 2021: esdegerlik kurulamaz': q2['p_tost'] < 0.05 and pc['p_value'] < 0.05
                                   and 'yazilabilir' not in out['Q3_reading']
                                   and 'kurulamaz' in out['confirmatory']['equivalence_claim'],
        'algoritma bazinda + tohum dagilimi': set(q2['per_algorithm']) == {'PPO', 'A2C'}
                                   and len(q2['per_algorithm']['PPO']['per_seed_delta_annual']) == 5,
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
