"""Faz 9 §9.39 (b): egitilmis M0' politikalari tahmin blogunu kullaniyor mu (on kayit: plan §9.39, commit b25012b).

Argmax kontrol noktasi modelleri (diskteki tek model; rota ara noktalari siler), yalniz 2022-2026 katlari
(2021 dogrulayici pencere yeniden okunmaz). Deneme kaydina YAZMAZ, yeniden egitim yok.

  Dogruluk kapisi: normal kosu deneme kaydindaki argmax test getirilerini yeniden uretmeli (|fark| <= 1e-9).
  Acik dongu: S_X = ort ||a - a_X||_1 / ort ||a||_1;  R = S_P,karistir / S_C,karistir (P tahmin blogu, C esit
  boyutlu piyasa ozellik blogu, permutasyon tohum 0). Kapali dongu (PC060): tahmin sozlugu test tarihleri
  arasinda karistirilir, dSR = SR(karistirilmis) - SR(normal), havuzlu HAC.

    CUDA_VISIBLE_DEVICES=-1 python scripts/m0_pred_block_ablation.py --out docs/development/measurements/phase10-m0/pred_block_ablation.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PROTOCOL = os.path.join('docs', 'development', 'milestone0', 'protocol-v2.json')
FOLDS = ['2022', '2023', '2024', '2025', '2026']
ANN = np.sqrt(252.0)


def _am():
    spec = importlib.util.spec_from_file_location('am', os.path.join(ROOT, 'scripts', 'analyze_milestone0.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def actions(model, O: np.ndarray) -> np.ndarray:
    a, _ = model.predict(O, deterministic=True)
    return np.asarray(a, dtype=float).reshape(len(O), -1)


def sens(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum(1).mean() / max(np.abs(a).sum(1).mean(), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=None, help='yalniz deneme: ilk n model')
    a = ap.parse_args()
    am = _am()
    from stable_baselines3 import A2C, PPO
    from app.api.routes import trading as tr
    from data.data_fetcher import DataFetcher
    from evaluation import rl_folds
    from evaluation.risk_free import risk_free_series
    from evaluation.sharpe_tests import sharpe_diff_hac_pooled
    from evaluation.splits import get_spec
    from evaluation.trial_registry import load_returns
    from prediction.pit_table import env_prediction_features, load_table

    p = json.load(open(PROTOCOL, encoding='utf-8'))
    mroot = os.path.join('results', 'milestone0', p['protocol_id'])
    _, cells, _ = am.read_ledger(os.path.join(mroot, 'ledger.jsonl'))
    _, allrec = am.count_n(am.registry_dirs(mroot, 'results'))
    by_id = {t['trial_id']: t for t in allrec.values()}

    raw = DataFetcher().load_data(tr.RL_PANEL_FILE)
    rp = rl_folds.load_rl_panel(price_norm='causal', start=tr._settings.RL_DATA_START, df=raw, adx_mode=tr.RL_ADX_MODE)
    spec = get_spec(p['fixed']['split_spec'])
    tables = {}
    for lv in ('DTF', 'PC060'):
        t, info = load_table(p['prediction_levels'][lv]['table'])
        if info['sha256'] != p['prediction_levels'][lv]['table_sha256']:
            raise SystemExit(f'{lv} tablo SHA uyusmuyor')
        tables[lv] = t
    rng0 = np.random.default_rng(0)
    rows, closed = [], {}
    todo = sorted((c for c in cells.values() if c['prediction'] in ('DTF', 'PC060') and str(c['fold']) in FOLDS),
                  key=lambda c: (str(c['fold']), c['prediction'], c['algorithm'], c['seed']))
    if a.limit:
        todo = todo[:a.limit]
    env_cache = {}
    for c in todo:
        f, lv = str(c['fold']), c['prediction']
        key = (f, lv)
        if key not in env_cache:
            fd = rl_folds.build_fold(rp, spec, spec.fold(f))
            pdates = sorted(set().union(*(pd.to_datetime(fd.frame(r).index.get_level_values('date'))
                                          for r in ('train', 'val', 'test'))))
            pfeat, _ = env_prediction_features(tables[lv], list(fd.symbols), pdates)
            test_keys = sorted({str(d)[:10] for d in pd.to_datetime(fd.frame('test').index.get_level_values('date'))})
            perm = np.random.default_rng(0).permutation(len(test_keys))
            pfeat_shuf = dict(pfeat)
            for k_src, k_dst in zip([test_keys[i] for i in perm], test_keys):
                pfeat_shuf[k_dst] = pfeat[k_src]
            base = dict(initial_balance=p['fixed']['initial_balance'], commission_rate=c['commission_rate'],
                        max_shares_per_trade=p['fixed']['max_shares_per_trade'], phase=1, reward_type='simple',
                        fundamental_df=None, macro_df=None, sharpe_term='ema_level')
            tr_env = rl_folds.make_fold_env(fd, 'train', price_norm='causal', **base, prediction_features=pfeat)
            mk = lambda pf: rl_folds.make_fold_env(fd, 'test', train_env=tr_env, price_norm='causal', **base,
                                                   prediction_features=pf)
            env_cache[key] = (fd, mk, pfeat, pfeat_shuf)
        fd, mk, pfeat, pfeat_shuf = env_cache[key]
        t_arg = by_id[c['trial_id']]
        mdir = os.path.join(os.path.dirname(c['results_dir']), 'models')
        mpath = os.path.join(mdir, t_arg['extra']['model_name'])
        model = (PPO if c['algorithm'] == 'PPO' else A2C).load(mpath, device='cpu')
        env = mk(pfeat)
        ep = rl_folds.run_episode(model, env, collect_obs=True)
        stored = load_returns(t_arg).to_numpy(dtype=float)
        if len(stored) != len(ep['returns']) or np.max(np.abs(stored - ep['returns'])) > 1e-9:
            raise SystemExit(f"{c['cell']}: yeniden kurulan ortam kayittaki getiriyi URETMEDI "
                             f"(len {len(stored)}/{len(ep['returns'])})")
        O = np.asarray(ep['seen_obs'], dtype=np.float32)
        n = len(fd.symbols)
        dim = O.shape[1]
        P = np.arange(dim - 4 * n, dim)
        market = np.arange(1 + n, dim - 4 * n)
        C = np.sort(rng0.choice(market, size=len(P), replace=False)) if c is todo[0] else C
        perm = np.random.default_rng(0).permutation(len(O))
        a0 = actions(model, O)

        def swap(dims, src):
            X = O.copy()
            X[:, dims] = src[:, dims]
            return X
        Z = O.copy()
        Z[:, P] = 0.0
        s_p = sens(a0, actions(model, swap(P, O[perm])))
        s_c = sens(a0, actions(model, swap(C, O[perm])))
        s_z = sens(a0, actions(model, Z))
        rows.append({'cell': c['cell'], 'fold': f, 'prediction': lv, 'algorithm': c['algorithm'], 'seed': c['seed'],
                     'S_P_shuffle': s_p, 'S_C_shuffle': s_c, 'S_P_zero': s_z, 'R': s_p / max(s_c, 1e-12),
                     'reproduced_max_abs': float(np.max(np.abs(stored - ep['returns'])))})
        if lv == 'PC060':
            ep_s = rl_folds.run_episode(model, mk(pfeat_shuf))
            idx = pd.to_datetime(ep['dates'])
            rf = risk_free_series(idx).reindex(idx).to_numpy()
            closed.setdefault((c['algorithm'], f), []).append((ep['returns'] - rf, ep_s['returns'] - rf))
        print(f"{c['cell']}: R {rows[-1]['R']:.3f} (S_P {s_p:.3f}, S_C {s_c:.3f}, S_0 {s_z:.3f})", flush=True)

    df = pd.DataFrame(rows)
    summary = {}
    for (al, lv), g in df.groupby(['algorithm', 'prediction']):
        summary[f'{al}/{lv}'] = {'n_models': int(len(g)), 'R_median': float(g['R'].median()),
                                 'R_q25_q75': [float(g['R'].quantile(.25)), float(g['R'].quantile(.75))],
                                 'S_P_shuffle_median': float(g['S_P_shuffle'].median()),
                                 'S_C_shuffle_median': float(g['S_C_shuffle'].median()),
                                 'S_P_zero_median': float(g['S_P_zero'].median())}
    closed_res = {}
    for al in sorted({k[0] for k in closed}):
        norm_parts, shuf_parts = [], []
        for f in FOLDS:
            if (al, f) in closed:
                arr = closed[(al, f)]
                norm_parts.append(np.mean([x for x, _ in arr], axis=0))
                shuf_parts.append(np.mean([y for _, y in arr], axis=0))
        r = sharpe_diff_hac_pooled([(np.concatenate(shuf_parts), np.concatenate(norm_parts))])
        closed_res[al] = {'delta_sharpe_annual_shuffled_minus_normal': r['delta'] * ANN, 'se_annual': r['se'] * ANN,
                          'p_value': r['p_value'], 'n_days': r['n']}
    if closed_res:
        pairs = []
        for al in sorted(closed_res):
            ns = [np.mean([x for x, _ in closed[(al, f)]], axis=0) for f in FOLDS if (al, f) in closed]
            ss = [np.mean([y for _, y in closed[(al, f)]], axis=0) for f in FOLDS if (al, f) in closed]
            pairs.append((np.concatenate(ss), np.concatenate(ns)))
        r = sharpe_diff_hac_pooled(pairs)
        closed_res['pooled'] = {'delta_sharpe_annual_shuffled_minus_normal': r['delta'] * ANN,
                                'se_annual': r['se'] * ANN, 'p_value': r['p_value'], 'n_days': r['n']}
    med_r = {k: v['R_median'] for k, v in summary.items()}
    pooled = closed_res.get('pooled', {})
    if all(v < 0.1 for v in med_r.values()):
        reading = 'medyan R < 0,1 (tum gruplar): politika tahmin blogunu eylem duzeyinde pratikte yok sayiyor'
    elif pooled and pooled['delta_sharpe_annual_shuffled_minus_normal'] < 0 and pooled['p_value'] < 0.05:
        reading = 'PC060 kapali dongu dSR < 0, p < 0,05: politika bloktaki bilgiyi kullaniyor'
    else:
        reading = 'medyan R >= 0,1 (en az bir grup) ve PC060 kapali dongu anlamli negatif degil: blok eylemi etkiliyor ama deger uretmiyor'
    rec = {'preregistration': 'plan §9.39, commit b25012b (sonuctan once)', 'models': 'argmax kontrol noktasi',
           'folds': FOLDS, 'n_models': int(len(df)), 'control_block_dims': [int(x) for x in C],
           'reproduction_gate_max_abs': float(df['reproduced_max_abs'].max()), 'open_loop': summary,
           'closed_loop_PC060': closed_res, 'reading': reading, 'per_model': rows,
           'note': 'kesifsel; deneme kaydina yazilmadi; last_checkpoint politikalarina dogrudan genellenmez'}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False, default=str)
    print(json.dumps({k: rec[k] for k in ('n_models', 'reproduction_gate_max_abs', 'open_loop', 'closed_loop_PC060',
                                          'reading')}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
