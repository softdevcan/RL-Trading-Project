"""Faz 10 M4a: KOMSU SINYALI pozitif kontrolu (M4 tasarimi §3) -> ortamin okudugu .npz. Ana venv.

Ajan j'nin gozlemindeki tahmin blogu, kendi hissesinin degil, gat grafinda bir KOMSUSU sigma(j)'nin PC060-K12
blogudur (sutun permutasyonu; sigma bir bijeksiyon, sigma(j) in N(j)). Hisse i'nin bilgisi yalniz komsu ajan
sigma^-1(i)'nin gozlemindedir: iletisimsiz (none) politika onu dogrudan kullanamaz; attn ve gat iletebilir.
Komsular korele oldugundan none komsunun sinyalinden kismi deger alabilir: bu sizinti `auc_holder_vs_own_label`
ile kosudan ONCE olculur. Girdi bloklari M3'unkulerdir (`build_pred_m3.py`, `--shuffle` dahil); ayni permutasyon
karistirilmis bloklara da uygulanir. BILEREK GELECEKTEN: boru hatti kontrolu, tez sonucu degil.

sigma: graf JSON'undaki kurala (varsayilan sector+mst) gore iki parcali eslemeyle (Hopcroft-Karp; dugumler panel
sirasiyla) bulunur, sabit noktasi yoktur ve `neighbor_map.json`'a yazilir; kosular o dosyayi kullanir.

    python scripts/marl/build_pred_m4.py --seeds 20261115,1,2 --out-dir data/marl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

MEAS = os.path.join('docs', 'development', 'measurements', 'phase10-m4')
GRAPH = os.path.join(MEAS, 'graph_rules.json')
FIELDS = ('p_up_cal', 'pred_logret', 'model_disagreement')


def neighbor_bijection(symbols: list, edges: list) -> list:
    """sigma[j] = j'nin gozlemine sinyali konan hisse; sigma[j] != j ve (j, sigma[j]) bir kenar."""
    import networkx as nx
    from networkx.algorithms import bipartite
    B = nx.Graph()
    top = [('h', s) for s in symbols]
    B.add_nodes_from(top, bipartite=0)
    B.add_nodes_from([('s', s) for s in symbols], bipartite=1)
    for u, v in edges:
        B.add_edge(('h', u), ('s', v))
        B.add_edge(('h', v), ('s', u))
    m = bipartite.hopcroft_karp_matching(B, top_nodes=top)
    idx = {s: i for i, s in enumerate(symbols)}
    sigma = [idx[m[('h', s)][1]] for s in symbols] if all(('h', s) in m for s in symbols) else None
    if sigma is None:
        raise SystemExit('grafta komsu bijeksiyonu yok')
    es = {frozenset(e) for e in edges}
    assert sorted(sigma) == list(range(len(symbols)))
    assert all(j != sigma[j] and frozenset((symbols[j], symbols[sigma[j]])) in es for j in range(len(symbols)))
    return sigma


def provenance() -> dict:
    git = lambda *c: subprocess.run(['git', *c], capture_output=True, text=True).stdout.strip()   # noqa: E731
    with open(os.path.abspath(__file__), 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return {'commit': git('rev-parse', 'HEAD'), 'git_status_porcelain': git('status', '--porcelain'),
            'script_sha256': sha}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='20261115,1,2')
    ap.add_argument('--rule', default='sector+mst')
    ap.add_argument('--out-dir', default=os.path.join('data', 'marl'))
    a = ap.parse_args()
    from evaluation.calibration import auc as _auc
    from marl import accounting as A
    from marl import panel as PN
    sys.path.insert(0, os.path.join(ROOT, 'scripts', 'marl'))
    from build_pred_m3 import PANEL, PANEL_SHA, _pc

    arrays = PN.load_npz(PANEL)
    if arrays['sha256'] != PANEL_SHA:
        raise SystemExit(f"panel SHA {arrays['sha256']} != {PANEL_SHA}")
    symbols = [str(s) for s in arrays['symbols']]
    g = json.load(open(GRAPH, encoding='utf-8'))
    sigma = neighbor_bijection(symbols, g['edges'][a.rule])
    graph_sha = hashlib.sha256(open(GRAPH, 'rb').read()).hexdigest()

    px = A.fill_prices(arrays['open'], arrays['close'], A.LISTING_LOCKOUT, A.limit_down_gaps(arrays['dates']))
    lab = _pc().k12_labels(arrays, px)
    lab['date'] = lab['date'].dt.strftime('%Y-%m-%d')
    Y = (lab.pivot_table(index='date', columns='symbol', values='y_up', aggfunc='first')
         .reindex(index=list(arrays['dates']), columns=symbols).to_numpy(dtype=np.float64))

    def auc_vs(P, L):
        m = np.isfinite(P) & np.isfinite(L)
        return float(_auc(P[m], L[m]))

    rec = {'note': 'M4a komsu sinyali bloklari; BILEREK GELECEKTEN, tez sonucu degil', **provenance(),
           'panel_sha256': arrays['sha256'], 'graph': GRAPH.replace(os.sep, '/'), 'graph_sha256': graph_sha,
           'rule': a.rule, 'sigma': {symbols[j]: symbols[sigma[j]] for j in range(len(symbols))}, 'blocks': []}
    os.makedirs(a.out_dir, exist_ok=True)
    for sd in [int(s) for s in a.seeds.split(',')]:
        for suffix in ('', '_shuf'):
            src = os.path.join(a.out_dir, f'pred_pc060_k12_g{sd}{suffix}.npz')
            d = np.load(src, allow_pickle=False)
            if not (np.array_equal(d['dates'], arrays['dates']) and np.array_equal(d['symbols'], arrays['symbols'])):
                raise SystemExit(f'blok panel ile hizali degil: {src}')
            blk = {k: d[k][:, sigma] for k in FIELDS}
            path = os.path.join(a.out_dir, f'pred_m4nb_g{sd}{suffix}.npz')
            np.savez_compressed(path, dates=arrays['dates'], symbols=arrays['symbols'], **blk)
            sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()
            row = {'seed': sd, 'shuffled_rows': bool(suffix), 'source_sha256':
                   hashlib.sha256(open(src, 'rb').read()).hexdigest(), 'path': path.replace(os.sep, '/'),
                   'sha256': sha,
                   'auc_source_vs_own_label': auc_vs(d['p_up_cal'], Y),          # M3: kendi sinyali
                   'auc_holder_vs_own_label': auc_vs(blk['p_up_cal'], Y),        # none'in eristigi (sizinti)
                   'auc_holder_vs_holder_label': auc_vs(blk['p_up_cal'], Y[:, sigma])}   # kontrol: = kaynak
            rec['blocks'].append(row)
            print(f"tohum {sd}{suffix}: kendi {row['auc_source_vs_own_label']:.4f}  komsu->kendi etiketi "
                  f"{row['auc_holder_vs_own_label']:.4f}  sha {sha[:8]}")
    out = os.path.join(MEAS, 'neighbor_map.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
