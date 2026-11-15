"""
Milestone 0' kosum betigi (Faz 9 §9.22 madde 8; protokol `docs/development/milestone0-protocol.md`).

Izgara DONDURULMUS protokol JSON'undan okunur (`docs/development/milestone0/protocol-v1.json`). Her hucre
egitim rotasinin kendisini (`_run_training_inner`) cagirir: veri kapisi, wf-v1 kati, olcek, kontrol noktasi
secimi, deneme kaydi ve P4 tahmin adaptoru rota ve HPO ile AYNI yoldan gecer (R1). Kosum defteri
`results_dir()/milestone0/<protocol_id>/ledger.jsonl`: ilk satir protokol SHA-256'si; protokol dosyasi
degisirse betik kosmayi REDDEDER (dondurma). Tamamlanan hucre yeniden kosulmaz (devam).

    python scripts/run_milestone0.py --dry-run                  # hucre sayisi + sure tahmini
    python scripts/run_milestone0.py --smoke [--steps 20000]    # tek hucre, adim maliyeti olcumu (deftere yazmaz)
    python scripts/run_milestone0.py --stage prescreen [--limit N] [--user <id>]
"""

import argparse
import asyncio
import contextlib
import hashlib
import io
import itertools
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

DEFAULT_PROTOCOL = os.path.join('docs', 'development', 'milestone0', 'protocol-v1.json')
FACTOR_KEYS = ('algorithm', 'reward', 'price_norm', 'prediction')


def load_protocol(path: str):
    with open(path, 'rb') as fh:
        raw = fh.read()
    return json.loads(raw.decode('utf-8')), hashlib.sha256(raw).hexdigest()


def cells(protocol: dict, stage: str):
    """Asama hucreleri: etkenlerin kartezyen carpimi x katlar x tohumlar (sira sabit -> devam guvenli)."""
    st = protocol['stages'][stage]
    f = protocol['factors']
    over = st.get('factor_override')
    if over is not None and not isinstance(over, dict):
        raise SystemExit(f"'{stage}' asamasinin etken listesi henuz yok: {over}")
    over = over or {}
    levels = [over.get(k, f[k]) for k in FACTOR_KEYS]
    for combo in itertools.product(*levels):
        for fold in st['folds']:
            for seed in st['seeds']:
                cfg = dict(zip(FACTOR_KEYS, combo))
                cfg.update(fold=fold, seed=seed)
                yield cfg


def cell_key(c: dict) -> str:
    return '|'.join(f'{k}={c[k]}' for k in (*FACTOR_KEYS, 'fold', 'seed'))


def to_request(c: dict, protocol: dict, steps: int):
    """Hucre -> TrainingRequest. Odul seviyesi 'simple' (faz 1) | 'psr_ema_level' | 'psr_differential' (faz 2)."""
    from app.schemas.trading import TrainingRequest
    reward = protocol['reward_levels'][c['reward']]
    pred = protocol['prediction_levels'][c['prediction']]
    return TrainingRequest(
        algorithm=c['algorithm'], phase=reward['phase'], sharpe_term=reward.get('sharpe_term', 'ema_level'),
        total_timesteps=steps, initial_balance=protocol['fixed']['initial_balance'],
        commission_rate=protocol['fixed']['commission_rate'],
        max_shares_per_trade=protocol['fixed']['max_shares_per_trade'],
        seed=c['seed'], split_spec=protocol['fixed']['split_spec'], fold=c['fold'],
        price_norm=c['price_norm'], prediction_table=pred.get('table'))


def run_cell(req, quiet: bool = True) -> dict:
    from app.api.routes import trading as tr
    state = tr._empty_training_state()
    t0 = time.time()
    out = io.StringIO()
    with (contextlib.redirect_stdout(out) if quiet else contextlib.nullcontext()):
        asyncio.run(tr._run_training_inner(req, state))
    return {'state': state.get('state'), 'error': state.get('error'),
            'trial_id': (state.get('metrics') or {}).get('trial_id'), 'seconds': round(time.time() - t0, 1),
            'warnings': state.get('warnings', [])}


def ledger_path(protocol: dict) -> str:
    from app.auth import workspace as ws
    d = os.path.join(ws.results_dir(), 'milestone0', protocol['protocol_id'])
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, 'ledger.jsonl')


def read_ledger(path: str, sha: str):
    done = {}
    if not os.path.exists(path):
        with open(path, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(json.dumps({'header': True, 'protocol_sha256': sha,
                                 'created': time.strftime('%Y-%m-%dT%H:%M:%S')}) + '\n')
        return done
    with open(path, encoding='utf-8') as fh:
        lines = [json.loads(x) for x in fh if x.strip()]
    if not lines or lines[0].get('protocol_sha256') != sha:
        raise SystemExit(f'Protokol dosyasi defterdekinden FARKLI ({path}). Dondurulmus protokol degistirilemez; '
                         'yeni surum = yeni protocol_id.')
    for r in lines[1:]:
        if r.get('state') == 'completed':
            done[r['cell']] = r
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', default=DEFAULT_PROTOCOL)
    ap.add_argument('--stage', default='prescreen')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--steps', type=int, default=None, help='yalniz --smoke: adim sayisi')
    ap.add_argument('--limit', type=int, default=None, help='bu cagrida en fazla N hucre')
    ap.add_argument('--user', default=None, help='calisma alani (workspaces/<id>); yoksa kok results/')
    a = ap.parse_args()
    if a.smoke:
        # Duman kosusu deneme kaydina (N, DSR) ve model dizinine YAZMAZ: uygulama ayarlari yuklenmeden
        # once ciktilar gecici dizine yonlenir (spike/duman sayisi rapora girmez; Faz 10 S1).
        import tempfile
        tmp = tempfile.mkdtemp(prefix='m0_smoke_')
        for k, sub in (('MODELS_DIR', 'models'), ('RESULTS_DIR', 'results'), ('LOGS_DIR', 'logs'),
                       ('HYPEROPT_DIR', 'hyperopt')):
            os.environ[k] = os.path.join(tmp, sub)
        print(f'duman: ciktilar {tmp} altina (deneme kaydina girmez)', flush=True)
    protocol, sha = load_protocol(a.protocol)
    st = protocol['stages'][a.stage]
    grid = list(cells(protocol, a.stage))
    from app.auth import workspace as ws
    with ws.use_workspace(a.user) if a.user else contextlib.nullcontext():
        if a.dry_run:
            from app.services.training_eta import estimate
            per = {alg: estimate(alg.lower(), 2, st['total_timesteps'], n_symbols=30)['total_seconds']
                   for alg in protocol['factors']['algorithm']}
            tot = sum(per[c['algorithm']] for c in grid) * (1 + protocol['fixed'].get('validation_overhead', 0))
            print(json.dumps({'protocol_id': protocol['protocol_id'], 'sha256': sha, 'stage': a.stage,
                              'n_cells': len(grid), 'n_configs': len(grid) // (len(st['folds']) * len(st['seeds'])),
                              'seconds_per_run_estimate': per, 'total_hours_single_process': round(tot / 3600, 1),
                              'note': 'ETA gecmisinden; --smoke ile olc'}, indent=1))
            return
        if a.smoke:
            c = grid[0]
            steps = a.steps or 20000
            r = run_cell(to_request(c, protocol, steps))
            per_step = r['seconds'] / steps
            full = per_step * st['total_timesteps']
            print(json.dumps({'cell': cell_key(c), 'steps': steps, **r, 'seconds_per_step': per_step,
                              'extrapolated_run_seconds': round(full),
                              'extrapolated_stage_hours_single_process': round(full * len(grid) / 3600, 1)},
                             indent=1, default=str))
            return
        path = ledger_path(protocol)
        done = read_ledger(path, sha)
        todo = [c for c in grid if cell_key(c) not in done]
        print(f'{a.stage}: {len(grid)} hucre, tamamlanmis {len(done)}, kalan {len(todo)} -> {path}')
        for i, c in enumerate(todo[:a.limit] if a.limit else todo):
            r = run_cell(to_request(c, protocol, st['total_timesteps']))
            rec = {'cell': cell_key(c), **c, 'stage': a.stage, **r, 'ts': time.strftime('%Y-%m-%dT%H:%M:%S')}
            with open(path, 'a', encoding='utf-8', newline='\n') as fh:
                fh.write(json.dumps(rec, default=str) + '\n')
            print(f"[{i + 1}] {rec['cell']} -> {r['state']} ({r['seconds']} sn) trial {r['trial_id']}", flush=True)


if __name__ == '__main__':
    main()
