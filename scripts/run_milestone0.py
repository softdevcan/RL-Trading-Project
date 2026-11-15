"""
Milestone 0' kosum betigi (Faz 9 §9.22 madde 8; protokol `docs/development/milestone0-protocol.md`).

Izgara DONDURULMUS protokol JSON'undan (`docs/development/milestone0/protocol-v1.json`). Her hucre egitim
rotasinin kendisini (`_run_training_inner`) cagirir: veri kapisi, wf-v1 kati, olcek, kontrol noktasi secimi,
deneme kaydi (argmax + son nokta) ve P4 tahmin adaptoru rota/HPO ile AYNI yoldan (R1).

Dondurma ve butunluk (hakem-2 kapanis #8, #12, #13):
  * Defter `results_dir()/milestone0/<protocol_id>/ledger.jsonl`; ilk satir protokol JSON'u ve belgesinin
    SHA-256'si. Ikisinden biri degisirse betik REDDEDER (yeni surum = yeni protocol_id).
  * Kosumdan ONCE: P tablolarinin dosya SHA'si ve panelin (rotayla ayni yoldan) SHA'si JSON'daki beklenenle
    eslesmeli. Her hucreden SONRA: kaydin panel SHA'si, gercek hiperparametreleri, odul w2'si ve tablo SHA'si
    beklenenle eslesmeli; eslesmezse kosum DURUR.
  * GPU gorulurse (C1) kosmaz; `CUDA_VISIBLE_DEVICES=-1` ya da `--allow-gpu` (cihaz deftere yazilir).
  * `--smoke` deneme kaydina YAZMAZ (gecici dizin) ve yalniz protokoldeki `smoke.fold` katinda kosar (acik pencere;
    2021 duman okumasi bir daha olmasin — §9.34).

    python scripts/run_milestone0.py --dry-run [--stage prescreen]
    python scripts/run_milestone0.py --print-input-shas
    CUDA_VISIBLE_DEVICES=-1 python scripts/run_milestone0.py --smoke [--steps 20000] [--algorithm TD3 --reward psr_differential]
    CUDA_VISIBLE_DEVICES=-1 python scripts/run_milestone0.py --stage prescreen [--limit N] [--user <id>]
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


def file_sha(path: str) -> str:
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_protocol(path: str):
    with open(path, 'rb') as fh:
        raw = fh.read()
    return json.loads(raw.decode('utf-8')), hashlib.sha256(raw).hexdigest()


def cells(protocol: dict, stage: str):
    """Asama hucreleri: etkenler x maliyet x katlar x tohumlar (sira sabit -> devam guvenli)."""
    st = protocol['stages'][stage]
    over = st.get('factor_override')
    if over is not None and not isinstance(over, dict):
        raise SystemExit(f"'{stage}' asamasinin etken listesi henuz yok: {over}")
    over = over or {}
    levels = [over.get(k, protocol['factors'][k]) for k in FACTOR_KEYS]
    costs = st.get('commission_rates', [protocol['fixed']['commission_rate']])
    for combo in itertools.product(*levels):
        for cr in costs:
            for fold in st['folds']:
                for seed in st['seeds']:
                    cfg = dict(zip(FACTOR_KEYS, combo))
                    cfg.update(commission_rate=cr, fold=fold, seed=seed)
                    yield cfg


def cell_key(c: dict) -> str:
    return '|'.join(f'{k}={c[k]}' for k in (*FACTOR_KEYS, 'commission_rate', 'fold', 'seed'))


def to_request(c: dict, protocol: dict, steps: int):
    """Hucre -> TrainingRequest (odul seviyesi faz, Sharpe terimi ve w2'yi tasir)."""
    from app.schemas.trading import TrainingRequest
    reward = protocol['reward_levels'][c['reward']]
    pred = protocol['prediction_levels'][c['prediction']]
    return TrainingRequest(
        algorithm=c['algorithm'], phase=reward['phase'], sharpe_term=reward.get('sharpe_term', 'ema_level'),
        sharpe_weight=reward.get('sharpe_weight'),
        total_timesteps=steps, initial_balance=protocol['fixed']['initial_balance'],
        commission_rate=c['commission_rate'], max_shares_per_trade=protocol['fixed']['max_shares_per_trade'],
        seed=c['seed'], split_spec=protocol['fixed']['split_spec'], fold=c['fold'],
        price_norm=c['price_norm'], prediction_table=pred.get('table'))


def panel_sha(price_norm: str) -> str:
    """Rotanin egitim panelinin SHA'si — rotayla AYNI yol (`_run_training_inner` C3 anlik goruntusu)."""
    from app.api.routes import trading as tr
    from data.data_fetcher import DataFetcher
    from evaluation import rl_folds
    from evaluation.reproducibility import frame_sha256
    raw = DataFetcher().load_data(tr.RL_PANEL_FILE)
    rp = rl_folds.load_rl_panel(price_norm=price_norm, start=tr._settings.RL_DATA_START, df=raw,
                                adx_mode=tr.RL_ADX_MODE)
    return frame_sha256(rp.panel)


def input_shas(protocol: dict) -> dict:
    from app.auth import workspace as ws
    d = os.path.join(ws.results_dir(), 'predictions_pit')
    tables = {lv: (file_sha(os.path.join(d, f"{v['table']}.parquet")) if v.get('table') else None)
              for lv, v in protocol['prediction_levels'].items()}
    panels = {pn: panel_sha(pn) for pn in protocol['factors']['price_norm']}
    return {'tables': tables, 'panels': panels}


def verify_inputs(protocol: dict) -> None:
    got = input_shas(protocol)
    bad = []
    for lv, v in protocol['prediction_levels'].items():
        if v.get('table') and got['tables'][lv] != v.get('table_sha256'):
            bad.append(f"P tablosu {lv} ({v['table']}): {got['tables'][lv]} != {v.get('table_sha256')}")
    for pn, sha in protocol['expected']['panel_sha256'].items():
        if got['panels'].get(pn) != sha:
            bad.append(f"panel [{pn}]: {got['panels'].get(pn)} != {sha}")
    if bad:
        raise SystemExit('Girdiler dondurulmus protokolle eslesmiyor (veri tazelendiyse yeni protokol surumu):\n  '
                         + '\n  '.join(bad))


def _close(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(b)))
    return a == b


def verify_trial(trial_id: str, c: dict, protocol: dict) -> list:
    """Hucrenin deneme kaydi protokolle uyusuyor mu; uyusmazlik listesi."""
    from evaluation.trial_registry import load_trials
    recs = [t for t in load_trials(exclude_tags=()) if t['trial_id'] == trial_id]
    if not recs:
        return [f'kayit yok: {trial_id}']
    t, cfg, bad = recs[0], recs[0]['config'], []
    if t.get('data_sha256') != protocol['expected']['panel_sha256'][c['price_norm']]:
        bad.append(f"panel SHA {t.get('data_sha256')}")
    want = protocol['hyperparameters'][c['algorithm']]
    have = cfg.get('effective_hyperparameters') or {}
    diff = {k: (have.get(k), v) for k, v in want.items() if not _close(have.get(k), v)}
    if diff or set(have) - set(want):
        bad.append(f"hiperparametre {diff} fazla {sorted(set(have) - set(want))}")
    rl = protocol['reward_levels'][c['reward']]
    w2 = (cfg.get('reward_spec') or {}).get('w2')
    if rl['phase'] == 2 and not _close(w2, rl.get('sharpe_weight', 0.30)):
        bad.append(f"w2 {w2}")
    pt = cfg.get('prediction_table') or {}
    exp = protocol['prediction_levels'][c['prediction']].get('table_sha256')
    if (pt.get('sha256') if pt else None) != exp:
        bad.append(f"P tablosu {pt.get('sha256') if pt else None}")
    return bad


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


def device_guard(allow_gpu: bool) -> str:
    import torch
    if torch.cuda.is_available():
        if not allow_gpu:
            raise SystemExit('GPU gorunuyor: protokol CPU (C1; bit-eslik yalniz CPU). CUDA_VISIBLE_DEVICES=-1 '
                             'ile kos ya da --allow-gpu (cihaz deftere yazilir).')
        return f'cuda:{torch.cuda.get_device_name(0)}'
    return 'cpu'


def ledger_path(protocol: dict) -> str:
    from app.auth import workspace as ws
    d = os.path.join(ws.results_dir(), 'milestone0', protocol['protocol_id'])
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, 'ledger.jsonl')


def read_ledger(path: str, header: dict):
    done = {}
    if not os.path.exists(path):
        with open(path, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(json.dumps({'header': True, **header}) + '\n')
        return done
    with open(path, encoding='utf-8') as fh:
        lines = [json.loads(x) for x in fh if x.strip()]
    h = lines[0] if lines else {}
    for k in ('protocol_sha256', 'protocol_doc_sha256'):
        if h.get(k) != header[k]:
            raise SystemExit(f'{k} defterdekinden FARKLI ({path}). Dondurulmus protokol degistirilemez; '
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
    ap.add_argument('--print-input-shas', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--steps', type=int, default=None, help='yalniz --smoke')
    ap.add_argument('--algorithm', default=None, help='yalniz --smoke')
    ap.add_argument('--reward', default=None, help='yalniz --smoke')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--user', default=None, help='calisma alani (workspaces/<id>); yoksa kok results/')
    ap.add_argument('--allow-gpu', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        # Duman kosusu deneme kaydina (N, DSR) ve model dizinine YAZMAZ: ayarlar yuklenmeden once ciktilar
        # gecici dizine; P tablolari okunabilsin diye kok tablo dizini kopyalanmaz, smoke DT'de kosar.
        import tempfile
        tmp = tempfile.mkdtemp(prefix='m0_smoke_')
        for k, sub in (('MODELS_DIR', 'models'), ('RESULTS_DIR', 'results'), ('LOGS_DIR', 'logs'),
                       ('HYPEROPT_DIR', 'hyperopt')):
            os.environ[k] = os.path.join(tmp, sub)
        print(f'duman: ciktilar {tmp} altina (deneme kaydina girmez)', flush=True)
    protocol, sha = load_protocol(a.protocol)
    doc_sha = file_sha(protocol['doc'])
    from app.auth import workspace as ws
    with ws.use_workspace(a.user) if a.user else contextlib.nullcontext():
        if a.print_input_shas:
            print(json.dumps(input_shas(protocol), indent=1))
            return
        if a.dry_run:
            st = protocol['stages'][a.stage]
            grid = list(cells(protocol, a.stage))
            print(json.dumps({'protocol_id': protocol['protocol_id'], 'sha256': sha, 'doc_sha256': doc_sha,
                              'stage': a.stage, 'n_cells': len(grid),
                              'hours_lower_bound_single_process': round(
                                  len(grid) * protocol['budget']['seconds_per_run_measured_ppo_phase1'] / 3600, 1),
                              'note': 'PPO faz 1 olcumunden alt sinir; TD3/faz 2 --smoke ile olc'}, indent=1))
            return
        device = device_guard(a.allow_gpu)
        if a.smoke:
            sm = protocol['smoke']
            c = {'algorithm': a.algorithm or sm['algorithm'], 'reward': a.reward or sm['reward'],
                 'price_norm': sm['price_norm'], 'prediction': 'DT',
                 'commission_rate': protocol['fixed']['commission_rate'], 'fold': sm['fold'], 'seed': sm['seed']}
            if c['fold'] in protocol['unexposed_folds']:
                raise SystemExit('duman kosusu okunmamis katta kosmaz')
            steps = a.steps or 20000
            r = run_cell(to_request(c, protocol, steps))
            per_step = r['seconds'] / steps
            # duman kaydi gecici kayitta: protokol denetimi uctan uca (panel SHA, hiperparametre, w2, tablo)
            r['verify'] = verify_trial(r['trial_id'], c, protocol) if r['state'] == 'completed' else None
            print(json.dumps({'cell': cell_key(c), 'device': device, 'steps': steps, **r,
                              'seconds_per_step': per_step,
                              'extrapolated_run_seconds': round(per_step * protocol['stages']['prescreen']['total_timesteps'])},
                             indent=1, default=str))
            return
        st = protocol['stages'][a.stage]
        grid = list(cells(protocol, a.stage))
        verify_inputs(protocol)
        path = ledger_path(protocol)
        done = read_ledger(path, {'protocol_sha256': sha, 'protocol_doc_sha256': doc_sha, 'device': device,
                                  'created': time.strftime('%Y-%m-%dT%H:%M:%S')})
        todo = [c for c in grid if cell_key(c) not in done]
        print(f'{a.stage}: {len(grid)} hucre, tamamlanmis {len(done)}, kalan {len(todo)} -> {path}')
        for i, c in enumerate(todo[:a.limit] if a.limit else todo):
            r = run_cell(to_request(c, protocol, st['total_timesteps']))
            bad = verify_trial(r['trial_id'], c, protocol) if r['state'] == 'completed' else []
            if bad:
                r['state'] = 'mismatch'
            rec = {'cell': cell_key(c), **c, 'stage': a.stage, **r, 'mismatch': bad, 'device': device,
                   'ts': time.strftime('%Y-%m-%dT%H:%M:%S')}
            with open(path, 'a', encoding='utf-8', newline='\n') as fh:
                fh.write(json.dumps(rec, default=str) + '\n')
            print(f"[{i + 1}] {rec['cell']} -> {r['state']} ({r['seconds']} sn) trial {r['trial_id']}", flush=True)
            if bad:
                raise SystemExit(f'Kayit protokolle uyusmuyor, kosum durdu: {bad}')


if __name__ == '__main__':
    main()
