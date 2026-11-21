"""Faz 10 M8 nihai kosumlar duzenleyicisi (M8 tasarimi §6 ON KAYIT v1.0). venv-marl.

Kollar (§6.2): M-none, S-esm, S-agr, M-gat, M-self (tohum 201-212) ve bos kol (M-none'in aynisi, tohum 213-224).
Katlar wf-v1 (§6.3; tanim ana venv'de evaluation/splits.py, okumada denetlenir). Her (kat, kol, tohum):
train_m3.py (--checkpoints 10, panel_m8, egitim sonu = dogrulama baslangicindan bir gun once) ->
eval_fold_m8.py --also-last (dogrulamada secim, test secilenle BIR KEZ; son nokta ayri kayit).

Sira kat -> tohum -> kol (yarida kalan kosum her kolda dengeli kalir). Degerlendirmesi olan kosu atlanir (yeniden
baslatilabilir). Coken kosu ayni yapilandirma ve tohumla BIR KEZ yeniden baslatilir; ikinci cokus `failures.jsonl`e
yazilir ve hucre eksik sayilir (§6.6). Her kosunun baslangic/bitis zamani `timing.jsonl`de.
GPU: yalniz kendi kosumuz yokken disaridan dolu mu diye bakilir (oyun vb.); kendi kosularimiz GPU'yu doldurdugu icin
her gorevden once bakmak eszamanliligi bozardi.

  venv-marl\\Scripts\\python scripts/marl/run_m8.py --dry-run
  venv-marl\\Scripts\\python scripts/marl/run_m8.py --jobs 3
  venv-marl\\Scripts\\python scripts/marl/run_m8.py --jobs 3 --folds 2021      # kat kat da kosulabilir
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_m3a import ROOT  # noqa: E402
from run_m4a import PY, wait_gpu  # noqa: E402
from run_m4b import run_logged  # noqa: E402

os.chdir(ROOT)
FRAMES = 600_000
CHECKPOINTS = 10
PANEL = os.path.join('data', 'marl', 'panel_m8.npz')
MAIN_SEEDS = tuple(range(201, 213))
NULL_SEEDS = tuple(range(213, 225))
# wf-v1 (evaluation/splits.py:115-124): test takvim yili, dogrulama onceki yil, egitim 2013-06'dan; 2026 testi 11-01'de biter
FOLDS = {str(y): {'val_start': f'{y - 1}-01-01', 'test_start': f'{y}-01-01',
                  'test_end': '2026-11-01' if y == 2026 else f'{y + 1}-01-01', 'train_end': f'{y - 2}-12-31'}
         for y in range(2021, 2027)}
ARMS = {
    'M-none': [],
    'S-esm': ['--sarl', 'esmimari'],
    'S-agr': ['--sarl', 'agirlik'],
    'M-gat': ['--comm', 'gat'],
    'M-self': ['--comm', 'self'],
    'null': [],
}
ARM_ORDER = ('M-none', 'S-esm', 'S-agr', 'M-gat', 'M-self', 'null')


def train_cmd(fold: str, arm: str, seed: int, run_dir: str, device: str) -> list:
    f = FOLDS[fold]
    return [PY, 'scripts/marl/train_m3.py', '--panel', PANEL, '--train-end', f['train_end'],
            '--checkpoints', str(CHECKPOINTS), '--frames', str(FRAMES), '--seed', str(seed), '--cost-bps', '20',
            '--lr', '5e-4', '--batched-env', '--train-device', device, *ARMS[arm], '--save-dir', run_dir,
            '--out', os.path.join(run_dir, 'train.json')]


def eval_cmd(fold: str, run_dir: str) -> list:
    f = FOLDS[fold]
    return [PY, 'scripts/marl/eval_fold_m8.py', '--run-dir', run_dir, '--fold', fold, '--val-start', f['val_start'],
            '--test-start', f['test_start'], '--test-end', f['test_end'], '--also-last',
            '--out-dir', os.path.join(run_dir, 'eval')]


def tasks(folds, arms) -> list:
    """(kat, kol, tohum) sirasi: kat -> tohum indeksi -> kol."""
    out = []
    for fold in folds:
        for i in range(len(MAIN_SEEDS)):
            for arm in arms:
                out.append((fold, arm, (NULL_SEEDS if arm == 'null' else MAIN_SEEDS)[i]))
    return out


def run_dir_of(root: str, fold: str, arm: str, seed: int) -> str:
    return os.path.join(root, fold, arm, f's{seed}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join('results', 'marl_m8'))
    ap.add_argument('--train-device', default='cuda')
    ap.add_argument('--folds', default=','.join(FOLDS), help='virgulle; varsayilan tum wf-v1 katlari')
    ap.add_argument('--arms', default=','.join(ARM_ORDER), help='yalniz duman testi icin degistir')
    ap.add_argument('--gpu-busy-pct', type=int, default=50)
    ap.add_argument('--max-wait', type=int, default=12 * 3600)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--jobs', type=int, default=3, help='es zamanli kosu (plan §3.2.8: GPU 8 GB -> 3-4)')
    a = ap.parse_args()
    folds = [f for f in a.folds.split(',') if f]
    arms = [x for x in a.arms.split(',') if x]
    bad = [f for f in folds if f not in FOLDS] + [x for x in arms if x not in ARMS]
    if bad:
        raise SystemExit(f'bilinmeyen kat/kol: {bad}')
    if not os.path.exists(PANEL):
        raise SystemExit(f'panel yok: {PANEL} (prepare_panel_m2.py --data-fix m8-v1)')
    todo = [t for t in tasks(folds, arms)
            if not os.path.exists(os.path.join(run_dir_of(a.root, *t), 'eval', 'fold_eval.json'))]
    print(f'{len(todo)} kosu kaldi ({len(folds)} kat x {len(arms)} kol x {len(MAIN_SEEDS)} tohum uzerinden)', flush=True)
    os.makedirs(a.root, exist_ok=True)
    lock, active = threading.Lock(), [0]

    def append(name, rec):
        if a.dry_run:
            return
        with lock, open(os.path.join(a.root, name), 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')

    def one(t):
        fold, arm, seed = t
        d = run_dir_of(a.root, fold, arm, seed)
        log = None if a.dry_run else os.path.join(d, 'run.log')
        if not a.dry_run:
            os.makedirs(d, exist_ok=True)
            while True:   # disaridan dolu GPU: yalniz kendi kosumuz yokken bak
                with lock:
                    if active[0] > 0:
                        active[0] += 1
                        break
                wait_gpu(a)
                with lock:
                    active[0] += 1
                    break
        t0 = time.time()
        try:
            for attempt in (1, 2):
                try:
                    if not os.path.exists(os.path.join(d, 'policy.pt')):
                        run_logged(train_cmd(fold, arm, seed, d, a.train_device), a.dry_run, log)
                    run_logged(eval_cmd(fold, d), a.dry_run, log)
                    append('timing.jsonl', {'fold': fold, 'arm': arm, 'seed': seed, 'attempt': attempt,
                                            'start': dt.datetime.fromtimestamp(t0).isoformat(timespec='seconds'),
                                            'sec': round(time.time() - t0, 1)})
                    return
                except subprocess.CalledProcessError as e:
                    append('failures.jsonl', {'fold': fold, 'arm': arm, 'seed': seed, 'attempt': attempt,
                                              'returncode': e.returncode,
                                              'time': dt.datetime.now().isoformat(timespec='seconds')})
                    print(f'COKTU {fold}/{arm}/s{seed} (deneme {attempt})', flush=True)
                    # test okunduysa (eval npz var, json yok) ikinci deneme yapilmaz: test ikinci kez okunmaz
                    ev = os.path.join(d, 'eval')
                    if os.path.isdir(ev) and any(n.startswith('test_') for n in os.listdir(ev)):
                        return
        finally:
            if not a.dry_run:
                with lock:
                    active[0] -= 1

    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for f in [ex.submit(one, t) for t in todo]:
            f.result()


if __name__ == '__main__':
    main()
