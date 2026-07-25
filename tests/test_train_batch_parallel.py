"""
Faz 6 (2.2 · B3) — Batch Egitim Paralelligi Testi

`train_batch` sembolleri is parcaciklariyla paralel egitebilir. Gercek egitim
pahali oldugu icin burada `train_model` mock'lanir; test edilen sey egitimin
kendisi degil **orkestrasyon**:

  - varsayilan (TRAIN_PARALLEL_SYMBOLS=1) seri yol degismedi
  - paralel modda tum semboller egitilir, manifest eksiksiz
  - es zamanlilik gercekten olusur (calisan sayisi > 1 gozlemlenir)
  - calisma alani baglami is parcacigina tasinir (Faz 7 — aksi halde
    modeller yanlis kullanicinin dizinine yazilir)
  - strict mod hatada durur, degraded/failed ayirt edilir
  - resume biten sembolleri atlar

Kullanim:
    python tests/test_train_batch_parallel.py
"""

import os
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

from app.auth import workspace as ws
from app.core.config import get_settings
from app.services.prediction_service import PredictionService

FULL_MODELS = ['xgboost', 'lightgbm', 'catboost', 'bilstm', 'tft']
FAILS = []


def check(name, cond, detail=''):
    print(('  [OK]   ' if cond else '  [FAIL] ') + name + (f'  {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


class _Recorder:
    """Mock train_model: es zamanlilik ve calisma alani baglamini kaydeder."""

    def __init__(self, delay=0.15, fail_on=(), degraded_on=()):
        self.delay = delay
        self.fail_on = set(fail_on)
        self.degraded_on = set(degraded_on)
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = []
        self.workspaces = {}

    def __call__(self, symbol, horizon='daily', **kwargs):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append(symbol)
            self.workspaces[symbol] = ws.current_user_id()
        try:
            time.sleep(self.delay)
            if symbol in self.fail_on:
                raise RuntimeError(f"{symbol} kasitli hata")
            models = ['xgboost'] if symbol in self.degraded_on else list(FULL_MODELS)
            return {
                'symbol': symbol,
                'models_trained': models,
                'n_models': len(models),
                'ensemble_test_metrics': {'mape': 12.3, 'direction_accuracy': 55.0},
            }
        finally:
            with self.lock:
                self.active -= 1


def run_batch(symbols, workers, **kwargs):
    """Mock'lu batch kosumu — (sonuc, recorder)."""
    s = get_settings()
    s.TRAIN_PARALLEL_SYMBOLS = workers
    svc = PredictionService()
    rec = _Recorder(**{k: v for k, v in kwargs.items()
                       if k in ('delay', 'fail_on', 'degraded_on')})
    svc.train_model = rec
    out = svc._train_batch(
        symbols,
        strict=kwargs.get('strict', False),
        resume_from=kwargs.get('resume_from'),
    )
    return out, rec


def main():
    s = get_settings()
    tmp = tempfile.mkdtemp(prefix='batch_par_')
    s.WORKSPACES_DIR = tmp
    written = []

    syms4 = ['AAA', 'BBB', 'CCC', 'DDD']

    print("1) Seri mod (varsayilan) — davranis degismedi")
    out, rec = run_batch(syms4, workers=1)
    written.append(out['manifest_path'])
    check('tum semboller egitildi', sorted(rec.calls) == syms4, f'{rec.calls}')
    check('es zamanlilik yok (max_active=1)', rec.max_active == 1, f'{rec.max_active}')
    check('hepsi ok', out['summary']['ok'] == 4, f"{out['summary']}")
    check('sira korunmus', rec.calls == syms4, f'{rec.calls}')

    print("\n2) Paralel mod — 3 is parcacigi")
    out, rec = run_batch(syms4, workers=3)
    written.append(out['manifest_path'])
    check('tum semboller egitildi', sorted(rec.calls) == syms4, f'{sorted(rec.calls)}')
    check('gercekten es zamanli calisti', rec.max_active > 1, f'max_active={rec.max_active}')
    check('manifest eksiksiz', out['summary']['ok'] == 4, f"{out['summary']}")

    print("\n3) Paralel hizlanma (mock gecikmesiyle)")
    t0 = time.perf_counter()
    out, _ = run_batch(syms4, workers=1, delay=0.3)
    serial = time.perf_counter() - t0
    written.append(out['manifest_path'])
    t0 = time.perf_counter()
    out, _ = run_batch(syms4, workers=4, delay=0.3)
    parallel = time.perf_counter() - t0
    written.append(out['manifest_path'])
    check('paralel belirgin hizli', parallel < serial * 0.6,
          f'seri={serial:.2f}s paralel={parallel:.2f}s')

    print("\n4) Calisma alani baglami is parcacigina tasiniyor (Faz 7)")
    uid = 'abc123def456'
    with ws.use_workspace(uid):
        out, rec = run_batch(syms4, workers=3)
        written.append(out['manifest_path'])
    seen = set(rec.workspaces.values())
    check('her is parcaciginda dogru kullanici', seen == {uid}, f'{seen}')
    check('manifest kullanici alanina yazildi', uid in out['manifest_path'],
          out['manifest_path'])

    print("\n5) degraded / failed ayrimi (paralel)")
    out, rec = run_batch(syms4, workers=3, fail_on=['CCC'], degraded_on=['BBB'])
    written.append(out['manifest_path'])
    check('failed tespit edildi', out['summary']['failed'] == 1 and 'CCC' in out['failed'],
          f"{out['summary']} {out['failed']}")
    check('degraded tespit edildi', out['summary']['degraded'] == 1, f"{out['summary']}")
    check('ok sayisi 2', out['summary']['ok'] == 2, f"{out['summary']}")

    print("\n6) Resume — biten semboller atlanir")
    prev_run = out['run_id']
    out2, rec2 = run_batch(syms4, workers=3, resume_from=prev_run)
    written.append(out2['manifest_path'])
    check('ok bitenler atlandi', len(out2['skipped']) == 2, f"{out2['skipped']}")
    check('sadece kalanlar egitildi', sorted(rec2.calls) == ['BBB', 'CCC'],
          f'{sorted(rec2.calls)}')

    print("\n7) strict mod — hatada durur")
    out3, rec3 = run_batch(['AAA', 'BBB', 'CCC', 'DDD'], workers=1,
                           fail_on=['BBB'], strict=True)
    written.append(out3['manifest_path'])
    check('strict seri: hatadan sonra durdu', len(rec3.calls) == 2, f'{rec3.calls}')

    # temizlik
    s.TRAIN_PARALLEL_SYMBOLS = 1
    for p in written:
        try:
            os.remove(p)
        except OSError:
            pass
    shutil.rmtree(tmp, ignore_errors=True)

    print('\n' + '=' * 60)
    if FAILS:
        print(f'BASARISIZ: {len(FAILS)} kontrol -> {FAILS}')
        return 1
    print('TUM KONTROLLER GECTI')
    return 0


if __name__ == '__main__':
    sys.exit(main())
