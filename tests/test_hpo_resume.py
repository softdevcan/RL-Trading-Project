"""
Faz 6 (3.2 · B6) — HPO SQLite Resume Testi

Optuna study'si kalici depoya (sqlite) yazildiginda yarida kesilen HPO
kaldigi trial'dan devam etmeli, sifirdan baslamamali.

Ucuz model (xgboost) + kucuk sentetik veri + az trial ile gercek HPO
kosturur; agir DL gerekmez. Test edilen: kalicilik + resume trial sayaci.

Kullanim:
    python tests/test_hpo_resume.py
"""

import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

import numpy as np

FAILS = []


def check(name, cond, detail=''):
    print(('  [OK]   ' if cond else '  [FAIL] ') + name + (f'  {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


def _data(n=240, f=8):
    rng = np.random.RandomState(7)
    X = rng.randn(n, f)
    y = X[:, 0] * 0.5 + rng.randn(n) * 0.1
    return X, y


def main():
    from prediction.hyperopt import PredictionHyperOptimizer

    tmp = tempfile.mkdtemp(prefix='hpo_resume_')
    db = os.path.join(tmp, 'hpo.db').replace('\\', '/')
    storage = f'sqlite:///{db}'
    X, y = _data()

    print("1) Bellekte (storage=None) — eski davranis")
    opt_mem = PredictionHyperOptimizer(n_trials=3, n_cv_splits=2, timeout=None)
    check('storage None', opt_mem.storage is None)
    r = opt_mem.optimize('xgboost', X, y, horizon='daily', study_name='mem_test')
    check('bellekte HPO calisti', 'best_params' in r and r['n_trials'] >= 1,
          f"n_trials={r.get('n_trials')}")

    print("\n2) Kalici depo — ilk kosum (5 trial)")
    opt1 = PredictionHyperOptimizer(n_trials=5, n_cv_splits=2, timeout=None, storage=storage)
    check('storage ayarlandi', opt1.storage == storage)
    check('db dosyasi dizini hazir', os.path.isdir(tmp))
    r1 = opt1.optimize('xgboost', X, y, horizon='daily', study_name='resume_test')
    check('db dosyasi olustu', os.path.exists(db))
    n_after_first = r1['n_trials']
    check('5 trial tamamlandi', n_after_first == 5, f'{n_after_first}')

    print("\n3) Resume — ayni study, ayni n_trials (biten atlanir)")
    import optuna
    opt2 = PredictionHyperOptimizer(n_trials=5, n_cv_splits=2, timeout=None, storage=storage)
    # Resume oncesi study'de kac trial var?
    study = optuna.load_study(study_name='resume_test', storage=storage)
    before = len(study.trials)
    check('onceki trial\'lar korundu', before == 5, f'{before}')

    r2 = opt2.optimize('xgboost', X, y, horizon='daily', study_name='resume_test')
    study2 = optuna.load_study(study_name='resume_test', storage=storage)
    after = len(study2.trials)
    check('resume yeni trial EKLEMEDI (5 zaten doluydu)', after == 5,
          f'before={before} after={after}')
    check('best_params yine dondu', 'best_params' in r2)

    print("\n4) Kismi resume — butce artir (5 -> 8), 3 yeni trial")
    opt3 = PredictionHyperOptimizer(n_trials=8, n_cv_splits=2, timeout=None, storage=storage)
    r3 = opt3.optimize('xgboost', X, y, horizon='daily', study_name='resume_test')
    study3 = optuna.load_study(study_name='resume_test', storage=storage)
    check('toplam 8 trial\'a ulasti', len(study3.trials) == 8, f'{len(study3.trials)}')

    print("\n5) Config knob (HPO_STORAGE) okunuyor")
    from app.core.config import get_settings
    s = get_settings()
    old = s.HPO_STORAGE
    s.HPO_STORAGE = storage
    opt_cfg = PredictionHyperOptimizer(n_trials=2, n_cv_splits=2, timeout=None)
    check('config\'ten storage alindi', opt_cfg.storage == storage, f'{opt_cfg.storage}')
    s.HPO_STORAGE = old

    shutil.rmtree(tmp, ignore_errors=True)

    print('\n' + '=' * 60)
    if FAILS:
        print(f'BASARISIZ: {len(FAILS)} kontrol -> {FAILS}')
        return 1
    print('TUM KONTROLLER GECTI')
    return 0


if __name__ == '__main__':
    sys.exit(main())
