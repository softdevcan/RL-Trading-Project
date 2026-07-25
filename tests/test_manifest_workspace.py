"""Manifest calisma alani cozumlemesi — Faz 6 G.3 x Faz 7 izolasyon."""
import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.abspath('.'))

import warnings, logging
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

from app.core.config import get_settings
from app.auth import workspace as ws
from prediction.manifest import (
    TrainingManifest, runs_dir, run_dirs, find_manifest, latest_run_id,
    LEGACY_RUNS_DIR,
)

FAILS = []


def check(name, cond, detail=''):
    print(('  [OK]   ' if cond else '  [FAIL] ') + name + (f'  {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


def main():
    s = get_settings()
    tmp = tempfile.mkdtemp(prefix='ws_manifest_')
    s.WORKSPACES_DIR = tmp
    print(f"Gecici calisma alani koku: {tmp}")
    print(f"AUTH_ENABLED={s.AUTH_ENABLED} WORKSPACE_ISOLATION={s.WORKSPACE_ISOLATION}")

    # --- 1. Betik baglami (kullanici yok) -> ortak/legacy dizin ---
    print("\n1) Betik baglami (aktif kullanici yok)")
    d = runs_dir()
    check('runs_dir() ortak dizine duser',
          os.path.abspath(d) == os.path.abspath(LEGACY_RUNS_DIR), f'({d})')

    m = TrainingManifest(run_kind='test', symbols=['AAA'])
    m.run_id = 'TEST_LEGACY_RUN'
    m.record_symbol('AAA', result={'models_trained':
                                   ['xgboost', 'lightgbm', 'catboost', 'bilstm', 'tft'],
                                   'n_models': 5})
    p_legacy = m.finalize()
    check('manifest ortak dizine yazildi',
          os.path.exists(p_legacy) and 'workspaces' not in p_legacy, f'({p_legacy})')
    legacy_run_id = m.run_id
    check('status=ok (5 model)', m.symbols['AAA']['status'] == 'ok')

    # --- 2. Kullanici baglami -> izole dizin ---
    print("\n2) Kullanici baglami (use_workspace)")
    uid = 'a1b2c3d4e5f6'
    with ws.use_workspace(uid):
        d2 = runs_dir()
        check('runs_dir() kullanici alanina cozulur',
              uid in d2 and 'training_runs' in d2, f'({d2})')

        m2 = TrainingManifest(run_kind='test', symbols=['BBB'])
        m2.run_id = 'TEST_USER_A_RUN'
        m2.record_symbol('BBB', result={'models_trained': ['xgboost'], 'n_models': 1})
        p_user = m2.finalize()
        check('manifest kullanici alanina yazildi', uid in p_user, f'({p_user})')
        check('degraded tespiti (1/5 model)', m2.symbols['BBB']['status'] == 'degraded',
              f"missing={m2.symbols['BBB']['missing_models']}")

        # Okuma: once kendi alani, sonra ortak
        dirs = run_dirs()
        check('run_dirs() = [kullanici, ortak]',
              len(dirs) == 2 and uid in dirs[0], f'({dirs})')
        check('kendi manifestini bulur', find_manifest(m2.run_id) is not None)
        check('eski ortak kosumu da bulur (resume geriye uyumlu)',
              find_manifest(legacy_run_id) is not None)
        check('latest_run_id() calisir', latest_run_id() != '')

    # --- 3. Izolasyon: baska kullanici digerinin manifestini gormemeli ---
    print("\n3) Kullanicilar arasi izolasyon")
    other = 'f6e5d4c3b2a1'
    with ws.use_workspace(other):
        own = find_manifest(m2.run_id)
        check('digerinin manifesti gorunmuyor', own is None,
              f'(bulunan={own})')

    # --- 4. Baglam disina cikinca ortak dizine geri doner ---
    print("\n4) Baglam sonrasi")
    check('baglam disinda yine ortak dizin',
          os.path.abspath(runs_dir()) == os.path.abspath(LEGACY_RUNS_DIR))

    # temizlik
    try:
        os.remove(p_legacy)
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
