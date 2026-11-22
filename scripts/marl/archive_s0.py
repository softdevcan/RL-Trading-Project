"""Faz 11 S0: dondurulmus M8 politika arsivi + harici disk yedegi + geri yukleme denemesi.

Plan: docs/development/phase-11-integration-plan.md §3 S0. Ana venv.

    python scripts/marl/archive_s0.py --archive                      # data/marl/frozen/m8-2026 (git disi)
    python scripts/marl/archive_s0.py --backup <hedef_dizin>         # results/marl_m8 + data/marl -> hedef
    python scripts/marl/archive_s0.py --restore-test <hedef_dizin>   # yedekten gecici dizine, verify_archive
    ... --out docs/development/measurements/phase11-s0/s0_archive.json   (yerel yol YAZILMAZ)

Yedek artimlidir: hedefte ayni SHA'li dosya varsa yeniden kopyalanmaz. Her kopya hedefte yeniden hash'lenir.
Hedefte `backup_manifest.json` (goreli yol -> sha256, bayt) tutulur.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from marl import frozen as FZ  # noqa: E402

READING = os.path.join('docs', 'development', 'measurements', 'phase10-m8', 'm8_reading.json')
NAME, FOLD = 'm8-2026', '2026'
BACKUP_SOURCES = (os.path.join('results', 'marl_m8'), os.path.join('data', 'marl'))


def forward_policies() -> tuple[dict, str]:
    with open(READING, encoding='utf-8') as fh:
        d = json.load(fh)
    return d[f'forward_policies_{FOLD}'], FZ.file_sha256(READING)


def git_head() -> str:
    import subprocess
    return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def do_archive() -> dict:
    fwd, reading_sha = forward_policies()
    dest = os.path.join(FZ.FROZEN_ROOT, NAME)
    if os.path.exists(os.path.join(dest, 'manifest.json')):
        built = False
    else:
        FZ.build_archive(os.path.join('results', 'marl_m8'), FOLD, fwd, dest,
                         meta={'name': NAME, 'source_reading': READING.replace(os.sep, '/'),
                               'source_reading_sha256': reading_sha, 'prereg': 'phase-10-m8-design.md §6.7',
                               'built': datetime.now().isoformat(timespec='seconds'), 'commit': git_head()})
        built = True
    rep = FZ.verify_archive(dest, forward=fwd, digest=True)
    return {'built_now': built, 'manifest_sha256': FZ.file_sha256(os.path.join(dest, 'manifest.json')),
            'source_reading_sha256': reading_sha, **rep}


def _walk(src: str):
    for dp, _, fns in os.walk(src):
        for fn in fns:
            yield os.path.relpath(os.path.join(dp, fn), ROOT)


def do_backup(target: str) -> dict:
    mpath = os.path.join(target, 'backup_manifest.json')
    old = {}
    if os.path.exists(mpath):
        with open(mpath, encoding='utf-8') as fh:
            old = json.load(fh).get('files', {})
    files, copied, skipped, nbytes = {}, 0, 0, 0
    for src in BACKUP_SOURCES:
        for rel in _walk(src):
            key = rel.replace(os.sep, '/')
            sha = FZ.file_sha256(rel)
            size = os.path.getsize(rel)
            dst = os.path.join(target, rel)
            if old.get(key, {}).get('sha256') == sha and os.path.exists(dst) and os.path.getsize(dst) == size:
                skipped += 1
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(rel, dst)
                if FZ.file_sha256(dst) != sha:
                    raise SystemExit(f'kopya SHA farkli: {key}')
                copied += 1
            files[key] = {'sha256': sha, 'bytes': size}
            nbytes += size
    man = {'written': datetime.now().isoformat(timespec='seconds'), 'commit': git_head(),
           'sources': [s.replace(os.sep, '/') for s in BACKUP_SOURCES], 'files': files}
    with open(mpath, 'w', encoding='utf-8') as fh:
        json.dump(man, fh, indent=0, ensure_ascii=False)
    listing = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {'n_files': len(files), 'bytes': nbytes, 'copied': copied, 'skipped_same_sha': skipped,
            'files_listing_sha256': listing}


def do_verify_backup(target: str) -> dict:
    with open(os.path.join(target, 'backup_manifest.json'), encoding='utf-8') as fh:
        files = json.load(fh)['files']
    bad = [k for k, v in files.items() if not os.path.exists(os.path.join(target, k))
           or FZ.file_sha256(os.path.join(target, k)) != v['sha256']]
    if bad:
        raise SystemExit(f'yedek dogrulanamadi ({len(bad)} dosya): {bad[:5]}')
    return {'n_files_rehashed': len(files), 'all_match': True}


def do_restore_test(target: str) -> dict:
    fwd, _ = forward_policies()
    src = os.path.join(target, FZ.FROZEN_ROOT, NAME)
    with tempfile.TemporaryDirectory() as tmp:
        dst = os.path.join(tmp, NAME)
        shutil.copytree(src, dst)
        rep = FZ.verify_archive(dst, forward=fwd, digest=True)
    return {'restored_from_backup': True, **rep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', action='store_true')
    ap.add_argument('--backup', default=None, help='hedef dizin (harici disk)')
    ap.add_argument('--restore-test', default=None, help='yedek dizini')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    os.chdir(ROOT)
    out = {'plan': 'phase-11-integration-plan.md §3 S0', 'name': NAME, 'fold': FOLD}
    if a.archive:
        out['archive'] = do_archive()
        print('arsiv:', out['archive'])
    if a.backup:
        out['backup'] = {'target': 'harici disk (komut satirinda verildi; yol kaydedilmez)', **do_backup(a.backup)}
        out['backup']['verify'] = do_verify_backup(a.backup)
        print('yedek:', out['backup'])
    if a.restore_test:
        out['restore_test'] = do_restore_test(a.restore_test)
        print('geri yukleme:', out['restore_test'])
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
