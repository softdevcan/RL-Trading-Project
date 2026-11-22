"""Faz 11 S0: dondurulmus M8 politika arsivi (ileri veri testi, M8 tasarimi §6.7).

Ana venv'de calisir (torch ana venv'de var; BenchMARL/TorchRL GEREKMEZ). Kontrol noktasi dosyasi duz bir
state_dict'tir; `state_digest` egitimde yazilan `params_sha` ile ayni tanimdir (m3_runner buradan ithal eder).

Arsiv duzeni (git disi, ortak, salt okunur; kullanici calisma alanina kopyalanmaz, Faz 11 K-F11-3 onerisi):
    data/marl/frozen/<ad>/<kol>/s<tohum>/{ckpt_XXXX.pt, meta.json, norm_stats.json, train.json, fold_eval.json}
    data/marl/frozen/<ad>/manifest.json

Gunluk is (S4) her calismadan once `verify_archive` cagirir; uyusmazlikta ValueError ve o gun yazilmaz.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil

FROZEN_ROOT = os.path.join('data', 'marl', 'frozen')
POLICY_FILES = ('meta.json', 'norm_stats.json', 'train.json')     # kosu dizininden
EVAL_FILE = os.path.join('eval', 'fold_eval.json')                # kosu dizininden, arsivde duz ad


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def state_digest(sd: dict) -> str:
    """state_dict ozeti; tensor olmayan girdiler (TensorDict ust verisi, or. torch.Size) repr ile."""
    import torch
    h = hashlib.sha256()
    for name in sorted(sd):
        v = sd[name]
        h.update(name.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes() if torch.is_tensor(v) else repr(v).encode())
    return h.hexdigest()


def ckpt_digest(path: str) -> str:
    import torch
    return state_digest(torch.load(path, map_location='cpu'))


def build_archive(src_root: str, fold: str, forward: dict, dest: str, meta: dict | None = None) -> dict:
    """forward = {kol: {'s<tohum>': {'ckpt', 'params_sha'}}} (m8_reading.json -> forward_policies_<kat>).
    Kaynaktaki secilen kontrol noktasini ve kayitlarini kopyalar, manifest yazar. Hedef bos olmali."""
    if os.path.exists(dest) and os.listdir(dest):
        raise ValueError(f'hedef bos degil: {dest} (arsiv ustune yazilmaz)')
    policies = []
    for arm in sorted(forward):
        for sk in sorted(forward[arm]):
            ent = forward[arm][sk]
            run = os.path.join(src_root, fold, arm, sk)
            out = os.path.join(dest, arm, sk)
            os.makedirs(out, exist_ok=True)
            pairs = [(os.path.join(run, 'ckpt', ent['ckpt']), ent['ckpt'])]
            pairs += [(os.path.join(run, f), f) for f in POLICY_FILES]
            pairs += [(os.path.join(run, EVAL_FILE), os.path.basename(EVAL_FILE))]
            files = {}
            for src, name in pairs:
                shutil.copy2(src, os.path.join(out, name))
                files[name] = file_sha256(os.path.join(out, name))
            policies.append({'arm': arm, 'seed': sk, 'ckpt': ent['ckpt'], 'params_sha': ent['params_sha'],
                             'files': files})
    manifest = {'fold': fold, 'n_policies': len(policies), 'policies': policies, **(meta or {})}
    with open(os.path.join(dest, 'manifest.json'), 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)
    return manifest


def verify_archive(dest: str, forward: dict | None = None, digest: bool = True) -> dict:
    """Dosya SHA'lari manifestle, (digest=True) politika ozeti params_sha ile, (forward verilirse) manifest
    on kayit listesiyle ayni mi. Herhangi bir uyusmazlik -> ValueError. Donus: ozet."""
    with open(os.path.join(dest, 'manifest.json'), encoding='utf-8') as fh:
        man = json.load(fh)
    errors = []
    listed = {(p['arm'], p['seed']) for p in man['policies']}
    if forward is not None:
        want = {(a, s) for a in forward for s in forward[a]}
        if want != listed:
            errors.append(f'politika kumesi farkli: eksik {sorted(want - listed)}, fazla {sorted(listed - want)}')
    for p in man['policies']:
        d = os.path.join(dest, p['arm'], p['seed'])
        if forward is not None and (p['arm'], p['seed']) in {(a, s) for a in forward for s in forward[a]}:
            f = forward[p['arm']][p['seed']]
            if (f['ckpt'], f['params_sha']) != (p['ckpt'], p['params_sha']):
                errors.append(f"{p['arm']}/{p['seed']}: manifest on kayit listesiyle ayni degil")
        for name, sha in p['files'].items():
            path = os.path.join(d, name)
            if not os.path.exists(path):
                errors.append(f"{p['arm']}/{p['seed']}/{name}: yok")
            elif file_sha256(path) != sha:
                errors.append(f"{p['arm']}/{p['seed']}/{name}: dosya SHA farkli")
        if digest and os.path.exists(os.path.join(d, p['ckpt'])):
            if ckpt_digest(os.path.join(d, p['ckpt'])) != p['params_sha']:
                errors.append(f"{p['arm']}/{p['seed']}: params_sha farkli")
    if errors:
        raise ValueError('dondurulmus arsiv dogrulanamadi:\n  ' + '\n  '.join(errors))
    return {'n_policies': len(man['policies']), 'n_files': sum(len(p['files']) for p in man['policies']),
            'digest_checked': digest, 'forward_checked': forward is not None}
