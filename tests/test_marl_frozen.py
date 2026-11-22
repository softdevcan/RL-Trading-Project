"""
Faz 11 S0: dondurulmus M8 politika arsivi `marl/frozen.py` (plan phase-11-integration-plan.md §3 S0).
ANA venv (torch). Sentetik kosu agaci; gercek arsiv varsa (data/marl/frozen/m8-2026) onu da dogrular.

Kontroller:
  1. Kur + dogrula: secilen kontrol noktasi ve kayitlar kopyalanir, manifest yazilir; dosya SHA, politika ozeti
     (params_sha, egitimdeki tanim) ve on kayit listesi tutar. Dolu hedefe yeniden kurulmaz.
  2. Bozulma yakalanir: (a) dosyada tek bayt, (b) kontrol noktasi farkli agirlikla degistirilip manifestteki dosya
     SHA'si da guncellense bile params_sha, (c) on kayit listesinde olmayan / eksik politika.
  3. Gercek arsiv (varsa): 72 politika, on kayit listesi (m8_reading.json forward_policies_2026) ile ayni.

Ag yok. ~5 sn (gercek arsivle ~20 sn).
    python tests/test_marl_frozen.py
"""

import json
import os
import shutil
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from marl import frozen as FZ  # noqa: E402

passed = failed = 0


def check(name, ok):
    global passed, failed
    print(f"[{'OK' if ok else 'HATA'}] {name}")
    passed, failed = passed + bool(ok), failed + (not ok)


def raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


def make_run(root, fold, arm, seed, val):
    d = os.path.join(root, fold, arm, f's{seed}')
    os.makedirs(os.path.join(d, 'ckpt'))
    os.makedirs(os.path.join(d, 'eval'))
    sd = {'w': torch.full((3, 2), float(val)), 'b': torch.zeros(2), 'shape': torch.Size([3, 2])}
    torch.save(sd, os.path.join(d, 'ckpt', 'ckpt_0020.pt'))
    for f in FZ.POLICY_FILES:
        with open(os.path.join(d, f), 'w', encoding='utf-8') as fh:
            json.dump({'f': f, 'seed': seed}, fh)
    with open(os.path.join(d, 'eval', 'fold_eval.json'), 'w', encoding='utf-8') as fh:
        json.dump({'seed': seed}, fh)
    return {'ckpt': 'ckpt_0020.pt', 'params_sha': FZ.state_digest(sd)}


with tempfile.TemporaryDirectory() as tmp:
    src = os.path.join(tmp, 'runs')
    fwd = {arm: {f's{s}': make_run(src, '2026', arm, s, s * (i + 1)) for s in (201, 202)}
           for i, arm in enumerate(('M-none', 'S-esm'))}
    dest = os.path.join(tmp, 'frozen', 'x')
    man = FZ.build_archive(src, '2026', fwd, dest, meta={'name': 'x'})
    rep = FZ.verify_archive(dest, forward=fwd)
    ok1 = man['n_policies'] == 4 and rep['n_files'] == 4 * 5 and rep['digest_checked'] and rep['forward_checked']
    ok1 &= raises(lambda: FZ.build_archive(src, '2026', fwd, dest))
    check("1. kur + dogrula: 4 politika x 5 dosya, params_sha ve on kayit listesi tutar; dolu hedefe kurulmaz", ok1)

    # 2a tek bayt
    t = os.path.join(tmp, 'a')
    shutil.copytree(dest, t)
    p = os.path.join(t, 'S-esm', 's202', 'meta.json')
    with open(p, 'ab') as fh:
        fh.write(b' ')
    ok2 = raises(lambda: FZ.verify_archive(t, forward=fwd))
    # 2b agirlik degisti, dosya SHA manifestte guncellendi -> params_sha yakalar
    t = os.path.join(tmp, 'b')
    shutil.copytree(dest, t)
    ck = os.path.join(t, 'M-none', 's201', 'ckpt_0020.pt')
    torch.save({'w': torch.full((3, 2), 9.0), 'b': torch.zeros(2), 'shape': torch.Size([3, 2])}, ck)
    mp = os.path.join(t, 'manifest.json')
    m = json.load(open(mp, encoding='utf-8'))
    for q in m['policies']:
        if (q['arm'], q['seed']) == ('M-none', 's201'):
            q['files']['ckpt_0020.pt'] = FZ.file_sha256(ck)
    json.dump(m, open(mp, 'w', encoding='utf-8'))
    ok2 &= raises(lambda: FZ.verify_archive(t, forward=fwd))
    ok2 &= FZ.verify_archive(t, forward=None, digest=False)['n_policies'] == 4      # yalniz dosya SHA'si kandirilir
    # 2c on kayit listesi farkli
    extra = {**fwd, 'M-gat': {'s201': fwd['M-none']['s201']}}
    less = {'M-none': fwd['M-none']}
    ok2 &= raises(lambda: FZ.verify_archive(dest, forward=extra)) and raises(lambda: FZ.verify_archive(dest, forward=less))
    check("2. bozulma yakalanir: tek bayt; agirlik degisip dosya SHA'si guncellense de params_sha; on kayit listesi farki",
          ok2)

real = os.path.join(FZ.FROZEN_ROOT, 'm8-2026')
if os.path.exists(os.path.join(real, 'manifest.json')):
    with open(os.path.join('docs', 'development', 'measurements', 'phase10-m8', 'm8_reading.json'),
              encoding='utf-8') as fh:
        fwd_real = json.load(fh)['forward_policies_2026']
    r = FZ.verify_archive(real, forward=fwd_real)
    check(f"3. gercek arsiv m8-2026: {r['n_policies']} politika, on kayit listesiyle ayni, params_sha tutar",
          r['n_policies'] == 72)
else:
    print("[ATLANDI] 3. gercek arsiv yok (scripts/marl/archive_s0.py --archive)")

# 4. Faz 11 S2: sade ileri gecis (marl/frozen_policy.py) venv-marl referansiyla bit-es, her kol turunden bir politika
REF = os.path.join('results', 'marl_s2', 'ref')
picks = [('M-none', 's201'), ('null', 's213'), ('S-esm', 's201'), ('S-agr', 's201'), ('M-gat', 's201'),
         ('M-self', 's201')]
if os.path.exists(os.path.join(real, 'manifest.json')) and all(
        os.path.exists(os.path.join(REF, a, f'{s}.npz')) for a, s in picks):
    import numpy as np
    from marl.frozen_policy import FrozenPolicy
    ok4, kinds = True, set()
    for a, s in picks:
        fp = FrozenPolicy(os.path.join(real, a, s))
        r = np.load(os.path.join(REF, a, f'{s}.npz'))
        obs = r['win_obs'].reshape(-1, fp.n_agents, fp.obs_dim)
        ok4 &= np.array_equal(np.stack([fp.act(o) for o in obs]), r['win_action'].reshape(obs.shape[0], -1))
        ok4 &= np.array_equal(fp.act(r['rand_obs'].reshape(-1, fp.n_agents, fp.obs_dim)),
                              r['rand_action'].reshape(-1, fp.n_agents))
        kinds.add(fp.kind)
    # mutasyon: gat maskesi kaldirilinca (tam bagli dikkat) ayni gozlemde eylem degisir
    fp = FrozenPolicy(os.path.join(real, 'M-gat', 's201'))
    r = np.load(os.path.join(REF, 'M-gat', 's201.npz'))
    fp.mask = torch.zeros_like(fp.mask)
    ok4 &= not np.array_equal(fp.act(r['rand_obs']), r['rand_action'])
    check(f"4. sade ileri gecis venv-marl referansiyla bit-es ({len(picks)} politika, {len(kinds)} tur); "
          f"gat maskesi bozulunca ayrisir", ok4 and len(kinds) == 4)
else:
    print("[ATLANDI] 4. S2 referansi yok (venv-marl: scripts/marl/s2_parity_ref.py)")

print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
