"""Faz 10 M4a: egitim egrisi tanisi (BETIMSEL; hakem-rl M4a #1 "ogrenme oldu mu?"). Ana venv.

BenchMARL kaydedicileri kapali (M3'ten beri `loggers = []`); tek egri kaynagi kosucu logundaki ilerleme cubugu:
her guncellemede (100 guncelleme = 600 bin kare) toplanan partinin bolum basina ortalama getirisi ("mean return",
egitim odulu, rastlantisal eylem). Bu deterministik degerlendirme SR'si DEGILDIR; okuma kuralini degistirmez.

  python scripts/marl/curves_m4a.py --log results/marl_m4a/run_stage1.log \
      --out docs/development/measurements/phase10-m4/m4a_stage1_curves.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess

TRAIN = re.compile(r"scripts/marl/train_m3\.py .*?--save-dir (\S+)")
STEP = re.compile(r"mean return = (-?[0-9.]+(?:e-?\d+)?):\s+\d+%\S*\s*\|\S*\s*(\d+)/(\d+)")


def parse(text: str) -> dict:
    """Log -> {kosu dizini: [guncelleme basina ortalama getiri]} (ayni guncellemenin tekrarlari son degerle)."""
    runs, cur = {}, None
    for chunk in re.split(r"(?=> \S*python\S* )", text):
        m = TRAIN.search(chunk)
        if m:
            cur = re.split(r'marl_m4[ab]/', m.group(1).replace('\\', '/'))[-1]
            runs[cur] = {}
        if cur is None:
            continue
        for v, i, n in STEP.findall(chunk):
            runs[cur][int(i)] = float(v)
    return {k: [d[i] for i in sorted(d)] for k, d in runs.items() if d}


def summary(vals: list) -> dict:
    n = len(vals)
    seg = lambda a, b: round(sum(vals[a:b]) / max(len(vals[a:b]), 1), 3)   # noqa: E731
    return {'n_updates': n, 'first10_mean': seg(0, 10), 'mid10_mean': seg(n // 2 - 5, n // 2 + 5),
            'last10_mean': seg(n - 10, n), 'max': round(max(vals), 3), 'curve': [round(v, 3) for v in vals]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True, nargs='+', help='bir ya da daha fazla log (M4b: her kosunun run.log u)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    text = ''
    for path in a.log:
        with open(path, encoding='utf-8', errors='replace') as fh:
            text += fh.read() + '
'
    curves = parse(text)
    git = lambda *c: subprocess.run(['git', *c], capture_output=True, text=True).stdout.strip()   # noqa: E731
    with open(os.path.abspath(__file__), 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    rec = {'note': 'BETIMSEL egitim egrisi (ilerleme cubugu mean return); okuma kuralini degistirmez',
           'log': [p.replace(os.sep, '/') for p in a.log],
           'log_sha256': hashlib.sha256(text.encode('utf-8', 'replace')).hexdigest(),
           'commit': git('rev-parse', 'HEAD'), 'git_status_porcelain': git('status', '--porcelain'),
           'script_sha256': sha, 'runs': {k: summary(v) for k, v in sorted(curves.items())}}
    for k, s in rec['runs'].items():
        print(f"{k:<14} n {s['n_updates']:>3}  ilk10 {s['first10_mean']:>7.2f}  orta {s['mid10_mean']:>7.2f}  "
              f"son10 {s['last10_mean']:>7.2f}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(rec, fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
