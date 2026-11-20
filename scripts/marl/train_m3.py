"""Faz 10 M3: MarlPortfolioEnv uzerinde BenchMARL MAPPO / IPPO egitimi (venv-marl).

Performans sayisi YAZMAZ: degerlendirme `scripts/marl/eval_m3.py` (deterministik, tam pencere), okuma M3 on
kaydindaki kuralla `scripts/marl/analyze_m3.py`. Kayit: sure, kare, parametre sayisi, politika ozeti, ortam
model_meta'si (K1), is parcacigi, git izi; `--save-dir` verilirse politika + norm_stats + meta (eval icin).

  # sure olcumu (M3 butcesi icin; GPU bos olmali)
  venv-marl\\Scripts\\python scripts/marl/train_m3.py --frames 30000 --train-device cuda --parallel --out <json>
  # M3 kosusu (on kayittaki kol/tohum)
  venv-marl\\Scripts\\python scripts/marl/train_m3.py --frames <butce> --pred-path data/marl/pred_pc060_k12_g1.npz \\
      --seed 1 --train-device cuda --parallel --save-dir results/marl_m3/<kol>/s1 --out <json>

  # M5 S-msg kolu (M5 tasarimi §4): guvenle agirlikli gat; z_scale egitim satirlarindan burada olculur
  venv-marl\\Scripts\\python scripts/marl/train_m3.py ... --comm gat --conf-mode zlogodds --beta-mode learned

Pencere: egitim WARMUP_ROWS .. 2018-12-31 (M2 §9.1 bolmesi; 2021+ okunmaz). Varsayilanlar M3-M7 kosulariyla ayni.

  # M8 kat kosusu (M8 tasarimi A1): wf-v1 kati, panel_m8, 10 kontrol noktasi (degerlendirme eval_fold_m8.py)
  venv-marl\\Scripts\\python scripts/marl/train_m3.py --panel data/marl/panel_m8.npz --train-end 2019-12-31 \\
      --checkpoints 10 --batched-env ... --save-dir <kosu> --out <json>
Tohum bit-esligi: OMP/MKL_NUM_THREADS=1 torch import'undan once (test_marl_m3 [3]).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PANEL = os.path.join('data', 'marl', 'panel_m2.npz')
TRAIN_END = '2018-12-31'


def policy_digest(exp) -> str:
    h = hashlib.sha256()
    for name, p in sorted(exp.policy.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def critic_summary(exp) -> dict:
    """Elestirmen kapasitesi (hakem-rl M3b #2): MAPPO, IPPO ve state() kollarinda boyut esit DEGIL, kayda gecer."""
    params = dict(exp.losses['agent'].critic_network_params.flatten_keys('.').items())
    first = next(v for k, v in params.items() if k.endswith('weight'))
    return {'params': int(sum(v.numel() for v in params.values())), 'input_dim': int(first.shape[-1])}


def provenance() -> dict:
    git = lambda *c: subprocess.run(['git', *c], capture_output=True, text=True).stdout.strip()   # noqa: E731
    with open(os.path.abspath(__file__), 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return {'commit': git('rev-parse', 'HEAD'), 'git_status_porcelain': git('status', '--porcelain'),
            'script_sha256': sha}


def comm_model(a, symbols: list, conf: dict | None = None):
    """M4 iletisim kolu -> CommPolicyConfig alanlari (None = MLP). gat kenarlari graf JSON'undaki sembol
    ciftlerinden panel sirasiyla indekslenir; graf dosyasinin SHA'si kayda gecer. `conf`: M5 S-msg alanlari
    (`conf_fields`), conf_mode = 'none' iken bos."""
    if a.comm == 'none':
        return None
    spec = dict(kind=a.comm, hidden=a.hidden, heads=a.heads, rounds=a.rounds, edges=None, **(conf or {}))
    if a.comm == 'gat':
        with open(a.graph, encoding='utf-8') as fh:
            g = json.load(fh)
        idx = {s: i for i, s in enumerate(symbols)}
        spec['edges'] = [[idx[u], idx[v]] for u, v in g['edges'][a.graph_rule]]
    return spec


def conf_fields(a, arrays: dict, end_row: int) -> dict:
    """M5 S-msg (M5 tasarimi §4): guven sutunlari gozlem duzeninden (obs_spec; 11 ozellik + feat_ok + w_i + w_nakit
    + N kimlik + [p - 0,5, z(logret), z(dis), pred_ok]) ve z_scale EGITIM satirlarindan (WARMUP_ROWS..end_row,
    ortamin pred_ok'u ile ayni hucreler). GLM'de z_scale bozulmus terimden (gonderici sicakligi)."""
    if a.conf_mode == 'none':
        if a.beta_mode != 'fixed' or a.beta_init or a.gate or a.shuffle_conf or a.conf_temp_seed is not None:
            raise SystemExit('--beta-mode/--beta-init/--gate/--shuffle-conf/--conf-temp-seed yalniz --conf-mode ile')
        return {}
    import numpy as np
    if a.comm not in ('attn', 'gat') or not a.pred_path:
        raise SystemExit('--conf-mode attn/gat iletisim kolu ve --pred-path ister')
    from marl import obs_spec as OS
    from marl.benchmarl_task import load_pred
    from marl.comm_models import conf_scale, sender_temperatures
    N = len(arrays['symbols'])
    # Tahmin blogu [p-0,5, z(logret), z(dis), pred_ok] yayin blogundan (S-reg, M6b) ONCE gelir (obs_spec.local_obs):
    # ek bloksuz boyuttan sayilan indeksler ek blok varken de dogrudur.
    dim = OS.ObsSpec(n_agents=N, use_pred=True).local_dim     # trainer grup ozelligi kullanmaz
    pred = load_pred(a.pred_path, arrays)
    rows = slice(OS.WARMUP_ROWS, end_row + 1)
    ok = np.isfinite(pred['p_up_cal']) & np.isfinite(pred['pred_logret']) & np.isfinite(pred['model_disagreement'])
    temp = None if a.conf_temp_seed is None else sender_temperatures(N, a.conf_temp_seed)
    z = conf_scale(pred['p_up_cal'][rows], ok[rows], temp) if a.conf_mode == 'zlogodds' else 1.0
    return dict(conf_mode=a.conf_mode, conf_index=dim - 4, conf_ok_index=dim - 1, beta_mode=a.beta_mode,
                beta_init=a.beta_init, gate=a.gate, z_scale=z, shuffle_conf=a.shuffle_conf,
                shuffle_seed=(a.seed if a.shuffle_seed is None else a.shuffle_seed), conf_temp=temp)


def conf_scalars(exp) -> dict | None:
    """Egitim sonu ogrenilen guven skalerleri (manipulasyon denetimi, M5 tasarimi §4): beta, gecit a/b, karistirma."""
    from marl.comm_models import CommPolicy
    mods = [m for m in exp.policy.modules() if isinstance(m, CommPolicy)]
    if not mods or mods[0].conf_mode == 'none':
        return None
    m = mods[0]
    out = {'beta': float(m.beta.detach().cpu())}
    if m.gate:
        out.update(gate_a=float(m.gate_a.detach().cpu()), gate_b=float(m.gate_b.detach().cpu()))
    if m.conf_perm is not None:
        out['conf_perm'] = [int(i) for i in m.conf_perm.cpu()]
    return out


def train_task_cfg(a, end_row: int, warmup: int) -> dict:
    return dict(panel_path=getattr(a, 'panel', PANEL), start_row=warmup, end_row=end_row, mode='train', alpha=a.alpha,
                cost_bps_roundtrip=a.cost_bps, reward_term='excess', reward_scale=100.0, episode_len=a.episode_len,
                use_state=a.use_state, pred_path=a.pred_path, extras_path=a.extras_path,
                **({'batched': True} if a.batched_env else {}))   # kapaliyken gorev konfigurasyonu eskisiyle ayni


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--algo', choices=('mappo', 'ippo'), default='mappo')
    ap.add_argument('--frames', type=int, default=30_000)
    ap.add_argument('--frames-per-batch', type=int, default=6000)
    ap.add_argument('--epochs', type=int, default=10, help='MAPPO onerisi <= 10; mini-parti bolunmez')
    ap.add_argument('--n-envs', type=int, default=10)
    ap.add_argument('--parallel', action='store_true')
    ap.add_argument('--batched-env', action='store_true',
                    help='Faz 10 §3.2.8: n-envs ortami tek surecte toplu adimla (marl/batched_env.py); --parallel yok sayilir')
    ap.add_argument('--episode-len', type=int, default=252)
    ap.add_argument('--alpha', type=float, default=0.25)
    ap.add_argument('--cost-bps', type=float, default=20.0, help='gidis-donus bps (D4)')
    ap.add_argument('--init-std', type=float, default=0.12, help='tanh oncesi baslangic std (hakem-rl R7)')
    ap.add_argument('--lr', type=float, default=5e-4, help='M3a on kaydi v0.2 (hakem-rl M3a #4); BenchMARL varsayilani 5e-5')
    ap.add_argument('--use-state', action='store_true', help='elestirmen girdisi ortam state() (varsayilan: tum gozlemler)')
    ap.add_argument('--pred-path', default=None)
    ap.add_argument('--extras-path', default=None, help='M6b S-reg yayin blogu (scripts/marl/build_sreg_m6.py)')
    ap.add_argument('--train-device', default='cpu')
    ap.add_argument('--sampling-device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=1, help='OMP/MKL/torch is parcacigi (bit-eslik icin 1)')
    ap.add_argument('--comm', choices=('none', 'attn', 'gat', 'self'), default='none',
                    help='M4 iletisim kolu; none = BenchMARL MLP (M3 ile ayni); self = mimari-es iletisimsiz (M4b)')
    ap.add_argument('--hidden', type=int, default=110, help='attn/gat gizli boyut (M4 esit butce, tasarim §1)')
    ap.add_argument('--heads', type=int, default=5)
    ap.add_argument('--rounds', type=int, default=1)
    ap.add_argument('--graph', default=os.path.join('docs', 'development', 'measurements', 'phase10-m4',
                                                    'graph_rules.json'), help='gat kenar kaynagi (graph_m4.py)')
    ap.add_argument('--graph-rule', default='sector+mst')
    ap.add_argument('--conf-mode', choices=('none', 'prob', 'zlogodds'), default='none',
                    help='M5 S-msg guven terimi (M5 tasarimi §4); none = M4 ile bit-es')
    ap.add_argument('--beta-mode', choices=('fixed', 'learned'), default='fixed')
    ap.add_argument('--beta-init', type=float, default=0.0)
    ap.add_argument('--gate', action='store_true', help='alici oz-guven gecidi')
    ap.add_argument('--shuffle-conf', action='store_true', help='karistirilmis guven (sabit derangement)')
    ap.add_argument('--shuffle-seed', type=int, default=None, help='derangement tohumu (varsayilan --seed)')
    ap.add_argument('--conf-temp-seed', type=int, default=None,
                    help='GLM: gonderici basina sicaklik T_j ~ log-tekduze comm_models.GLM_TEMP_RANGE [0,25; 4] ureteci (yalniz guven terimi)')
    ap.add_argument('--panel', default=PANEL, help='M8: data/marl/panel_m8.npz (M8 tasarimi A1)')
    ap.add_argument('--train-end', default=TRAIN_END, help='egitim son tarihi (dahil); M8: wf-v1 katinin dogrulama oncesi')
    ap.add_argument('--checkpoints', type=int, default=0,
                    help='M8: esit aralikli kontrol noktasi sayisi (--save-dir/ckpt); 0 = kapali (M3-M7 ile ayni)')
    ap.add_argument('--save-dir', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.checkpoints and not a.save_dir:
        raise SystemExit('--checkpoints --save-dir ister')
    os.environ.setdefault('OMP_NUM_THREADS', str(a.threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(a.threads))

    import numpy as np
    import torch

    from marl import obs_spec as OS
    from marl import panel as PN
    from marl.m3_runner import build_experiment, save_policy

    torch.set_num_threads(a.threads)
    arrays = PN.load_npz(a.panel)
    dates = arrays['dates']
    end_row = int(np.flatnonzero(dates <= a.train_end)[-1])
    tcfg = train_task_cfg(a, end_row, OS.WARMUP_ROWS)
    build = dict(algo=a.algo, seed=a.seed, frames=a.frames, frames_per_batch=a.frames_per_batch, epochs=a.epochs,
                 n_envs=a.n_envs, parallel=a.parallel, init_std=a.init_std, lr=a.lr,
                 model=comm_model(a, [str(s) for s in arrays['symbols']], conf_fields(a, arrays, end_row)))
    ckpt_dir = os.path.join(a.save_dir, 'ckpt') if a.checkpoints else None
    exp = build_experiment(tcfg, train_device=a.train_device, sampling_device=a.sampling_device, **build,
                           **({'checkpoints': a.checkpoints, 'checkpoint_dir': ckpt_dir} if a.checkpoints else {}))
    t0 = time.perf_counter()
    exp.run()
    wall = time.perf_counter() - t0
    rec = {
        'note': 'egitim kaydi; performans sayisi YOK (degerlendirme eval_m3, okuma M3 on kaydi)',
        **provenance(), 'args': vars(a), 'build': build, 'panel_sha256': arrays['sha256'],
        'pred_sha256': (hashlib.sha256(open(a.pred_path, 'rb').read()).hexdigest() if a.pred_path else None),
        'extras_sha256': (hashlib.sha256(open(a.extras_path, 'rb').read()).hexdigest() if a.extras_path else None),
        'train_window': [str(dates[OS.WARMUP_ROWS]), str(dates[end_row])],
        'wall_seconds': round(wall, 1), 'frames': exp.total_frames,
        'seconds_per_200k': round(wall / max(exp.total_frames, 1) * 200_000, 1),
        'policy_params': sum(p.numel() for p in exp.policy.parameters()), 'critic': critic_summary(exp),
        'graph_sha256': (hashlib.sha256(open(a.graph, 'rb').read()).hexdigest() if a.comm == 'gat' else None),
        'policy_sha256': policy_digest(exp), 'conf_scalars_final': conf_scalars(exp),
        'checkpoints': (json.load(open(os.path.join(ckpt_dir, 'index.json'), encoding='utf-8')) if ckpt_dir else None),
        'env': {'python': platform.python_version(), 'os': platform.platform(), 'torch': torch.__version__,
                'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'threads': torch.get_num_threads(), 'omp': os.environ.get('OMP_NUM_THREADS')},
    }
    if a.save_dir:
        rec['model_meta'] = save_policy(exp, tcfg, {'build': build, 'provenance': provenance(),
                                                    'policy_sha256': rec['policy_sha256']}, a.save_dir)['model_meta']
    print(json.dumps({k: rec[k] for k in ('wall_seconds', 'frames', 'seconds_per_200k', 'policy_params',
                                          'policy_sha256')}, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(rec, fh, indent=1, ensure_ascii=False, default=str)


if __name__ == '__main__':
    main()
