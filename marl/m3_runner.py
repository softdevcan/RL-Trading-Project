"""Faz 10 M3: deney kurucu, politika kaydi/yuklemesi, deterministik degerlendirme ve ayni muhasebeyle baseline'lar.
YALNIZ `venv-marl/`. Egitim `scripts/marl/train_m3.py`, degerlendirme `scripts/marl/eval_m3.py` bunu kullanir.

Hiperparametre kaynagi (M3 on kaydi):
- BenchMARL 1.4 varsayilanlari; Faz 9 plani §M3'teki MAPPO onerileri: mini-parti bolmeme + <=10 epoch
  (`minibatch_size = frames_per_batch`, `n_minibatch_iters = epochs`), clip eps 0,2 (varsayilan).
  Deger normalizasyonu (oneri 1) BenchMARL 1.4 MAPPO'da YOK -> beyan.
- Baslangic eylem gurultusu (hakem-rl R7): TanhNormal, tanh oncesi std `init_std` (biased_softplus); eylem
  [-B, B] logit oldugundan std 1,0 her gun ~tam devir uretir.
- Degerlendirme: deterministik eylem (TanhNormal modu), tam pencere, kesirli mod, egitim norm_stats'i (B9).

Baseline'lar (M2 §9 #11, hakem-finans F7/F10): ortamin AYNI muhasebesi (`accounting.step_fractional`), ayni
`fill_prices` (kilit 20 + v0.7 marj takvimi), `project` (ayni Constraints), A8 rf, ayni maliyet:
  ew   : her gun tam yatirimli 1/N (t'de gecerli semboller)
  bh   : ilk gun 1/N, sonra hedef = kaymis agirlik (islem yok; yalniz zorunlu dondurma)
  cash : tam nakit (X = 0)
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

import numpy as np
import torch

from marl import accounting as A
from marl import weights as W
from marl.benchmarl_task import MarlTask, make_env


def build_experiment(task_cfg: dict, algo: str = 'mappo', seed: int = 0, frames: int = 30_000,
                     frames_per_batch: int = 6000, epochs: int = 10, n_envs: int = 10, parallel: bool = False,
                     train_device: str = 'cpu', sampling_device: str = 'cpu', init_std: float = 0.12,
                     lr: Optional[float] = None, save_folder: Optional[str] = None):
    from benchmarl.algorithms import IppoConfig, MappoConfig
    from benchmarl.experiment import Experiment, ExperimentConfig
    from benchmarl.models.mlp import MlpConfig

    cfg = ExperimentConfig.get_from_yaml()
    cfg.max_n_frames = frames
    if lr is not None:
        cfg.lr = float(lr)                               # M3a on kaydi v0.2: 5e-4 (hakem-rl M3a #4)
    cfg.on_policy_collected_frames_per_batch = frames_per_batch
    cfg.on_policy_n_envs_per_worker = n_envs
    cfg.on_policy_minibatch_size = frames_per_batch      # MAPPO onerisi: mini-parti bolmeme
    cfg.on_policy_n_minibatch_iters = epochs             # MAPPO onerisi: <= 10 epoch
    cfg.parallel_collection = parallel
    cfg.sampling_device = sampling_device
    cfg.train_device = cfg.buffer_device = train_device
    cfg.share_policy_params = True                       # K24
    cfg.evaluation = False
    cfg.render = False
    cfg.loggers = []
    cfg.create_json = False
    cfg.checkpoint_at_end = False
    cfg.save_folder = save_folder or tempfile.mkdtemp(prefix='marl_m3_')
    algo_cfg = (MappoConfig if algo == 'mappo' else IppoConfig).get_from_yaml()
    algo_cfg.scale_mapping = f'biased_softplus_{init_std}'
    task = MarlTask.PORTFOLIO.update_config(dict(task_cfg))
    return Experiment(task=task, algorithm_config=algo_cfg, model_config=MlpConfig.get_from_yaml(),
                      critic_model_config=MlpConfig.get_from_yaml(), seed=seed, config=cfg)


def save_policy(exp, task_cfg: dict, run_meta: dict, out_dir: str) -> dict:
    """Politika agirliklari + egitim ortaminin norm_stats'i + yeniden kurulum bilgisi (K1 alanlari)."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(exp.policy.state_dict(), os.path.join(out_dir, 'policy.pt'))
    env = make_env({k: v for k, v in task_cfg.items() if k != 'use_state'})
    with open(os.path.join(out_dir, 'norm_stats.json'), 'w', encoding='utf-8') as fh:
        json.dump(env.norm_stats, fh)
    meta = {'task_cfg': task_cfg, **run_meta, 'model_meta': {k: v for k, v in env.model_meta().items()
                                                             if k != 'norm_stats'}}
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False, default=str)
    return meta


def load_policy(run_dir: str):
    """Kayitli politikayi ayni yapiyla yeniden kur (egitim yapilmaz)."""
    with open(os.path.join(run_dir, 'meta.json'), encoding='utf-8') as fh:
        meta = json.load(fh)
    b = meta['build']
    exp = build_experiment(meta['task_cfg'], algo=b['algo'], seed=b['seed'], frames=b['frames_per_batch'],
                           frames_per_batch=b['frames_per_batch'], epochs=b['epochs'], n_envs=1,
                           init_std=b['init_std'], lr=b.get('lr'))
    exp.policy.load_state_dict(torch.load(os.path.join(run_dir, 'policy.pt'), map_location='cpu'))
    return exp, meta


def deterministic_rollout(policy, env_cfg: dict) -> dict:
    """Deterministik politika ile tam pencere (eval modu). Donus: gunluk R, X, devir, nakit agirligi, tarih."""
    from torchrl.envs import PettingZooWrapper
    from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

    base = make_env(env_cfg)
    if base.mode != 'eval':
        raise ValueError("degerlendirme ortami mode='eval' olmali (tam pencere)")
    env = PettingZooWrapper(env=base, categorical_actions=True, return_state=False, use_mask=False)
    out = {k: [] for k in ('date', 'R', 'X', 'turnover', 'cash_weight', 'n_sell_blocked')}
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        td = env.reset()
        for _ in range(base.end_row - base.start_row):
            td = policy(td)
            td = env.step(td)
            li = base.last_info
            for k in out:
                out[k].append(li[k])
            if bool(td['next', 'done'].all()):
                break
            td = step_mdp(td)
    return {k: (np.asarray(v) if k != 'date' else list(v)) for k, v in out.items()}


def rule_series(arrays: dict, start_row: int, end_row: int, kind: str, cost_bps_roundtrip: float = 20.0,
                limit_down_gap='schedule', constraints: W.Constraints = W.Constraints()) -> dict:
    """Ortamin muhasebesiyle kural portfoyu (ew | bh | cash); karar satirlari [start_row, end_row)."""
    if kind not in ('ew', 'bh', 'cash'):
        raise ValueError(f"kind ew | bh | cash: {kind}")
    ldg = A.limit_down_gaps(arrays['dates']) if limit_down_gap == 'schedule' else float(limit_down_gap)
    px = A.fill_prices(arrays['open'], arrays['close'], A.LISTING_LOCKOUT, ldg)
    cost = A.one_way_cost(cost_bps_roundtrip)
    N = arrays['open'].shape[1]
    w = np.r_[1.0, np.zeros(N)]
    out = {k: [] for k in ('date', 'R', 'X', 'turnover', 'cash_weight')}
    for t in range(start_row, end_row):
        g_on, g_day, tradable = A.step_growth(px, t)
        rf = float(arrays['rf'][t + 1])
        v = px['valid'][t]
        if kind == 'cash':
            u = np.r_[1.0, np.zeros(N)]
        elif kind == 'ew' or (kind == 'bh' and t == start_row):
            u = np.r_[0.0, v.astype(float) / max(float(v.sum()), 1.0)]
        else:   # bh: hedef = kaymis agirlik (gece kaymasi project icinde w~'ye esit -> islem yok)
            drift = w * np.concatenate([[1.0 + rf], g_on])
            u = drift / drift.sum()
        res = A.step_fractional(w, u, g_on, g_day, tradable, rf, cost, constraints, None, px['no_sell'][t + 1])
        w = res.w_next
        for k, val in (('date', str(arrays['dates'][t + 1])), ('R', res.R), ('X', res.X),
                       ('turnover', res.turnover), ('cash_weight', float(w[0]))):
            out[k].append(val)
    return {k: (np.asarray(v) if k != 'date' else list(v)) for k, v in out.items()}
