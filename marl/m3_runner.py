"""Faz 10 M3: deney kurucu, politika kaydi/yuklemesi, deterministik degerlendirme ve ayni muhasebeyle baseline'lar.
YALNIZ `venv-marl/`. Egitim `scripts/marl/train_m3.py`, degerlendirme `scripts/marl/eval_m3.py` bunu kullanir.

Hiperparametre kaynagi (M3 on kaydi):
- BenchMARL 1.4 varsayilanlari; Faz 9 plani §M3'teki MAPPO onerileri: mini-parti bolmeme + <=10 epoch
  (`minibatch_size = frames_per_batch`, `n_minibatch_iters = epochs`), clip eps 0,2 (varsayilan).
  Deger normalizasyonu (oneri 1) BenchMARL 1.4 MAPPO'da YOK -> beyan.
- Baslangic eylem gurultusu (hakem-rl R7): TanhNormal, tanh oncesi std `init_std` (biased_softplus); eylem
  [-B, B] logit oldugundan std 1,0 her gun ~tam devir uretir.
- Degerlendirme: deterministik eylem (TanhNormal modu), tam pencere, kesirli mod, egitim norm_stats'i (B9).

Baseline'lar (`rule_series`): `marl/rule_baselines.py` (saf cekirdek; ew, bh, cash + M8 minvar, hrp), buradan
yeniden ihrac edilir.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

import numpy as np
import torch

from marl.benchmarl_task import MarlTask, make_env


def checkpoint_iters(frames: int, frames_per_batch: int, k: int) -> list:
    """M8 (tasarim A1): egitim butcesi boyunca esit aralikli k kontrol noktasinin iterasyon numaralari (1 tabanli;
    sonuncusu son iterasyon = egitimin son modeli; `rl_folds` izgarasiyla ayni ilke)."""
    n = -(-int(frames) // int(frames_per_batch))
    if not 1 <= k <= n:
        raise ValueError(f"kontrol noktasi sayisi 1..{n} olmali: {k}")
    return [int(round(i * n / k)) for i in range(1, k + 1)]


from marl.frozen import state_digest  # noqa: E402,F401  (tek tanim; Faz 11 S0 arsivi ayni ozeti ana venv'de denetler)


def _checkpoint_saver(out_dir: str, iters: list):
    """Yalniz politika agirliklarini secili iterasyonlarin sonunda yazar; RNG'ye ve egitime dokunmaz."""
    from benchmarl.experiment.callback import Callback

    class CheckpointSaver(Callback):
        def __init__(self):
            super().__init__()
            self.k, self.index = 0, []

        def on_setup(self):
            if len(self.experiment.group_map) != 1:
                raise ValueError("kontrol noktasi kaydi tek ajan grubu ister")

        def on_train_end(self, training_td, group):
            self.k += 1
            if self.k not in iters:
                return
            sd = {n: (t.detach().cpu().clone() if torch.is_tensor(t) else t)
                  for n, t in self.experiment.policy.state_dict().items()}
            name = f'ckpt_{self.k:04d}.pt'
            torch.save(sd, os.path.join(out_dir, name))
            self.index.append({'iter': self.k, 'frames': int(self.experiment.total_frames), 'file': name,
                               'params_sha': state_digest(sd)})
            with open(os.path.join(out_dir, 'index.json'), 'w', encoding='utf-8') as fh:
                json.dump(self.index, fh, indent=1)

    return CheckpointSaver()


def build_experiment(task_cfg: dict, algo: str = 'mappo', seed: int = 0, frames: int = 30_000,
                     frames_per_batch: int = 6000, epochs: int = 10, n_envs: int = 10, parallel: bool = False,
                     train_device: str = 'cpu', sampling_device: str = 'cpu', init_std: float = 0.12,
                     lr: Optional[float] = None, save_folder: Optional[str] = None, model: Optional[dict] = None,
                     checkpoints: int = 0, checkpoint_dir: Optional[str] = None):
    """`model`: None -> BenchMARL MLP politikasi (M3, M4 `none` kolu); dict -> M4 iletisim kolu
    (`marl.comm_models.CommPolicyConfig` alanlari: kind, hidden, heads, rounds, edges). Elestirmen her zaman MLP.
    `checkpoints` > 0 (M8, tasarim A1): esit aralikli politika agirliklari `checkpoint_dir`'e (index.json ile);
    0 iken Experiment cagrisi oncekiyle ayni."""
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
    if model is None:
        policy_cfg = MlpConfig.get_from_yaml()
    elif model.get('kind') == 'rowshared':               # M8 A2: SARL-esmimari (marl/sarl.py)
        from marl.sarl import RowSharedConfig
        policy_cfg = RowSharedConfig(n_rows=int(model['n_rows']), num_cells=list(model['num_cells']))
    else:
        from marl.comm_models import CommPolicyConfig
        policy_cfg = CommPolicyConfig(**model)
    kw = {}
    if checkpoints:
        if not checkpoint_dir:
            raise ValueError("checkpoints > 0 icin checkpoint_dir gerekli")
        os.makedirs(checkpoint_dir, exist_ok=True)
        kw['callbacks'] = [_checkpoint_saver(checkpoint_dir, checkpoint_iters(frames, frames_per_batch, checkpoints))]
    return Experiment(task=task, algorithm_config=algo_cfg, model_config=policy_cfg,
                      critic_model_config=MlpConfig.get_from_yaml(), seed=seed, config=cfg, **kw)


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


def load_policy(run_dir: str, weights: Optional[str] = None):
    """Kayitli politikayi ayni yapiyla yeniden kur (egitim yapilmaz). `weights`: varsayilan `policy.pt`;
    M8 kontrol noktasi icin `ckpt/ckpt_XXXX.pt`."""
    with open(os.path.join(run_dir, 'meta.json'), encoding='utf-8') as fh:
        meta = json.load(fh)
    b = meta['build']
    exp = build_experiment(meta['task_cfg'], algo=b['algo'], seed=b['seed'], frames=b['frames_per_batch'],
                           frames_per_batch=b['frames_per_batch'], epochs=b['epochs'], n_envs=1,
                           init_std=b['init_std'], lr=b.get('lr'), model=b.get('model'))
    exp.policy.load_state_dict(torch.load(os.path.join(run_dir, weights or 'policy.pt'), map_location='cpu'))
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


# Kural portfoyleri M8 A4'te saf cekirdege tasindi (ana venv de okur); davranis ayni (test_marl_m3 baseline testi).
from marl.rule_baselines import rule_series  # noqa: E402,F401
