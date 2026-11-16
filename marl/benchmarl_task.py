"""Faz 10 M3: `MarlPortfolioEnv` icin BenchMARL 1.4 gorevi (tasarim M2 §9 #4; spike kalibi
`scripts/spike_m1/benchmarl_glue.py`). YALNIZ `venv-marl/` (benchmarl, torchrl). Uretim kodu import etmez.

    from marl.benchmarl_task import MarlTask
    task = MarlTask.PORTFOLIO.update_config(dict(panel_path='data/marl/panel_m2.npz', start_row=59, end_row=1393, ...))

Gruplama: PettingZooWrapper tum ajanlari tek grup 'agent' altinda toplar; ajan ekseni tensor boyutudur
(K24 tam parametre paylasimi + ajan kimligi one-hot gozlemde). Elestirmen girdisi: `use_state=False` (varsayilan)
-> BenchMARL merkezi MLP elestirmeni tum ajanlarin yerel gozlemlerini birlestirir (her ajanin z, w, kimlik);
`use_state=True` -> ortamin `state()` vektoru (w, kesitsel ortalama z, gecerli oran; M2 §5.3). Secim M3 on kaydinda.

Bolum sonu: ortam her bolum sonunda terminated=False, truncated=True verir (sabit uzunluklu pencere, gercek bir
son durum yok). BenchMARL MAPPO/IPPO kaybi grup duzeyinde `terminated` anahtarini ayri verir (mappo.py set_keys),
TorchRL GAE yalniz terminated'da bootstrap'i keser -> V(s_T) bootstrap edilir (test_marl_m3 [2]).
"""
from __future__ import annotations

import copy
import json
from typing import Callable, Dict, List, Optional

import numpy as np
from torchrl.data import Composite
from torchrl.envs import EnvBase, PettingZooWrapper

from benchmarl.environments.common import Task

from marl import panel as PN
from marl.env import MarlPortfolioEnv

TASK_KEYS = ('panel_path', 'start_row', 'end_row', 'mode', 'alpha', 'cost_bps_roundtrip', 'reward_term',
             'reward_scale', 'episode_len', 'pred_path', 'norm_stats_path', 'use_state', 'limit_down_gap')


def load_pred(path: Optional[str], panel: dict) -> Optional[dict]:
    """Tahmin blogu .npz -> {p_up_cal, pred_logret, model_disagreement} [T, N]; panel tarih/sembol hizasi zorunlu."""
    if not path:
        return None
    d = np.load(path, allow_pickle=False)
    if not (np.array_equal(d['dates'], panel['dates']) and np.array_equal(d['symbols'], panel['symbols'])):
        raise ValueError(f"tahmin blogu panel ile hizali degil: {path}")
    return {k: d[k] for k in ('p_up_cal', 'pred_logret', 'model_disagreement')}


def make_env(cfg: dict) -> MarlPortfolioEnv:
    """Gorev konfigurasyonundan ortam. norm_stats: 'fit' (egitim) ya da egitim ortaminin JSON'u (held-out)."""
    unknown = set(cfg) - set(TASK_KEYS)
    if unknown:
        raise ValueError(f"bilinmeyen gorev anahtari: {sorted(unknown)}")
    arrays = PN.load_npz(cfg['panel_path'])
    ns = 'fit'
    if cfg.get('norm_stats_path'):
        with open(cfg['norm_stats_path'], encoding='utf-8') as fh:
            ns = json.load(fh)
    return MarlPortfolioEnv(
        arrays, int(cfg['start_row']), int(cfg['end_row']), norm_stats=ns, mode=cfg.get('mode', 'train'),
        alpha=float(cfg.get('alpha', 0.25)), cost_bps_roundtrip=float(cfg.get('cost_bps_roundtrip', 20.0)),
        reward_term=cfg.get('reward_term', 'excess'), reward_scale=float(cfg.get('reward_scale', 100.0)),
        episode_len=int(cfg.get('episode_len', 252)), pred=load_pred(cfg.get('pred_path'), arrays),
        limit_down_gap=cfg.get('limit_down_gap', 'schedule'))


class MarlTask(Task):
    """Faz 10 MARL portfoy gorevi (tek uye; konfigurasyon `update_config` ile)."""

    PORTFOLIO = None

    def get_env_fun(self, num_envs: int, continuous_actions: bool, seed: Optional[int], device) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        use_state = bool(config.get('use_state', False))
        return lambda: PettingZooWrapper(env=make_env(config), seed=seed, device=device, categorical_actions=True,
                                         return_state=use_state, use_mask=False)

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return False

    def has_render(self, env: EnvBase) -> bool:
        return False

    def max_steps(self, env: EnvBase) -> int:
        cfg = self.config
        span = int(cfg['end_row']) - int(cfg['start_row'])
        return span if cfg.get('mode', 'train') == 'eval' else min(int(cfg.get('episode_len', 252)), span)

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        if not self.config.get('use_state', False):
            return None
        return Composite({'state': env.observation_spec['state'].clone()})

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def observation_spec(self, env: EnvBase) -> Composite:
        spec = env.observation_spec.clone()
        for group in self.group_map(env):
            for key in list(spec[group].keys()):
                if key != 'observation':
                    del spec[group][key]
        if 'state' in spec.keys():
            del spec['state']
        return spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec

    @staticmethod
    def env_name() -> str:
        return 'marl_portfolio'
