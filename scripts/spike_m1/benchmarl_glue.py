"""Faz 10 M1 spike'i: BenchMARL 1.4 baglantisi — spike gorevi ve `attn` iletisim modeli (venv-marl).

Gorev: SpikePortfolioEnv (PettingZoo Parallel) -> torchrl PettingZooWrapper. Tek grup 'agent' (30 ajan).
Model (`attn`, TarMAC tipi; Das vd. 2019; Faz 10 plani §3.2.5): paylasimli kodlayici -> ajanlar arasi cok
basli oz-dikkat (ajan ekseni tensor boyutu; tum ajanlar tek ileri geciste) -> paylasimli cikti basi.
`none` kolu BenchMARL'in kendi MlpConfig'i (ayni paylasim). Spike'a ozgudur; M4 eşit-butce (±%10 parametre)
kurali burada uygulanmaz, yalniz parametre sayilari raporlanir.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, MISSING
from typing import Callable, Dict, List, Optional

import torch
from tensordict import TensorDictBase
from torch import nn
from torchrl.data import Composite
from torchrl.envs import EnvBase, PettingZooWrapper

from benchmarl.environments.common import Task
from benchmarl.models.common import Model, ModelConfig

from portfolio_env import SpikePortfolioEnv


class SpikeTask(Task):
    """Spike gorevi. Config: panel_path, episode_len, alpha, cost, reward_scale, n_agents."""

    PORTFOLIO = None

    def get_env_fun(self, num_envs: int, continuous_actions: bool, seed: Optional[int], device) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        return lambda: PettingZooWrapper(env=SpikePortfolioEnv(**config), seed=seed, device=device,
                                         categorical_actions=True, return_state=False, use_mask=False)

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return False

    def has_render(self, env: EnvBase) -> bool:
        return False

    def max_steps(self, env: EnvBase) -> int:
        return int(self.config['episode_len'])

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def observation_spec(self, env: EnvBase) -> Composite:
        spec = env.observation_spec.clone()
        for group in self.group_map(env):
            for key in list(spec[group].keys()):
                if key != 'observation':
                    del spec[group][key]
        return spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec

    @staticmethod
    def env_name() -> str:
        return 'spike'


class AttnComm(Model):
    """Paylasimli kodlayici + ajanlar arasi oz-dikkat + paylasimli cikti. Yalniz input_has_agent_dim=True ve
    share_params=True (K24 tam paylasim) desteklenir; merkezi elestirmen icin BenchMARL Mlp kullanilir."""

    def __init__(self, hidden: int, heads: int, rounds: int, **kwargs):
        super().__init__(
            input_spec=kwargs.pop('input_spec'), output_spec=kwargs.pop('output_spec'),
            agent_group=kwargs.pop('agent_group'), input_has_agent_dim=kwargs.pop('input_has_agent_dim'),
            n_agents=kwargs.pop('n_agents'), centralised=kwargs.pop('centralised'),
            share_params=kwargs.pop('share_params'), device=kwargs.pop('device'),
            action_spec=kwargs.pop('action_spec'), model_index=kwargs.pop('model_index'),
            is_critic=kwargs.pop('is_critic'),
        )
        if not (self.input_has_agent_dim and self.share_params):
            raise ValueError('AttnComm yalniz ajan eksenli girdi ve tam parametre paylasimiyla (K24)')
        n_in = sum(s.shape[-1] for s in self.input_spec.values(True, True))
        n_out = self.output_leaf_spec.shape[-1]
        self.enc = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh()).to(self.device)
        self.attn = nn.ModuleList(nn.MultiheadAttention(hidden, heads, batch_first=True, device=self.device)
                                  for _ in range(rounds))
        self.norm = nn.ModuleList(nn.LayerNorm(hidden, device=self.device) for _ in range(rounds))
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_out)).to(self.device)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        x = torch.cat([tensordict.get(k) for k in self.in_keys], dim=-1)  # [..., N, F]
        lead = x.shape[:-2]
        h = self.enc(x.reshape(-1, *x.shape[-2:]))  # [B, N, H]
        own = h
        for att, ln in zip(self.attn, self.norm):
            m, _ = att(h, h, h, need_weights=False)  # her ajan tum ajanlarin mesajina bakar
            h = ln(h + m)
        out = self.head(torch.cat([own, h], dim=-1))
        res = out.reshape(*lead, *out.shape[-2:])
        if not self.output_has_agent_dim:  # merkezi + paylasimli: ajan ekseni gereksiz
            res = res[..., 0, :]
        tensordict.set(self.out_key, res)
        return tensordict


@dataclass
class AttnCommConfig(ModelConfig):
    hidden: int = MISSING
    heads: int = MISSING
    rounds: int = MISSING

    @staticmethod
    def associated_class():
        return AttnComm


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
