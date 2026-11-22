"""Faz 10 M8 (tasarim A2, plan K15): SARL kontrol kollari. YALNIZ `venv-marl/` (pettingzoo, benchmarl, torchrl).

SarlPortfolioEnv: `MarlPortfolioEnv`'in TEK-AJAN gorunumu; ortamin kendisi degismez.
  - tek ajan `agent_0` (PettingZooWrapper ve BatchedMarlEnv'de grup adi MARL'la ayni: 'agent')
  - gozlem: N yerel gozlem panel sirasiyla yan yana [N*d]; her satir MARL ajaninin gozlemiyle ayni (kimlik dahil)
  - eylem: N logit [-B, B], i. bilesen = ajan i'nin logiti; hedef agirlik, projeksiyon, muhasebe ic ortamda (AYNI cekirdek)
  - odul: ajan odullerinin ortalamasi = olcek * (alpha * ort(k_i) + (1 - alpha) * G) (takim odulu; alpha = 0'da MARL'a esit)
  - bitis ic ortamla ayni; ortam alanlari (mode, start_row, end_row, last_info, norm_stats, bound, spec) ic ortamdan.

RowSharedPolicy (SARL-esmimari): duz gozlemi N satira boler ve her satira AYNI MLP'yi (MARL paylasimli ajan MLP'siyle ayni
katmanlar: d -> num_cells -> 2) uygular; cikti [loc_1..loc_N, olcek_1..olcek_N] (NormalParamExtractor yarilari). Ayni
agirliklarla eylem dagilimi MARL ajanlarinin ortak dagilimina esittir. SARL-agirlik: BenchMARL MLP duz gozlem uzerinde
(merkezi aktor). Iki kol da MAPPO ile tek uyeli grupta egitilir (= PPO; oran kirpmasi ortak log-olasilik uzerinden).
Testler: `tests/test_marl_sarl.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from gymnasium import spaces
from pettingzoo import ParallelEnv
from tensordict import TensorDictBase
from torch import nn
from torchrl.modules import MLP

from benchmarl.models.common import Model, ModelConfig

AGENT = 'agent_0'


class SarlPortfolioEnv(ParallelEnv):
    metadata = {'name': 'sarl_portfolio_v1', 'render_modes': []}

    def __init__(self, inner):
        self.inner = inner
        self.inner_agents = list(inner.possible_agents)
        self.N = len(self.inner_agents)
        self.possible_agents = [AGENT]
        self.agents: list[str] = []
        self._d = int(inner.observation_space(self.inner_agents[0]).shape[0])

    # --- ic ortam alanlari (degerlendirme ve kayit kodu ayni adlari kullanir) --------------------------------------
    mode = property(lambda self: self.inner.mode)
    start_row = property(lambda self: self.inner.start_row)
    end_row = property(lambda self: self.inner.end_row)
    last_info = property(lambda self: self.inner.last_info)
    norm_stats = property(lambda self: self.inner.norm_stats)
    bound = property(lambda self: self.inner.bound)
    spec = property(lambda self: self.inner.spec)

    def observation_space(self, agent):
        return spaces.Box(-np.inf, np.inf, shape=(self.N * self._d,), dtype=np.float32)

    def action_space(self, agent):
        return spaces.Box(-self.inner.bound, self.inner.bound, shape=(self.N,), dtype=np.float32)

    def state_space(self):
        return self.inner.state_space()

    def state(self) -> np.ndarray:
        return self.inner.state()

    def _flat(self, obs: dict) -> np.ndarray:
        return np.concatenate([np.asarray(obs[a], dtype=np.float32).reshape(-1) for a in self.inner_agents])

    def reset(self, seed=None, options=None):
        obs, _ = self.inner.reset(seed=seed, options=options)
        self.agents = [AGENT]
        return {AGENT: self._flat(obs)}, {AGENT: {}}

    def step(self, actions):
        a = np.asarray(actions[AGENT], dtype=np.float32).reshape(-1)
        if a.size != self.N:
            raise ValueError(f"SARL eylemi {self.N} logit olmali: {a.size}")
        obs, rew, term, trunc, _ = self.inner.step({ag: a[i:i + 1] for i, ag in enumerate(self.inner_agents)})
        te, tr = [term[ag] for ag in self.inner_agents], [trunc[ag] for ag in self.inner_agents]
        if len(set(te)) != 1 or len(set(tr)) != 1:
            raise ValueError("ic ortamda ajanlarin bitis bayraklari farkli (SARL tek bayrak ister)")
        if not self.inner.agents:
            self.agents = []
        r = float(np.mean([rew[ag] for ag in self.inner_agents]))
        return {AGENT: self._flat(obs)}, {AGENT: r}, {AGENT: te[0]}, {AGENT: tr[0]}, {AGENT: {}}

    def model_meta(self) -> dict:
        m = self.inner.model_meta()
        return {**m, 'kind': 'sarl', 'agent_ids': [AGENT], 'inner_agent_ids': self.inner_agents,
                'reward': {**m['reward'], 'aggregation': 'mean_over_agents'}}


class RowSharedPolicy(Model):
    """SARL-esmimari politikasi: [..., 1, N*d] -> satir basina ayni MLP -> [..., 1, 2N]."""

    def __init__(self, n_rows: int, num_cells: Sequence[int], **kwargs):
        super().__init__(
            input_spec=kwargs.pop('input_spec'), output_spec=kwargs.pop('output_spec'),
            agent_group=kwargs.pop('agent_group'), input_has_agent_dim=kwargs.pop('input_has_agent_dim'),
            n_agents=kwargs.pop('n_agents'), centralised=kwargs.pop('centralised'),
            share_params=kwargs.pop('share_params'), device=kwargs.pop('device'),
            action_spec=kwargs.pop('action_spec'), model_index=kwargs.pop('model_index'),
            is_critic=kwargs.pop('is_critic'),
        )
        if self.is_critic or self.n_agents != 1 or not self.input_has_agent_dim or self.centralised:
            raise ValueError("RowSharedPolicy yalniz tek uyeli grubun merkezi olmayan politikasi (SARL-esmimari)")
        n_in = sum(s.shape[-1] for s in self.input_spec.values(True, True))
        n_out = self.output_leaf_spec.shape[-1]
        if n_rows < 1 or n_in % n_rows or n_out != 2 * n_rows:
            raise ValueError(f"girdi {n_in} satira ({n_rows}) bolunmeli ve cikti 2N ({n_out}) olmali")
        self.n_rows, self.row_dim = int(n_rows), n_in // int(n_rows)
        self.mlp = MLP(in_features=self.row_dim, out_features=2, num_cells=list(num_cells), activation_class=nn.Tanh,
                       layer_class=nn.Linear, device=self.device)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        x = torch.cat([tensordict.get(k) for k in self.in_keys], dim=-1)          # [..., 1, N*d]
        y = self.mlp(x.reshape(*x.shape[:-1], self.n_rows, self.row_dim))          # [..., 1, N, 2]
        tensordict.set(self.out_key, torch.cat([y[..., 0], y[..., 1]], dim=-1))   # [..., 1, 2N]
        return tensordict


@dataclass
class RowSharedConfig(ModelConfig):
    n_rows: int = 30
    num_cells: list = field(default_factory=lambda: [256, 256])

    @staticmethod
    def associated_class():
        return RowSharedPolicy
