"""Faz 10 M4: iletisim kollu politika modelleri (BenchMARL 1.4 `Model`). YALNIZ `venv-marl/`.

Kollar (plan §3.2.5; M4 tasarimi `docs/development/phase-10-m4-design.md` §1):
  none : BenchMARL MLP (iletisimsiz; ajan basina paylasimli MLP)  -> bu modul KULLANILMAZ
  attn : paylasimli kodlayici -> ajanlar arasi cok basli oz-dikkat (tam bagli; TarMAC tipi, Das vd. 2019)
         -> paylasimli cikti (spike `scripts/spike_m1/benchmarl_glue.py::AttnComm` ile ayni yapi)
  gat  : attn ile AYNI katmanlar ve AYNI parametre sayisi; dikkat yalniz graf komsulari + kendisi ile sinirli
         (sabit ikili maske; graf egitim penceresinden, `scripts/marl/graph_m4.py`). Iki kol arasindaki tek fark
         graf onculudur (plan §2 Y-tablosu #3: "fark graf onculunden gelir").

Yalniz ajan eksenli girdi + tam parametre paylasimi (K24) + merkezi olmayan politika desteklenir; elestirmen
BenchMARL MLP'dir (kollar arasinda ayni).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
from tensordict import TensorDictBase
from torch import nn

from benchmarl.models.common import Model, ModelConfig

KINDS = ('attn', 'gat')


def graph_mask(n_agents: int, edges: Sequence[Sequence[int]]) -> torch.Tensor:
    """Yonsuz kenar listesi -> MultiheadAttention maskesi [N, N], True = ENGELLI. Kendine dikkat her zaman acik
    (her satirda en az bir izinli giris; softmax NaN vermez)."""
    allowed = torch.eye(n_agents, dtype=torch.bool)
    for e in edges:
        i, j = int(e[0]), int(e[1])
        if not (0 <= i < n_agents and 0 <= j < n_agents) or i == j:
            raise ValueError(f"gecersiz kenar {e} (N = {n_agents})")
        allowed[i, j] = allowed[j, i] = True
    return ~allowed


class CommPolicy(Model):
    """Kodlayici -> `rounds` tur (dikkat + artik + LayerNorm) -> [kendi kod || mesaj sonrasi] -> cikti."""

    def __init__(self, kind: str, hidden: int, heads: int, rounds: int, edges: Optional[Sequence] = None, **kwargs):
        super().__init__(
            input_spec=kwargs.pop('input_spec'), output_spec=kwargs.pop('output_spec'),
            agent_group=kwargs.pop('agent_group'), input_has_agent_dim=kwargs.pop('input_has_agent_dim'),
            n_agents=kwargs.pop('n_agents'), centralised=kwargs.pop('centralised'),
            share_params=kwargs.pop('share_params'), device=kwargs.pop('device'),
            action_spec=kwargs.pop('action_spec'), model_index=kwargs.pop('model_index'),
            is_critic=kwargs.pop('is_critic'),
        )
        if kind not in KINDS:
            raise ValueError(f"kind {KINDS}: {kind}")
        if not (self.input_has_agent_dim and self.share_params) or self.centralised or self.is_critic:
            raise ValueError('CommPolicy yalniz ajan eksenli, tam paylasimli, merkezi olmayan politika (K24)')
        if kind == 'gat' and not edges:
            raise ValueError("gat kolu kenar listesi ister")
        if kind == 'attn' and edges:
            raise ValueError("attn kolu tam baglidir; kenar listesi verilmez")
        if hidden % heads:
            raise ValueError(f"hidden ({hidden}) heads'e ({heads}) bolunmeli")
        self.kind = kind
        n_in = sum(s.shape[-1] for s in self.input_spec.values(True, True))
        n_out = self.output_leaf_spec.shape[-1]
        self.enc = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh()).to(self.device)
        self.attn = nn.ModuleList(nn.MultiheadAttention(hidden, heads, batch_first=True, device=self.device)
                                  for _ in range(rounds))
        self.norm = nn.ModuleList(nn.LayerNorm(hidden, device=self.device) for _ in range(rounds))
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_out)).to(self.device)
        mask = graph_mask(self.n_agents, edges) if kind == 'gat' else None
        self.register_buffer('attn_mask', None if mask is None else mask.to(self.device), persistent=False)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        x = torch.cat([tensordict.get(k) for k in self.in_keys], dim=-1)  # [..., N, F]
        lead = x.shape[:-2]
        h = self.enc(x.reshape(-1, *x.shape[-2:]))  # [B, N, H]
        own = h
        for att, ln in zip(self.attn, self.norm):
            m, _ = att(h, h, h, attn_mask=self.attn_mask, need_weights=False)
            h = ln(h + m)
        out = self.head(torch.cat([own, h], dim=-1))
        tensordict.set(self.out_key, out.reshape(*lead, *out.shape[-2:]))
        return tensordict


@dataclass
class CommPolicyConfig(ModelConfig):
    kind: str = 'attn'
    hidden: int = 112
    heads: int = 4
    rounds: int = 1
    edges: Optional[list] = field(default=None)

    @staticmethod
    def associated_class():
        return CommPolicy


def comm_param_count(n_in: int, n_out: int, hidden: int, rounds: int = 1) -> int:
    """CommPolicy parametre sayisi (attn ve gat ayni): kodlayici + rounds x (MHA 4H^2+4H + LN 2H) + cikti."""
    H = hidden
    return (n_in * H + H) + rounds * (4 * H * H + 4 * H + 2 * H) + (2 * H * H + H) + (H * n_out + n_out)


def mlp_param_count(n_in: int, n_out: int, cells: Sequence[int] = (256, 256)) -> int:
    """BenchMARL MLP (ajan basina, paylasimli) parametre sayisi."""
    dims = [n_in, *cells, n_out]
    return sum(a * b + b for a, b in zip(dims[:-1], dims[1:]))
