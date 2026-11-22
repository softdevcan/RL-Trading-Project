"""Faz 11 S2: dondurulmus M8 politikasinin sade ileri gecisi (K9). ANA venv; BenchMARL / TorchRL GEREKMEZ.

Plan `docs/development/phase-11-integration-plan.md` §3 S2. Egitimdeki politikanin deterministik eylemi
(`m3_runner.deterministic_rollout`, ExplorationType.DETERMINISTIC) burada duz torch modulleriyle yeniden kurulur:

    cikti  = ag(gozlem)                       # [N, 2] (ajan basina) ya da [2N] (SARL)
    loc    = ilk yari (NormalParamExtractor: loc, olcek = chunk(2))
    eylem  = B * clamp(tanh(loc), +-(1 - eps))   # TorchRL 0.7 TanhNormal.deterministic_sample, safe_tanh; eps =
                                                 # finfo(float32).resolution; B = logit_bound (eylem uzayi float32)

Kollar (M8 tasarimi; ag yapilari egitim kodundan, agirlik adlari kontrol noktasindan):
  mlp       M-none, null : ajan basina paylasimli MLP d -> 256 -> 256 -> 2 (tanh)   [BenchMARL MultiAgentMLP]
  sarl_agr  S-agr        : merkezi MLP N*d -> 256 -> 256 -> 2N                       [tek uyeli grup]
  rowshared S-esm        : duz gozlemin her satirina ayni MLP d -> 256 -> 256 -> 2   [marl/sarl.py RowSharedPolicy]
  comm      M-gat, M-self: kodlayici -> MHA (+ graf / birim maske) -> LN -> cikis    [marl/comm_models.py CommPolicy,
                           conf_mode='none'; M7 degerlendirme kipleri yok]
Girdi her kolda ajan eksenli gozlem [N, d] (panel sirasi; `obs_spec.build_obs_at` + egitim norm_stats'i);
SARL kollari duz gozlemi ajan sirasiyla yan yana birlestirir (`SarlPortfolioEnv._flat` ile ayni).

Parite: `scripts/marl/s2_parity.py` (venv-marl referansina karsi, logit farki <= 1e-6, 72/72).
CAGRI BICIMI: gunluk karar TEK gozlemle (`act(obs[N, d])`) cagrilir; egitimdeki degerlendirme de boyledir. Toplu
cagri (`[T, N, d]`) float32 toplama sirasini degistirir (S-agr'de ~1e-6 mertebesi) -> canli/simulasyon yolunda
kullanma; toplu yol yalniz cozumleme icindir.
"""
from __future__ import annotations

import json
import os

import numpy as np

MLP_CELLS = (256, 256)


def _kind(meta: dict) -> str:
    model = meta['build'].get('model')
    single = bool(meta['task_cfg'].get('single_agent'))
    if model is None:
        return 'sarl_agr' if single else 'mlp'
    if model.get('kind') == 'rowshared':
        return 'rowshared'
    if model.get('kind') in ('gat', 'self'):
        if model.get('conf_mode', 'none') != 'none':
            raise ValueError("guven terimli iletisim kolu (M5 S-msg) aktarilmadi")
        return 'comm'
    raise ValueError(f"desteklenmeyen model: {model}")


def _strip(sd: dict, prefix: str = 'module.0.module.0.module.0.') -> dict:
    out = {}
    for k, v in sd.items():
        if not k.startswith(prefix):
            raise ValueError(f"beklenmeyen agirlik adi: {k}")
        out[k[len(prefix):]] = v
    return out


def _graph_mask(n: int, edges) -> 'torch.Tensor':
    """comm_models.graph_mask ile ayni tanim (True = ENGELLI; kendine dikkat acik). Esitlik parite testinde."""
    import torch
    allowed = torch.eye(n, dtype=torch.bool)
    for i, j in edges or []:
        allowed[int(i), int(j)] = allowed[int(j), int(i)] = True
    return ~allowed


class FrozenPolicy:
    """`act(obs [N, d]) -> eylem [N]` (logit, [-B, B]); `loc(obs)` tanh oncesi. Egitim yapilmaz, durum tutmaz."""

    def __init__(self, policy_dir: str, ckpt: str | None = None):
        import torch
        from torch import nn

        with open(os.path.join(policy_dir, 'meta.json'), encoding='utf-8') as fh:
            self.meta = json.load(fh)
        if ckpt is None:
            cands = sorted(f for f in os.listdir(policy_dir) if f.startswith('ckpt_') and f.endswith('.pt'))
            if len(cands) != 1:
                raise ValueError(f"tek kontrol noktasi bekleniyordu: {cands}")
            ckpt = cands[0]
        self.ckpt = ckpt
        self.kind = _kind(self.meta)
        mm = self.meta['model_meta']
        self.n_agents = int(mm['obs_spec']['n_agents'])
        self.obs_dim = int(mm['obs_spec']['local_dim'])
        self.bound = torch.tensor(float(mm['logit_bound']), dtype=torch.float32)
        sd = _strip(torch.load(os.path.join(policy_dir, ckpt), map_location='cpu'))
        N, d = self.n_agents, self.obs_dim

        def mlp(n_in, n_out):
            dims = [n_in, *MLP_CELLS, n_out]
            layers = []
            for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
                layers.append(nn.Linear(a, b))
                if i < len(dims) - 2:
                    layers.append(nn.Tanh())
            return nn.Sequential(*layers)

        if self.kind in ('mlp', 'sarl_agr'):
            n_in, n_out = (d, 2) if self.kind == 'mlp' else (N * d, 2 * N)
            net = mlp(n_in, n_out)
            w = {k[len('mlp.params.'):]: v for k, v in sd.items()
                 if k.startswith('mlp.params.') and torch.is_tensor(v)}
        elif self.kind == 'rowshared':
            net = mlp(d, 2)
            w = {k[len('mlp.'):]: v for k, v in sd.items()}
        else:
            m = self.meta['build']['model']
            H, heads, rounds = int(m['hidden']), int(m['heads']), int(m['rounds'])
            net = nn.ModuleDict({
                'enc': nn.Sequential(nn.Linear(d, H), nn.Tanh()),
                'attn': nn.ModuleList(nn.MultiheadAttention(H, heads, batch_first=True) for _ in range(rounds)),
                'norm': nn.ModuleList(nn.LayerNorm(H) for _ in range(rounds)),
                'head': nn.Sequential(nn.Linear(2 * H, H), nn.Tanh(), nn.Linear(H, 2)),
            })
            w = sd
            self.mask = _graph_mask(N, m['edges'] if m['kind'] == 'gat' else [])
        net.load_state_dict(w, strict=True)
        net.eval()
        self.net = net

    def _raw(self, x):
        """x [B, N, d] float32 -> loc [B, N]."""
        import torch
        B, N, d = x.shape
        if self.kind == 'mlp':
            y = self.net(x)                                   # [B, N, 2]
            return y[..., 0]
        if self.kind == 'rowshared':
            y = self.net(x)                                   # [B, N, 2] (satir = ajan)
            return y[..., 0]
        if self.kind == 'sarl_agr':
            y = self.net(x.reshape(B, N * d))                 # [B, 2N]
            return y[:, :N]
        h = self.net['enc'](x)
        own = h
        for att, ln in zip(self.net['attn'], self.net['norm']):
            m, _ = att(h, h, h, attn_mask=self.mask, need_weights=False)
            h = ln(h + m)
        return self.net['head'](torch.cat([own, h], dim=-1))[..., 0]

    def loc(self, obs) -> np.ndarray:
        import torch
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        one = x.ndim == 2
        if one:
            x = x[None]
        if tuple(x.shape[1:]) != (self.n_agents, self.obs_dim):
            raise ValueError(f"gozlem [*, {self.n_agents}, {self.obs_dim}] olmali: {tuple(x.shape)}")
        with torch.no_grad():
            out = self._raw(x)
        out = out.numpy()
        return out[0] if one else out

    def act(self, obs) -> np.ndarray:
        """Deterministik eylem (logit, [-B, B]); TorchRL TanhNormal.deterministic_sample ile ayni islem sirasi."""
        import torch
        loc = torch.as_tensor(self.loc(obs))
        lim = 1.0 - torch.finfo(loc.dtype).resolution
        z = loc.tanh().clamp(-lim, lim)
        return (0.0 + self.bound * z).numpy()
