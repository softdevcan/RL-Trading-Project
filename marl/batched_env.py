"""Faz 10 §3.2.8 adim 2 (kullanici karari 2026-11-20): toplu yerel TorchRL ortami. YALNIZ `venv-marl/`.

Profil (`measurements/phase10-m8/env_profile.json`): ham ortam adimi 0,12 ms, TorchRL `PettingZooWrapper` icinden 1,91 ms;
egitimin ~%84'u toplama ve darbogaz adim basina sabit ek yuk. Bu ortam B tane `MarlPortfolioEnv`'i TEK SURECTE adimlar ve
tensordict'i parti basina bir kez kurar. Cekirdek fonksiyonlar (muhasebe, projeksiyon, gozlem) AYNIDIR: her alt ortam
tek ortamla ayni kodu calistirir; K2/K5/K6 paritesi degismez.

Veri paketi `PettingZooWrapper(categorical_actions=True, return_state=False, use_mask=False)` ile ayni anahtarlari ve
ayni dtype'lari uretir, basa [B] parti boyutu eklenir (BenchMARL vektorlesmis ortam yolu: `test_env.batch_size != ()`).
Tohum: `PettingZooWrapper._set_seed` ile ayni: tohum verilince alt ortam i hemen `reset(seed=seed + i)` ile bir kez
sifirlanir (SerialEnv/ParallelEnv'in seed + i duzeni); sonraki her sifirlama tohumsuzdur ve ortamin RNG'sini surdurur.
Parite: `tests/test_marl_batched_env.py` (B = 1 sarmalayiciyla bit-es; B = 3 her alt ortam bagimsiz tek ortamla bit-es).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

GROUP = 'agent'


class BatchedMarlEnv(EnvBase):
    def __init__(self, env_fns: list, seed: Optional[int] = None, device='cpu'):
        envs = [f() for f in env_fns]
        super().__init__(device=device, batch_size=torch.Size([len(envs)]))
        self.envs = envs
        self.agents = list(envs[0].possible_agents)
        self.N = len(self.agents)
        first = envs[0]
        obs_dim = int(first.observation_space(self.agents[0]).shape[0])
        aspace = first.action_space(self.agents[0])
        B, N = len(envs), self.N
        self.group_map = {GROUP: [str(a) for a in self.agents]}
        self.observation_spec = Composite({GROUP: Composite({
            'observation': Unbounded(shape=(B, N, obs_dim), dtype=torch.float32, device=device)}, shape=(B, N))},
            shape=(B,))
        self.action_spec = Composite({GROUP: Composite({
            'action': Bounded(low=torch.as_tensor(aspace.low, dtype=torch.float32).expand(B, N, *aspace.shape),
                              high=torch.as_tensor(aspace.high, dtype=torch.float32).expand(B, N, *aspace.shape),
                              shape=(B, N, *aspace.shape), dtype=torch.float32, device=device)}, shape=(B, N))},
            shape=(B,))
        self.reward_spec = Composite({GROUP: Composite({
            'reward': Unbounded(shape=(B, N, 1), dtype=torch.float32, device=device)}, shape=(B, N))}, shape=(B,))
        done = lambda *s: Categorical(n=2, shape=s, dtype=torch.bool, device=device)  # noqa: E731
        self.done_spec = Composite({'done': done(B, 1), 'terminated': done(B, 1), 'truncated': done(B, 1),
                                    GROUP: Composite({'done': done(B, N, 1), 'terminated': done(B, N, 1),
                                                      'truncated': done(B, N, 1)}, shape=(B, N))}, shape=(B,))
        self._seed = None
        if seed is not None:
            self._set_seed(seed)

    def _set_seed(self, seed: Optional[int]):
        self._seed = seed
        if seed is not None:
            for i, env in enumerate(self.envs):
                env.reset(seed=int(seed) + i)

    def _obs(self, obs: dict) -> np.ndarray:
        return np.stack([obs[a] for a in self.agents]).astype(np.float32)

    def _flags(self, term: np.ndarray, trunc: np.ndarray) -> dict:
        t = torch.as_tensor(term, device=self.device)[..., None]        # [B, N, 1]
        u = torch.as_tensor(trunc, device=self.device)[..., None]
        d = t | u
        return {'agent': {'terminated': t, 'truncated': u, 'done': d},
                'root': {'terminated': t.all(dim=1), 'truncated': u.all(dim=1), 'done': d.all(dim=1)}}

    def _pack(self, agent: dict, flags: dict) -> TensorDict:
        """Sarmalayiciyla ayni yapi: grup alt paketi [B, N] parti boyutlu (toplayici yigma icin sart)."""
        B = len(self.envs)
        grp = TensorDict({**agent, **flags['agent']}, batch_size=(B, self.N), device=self.device)
        return TensorDict({GROUP: grp, **flags['root']}, batch_size=self.batch_size, device=self.device)

    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs) -> TensorDictBase:
        B = len(self.envs)
        mask = np.ones(B, dtype=bool)
        if tensordict is not None and '_reset' in tensordict.keys():
            mask = tensordict.get('_reset').reshape(B, -1).any(dim=1).cpu().numpy()
        obs = getattr(self, '_last_obs', None)
        if obs is None:
            obs = np.zeros((B, self.N, self.observation_spec[GROUP, 'observation'].shape[-1]), dtype=np.float32)
        obs = obs.copy()
        for i in np.flatnonzero(mask):
            o, _ = self.envs[i].reset()
            obs[i] = self._obs(o)
        self._last_obs = obs
        z = np.zeros((B, self.N), dtype=bool)
        return self._pack({'observation': torch.as_tensor(obs, device=self.device)}, self._flags(z, z))

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        act = tensordict.get((GROUP, 'action')).detach().cpu().numpy()     # [B, N, A]
        B = len(self.envs)
        obs = np.empty_like(self._last_obs)
        rew = np.empty((B, self.N, 1), dtype=np.float32)
        term = np.empty((B, self.N), dtype=bool)
        trunc = np.empty((B, self.N), dtype=bool)
        for i, env in enumerate(self.envs):
            o, r, te, tr, _ = env.step({a: act[i, k] for k, a in enumerate(self.agents)})
            obs[i] = self._obs(o)
            rew[i, :, 0] = [r[a] for a in self.agents]
            term[i] = [te[a] for a in self.agents]
            trunc[i] = [tr[a] for a in self.agents]
        self._last_obs = obs
        return self._pack({'observation': torch.as_tensor(obs, device=self.device),
                           'reward': torch.as_tensor(rew, device=self.device)}, self._flags(term, trunc))
