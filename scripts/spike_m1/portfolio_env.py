"""Faz 10 M1 spike'i: 30 ajanli portfoy ortami (PettingZoo Parallel; yalniz numpy + pettingzoo).

SPIKE ORTAMIDIR, M2 ortami degildir. Amaci cerceve secimi icin gercekci boyutta (30 ajan, gercek panel,
softmax agirlik, maliyet) sure ve tekrar uretim olcmektir. Tasarim Faz 10 plani §3.2.1-3.2.3'u izler ama
M2 kararlari (obs_spec, nakit faizi, risk projeksiyonu, tam sayi hisse, egitim penceresi normalizasyonu) burada
YOKTUR. Uretilen hicbir performans sayisi rapora girmez (§4 S1).

Zamanlama (K12): adim t'de ajanlar t kapanisindaki gozlemle karar verir; yurutme t+1 acilisinda.
  kapanis_t -> acilis_{t+1}: eski agirliklar kayar (nakit 1)
  acilis_{t+1}: hedefe yeniden dengeleme, maliyet c * sum_i |hedef_i - kaymis_i| (yalniz hisseler)
  acilis_{t+1} -> kapanis_{t+1}: hedef agirliklar
Odul (§3.2.1): r_i = alpha * k_i + (1 - alpha) * R, R portfoy getirisi (maliyet sonrasi), k_i ajanin birinci
derece katkisi (sum_i k_i = R'nin riskli kismi, birinci derecede).
Eylem (§3.2.2): ajan i skaler logit z_i; hedef w = softmax([0, z_1..z_N]) (nakit logiti sabit 0).
Gozlem (§3.2.3, K24): kendi F ozelligi + kendi agirligi + nakit orani + ajan kimligi (one-hot).
"""
from __future__ import annotations

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class SpikePortfolioEnv(ParallelEnv):
    metadata = {'name': 'spike_portfolio_v0', 'render_modes': []}

    def __init__(self, panel_path: str, episode_len: int = 252, alpha: float = 0.25, cost: float = 0.001,
                 reward_scale: float = 100.0, logit_bound: float = 5.0, n_agents: int | None = None):
        d = np.load(panel_path)
        self.feat = d['features']  # [T, N, F]
        self.open_px = d['open'].astype(np.float64)
        self.close_px = d['close'].astype(np.float64)
        n = self.feat.shape[1] if n_agents is None else n_agents
        self.feat, self.open_px, self.close_px = self.feat[:, :n], self.open_px[:, :n], self.close_px[:, :n]
        self.T, self.N, self.F = self.feat.shape
        self.episode_len = min(episode_len, self.T - 2)
        self.alpha, self.cost, self.scale, self.bound = alpha, cost, reward_scale, logit_bound
        self.possible_agents = [f'agent_{i}' for i in range(self.N)]
        self.agents: list[str] = []
        self._eye = np.eye(self.N, dtype=np.float32)
        self._rng = np.random.default_rng()
        self.render_mode = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(-np.inf, np.inf, shape=(self.F + 2 + self.N,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(-self.bound, self.bound, shape=(1,), dtype=np.float32)

    def _obs(self):
        f = self.feat[self.t]  # [N, F]
        own_w = self.w[1:, None].astype(np.float32)
        cash = np.full((self.N, 1), self.w[0], dtype=np.float32)
        o = np.concatenate([f, own_w, cash, self._eye], axis=1)
        return {a: o[i] for i, a in enumerate(self.agents)}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self.t0 = int(self._rng.integers(0, self.T - self.episode_len - 1))
        self.t = self.t0
        self.w = np.zeros(self.N + 1)
        self.w[0] = 1.0  # tamami nakit baslar
        self.value = 1.0
        return self._obs(), {a: {} for a in self.agents}

    def step(self, actions):
        z = np.array([float(np.asarray(actions[a]).reshape(-1)[0]) for a in self.agents])
        z = np.clip(z, -self.bound, self.bound)
        logits = np.concatenate([[0.0], z])
        target = np.exp(logits - logits.max())
        target /= target.sum()

        t, t1 = self.t, self.t + 1
        g_on = self.open_px[t1] / self.close_px[t]  # kapanis_t -> acilis_{t+1}
        g_day = self.close_px[t1] / self.open_px[t1]  # acilis_{t+1} -> kapanis_{t+1}
        drift = self.w * np.concatenate([[1.0], g_on])
        g1 = drift.sum()
        w_open = drift / g1
        trade = np.abs(target[1:] - w_open[1:])
        cost_frac = self.cost * trade.sum()
        stock_after = target[1:] * g_day
        g2 = target[0] + stock_after.sum()
        R = g1 * (1.0 - cost_frac) * g2 - 1.0
        k = self.w[1:] * (g_on - 1.0) + target[1:] * (g_day - 1.0) - self.cost * trade

        self.value *= 1.0 + R
        self.w = np.concatenate([[target[0]], stock_after]) / g2
        self.t = t1
        r = self.scale * (self.alpha * k + (1.0 - self.alpha) * R)
        done = (self.t - self.t0) >= self.episode_len or self.t >= self.T - 1
        rewards = {a: float(r[i]) for i, a in enumerate(self.agents)}
        infos = {a: {} for a in self.agents}
        obs = self._obs()
        terms = {a: False for a in self.agents}
        truncs = {a: bool(done) for a in self.agents}
        if done:
            self.agents = []
        return obs, rewards, terms, truncs, infos
