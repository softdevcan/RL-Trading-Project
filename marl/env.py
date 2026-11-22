"""Faz 10 M2 ortami: hisse basina ajan, PettingZoo Parallel (tasarim §7; plan §3.2).

YALNIZ `venv-marl/` icinde import edilir (pettingzoo). Muhasebe, projeksiyon, emir ve gozlem saf
modullerden gelir; bu sinif onlari zaman ekseninde dizer.

    arrays = marl.panel.load_npz('panel_m2.npz')
    env = MarlPortfolioEnv(arrays, start_row=..., end_row=..., norm_stats='fit', mode='train')
    val = MarlPortfolioEnv(arrays, start_row=..., end_row=..., norm_stats=env.norm_stats, mode='eval')

Pencere: karar satirlari [start_row, end_row); son gecis end_row - 1 -> end_row. start_row'dan once
en az `WARMUP_ROWS` satir baglam olmali (ozellik isinmasi; baglam satirlari yalniz gozleme girer).
`norm_stats` ZORUNLU: 'fit' yalniz egitim ortaminda (kendi penceresinden kurar); held-out ortama
egitim ortaminin `norm_stats`'i verilir (B9/Y13 dersi). Sessiz varsayilan yok.
"""
from __future__ import annotations

import functools
from typing import Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from marl import obs_spec as OS
from marl.accounting import (LISTING_LOCKOUT, DifferentialSharpe, limit_down_gaps, margin_schedule_meta, agent_rewards, fill_prices, one_way_cost,
                             step_fractional, step_growth)
from marl.orders import step_integer
from marl.weights import Constraints, logit_bound, project, target_weights

REWARD_TERMS = ('excess', 'differential')


class MarlPortfolioEnv(ParallelEnv):
    metadata = {'name': 'marl_portfolio_v1', 'render_modes': []}

    def __init__(self, arrays: dict, start_row: int, end_row: int, norm_stats, mode: str = 'train',
                 alpha: float = 0.25, cost_bps_roundtrip: float = 20.0,
                 constraints: Constraints = Constraints(), groups: Optional[np.ndarray] = None,
                 reward_term: str = 'excess', reward_scale: float = 100.0, dsr_eta: float = 1.0 / 252,
                 episode_len: int = 252, integer_shares: bool = False, initial_capital: float = 1_000_000.0,
                 pred: Optional[dict] = None, extras: Optional[np.ndarray] = None,
                 listing_lockout: int = LISTING_LOCKOUT, limit_down_gap='schedule'):
        if mode not in ('train', 'eval'):
            raise ValueError(f"mode 'train' | 'eval': {mode}")
        if reward_term not in REWARD_TERMS:
            raise ValueError(f"reward_term {REWARD_TERMS}: {reward_term}")
        if 'rf' not in arrays:
            raise ValueError("arrays['rf'] yok: A8 risksiz faiz zorunlu girdi (plan §3.2.1; sabite dusulmez)")
        self.symbols = [str(s) for s in arrays['symbols']]
        self.dates = np.asarray(arrays['dates'])
        T = len(self.dates)
        if not (OS.WARMUP_ROWS <= start_row < end_row < T):
            raise ValueError(f"pencere gecersiz: {OS.WARMUP_ROWS} <= start {start_row} < end {end_row} < T {T}")
        self.N = len(self.symbols)
        self.start_row, self.end_row = int(start_row), int(end_row)
        self.mode = mode
        self.rf = np.asarray(arrays['rf'], dtype=np.float64)
        if not np.all(np.isfinite(self.rf[start_row + 1:end_row + 1])):
            raise ValueError("pencerede rf eksik (A8 serisi kapsami)")
        self.listing_lockout = int(listing_lockout)
        # Satis engeli: 'schedule' = donem bagimli marj (tasarim §2.5 v0.7; dogrulanmis kapsam disinda hata),
        # sayi = sabit esik (duyarlilik / sentetik test), 0 = kapali
        if isinstance(limit_down_gap, str):
            if limit_down_gap != 'schedule':
                raise ValueError(f"limit_down_gap 'schedule' ya da sayi: {limit_down_gap}")
            ldg, self._limit_meta = limit_down_gaps(self.dates), margin_schedule_meta()
        else:
            ldg, self._limit_meta = float(limit_down_gap), {'rule': 'fixed', 'limit_down_gap': float(limit_down_gap)}
        self.px = fill_prices(arrays['open'], arrays['close'], self.listing_lockout, ldg)
        feat, ok = OS.compute_raw_features(arrays)
        self._raw_feat, self._ok = feat, ok
        rows = np.zeros(T, dtype=bool)
        rows[start_row:end_row + 1] = True
        if isinstance(norm_stats, str):
            if norm_stats != 'fit' or mode != 'train':
                raise ValueError("norm_stats='fit' yalniz mode='train' icin; held-out ortama egitim istatistigini ver")
            norm_stats = OS.fit_norm_stats(feat, ok, rows, pred)
            norm_stats['fit_rows'] = [self.start_row, self.end_row]
            norm_stats['fit_dates'] = [str(self.dates[start_row]), str(self.dates[end_row])]
        elif norm_stats is None:
            raise ValueError("norm_stats zorunlu ('fit' ya da egitim ortaminin norm_stats'i)")
        self.norm_stats = norm_stats
        self.spec = OS.ObsSpec(n_agents=self.N, use_pred=pred is not None,
                               n_groups=0 if groups is None else int(np.max(groups)) + 1,
                               n_extras=0 if extras is None else int(np.shape(extras)[1]))
        self.z = OS.normalize(feat, ok, norm_stats, self.spec.features, self.spec.clip)
        self.groups = None if groups is None else np.asarray(groups, dtype=np.int64)
        self._g1h = None if groups is None else np.eye(self.spec.n_groups)[self.groups]
        if pred is not None:
            self._pred, self._pred_ok = OS.pred_block(pred, norm_stats, self.spec.clip)
        self.extras = None if extras is None else np.asarray(extras, dtype=np.float64)
        self.alpha = alpha
        agent_rewards(np.zeros(1), 0.0, alpha)  # alpha dogrulamasi
        self.cost = one_way_cost(cost_bps_roundtrip)
        self.constraints = constraints
        self.reward_term, self.scale = reward_term, reward_scale
        self.dsr = DifferentialSharpe(dsr_eta)
        self.episode_len = int(episode_len)
        if reward_term == 'differential' and alpha != 0.0:
            # Kullanici karari 2026-11-16 (hakem-rl M2 #1): tanh(D/2) bireysel terimden ~2 mertebe buyuk
            # (sentetik olcum: std orani 242); alpha onelemesi yalniz 'excess'te. DSR = alpha 0 ablasyonu.
            raise ValueError(f"'differential' yalniz alpha=0 ile (ablasyon; birincil odul 'excess'): {alpha}")
        if reward_term == 'differential' and mode == 'train':
            span = end_row - start_row
            eff = min(self.episode_len, span)
            if eff <= self.dsr.warmup:
                # Isinma bolum basinda sifirlanir; bolum isinmadan kisaysa G hep 0 olur ve odul
                # sessizce yalniz alpha*k_i'ye iner (oz-kontrol 2026-11-16: 252 adim, eta=1/252).
                raise ValueError(f"'differential': bolum uzunlugu {eff} <= DSR isinmasi {self.dsr.warmup} "
                                 f"(eta={dsr_eta}); G hep 0 olur. dsr_eta'yi buyut ya da bolumu uzat")
        self.integer_shares, self.initial_capital = integer_shares, float(initial_capital)
        self.bound = logit_bound(self.N)
        self.possible_agents = [f'agent_{i}' for i in range(self.N)]
        self.agents: list[str] = []
        self._rng = np.random.default_rng()
        self.render_mode = None
        self.last_info: dict = {}

    # --- uzaylar -------------------------------------------------------------------------------
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(-np.inf, np.inf, shape=(self.spec.local_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(-self.bound, self.bound, shape=(1,), dtype=np.float32)

    @property
    def state_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(self.spec.state_dim,), dtype=np.float32)

    # --- gozlem --------------------------------------------------------------------------------
    def _local(self, t: int, w: np.ndarray) -> np.ndarray:
        pred_t = pred_ok = None
        if self.spec.use_pred:
            pred_t, pred_ok = self._pred[t], self._pred_ok[t]
        ex = None if self.extras is None else self.extras[t]
        return OS.local_obs(self.spec, self.z[t], self._ok[t], w, self._g1h, pred_t, pred_ok, ex)

    def _obs(self):
        o = self._local(self.t, self.w)
        return {a: o[i] for i, a in enumerate(self.possible_agents)}

    def state(self) -> np.ndarray:
        ex = None if self.extras is None else self.extras[self.t]
        return OS.global_state(self.spec, self.z[self.t], self._ok[self.t], self.w, ex)

    # --- dongu ---------------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        span = self.end_row - self.start_row
        if self.mode == 'train' and self.episode_len < span:
            self.t0 = int(self._rng.integers(self.start_row, self.end_row - self.episode_len + 1))
            self.t_end = self.t0 + self.episode_len
        else:
            self.t0, self.t_end = self.start_row, self.end_row
        self.t = self.t0
        self.w = np.zeros(self.N + 1)
        self.w[0] = 1.0
        self.value = 1.0 if not self.integer_shares else self.initial_capital
        self.cash = self.value
        self.holdings = np.zeros(self.N, dtype=np.int64)
        self.dsr.reset()
        self.last_info = {'date': str(self.dates[self.t]), 'value': self.value}
        return self._obs(), {a: {} for a in self.agents}

    def _step_integer(self, u, g_on, g_day, tradable, rf, no_sell):
        # Faz 11 S5: govde `orders.step_integer`'a tasindi (defter ve urun portfoyu ayni fonksiyonu cagirir)
        r = step_integer(self.holdings, self.cash, self.value, self.w, u, self.px, self.t, g_on, g_day, tradable, rf,
                         no_sell, self.cost, self.symbols, self.constraints, self.groups)
        self.holdings, self.cash, self.value, self.w = r['holdings'], r['cash'], r['value'], r['w']
        return r['R'], r['X'], r['k'], r['turnover'], r['projection'], r['weight_gap'], r['decisions']

    def step(self, actions):
        z = np.array([float(np.asarray(actions[a]).reshape(-1)[0]) for a in self.possible_agents])
        u = target_weights(z, self.bound)
        t = self.t
        rf = float(self.rf[t + 1])
        g_on, g_day, tradable = step_growth(self.px, t)
        no_sell = self.px['no_sell'][t + 1]
        info = {'date': str(self.dates[t + 1]), 'rf': rf}
        if self.integer_shares:
            R, X, k, tau, rep, gap, dec = self._step_integer(u, g_on, g_day, tradable, rf, no_sell)
            info.update(weight_gap=gap, n_orders=sum(d['executed'] for d in dec))
        else:
            res = step_fractional(self.w, u, g_on, g_day, tradable, rf, self.cost, self.constraints, self.groups,
                                  no_sell)
            R, X, k, tau, rep = res.R, res.X, res.k, res.turnover, res.projection
            self.w = res.w_next
            self.value *= 1.0 + R
            info['contrib_residual'] = res.contrib_residual
        if self.reward_term == 'excess':
            G = X
        else:
            G, d_raw = self.dsr.update(X)
            info['dsr_raw'] = d_raw
        r = agent_rewards(k, G, self.alpha, self.scale)
        self.t = t + 1
        info.update(R=R, X=X, turnover=tau, value=self.value, cash_weight=float(self.w[0]), **rep)
        self.last_info = info
        done = self.t >= self.t_end
        obs = self._obs()
        rewards = {a: float(r[i]) for i, a in enumerate(self.possible_agents)}
        terms = {a: False for a in self.possible_agents}
        truncs = {a: bool(done) for a in self.possible_agents}
        infos = {a: {} for a in self.possible_agents}
        if done:
            self.agents = []
        return obs, rewards, terms, truncs, infos

    # --- meta (K1) -----------------------------------------------------------------------------
    def model_meta(self) -> dict:
        """Model paketine yazilacak ortam alanlari (plan §3.3 K1; egitim betigi run_id/seed ekler)."""
        return {
            'kind': 'marl', 'action_type': 'target_weights', 'symbols': self.symbols,
            'agent_ids': list(self.possible_agents), 'obs_spec': self.spec.to_dict(),
            'norm_stats': self.norm_stats, 'logit_bound': self.bound,
            'risk_projection': self.constraints.to_dict(), 'cost_one_way': self.cost,
            'execution': 'close_t_decision__open_t+1_execution', 'listing_lockout': self.listing_lockout,
            'limit_down_no_sell': self._limit_meta,
            'price_basis': 'yfinance auto_adjust (bolunme + temettu duzeltilmis)', 'reward': {
                'alpha': self.alpha, 'term': self.reward_term, 'scale': self.scale,
                'dsr_eta': self.dsr.eta, 'risk_free': 'A8 funding_rate per_period (t-1)'},
            'train_range': [str(self.dates[self.start_row]), str(self.dates[self.end_row])],
        }
