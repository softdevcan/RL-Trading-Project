"""Faz 11 S3: sanal para simulasyonu defteri (plan phase-11-integration-plan.md §3 S3; K14). ANA venv.

Strateji basina bir defter; kesirli mod, M8 degerlendirmesinin AYNI cekirdegi:
  karar t kapanisinda -> hedef u (politika: `weights.target_weights(eylem)`, kural: `rule_baselines`) ->
  `accounting.step_fractional` t+1 acilisinda yurutme (maliyet tek yon 20 bps / 2, A8 rf, `fill_prices` kilit 20 +
  donem bagimli marj satis engeli, `weights.project` varsayilan kisit).
Ortam (`marl/env.py`, venv-marl) ile tek fark yurutucu: burada TorchRL/BenchMARL yok, politika `FrozenPolicy` ile TEK
gozlemle cagrilir (toplu cagri float32 toplama sirasini degistirir; S2). Gozlem `obs_spec.local_obs` (ortamla ayni).

Durum (`LedgerState`): baslangic satiri, son yazilan karar satiri, kapanis agirligi w [N+1], deger. JSON'a kayipsiz
yazilir (float repr gidis-donus); kaldigi yerden devam tam kosuyla bit-es (test_marl_sim_ledger). Deger sanal sermaye
x prod(1 + R): kesirli defterde sermaye getiriyi degistirmez (K-F11-6).

    mk = Market(PN.load_npz(...))
    st = PolicyStrategy('data/marl/frozen/m8-2026/M-none/s201')     # ya da RuleStrategy('minvar')
    state, rows = run(mk, st, LedgerState.start(start_row, mk.N), end_row)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from marl import accounting as A
from marl import obs_spec as OS
from marl import rule_baselines as RB
from marl import weights as W

COST_BPS_ROUNDTRIP = 20.0          # M8 on kaydi (tum kollar ve kural portfoyleri)
INITIAL_CAPITAL = 100_000.0        # K-F11-6 onerisi (sanal TL); kesirli defterde R'yi degistirmez
ROW_KEYS = ('date', 'row', 'R', 'X', 'rf', 'turnover', 'cost_frac', 'cash_weight', 'value', 'n_sell_blocked')


class Market:
    """Panel -> defterin ortak girdileri (tasinmis fiyat, islem/satis maskeleri, ham ozellik). Bir kez kurulur."""

    def __init__(self, arrays: dict, cost_bps_roundtrip: float = COST_BPS_ROUNDTRIP, verified_until: str | None = None):
        if 'rf' not in arrays:
            raise ValueError("arrays['rf'] yok: A8 risksiz faiz zorunlu")
        self.arrays = arrays
        self.dates = np.asarray(arrays['dates'])
        self.symbols = [str(s) for s in arrays['symbols']]
        self.N = len(self.symbols)
        self.rf = np.asarray(arrays['rf'], dtype=np.float64)
        # Ortamla ayni: limit_down_gap='schedule' (dogrulanmis kapsam disi tarih -> ValueError, sessiz gecmez);
        # ileri gunlerde kapsam `margin_coverage.coverage_until` (S4)
        self.px = A.fill_prices(arrays['open'], arrays['close'], A.LISTING_LOCKOUT, A.limit_down_gaps(self.dates, verified_until))
        self.cost = A.one_way_cost(cost_bps_roundtrip)
        self.feat, self.ok = OS.compute_raw_features(arrays)
        self.panel_sha256 = arrays.get('sha256')

    def row_of(self, date: str) -> int:
        i = np.flatnonzero(self.dates == date)
        if i.size != 1:
            raise ValueError(f"tarih panelde yok: {date}")
        return int(i[0])


@dataclass
class LedgerState:
    start_row: int                 # ilk karar satiri
    t: int                         # siradaki karar satiri (son yazilan gun = t)
    w: np.ndarray                  # kapanis_t agirligi [N+1] (0 = nakit)
    value: float
    n_steps: int = 0
    meta: dict = field(default_factory=dict)

    @classmethod
    def start(cls, start_row: int, n: int, capital: float = INITIAL_CAPITAL, meta: dict | None = None):
        w = np.zeros(n + 1)
        w[0] = 1.0
        return cls(int(start_row), int(start_row), w, float(capital), 0, dict(meta or {}))

    def to_json(self) -> str:
        return json.dumps({'start_row': self.start_row, 't': self.t, 'w': [float(x) for x in self.w],
                           'value': self.value, 'n_steps': self.n_steps, 'meta': self.meta}, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str):
        d = json.loads(s)
        return cls(int(d['start_row']), int(d['t']), np.asarray(d['w'], dtype=np.float64), float(d['value']),
                   int(d['n_steps']), d.get('meta', {}))


class RuleStrategy:
    """M8 A4 kural portfoyleri; `rule_baselines.rule_series` ile ayni karar (ew, bh, minvar, hrp, cash)."""

    def __init__(self, kind: str):
        if kind not in RB.KINDS:
            raise ValueError(f"kind {RB.KINDS}: {kind}")
        self.kind = kind
        self.sid = f'rule/{kind}'

    def target(self, mk: Market, st: LedgerState, t: int, g_on: np.ndarray, rf: float) -> np.ndarray:
        N, k = mk.N, self.kind
        if k == 'cash':
            return np.r_[1.0, np.zeros(N)]
        if k == 'ew' or (k == 'bh' and t == st.start_row):
            v = mk.px['valid'][t]
            return np.r_[0.0, v.astype(float) / max(float(v.sum()), 1.0)]
        if k in RB.MONTHLY and (t == st.start_row or str(mk.dates[t])[:7] != str(mk.dates[t - 1])[:7]):
            return RB.monthly_target(mk.px, t, k)[0]          # rebalance_rows ile ayni satirlar
        drift = st.w * np.concatenate([[1.0 + rf], g_on])     # bh ve ay ici: hedef = kaymis agirlik
        return drift / drift.sum()


class PolicyStrategy:
    """Dondurulmus M8 politikasi (S2 `FrozenPolicy`); gozlem politikanin kendi egitim `norm_stats`'iyla."""

    def __init__(self, policy_dir: str, ckpt: str | None = None, sid: str | None = None):
        from marl.frozen_policy import FrozenPolicy
        self.policy = FrozenPolicy(policy_dir, ckpt)
        mm = self.policy.meta['model_meta']
        if mm['obs_spec'].get('use_pred') or mm['obs_spec'].get('n_groups') or mm['obs_spec'].get('n_extras'):
            raise ValueError("tahmin / grup / ek ozellikli gozlem bu defterde yok (M8 kollari kullanmaz)")
        with open(f'{policy_dir}/norm_stats.json', encoding='utf-8') as fh:
            self.norm_stats = json.load(fh)
        self.spec = OS.ObsSpec(n_agents=int(mm['obs_spec']['n_agents']))
        if self.spec.to_dict() != {k: mm['obs_spec'][k] for k in self.spec.to_dict()}:
            raise ValueError(f"gozlem tanimi egitimdekiyle ayni degil: {self.spec.to_dict()} vs {mm['obs_spec']}")
        self.bound = float(mm['logit_bound'])
        self.sid = sid or f'policy/{policy_dir}'
        self._z = None
        self._z_key = None

    def _zs(self, mk: Market) -> np.ndarray:
        if self._z_key is not mk:
            self._z = OS.normalize(mk.feat, mk.ok, self.norm_stats, self.spec.features, self.spec.clip)
            self._z_key = mk
        return self._z

    def obs(self, mk: Market, st: LedgerState, t: int) -> np.ndarray:
        return OS.local_obs(self.spec, self._zs(mk)[t], mk.ok[t], st.w)

    def target(self, mk: Market, st: LedgerState, t: int, g_on: np.ndarray, rf: float) -> np.ndarray:
        a = self.policy.act(self.obs(mk, st, t))                            # float32 [N], tek gozlem
        z = np.array([float(x) for x in np.asarray(a, dtype=np.float32).reshape(-1)])
        return W.target_weights(z, self.bound)


def step(mk: Market, strategy, st: LedgerState) -> dict:
    """Karar satiri st.t -> yurutme st.t+1; durumu yerinde ilerletir, gun satirini dondurur."""
    t = st.t
    if t + 1 >= len(mk.dates):
        raise ValueError(f"panelde {mk.dates[t]} sonrasi satir yok")
    g_on, g_day, tradable = A.step_growth(mk.px, t)
    rf = float(mk.rf[t + 1])
    u = strategy.target(mk, st, t, g_on, rf)
    res = A.step_fractional(st.w, u, g_on, g_day, tradable, rf, mk.cost, W.Constraints(), None,
                            mk.px['no_sell'][t + 1])
    st.w = res.w_next
    st.value *= 1.0 + res.R
    st.t = t + 1
    st.n_steps += 1
    return {'date': str(mk.dates[t + 1]), 'row': t + 1, 'R': res.R, 'X': res.X, 'rf': rf, 'turnover': res.turnover,
            'cost_frac': res.cost_frac, 'cash_weight': float(res.w_next[0]), 'value': st.value,
            'n_sell_blocked': int(res.projection.get('n_sell_blocked', 0)), 'w': res.w_next.copy()}


def run(mk: Market, strategy, st: LedgerState, end_row: int) -> tuple[LedgerState, list]:
    """Karar satirlari [st.t, end_row): son yurutme gunu end_row. Durum yerinde ilerler."""
    if not (st.t <= end_row < len(mk.dates)):
        raise ValueError(f"pencere gecersiz: t {st.t}, end {end_row}, T {len(mk.dates)}")
    rows = []
    while st.t < end_row:
        rows.append(step(mk, strategy, st))
    return st, rows


def series(rows: list) -> dict:
    return {k: (np.asarray([r[k] for r in rows]) if k != 'date' else [r[k] for r in rows]) for k in ROW_KEYS}
