"""Zaman indeksli muhasebe denklemi (tasarim §2; hakem-2 #13, plan §3.2.1-3.2.2, K12).

Karar t kapanisinda, yurutme t+1 acilisinda. Bir adim t -> t+1 uc parcadir:

  1. Gece (kapanis_t -> acilis_{t+1}): hisseler g_on = Of_{t+1}/Cf_t ile kayar; nakit rf_{t+1} ile
     faiz kazanir (A8, t'de bilinen oran, iki seans arasi takvim gunu; `evaluation.risk_free`).
     g1 = w0 (1 + rf) + sum_i w_i g_on_i,  w~ = kaymis agirlik.
  2. Acilis_{t+1}: hedef w* (`weights.project`), devir tau = sum_i |w*_i - w~_i| (yalniz hisseler),
     maliyet c * tau, c TEK YON = bps_gidis_donus / 2 / 1e4 (`evaluation/baselines.py` D4 ile ayni).
     Maliyet portfoy degerinden orantili duser (tasarim varsayimi; Jiang vd. 2017'nin kesin mu'su
     yerine birinci derece; hata O(c^2 tau)).
  3. Seans ici (acilis_{t+1} -> kapanis_{t+1}): g_day = Cf_{t+1}/Of_{t+1}; nakit seans icinde faiz
     kazanmaz (gecelik oran gece kazanilir).

  R = g1 (1 - c tau) g2 - 1,   X = R - rf  (fazla getiri; odul ve secim metrigi ayni seriyi kullanir).
  k_i = w_i (g_on_i - 1 - rf) + w*_i (g_day_i - 1) - c |w*_i - w~_i|
  sum_i k_i = X + (ikinci derece capraz terimler); nakdin fazla getiri katkisi tanim geregi 0.
  Katki ayrisimi yalniz bu birinci derece basit getiri icin iddia edilir (plan §8 #13).

Eksik hucre (M1: 69 gun): hucre atilmaz. `fill_prices` sembolu kendi son kapanisiyla tasir;
gecersiz gunde g = 1 (getiri 0) ve hisse o gun islem goremez (`tradable=False` -> dondurulur).
Hucre yeniden gecerli olunca g_on = O / son gecerli kapanis: aradaki hareket o gun yazilir.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from marl.weights import Constraints, project


def one_way_cost(bps_roundtrip: float) -> float:
    """D4 izgarasi gidis-donus bps -> tek yon oran (20 bps -> 0,001)."""
    if bps_roundtrip < 0:
        raise ValueError(f"maliyet negatif olamaz: {bps_roundtrip}")
    return float(bps_roundtrip) / 2.0 / 1e4


LISTING_LOCKOUT = 20   # kullanici karari 2026-11-16 (hakem-finans M2 #1): halka arz tavan serisi


def fill_prices(open_px, close_px, listing_lockout: int = 0) -> dict:
    """[T, N] acilis/kapanis -> tasinmis seriler + gecerlilik ve islem maskesi.

    valid[t,i]: O ve C sonlu ve > 0. Cf: son gecerli kapanis (listelenmeden once NaN).
    Of: gecerli gunde O, degilse Cf_{t-1} (getiri 0). listed[t,i] = Cf[t,i] sonlu.
    tradable[t,i] = valid ve listeleme kilidi disinda: sembol panelde ILK satirdan sonra listelendiyse
    (ilk gecerli satir > 0) ilk gecerli gunden itibaren `listing_lockout` seans islem yok (halka arz
    sonrasi tavan serisinde acilistan dolum gercekci degil). Panel basinda zaten listeli sembolun halka
    arz tarihi panelden gorunmez (beyan, tasarim §2.3).
    Nedenseldir: satir t yalniz <= t satirlarindan uretilir (ilk gecerli gun o gun bilinir).
    """
    if listing_lockout < 0:
        raise ValueError(f"listing_lockout >= 0 olmali: {listing_lockout}")
    o = np.asarray(open_px, dtype=np.float64)
    c = np.asarray(close_px, dtype=np.float64)
    if o.shape != c.shape or o.ndim != 2:
        raise ValueError(f"open/close [T, N] ve ayni boyut olmali: {o.shape}, {c.shape}")
    valid = np.isfinite(o) & np.isfinite(c) & (o > 0) & (c > 0)
    T, N = c.shape
    cf = np.full((T, N), np.nan)
    of = np.full((T, N), np.nan)
    last = np.full(N, np.nan)
    for t in range(T):
        v = valid[t]
        of[t] = np.where(v, o[t], last)
        last = np.where(v, c[t], last)
        cf[t] = last
    tradable = valid.copy()
    lockout = np.zeros((T, N), dtype=bool)
    if listing_lockout > 0:
        for i in range(N):
            first = np.flatnonzero(valid[:, i])
            if first.size and first[0] > 0:
                lockout[first[0]:first[0] + listing_lockout, i] = True
        tradable &= ~lockout
    return {'open': of, 'close': cf, 'valid': valid, 'listed': np.isfinite(cf),
            'tradable': tradable, 'lockout': lockout}


def step_growth(px: dict, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """t -> t+1 icin (g_on [N], g_day [N], tradable [N]). Listelenmemis sembolde g = 1."""
    cf, of = px['close'], px['open']
    t1 = t + 1
    with np.errstate(invalid='ignore', divide='ignore'):
        g_on = of[t1] / cf[t]
        g_day = cf[t1] / of[t1]
    g_on = np.where(np.isfinite(g_on), g_on, 1.0)
    g_day = np.where(np.isfinite(g_day), g_day, 1.0)
    tradable = px['tradable'][t1].copy()
    return g_on, g_day, tradable


@dataclass
class StepResult:
    w_next: np.ndarray       # [N+1] kapanis_{t+1} agirligi
    w_drift: np.ndarray      # [N+1] acilis_{t+1} kaymis agirlik (w~)
    w_target: np.ndarray     # [N+1] uygulanan hedef (w*)
    R: float                 # maliyet sonrasi basit portfoy getirisi
    X: float                 # fazla getiri R - rf
    k: np.ndarray            # [N] ajan katkilari (fazla getiri, birinci derece)
    turnover: float          # tau
    cost_frac: float         # c * tau
    rf: float
    projection: dict

    @property
    def contrib_residual(self) -> float:
        return float(self.X - self.k.sum())


def step_fractional(w, u, g_on, g_day, tradable, rf: float, cost: float,
                    constraints: Constraints = Constraints(),
                    groups: Optional[np.ndarray] = None) -> StepResult:
    """Kesirli mod bir adim (modul docstring'indeki denklem). w: kapanis_t agirligi [N+1]."""
    w = np.asarray(w, dtype=np.float64)
    if not np.isfinite(rf):
        raise ValueError("rf sonlu olmali (A8 serisi eksik; sabite dusulmez)")
    drift = w * np.concatenate([[1.0 + rf], g_on])
    g1 = float(drift.sum())
    w_tilde = drift / g1
    w_star, rep = project(u, w_tilde, tradable, constraints, groups)
    trade = np.abs(w_star[1:] - w_tilde[1:])
    tau = float(trade.sum())
    cf = cost * tau
    after = np.concatenate([[w_star[0]], w_star[1:] * g_day])
    g2 = float(after.sum())
    R = g1 * (1.0 - cf) * g2 - 1.0
    X = R - rf
    k = w[1:] * (g_on - 1.0 - rf) + w_star[1:] * (g_day - 1.0) - cost * trade
    return StepResult(w_next=after / g2, w_drift=w_tilde, w_target=w_star, R=R, X=X, k=k,
                      turnover=tau, cost_frac=cf, rf=rf, projection=rep)


class DifferentialSharpe:
    """Moody & Saffell (2001) D_t, fazla getiri uzerinde (G1 'differential' ile ayni tanim).

    `env/reward_functions.py`'deki uygulamanin aynisi (isinma 1/eta adim D=0, varyans tabani 1e-9,
    tanh(D/2) sikistirma); tek fark girdinin sabit 0,15 degil A8 fazla getirisi olmasi (G2).
    """

    def __init__(self, eta: float = 1.0 / 252):
        if not (0 < eta < 1):
            raise ValueError(f"eta (0, 1) olmali: {eta}")
        self.eta = eta
        self.warmup = int(round(1.0 / eta))   # ilk `warmup` adimda D = 0
        self.reset()

    def reset(self):
        self.a = 0.0
        self.b = 0.0
        self.n = 0

    def update(self, x: float) -> tuple[float, float]:
        a_prev, b_prev = self.a, self.b
        d_a = x - a_prev
        d_b = x * x - b_prev
        var_prev = b_prev - a_prev * a_prev
        self.a = a_prev + self.eta * d_a
        self.b = b_prev + self.eta * d_b
        self.n += 1
        if self.n <= self.warmup or not np.isfinite(var_prev):
            return 0.0, 0.0
        d_t = (b_prev * d_a - 0.5 * a_prev * d_b) / max(var_prev, 1e-9) ** 1.5
        if not np.isfinite(d_t):
            return 0.0, 0.0
        return float(np.tanh(d_t / 2)), float(d_t)


def agent_rewards(k: np.ndarray, portfolio_term: float, alpha: float, scale: float = 1.0) -> np.ndarray:
    """r_i = scale * (alpha k_i + (1 - alpha) G). alpha in [0, 1) (alpha=1 kullanilmaz; vision §3.3 beta)."""
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha [0, 1) olmali (alpha=1 plan disi): {alpha}")
    return scale * (alpha * np.asarray(k, dtype=np.float64) + (1.0 - alpha) * float(portfolio_term))
