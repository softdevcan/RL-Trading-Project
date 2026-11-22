"""Faz 10 kural portfoyleri (baseline) — ortamin AYNI muhasebesiyle; saf numpy (ana venv ve venv-marl).

M3'ten (M2 §9 #11, hakem-finans F7/F10): `accounting.step_fractional`, ayni `fill_prices` (kilit 20 + v0.7 marj
takvimi), ayni `project` (Constraints), A8 rf, ayni maliyet:
  ew     : her gun tam yatirimli 1/N (t'de gecerli semboller)
  bh     : ilk gun 1/N, sonra hedef = kaymis agirlik (islem yok; yalniz zorunlu dondurma)
  cash   : tam nakit (X = 0)
M8 on kaydi §6.2 (A4; hakem-finans #6), aylik yeniden dengeleme:
  minvar : uzun-yalniz minimum varyans, Ledoit-Wolf daraltmali kovaryans (sabit varyansli birim hedefe; formul
           `sklearn.covariance.ledoit_wolf` ile ayni, test paritesi)
  hrp    : hiyerarsik risk paritesi (korelasyon uzakligi, tekli baglanti agaci, yari-kosegen sira, ozyineli ikiye
           bolme, kume ici ters varyans)
Aylik kurallarin ileriye bakmama kosullari (§6.2 a-e):
  (a) agirlik ayin ilk seansinda (ve pencerenin ilk karar satirinda) t kapanisina kadarki WINDOW = 252 gunluk
      getiriden hesaplanir, t+1 acilisinda uygulanir; ay icinde hedef = kaymis agirlik (fiyatla kayar).
  (b) uzun-yalniz, toplam <= 1; kalan nakit A8 faizi kazanir; satis engeli, kilit ve maliyet ortamla ayni (`project`).
  (c) pencerede 252 gecerli getirisi olmayan ya da t'de islem goremeyen (kilitli) hisse o ay evrene girmez.
  (d) kovaryans, daraltma ve baglanti agaci yalniz o pencereden.
  (e) `monthly_target(px, t, ...)` yalniz <= t satirlarini okur (test: t'den sonrasi kesilmis panelle ayni sonuc).
"""
from __future__ import annotations

import numpy as np

from marl import accounting as A
from marl import weights as W

WINDOW = 252
KINDS = ('ew', 'bh', 'cash', 'minvar', 'hrp')
MONTHLY = ('minvar', 'hrp')


def ledoit_wolf(x: np.ndarray) -> tuple[np.ndarray, float]:
    """x [n, p] getiri -> (daraltilmis kovaryans, daraltma). Ortalama cikarilir; ampirik kovaryans 1/n."""
    x = np.asarray(x, dtype=np.float64)
    n, p = x.shape
    x = x - x.mean(axis=0)
    s = x.T @ x / n
    mu = float(np.trace(s)) / p
    delta = float(((s - mu * np.eye(p)) ** 2).sum()) / p
    x2 = x ** 2
    beta = float((x2.T @ x2 / n - s ** 2).sum()) / (p * n)
    shrink = 0.0 if delta == 0 else min(beta, delta) / delta
    return (1.0 - shrink) * s + shrink * mu * np.eye(p), shrink


def _simplex(v: np.ndarray) -> np.ndarray:
    """Oklid izdusumu {w >= 0, sum w = 1}."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    k = np.flatnonzero(u - css / np.arange(1, v.size + 1) > 0)[-1]
    return np.maximum(v - css[k] / (k + 1), 0.0)


def minvar_long_only(cov: np.ndarray, iters: int = 20_000, tol: float = 1e-12) -> np.ndarray:
    """min w' S w, w >= 0, sum w = 1: hizlandirilmis izdusumlu gradyan (dis bagimlilik yok; KKT testli)."""
    cov = np.asarray(cov, dtype=np.float64)
    p = cov.shape[0]
    if p == 1:
        return np.ones(1)
    lr = 1.0 / (2.0 * float(np.linalg.eigvalsh(cov)[-1]))
    w = y = np.full(p, 1.0 / p)
    t = 1.0
    for _ in range(iters):
        w_new = _simplex(y - lr * 2.0 * cov @ y)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = w_new + (t - 1.0) / t_new * (w_new - w)
        if np.abs(w_new - w).max() < tol:
            w = w_new
            break
        w, t = w_new, t_new
    return w


def _single_linkage_order(dist: np.ndarray) -> list:
    """Tekli baglanti agaci -> yaprak sirasi (birlesmede sol = daha kucuk ilk indeksli kume; belirlenimci)."""
    clusters = {i: [i] for i in range(dist.shape[0])}
    d = {(i, j): dist[i, j] for i in clusters for j in clusters if i < j}
    nxt = dist.shape[0]
    while len(clusters) > 1:
        (a, b) = min(d, key=lambda k: (d[k], k))
        merged = clusters[a] + clusters[b]
        del clusters[a], clusters[b]
        d = {k: v for k, v in d.items() if a not in k and b not in k}
        for c, members in clusters.items():
            d[(c, nxt)] = min(dist[i, j] for i in merged for j in members)
        clusters[nxt] = merged
        nxt += 1
    return next(iter(clusters.values()))


def _ivp(cov: np.ndarray) -> np.ndarray:
    iv = 1.0 / np.diag(cov)
    return iv / iv.sum()


def hrp(cov: np.ndarray) -> np.ndarray:
    """Hiyerarsik risk paritesi; cov -> agirlik (uzun-yalniz, toplam 1)."""
    cov = np.asarray(cov, dtype=np.float64)
    p = cov.shape[0]
    if p == 1:
        return np.ones(1)
    sd = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(sd, sd), -1.0, 1.0)
    dcorr = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    dist = np.sqrt(((dcorr[:, None, :] - dcorr[None, :, :]) ** 2).sum(axis=2))
    order = _single_linkage_order(dist)
    w = np.ones(p)
    stack = [order]
    while stack:
        items = stack.pop()
        if len(items) < 2:
            continue
        h = len(items) // 2
        left, right = items[:h], items[h:]
        var = []
        for c in (left, right):
            sub = cov[np.ix_(c, c)]
            wc = _ivp(sub)
            var.append(float(wc @ sub @ wc))
        alpha = 1.0 - var[0] / (var[0] + var[1])
        w[left] *= alpha
        w[right] *= 1.0 - alpha
        stack += [left, right]
    return w / w.sum()


def monthly_target(px: dict, t: int, kind: str, window: int = WINDOW) -> tuple[np.ndarray, dict]:
    """t kapanisinda karar: [N+1] ham hedef (0 = nakit) ve evren bilgisi. YALNIZ <= t satirlari okunur."""
    if kind not in MONTHLY:
        raise ValueError(f"kind {MONTHLY}: {kind}")
    N = px['close'].shape[1]
    u = np.r_[1.0, np.zeros(N)]
    if t < window:
        return u, {'n_universe': 0}
    rows = slice(t - window, t + 1)
    ok = px['valid'][rows].all(axis=0) & px['tradable'][t]          # (c): 253 gecerli kapanis = 252 getiri
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return u, {'n_universe': 0}
    c = px['close'][rows][:, idx]
    r = c[1:] / c[:-1] - 1.0
    cov, shrink = ledoit_wolf(r)
    w = minvar_long_only(cov) if kind == 'minvar' else hrp(cov)
    u = np.zeros(N + 1)
    u[1 + idx] = w
    return u, {'n_universe': int(idx.size), 'shrinkage': float(shrink)}


def rebalance_rows(dates, start_row: int, end_row: int) -> list:
    """Aylik kurallarin karar satirlari: pencerenin ilk karar satiri + ayin ilk seansi olan satirlar."""
    out = [start_row]
    for t in range(start_row + 1, end_row):
        if str(dates[t])[:7] != str(dates[t - 1])[:7]:
            out.append(t)
    return out


def rule_series(arrays: dict, start_row: int, end_row: int, kind: str, cost_bps_roundtrip: float = 20.0,
                limit_down_gap='schedule', constraints: W.Constraints = W.Constraints()) -> dict:
    """Ortamin muhasebesiyle kural portfoyu; karar satirlari [start_row, end_row)."""
    if kind not in KINDS:
        raise ValueError(f"kind {' | '.join(KINDS)}: {kind}")
    ldg = A.limit_down_gaps(arrays['dates']) if limit_down_gap == 'schedule' else float(limit_down_gap)
    px = A.fill_prices(arrays['open'], arrays['close'], A.LISTING_LOCKOUT, ldg)
    cost = A.one_way_cost(cost_bps_roundtrip)
    N = arrays['open'].shape[1]
    w = np.r_[1.0, np.zeros(N)]
    reb = set(rebalance_rows(arrays['dates'], start_row, end_row)) if kind in MONTHLY else set()
    out = {k: [] for k in ('date', 'R', 'X', 'turnover', 'cash_weight')}
    universe = []
    for t in range(start_row, end_row):
        g_on, g_day, tradable = A.step_growth(px, t)
        rf = float(arrays['rf'][t + 1])
        v = px['valid'][t]
        if kind == 'cash':
            u = np.r_[1.0, np.zeros(N)]
        elif kind == 'ew' or (kind == 'bh' and t == start_row):
            u = np.r_[0.0, v.astype(float) / max(float(v.sum()), 1.0)]
        elif t in reb:
            u, info = monthly_target(px, t, kind)
            universe.append({'date': str(arrays['dates'][t]), **info})
        else:   # bh ve ay ici: hedef = kaymis agirlik (gece kaymasi project icinde w~'ye esit -> islem yok)
            drift = w * np.concatenate([[1.0 + rf], g_on])
            u = drift / drift.sum()
        res = A.step_fractional(w, u, g_on, g_day, tradable, rf, cost, constraints, None, px['no_sell'][t + 1])
        w = res.w_next
        for k, val in (('date', str(arrays['dates'][t + 1])), ('R', res.R), ('X', res.X),
                       ('turnover', res.turnover), ('cash_weight', float(w[0]))):
            out[k].append(val)
    ser = {k: (np.asarray(v) if k != 'date' else list(v)) for k, v in out.items()}
    if kind in MONTHLY:
        ser['rebalances'] = universe
    return ser
