"""K5 — hedef agirliktan tam sayi emir uretimi (tasarim §4; plan §3.3 K5).

Ortamin tam sayi modu ve canli karar (`POST /api/marl/daily-decision`, ileride) AYNI fonksiyonu
cagirir. Cikti mevcut karar sozlugu semasindadir (`daily_trading.interpret_actions_with_risk`), bu
yuzden `app.services.portfolio.apply_decisions` DEGISMEDEN calisir.

Nakit tanimlari `interpret_actions_with_risk` ile ayni (CLAUDE.md "Kagit portfoy"):
  BUY  -> cost    = adet * fiyat * (1 + komisyon)   (komisyon DAHIL)
  SELL -> revenue = adet * fiyat * (1 - komisyon)   (komisyon DUSULMUS)
`commission` TEK YON orandir (`accounting.one_way_cost`).

Algoritma (deterministik):
  1. V = nakit + sum_i h_i p_i (islem goremeyen hisse verilen fiyatla degerlenir).
  2. Islem goren i icin hedef adet n_i = floor(w*_i V / p_i); islem goremeyen i icin n_i = h_i.
  3. Once SATISLAR (n_i < h_i), gelir nakde.
  4. Sonra ALIMLAR, eksik deger buyukten kucuge (esitlikte sembol sirasi); her alim nakdin
     yettigi kadarla sinirli: q = min(n_i - h_i, floor(nakit / (p_i (1 + komisyon)))).
  Hedef komisyonsuz V ile kuruldugu icin son alimlar birkac adet eksik kalabilir; bu yuvarlama
  etkisi `summary.weight_gap` ile raporlanir (plan §3.2.2 "yuvarlama etkisi raporlanir").
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def rebalance_orders(target_w, prices, holdings, cash: float, commission: float,
                     symbols: Sequence[str], tradable: Optional[np.ndarray] = None,
                     logits: Optional[np.ndarray] = None,
                     no_sell: Optional[np.ndarray] = None) -> tuple[List[dict], dict]:
    """Hedef agirlik -> emir listesi + ozet.

    Args:
        target_w: [N+1] hedef (0 = nakit), `weights.project` ciktisi.
        prices: [N] yurutme fiyati (sonlu, > 0; islem goremeyen icin son kapanis).
        holdings: [N] tam sayi mevcut adet (>= 0).
        cash: mevcut nakit (>= 0).
        commission: tek yon oran.
        symbols: [N] sembol adlari (meta `symbols` sirasi).
        tradable: [N] bool; None -> hepsi.
        logits: [N] opsiyonel ham ajan logitleri; `raw_signal` alanina yazilir (yoksa hedef agirlik).
        no_sell: [N] bool satis engeli (taban acilis); hedef adet mevcut adedin altina inmez.
            `weights.project` ayni maskeyle hedefi w~_i'nin altina indirmez; bu, yuvarlamaya karsi bekci.
    Returns:
        (kararlar, ozet). ozet: cash_after, holdings_after, value, turnover, weight_gap, commission.
    """
    w = np.asarray(target_w, dtype=np.float64).reshape(-1)
    p = np.asarray(prices, dtype=np.float64).reshape(-1)
    h = np.asarray(holdings).reshape(-1)
    n = p.shape[0]
    if w.shape != (n + 1,) or h.shape != (n,) or len(symbols) != n:
        raise ValueError(f"boyut uyumsuz: w {w.shape}, p {p.shape}, h {h.shape}, symbols {len(symbols)}")
    if not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("fiyatlar sonlu ve > 0 olmali (islem goremeyen icin son kapanisi ver)")
    if np.any(h < 0) or not np.all(np.equal(np.mod(h, 1), 0)):
        raise ValueError("holdings negatif olmayan tam sayi olmali")
    if cash < 0 or not np.isfinite(cash):
        raise ValueError(f"nakit >= 0 olmali: {cash}")
    if np.any(w < -1e-12) or abs(w.sum() - 1.0) > 1e-9:
        raise ValueError(f"hedef uzun-yalniz ve toplam 1 olmali: min {w.min()}, toplam {w.sum()}")
    h = h.astype(np.int64)
    tr = np.ones(n, dtype=bool) if tradable is None else np.asarray(tradable, dtype=bool)
    sig = w[1:] if logits is None else np.asarray(logits, dtype=np.float64).reshape(-1)

    value = float(cash + np.dot(h, p))
    cur_w = (h * p) / value if value > 0 else np.zeros(n)
    tgt = h.copy()
    tgt[tr] = np.floor(np.maximum(w[1:][tr], 0.0) * value / p[tr] + 1e-9).astype(np.int64)
    ns = np.zeros(n, dtype=bool) if no_sell is None else np.asarray(no_sell, dtype=bool)
    tgt[ns] = np.maximum(tgt[ns], h[ns])

    decisions: List[Optional[dict]] = [None] * n
    cash_now = float(cash)
    comm_total = 0.0

    def _row(i, action, qty, cost=0.0, revenue=0.0, comm=0.0, reason=''):
        return {
            'symbol': symbols[i], 'action': action, 'raw_signal': float(sig[i]), 'shares': int(qty),
            'price': float(p[i]), 'cost': float(cost), 'revenue': float(revenue),
            'commission': float(comm), 'reason': reason, 'executed': action != 'HOLD',
            'target_weight': float(w[1 + i]), 'current_weight': float(cur_w[i]),
        }

    for i in range(n):  # satislar
        if tgt[i] < h[i]:
            q = int(h[i] - tgt[i])
            gross = q * p[i]
            decisions[i] = _row(i, 'SELL', q, revenue=gross * (1 - commission), comm=gross * commission,
                                reason=f"hedef agirlik {w[1 + i]:.4f} < mevcut {cur_w[i]:.4f}")
            cash_now += gross * (1 - commission)
            comm_total += gross * commission

    buys = [i for i in range(n) if tgt[i] > h[i]]
    buys.sort(key=lambda i: (-(tgt[i] - h[i]) * p[i], i))
    for i in buys:
        unit = p[i] * (1 + commission)
        q = int(min(tgt[i] - h[i], np.floor(cash_now / unit + 1e-12)))
        if q <= 0:
            decisions[i] = _row(i, 'HOLD', 0, reason='nakit yetmedi (yuvarlama)')
            continue
        gross = q * p[i]
        cost = gross * (1 + commission)
        if cost > cash_now:  # kayan nokta bekcisi
            q -= 1
            if q <= 0:
                decisions[i] = _row(i, 'HOLD', 0, reason='nakit yetmedi (yuvarlama)')
                continue
            gross = q * p[i]
            cost = gross * (1 + commission)
        decisions[i] = _row(i, 'BUY', q, cost=cost, comm=gross * commission,
                            reason=f"hedef agirlik {w[1 + i]:.4f} > mevcut {cur_w[i]:.4f}")
        cash_now -= cost
        comm_total += gross * commission

    for i in range(n):
        if decisions[i] is None:
            why = ('islem goremez (eksik hucre)' if not tr[i]
                   else 'satis engeli (taban acilis)' if ns[i] else 'hedef adet = mevcut')
            decisions[i] = _row(i, 'HOLD', 0, reason=why)

    h_after = h.copy()
    for i, d in enumerate(decisions):
        if d['action'] == 'BUY':
            h_after[i] += d['shares']
        elif d['action'] == 'SELL':
            h_after[i] -= d['shares']
    traded_value = sum(d['shares'] * d['price'] for d in decisions if d['executed'])
    value_after = cash_now + float(np.dot(h_after, p))
    w_after = np.concatenate([[cash_now], h_after * p]) / value_after if value_after > 0 else w * 0
    summary = {
        'value': value, 'cash_after': cash_now, 'holdings_after': h_after,
        'turnover': traded_value / value if value > 0 else 0.0, 'commission': comm_total,
        'weight_gap': float(np.abs(w_after - w).sum()), 'weights_after': w_after,
    }
    return decisions, summary


def apply_to_holdings(decisions: List[dict], holdings, cash: float) -> tuple[np.ndarray, float]:
    """Kararlari nakit/adet dizisine isle (ortamin tam sayi modu; `portfolio.apply_decisions` ile ayni kural)."""
    h = np.asarray(holdings, dtype=np.int64).copy()
    c = float(cash)
    for i, d in enumerate(decisions):
        if not d.get('executed'):
            continue
        if d['action'] == 'BUY':
            h[i] += d['shares']
            c -= d['cost']
        elif d['action'] == 'SELL':
            h[i] -= d['shares']
            c += d['revenue']
    return h, c


def decisions_by_symbol(decisions: List[dict]) -> Dict[str, dict]:
    return {d['symbol']: d for d in decisions}
