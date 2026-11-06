"""Faz 9 / B9 (+ K18 on adimi) — RL gozlem sagligi.

Neden (hakem-2 B9 #1, #9): B9 ile eski ortamin test fiyat ozellikleri egitim
penceresinin olcegiyle z-skorlaniyor ve TL fiyatlari o olcegin cok ustunde:
test fiyat degerlerinin %74,7'si +-10'da kirpiliyor (Faz 9 §9.5). Politika
egitimde hic gormedigi girdilerle karar veriyor. Bu modul bunu OLCER ki deneme
kaydi protokolu (`price_in_support`) makine tarafindan uygulanabilsin; belgede
yazmak yetmez.

Olculenler:
  * Piyasa bloklari (fiyat O/H/L/C z, hacim, gostergeler, faz 2'de temel
    oranlar) politikadan BAGIMSIZDIR. Env'in kendi gozlem fonksiyonu her tarih
    icin cagrilir (islem yapilmaz, bakiye/pozisyon degismez); boylece eksik
    hucre modu ve kirpma dahil politikanin gordugu degerler olculur.
  * Blok basina: kirpilan oran (|x| >= 10) ve sabit boyut orani (pencere
    boyunca degismeyen gozlem boyutu), egitim ve degerlendirme penceresinde ayri.
  * Destek disi oran: degerlendirme degerinin, AYNI gozlem boyutunun egitim
    penceresindeki [min, max] araliginin disinda kaldigi oran.
  * Pozisyon blogu politikaya bagli; yalnizca verilen `seen_obs`'tan (modelin
    karar verirken gordugu test gozlemleri) olculur.

`SUPPORT_EPS` bir TASARIM VARSAYIMIDIR: degerlendirme fiyat degerlerinin en
fazla %1'i egitim desteginin disinda olabilir. Esik degil OLCUM kayda girer.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

CLIP = 10.0
SUPPORT_EPS = 0.01

# Sembol basina gozlem blogu (env/trading_env.py::_get_observation sirasi)
BLOCKS = {
    'price': (0, 4),         # O/H/L/C z-skoru
    'volume': (4, 5),
    'indicators': (5, 10),   # macd, rsi, cci, adx, turbulence
    'fundamentals': (10, 17),  # yalnizca faz 2
}


def _base(env):
    return getattr(env, 'unwrapped', env)


def market_matrix(env) -> np.ndarray:
    """(T, n_stocks, features_per_stock): env'in her tarih icin urettigi piyasa blogu.

    Gozlem SINIF metoduyla uretilir (ornek uzerindeki sarmalayicilar —
    testlerin gozlem kaydedicisi gibi — tetiklenmez); env'in adimi geri yuklenir.
    """
    base = _base(env)
    n, fps = base.n_stocks, base.features_per_stock
    observe = type(base)._get_observation
    saved = base.current_step
    rows = []
    try:
        for i in range(base.frame_bound[0], base.frame_bound[1] + 1):
            base.current_step = i
            obs = observe(base)
            rows.append(obs[1 + n: 1 + n + n * fps].reshape(n, fps))
    finally:
        base.current_step = saved
    return np.asarray(rows, dtype=np.float32)


def block_stats(market: np.ndarray) -> Dict[str, dict]:
    """Blok basina kirpilan oran ve sabit boyut orani."""
    t, n, fps = market.shape
    out = {}
    for name, (a, b) in BLOCKS.items():
        if a >= fps:
            continue
        flat = market[:, :, a:b].reshape(t, -1)
        out[name] = {
            'clip_frac': float(np.mean(np.abs(flat) >= CLIP)),
            'constant_dims_frac': float(np.mean(np.ptp(flat, axis=0) == 0)) if t > 1 else None,
        }
    return out


def out_of_support(train_market: np.ndarray, eval_market: np.ndarray) -> Dict[str, float]:
    """Degerlendirme degerlerinin egitim [min, max] araligi disinda kalan orani (blok basina)."""
    lo = train_market.min(axis=0)
    hi = train_market.max(axis=0)
    outside = (eval_market < lo) | (eval_market > hi)
    fps = eval_market.shape[2]
    return {name: float(outside[:, :, a:b].mean())
            for name, (a, b) in BLOCKS.items() if a < fps}


def observation_health(train_env, eval_env, seen_obs=None) -> dict:
    """Egitim ve degerlendirme penceresinin gozlem sagligi (bkz. modul docstring'i)."""
    tr_env, ev_env = _base(train_env), _base(eval_env)
    tr, ev = market_matrix(tr_env), market_matrix(ev_env)
    health = {
        'train': block_stats(tr),
        'eval': block_stats(ev),
        'n_train_days': int(tr.shape[0]),
        'n_eval_days': int(ev.shape[0]),
        'eval_out_of_train_support': None,
        'support_eps': SUPPORT_EPS,
    }
    tsyms, esyms = list(tr_env.symbols), list(ev_env.symbols)
    if set(esyms) <= set(tsyms) and tr.shape[2] == ev.shape[2] and len(tr) and len(ev):
        idx = [tsyms.index(s) for s in esyms]
        health['eval_out_of_train_support'] = out_of_support(tr[:, idx, :], ev)
    if seen_obs is not None and len(seen_obs):
        seen = np.asarray(seen_obs, dtype=np.float32)
        n = ev_env.n_stocks
        health['eval_seen'] = {
            'n_obs': int(len(seen)),
            'position_clip_frac': float(np.mean(np.abs(seen[:, 1:1 + n]) >= CLIP)),
            'balance_clip_frac': float(np.mean(np.abs(seen[:, 0]) >= CLIP)),
            'obs_clip_frac': float(np.mean(np.abs(seen) >= CLIP)),
        }
    return health


def price_in_support(health: Optional[dict], eps: float = SUPPORT_EPS) -> Optional[bool]:
    """Protokol maddesi: degerlendirme fiyat degerlerinin <= eps'i egitim destegi disinda mi."""
    oos = (health or {}).get('eval_out_of_train_support')
    if not oos or oos.get('price') is None:
        return None
    return bool(oos['price'] <= eps)
