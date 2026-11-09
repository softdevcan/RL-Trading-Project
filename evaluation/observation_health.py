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

K18 (Faz 10 §6) — olcutun nihai tanimi (`obs_health_ok`, protokol maddesi
`obs_health`): egitim VE degerlendirme penceresinde, kapili piyasa bloklarinda
(fiyat, hacim, gostergeler) kirpilan oran (|x| >= 10) <= OBS_CLIP_EPS, pencere
boyunca sabit gozlem boyutu YOK ve hicbir ozellik doymus degil: ozellik basina
doygunluk (`sat_frac`: |x| >= %99 x sinir) <= OBS_SAT_MAX. Doymus ozellik gozlemde
neredeyse sabit bir boyuttur — ADX = 0 hatasinin RL karsiligi (hakem-rl K18 #1;
eski ortamda tanh(turbulence/2) egitimde %99,7 doygun). Dagilim disilik ayri
maddelerde (`price_in_support`, `features_in_support`). OBS_CLIP_EPS ve OBS_SAT_MAX
tasarim varsayimidir.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

CLIP = 10.0
SUPPORT_EPS = 0.01
OBS_CLIP_EPS = 0.01
SAT_LEVEL = 0.99
OBS_SAT_MAX = 0.90

# Sembol basina gozlem blogu (env/trading_env.py::_get_observation sirasi)
BLOCKS = {
    'price': (0, 4),         # O/H/L/C z-skoru
    'volume': (4, 5),
    'indicators': (5, 10),   # macd, rsi, cci, adx, turbulence
    'fundamentals': (10, 17),  # yalnizca faz 2
}
# Kapiya giren bloklar: faz 2 temel oranlari sembol basina SABIT bir tablodan gelir
# (zaman boyunca degismemesi tasarim geregi), sabit-boyut kurali onlara uygulanmaz.
GATED_BLOCKS = ('price', 'volume', 'indicators')

# Ozellik adi -> gozlemdeki donusumun mutlak siniri (doygunluk olcusu icin).
# tanh ozellikleri ve (rsi-50)/50: 1; dogrusal z/log ozellikler: kirpma siniri 10.
FEATURES = (
    ('open', CLIP), ('high', CLIP), ('low', CLIP), ('close', CLIP), ('volume', CLIP),
    ('macd', 1.0), ('rsi', 1.0), ('cci', 1.0), ('adx', CLIP), ('turbulence', 1.0),
    ('roe', 1.0), ('roa', 1.0), ('debt_to_equity', 1.0), ('current_ratio', CLIP),
    ('pe_ratio', 1.0), ('pb_ratio', 1.0), ('profit_margin', 1.0),
)


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


def feature_stats(market: np.ndarray) -> Dict[str, dict]:
    """Ozellik basina (tum semboller): kirpilan, doygun ve sabit boyut orani (K18)."""
    t, n, fps = market.shape
    out = {}
    for j, (name, bound) in enumerate(FEATURES[:fps]):
        x = market[:, :, j]
        out[name] = {
            'clip_frac': float(np.mean(np.abs(x) >= CLIP)),
            'sat_frac': float(np.mean(np.abs(x) >= SAT_LEVEL * bound)),
            'constant_dims_frac': float(np.mean(np.ptp(x, axis=0) == 0)) if t > 1 else None,
        }
    return out


def _outside(train_market: np.ndarray, eval_market: np.ndarray) -> np.ndarray:
    lo = train_market.min(axis=0)
    hi = train_market.max(axis=0)
    return (eval_market < lo) | (eval_market > hi)


def out_of_support(train_market: np.ndarray, eval_market: np.ndarray) -> Dict[str, float]:
    """Degerlendirme degerlerinin egitim [min, max] araligi disinda kalan orani (blok basina)."""
    outside = _outside(train_market, eval_market)
    fps = eval_market.shape[2]
    return {name: float(outside[:, :, a:b].mean())
            for name, (a, b) in BLOCKS.items() if a < fps}


def out_of_support_features(train_market: np.ndarray, eval_market: np.ndarray) -> Dict[str, float]:
    """Ayni olcu ozellik basina (K18)."""
    outside = _outside(train_market, eval_market)
    return {name: float(outside[:, :, j].mean())
            for j, (name, _) in enumerate(FEATURES[:eval_market.shape[2]])}


def observation_health(train_env, eval_env, seen_obs=None) -> dict:
    """Egitim ve degerlendirme penceresinin gozlem sagligi (bkz. modul docstring'i)."""
    tr_env, ev_env = _base(train_env), _base(eval_env)
    tr, ev = market_matrix(tr_env), market_matrix(ev_env)
    health = {
        'train': block_stats(tr),
        'eval': block_stats(ev),
        'train_features': feature_stats(tr),
        'eval_features': feature_stats(ev),
        'n_train_days': int(tr.shape[0]),
        'n_eval_days': int(ev.shape[0]),
        'eval_out_of_train_support': None,
        'eval_out_of_train_support_features': None,
        'support_eps': SUPPORT_EPS,
        'obs_clip_eps': OBS_CLIP_EPS,
        'obs_sat_max': OBS_SAT_MAX,
        'price_norm': {'train': getattr(tr_env, 'price_norm', 'stats'),
                       'eval': getattr(ev_env, 'price_norm', 'stats')},
        # K25 (b): gercekte kullanilan pencere (hakem-rl K18 #2) — pencere dolmadan ve
        # notr kalan kendi satir sayisi, baglam satiri (sembol basina en az)
        'price_norm_warmup': {
            split: ({'rows_window_not_full': int(sum(e.price_norm_warmup_rows.values())),
                     'rows_neutral': int(sum(e.price_norm_neutral_rows.values())),
                     'rows_total': int(sum(len(v) for v in e._causal_stats.values())),
                     'context_rows_min': int(min(e.price_norm_context_rows.values(), default=0)),
                     'by_symbol_window_not_full': {s: k for s, k in e.price_norm_warmup_rows.items() if k}}
                    if getattr(e, 'price_norm', 'stats') == 'causal' else None)
            for split, e in (('train', tr_env), ('eval', ev_env))},
    }
    tsyms, esyms = list(tr_env.symbols), list(ev_env.symbols)
    if set(esyms) <= set(tsyms) and tr.shape[2] == ev.shape[2] and len(tr) and len(ev):
        idx = [tsyms.index(s) for s in esyms]
        health['eval_out_of_train_support'] = out_of_support(tr[:, idx, :], ev)
        health['eval_out_of_train_support_features'] = out_of_support_features(tr[:, idx, :], ev)
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


def obs_health_failures(health: Optional[dict], eps: float = OBS_CLIP_EPS,
                        sat_max: float = OBS_SAT_MAX) -> list:
    """`obs_health` neden acik: ['eval.price.clip', 'train.indicators.constant',
    'train.turbulence.saturated', ...]. Kapsam: GATED_BLOCKS ve onlarin ozellikleri."""
    out = []
    gated = {n for b in GATED_BLOCKS for n, _ in FEATURES[BLOCKS[b][0]:BLOCKS[b][1]]}
    for split in ('train', 'eval'):
        for name in GATED_BLOCKS:
            st = ((health or {}).get(split) or {}).get(name)
            if not st:
                continue
            if st['clip_frac'] > eps:
                out.append(f'{split}.{name}.clip')
            if (st.get('constant_dims_frac') or 0) > 0:
                out.append(f'{split}.{name}.constant')
        for feat, st in ((health or {}).get(f'{split}_features') or {}).items():
            if feat in gated and st.get('sat_frac') is not None and st['sat_frac'] > sat_max:
                out.append(f'{split}.{feat}.saturated')
    return out


def obs_health_ok(health: Optional[dict], eps: float = OBS_CLIP_EPS,
                  sat_max: float = OBS_SAT_MAX) -> Optional[bool]:
    """Protokol maddesi `obs_health` (K18): bkz. modul docstring'i.

    Egitim ve degerlendirme penceresinde GATED_BLOCKS'un her birinde kirpilan oran
    <= eps, sabit boyut orani 0 ve ozellik basina doygunluk <= sat_max. Olcum yoksa
    (tek gunluk pencere) None.
    """
    if not health:
        return None
    for split in ('train', 'eval'):
        for name in GATED_BLOCKS:
            st = (health.get(split) or {}).get(name)
            if st is not None and st.get('constant_dims_frac') is None:
                return None
    return not obs_health_failures(health, eps, sat_max)


def features_in_support(health: Optional[dict], eps: float = SUPPORT_EPS) -> Optional[bool]:
    """Protokol maddesi `features_in_support` (K18): fiyat DISI her piyasa ozelligi
    (hacim, gostergeler, faz 2 temel oranlar) icin degerlendirme degerlerinin <= eps'i
    egitim destegi disinda mi. Ozellik basina (blok ortalamasi tek ozelligi gizler:
    eski ortamda MACD %28,7 iken gosterge blogu %5,8). Fiyat `price_in_support`'ta.
    """
    oos = (health or {}).get('eval_out_of_train_support_features')
    if not oos:
        return None
    price = {'open', 'high', 'low', 'close'}
    vals = [v for k, v in oos.items() if k not in price]
    return bool(vals) and bool(max(vals) <= eps)


def support_failures(health: Optional[dict], eps: float = SUPPORT_EPS) -> list:
    """Destek disi orani eps'i asan ozellikler (rapor/uyari icin)."""
    oos = (health or {}).get('eval_out_of_train_support_features') or {}
    return sorted(k for k, v in oos.items() if v > eps)


def price_in_support(health: Optional[dict], eps: float = SUPPORT_EPS) -> Optional[bool]:
    """Protokol maddesi: degerlendirme fiyat degerlerinin <= eps'i egitim destegi disinda mi."""
    oos = (health or {}).get('eval_out_of_train_support')
    if not oos or oos.get('price') is None:
        return None
    return bool(oos['price'] <= eps)
