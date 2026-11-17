"""M2 testleri icin ortak sentetik panel (test_marl_core.py, test_marl_env.py). Kendi basina test degil."""
import numpy as np


def synth_arrays(T=220, N=4, seed=0, with_rf=False):
    """Rastgele yuruyus paneli; sembol 2 gec listelenir (ilk 90 satir), sembol 1'de 3 eksik gun (150-152)."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0005, 0.02, (T, N))
    close = 10.0 * np.exp(np.cumsum(r, axis=0))
    open_ = close * np.exp(rng.normal(0, 0.005, (T, N)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, (T, N))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, (T, N))))
    vol = rng.integers(1_000, 100_000, (T, N)).astype(float)
    a = {'open': open_, 'high': high, 'low': low, 'close': close, 'volume': vol,
         'rsi': rng.uniform(20, 80, (T, N)), 'cci': rng.normal(0, 100, (T, N)),
         'adx': rng.uniform(10, 40, (T, N)), 'macd': rng.normal(0, 0.1, (T, N))}
    if N > 2:
        for k in a:
            a[k][:90, 2] = np.nan
    if N > 1:
        for k in a:
            a[k][150:153, 1] = np.nan
    if with_rf:
        a['rf'] = np.full(T, 0.0005)
        a['rf'][0] = np.nan
        a['dates'] = np.array([f'd{t:04d}' for t in range(T)])
        a['symbols'] = np.array([f'S{i}.IS' for i in range(N)])
    return a
