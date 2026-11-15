"""
Ikili yon olasiligi icin kalibrasyon olculeri ve sicaklik olcekleme (Faz 9 F2, F3).

Kaynak: Guo, Pleiss, Sun & Weinberger (2017), "On Calibration of Modern Neural Networks",
ICML — ECE: M = 15 esit genislikli guven kutusu, ECE = sum_m |B_m|/n * |acc(B_m) - conf(B_m)|;
sicaklik olcekleme: tek parametre T > 0, dogrulama kumesinde NLL en kucuklenir, p = sigma(z/T).
Brier (1950): ortalama (p - y)^2.

Ikili durumda "guven" tahmin edilen sinifin olasiligidir: c = max(p, 1-p); dogru = 1[(p>=0,5) == y].
`compute_calibration` (prediction/evaluator.py, tracker; 5 kutu) bu modulle KARISTIRILMAZ (Y18).
"""

from typing import Dict, Optional

import numpy as np

N_BINS = 15
_EPS = 1e-12


def _as_arrays(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    y = np.asarray(y, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError(f"p {p.shape} ve y {y.shape} ayni boyda olmali")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y yalniz 0/1 olmali")
    return p, y


def reliability(p_up, y_up, n_bins: int = N_BINS) -> list:
    """Guven kutulari: [{'lo','hi','n','conf','acc'}] (bos kutular dahil)."""
    p, y = _as_arrays(p_up, y_up)
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5) == (y == 1)).astype(np.float64)
    # Guo vd. gibi [0, 1] uzerinde esit genislik; ikili durumda c >= 0,5 oldugundan alt kutular bos
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        n = int(m.sum())
        out.append({'lo': float(lo), 'hi': float(hi), 'n': n,
                    'conf': float(conf[m].mean()) if n else None,
                    'acc': float(correct[m].mean()) if n else None})
    return out


def ece(p_up, y_up, n_bins: int = N_BINS) -> float:
    """Expected Calibration Error (Guo vd. 2017, denklem 3): esit genislikli kutular [0, 1]."""
    p, _ = _as_arrays(p_up, y_up)
    n = len(p)
    if n == 0:
        raise ValueError("bos dizi")
    return float(sum(b['n'] / n * abs(b['acc'] - b['conf'])
                     for b in reliability(p_up, y_up, n_bins) if b['n']))


def brier(p_up, y_up) -> float:
    p, y = _as_arrays(p_up, y_up)
    return float(np.mean((p - y) ** 2))


def nll(p_up, y_up) -> float:
    p, y = _as_arrays(p_up, y_up)
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(logits, y_up, bounds=(0.05, 20.0)) -> float:
    """NLL'yi en kucukleyen T (Guo vd. 2017 §4.2). log T uzerinde sinirli tek boyutlu arama."""
    from scipy.optimize import minimize_scalar
    z = np.asarray(logits, dtype=np.float64)
    _, y = _as_arrays(sigmoid(z), y_up)
    if len(z) < 2 or len(np.unique(y)) < 2:
        raise ValueError("sicaklik icin en az iki sinif gerekli")
    res = minimize_scalar(lambda lt: nll(sigmoid(z / np.exp(lt)), y),
                          bounds=(np.log(bounds[0]), np.log(bounds[1])), method='bounded')
    return float(np.exp(res.x))


def apply_temperature(logits, temperature: float):
    if not (np.isfinite(temperature) and temperature > 0):
        raise ValueError(f"gecersiz sicaklik: {temperature}")
    return sigmoid(np.asarray(logits, dtype=np.float64) / temperature)


def fit_isotonic(p_up, y_up):
    """Karsilastirma kolu (Niculescu-Mizil & Caruana 2005): monoton p -> p eslemesi."""
    from sklearn.isotonic import IsotonicRegression
    p, y = _as_arrays(p_up, y_up)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    iso.fit(p, y)
    return iso


def auc(p_up, y_up) -> Optional[float]:
    """Ayirt edicilik: ROC alani (Mann-Whitney U / (n+ n-)). Tek sinifta None."""
    p, y = _as_arrays(p_up, y_up)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    from scipy.stats import rankdata
    r = rankdata(np.r_[pos, neg])
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def brier_skill(p_up, y_up, base_rate: float) -> float:
    """1 - Brier / Brier(sabit taban orani). Taban orani TEST DISINDAN (nokta-zamanli) verilir."""
    p, y = _as_arrays(p_up, y_up)
    ref = float(np.mean((base_rate - y) ** 2))
    return float(1.0 - brier(p, y) / ref) if ref > 0 else float('nan')


def _block_bootstrap_vals(fn, p_up, y_up, block: int = 20, n_boot: int = 1000, seed: int = 0):
    """Hareketli blok bootstrap tekrarlarinda metrik degerleri; yetersizse None."""
    p, y = _as_arrays(p_up, y_up)
    n = len(p)
    if n < 2 * block:
        return None
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=k)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        v = fn(p[idx], y[idx])
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) < n_boot // 2:
        return None
    return np.asarray(vals, dtype=np.float64)


def block_bootstrap_ci(fn, p_up, y_up, block: int = 20, n_boot: int = 1000, seed: int = 0,
                       alpha: float = 0.05):
    """Hareketli blok bootstrap ile metrigin (1 - alpha) yuzdelik araligi (zaman bagimliligi).

    Blok 20 gun bir tasarim varsayimi (hakem-finans I2 #2). None/NaN donduren tekrarlar atlanir.
    """
    vals = _block_bootstrap_vals(fn, p_up, y_up, block, n_boot, seed)
    if vals is None:
        return None
    return [float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))]


def block_bootstrap_pvalue(fn, p_up, y_up, null: float = 0.5, block: int = 20,
                           n_boot: int = 1000, seed: int = 0) -> Optional[float]:
    """Tek tarafli H0: metrik <= null icin yuzdelik bootstrap p-degeri, (1 + #{v <= null}) / (1 + B).

    Faz 9 P (hakem-finans I2 ikinci okuma not 1): 30 sembolde bilgi kapisi coklu karsilastirmaya
    (BH) gore okunur; GA'nin alt ucu yerine p-degeri gerekir. Yuzdelik yontem bir yaklasiklik
    (tasarim varsayimi); GA ile ayni tekrarlari kullanir (ayni tohum -> ayni karar).
    """
    vals = _block_bootstrap_vals(fn, p_up, y_up, block, n_boot, seed)
    if vals is None:
        return None
    return float((1 + np.sum(vals <= null)) / (1 + len(vals)))


def _accuracy(p, y):
    return float(np.mean((p >= 0.5) == (y == 1)))


def summary(p_up, y_up, n_bins: int = N_BINS, with_bins: bool = True,
            base_rate: Optional[float] = None, ci: bool = True) -> Dict[str, object]:
    """ECE, Brier, NLL, isabet, AUC; base_rate verilirse BSS; blok bootstrap GA'lari.

    `n_nonempty_bins`: ikili durumda guven >= 0,5 oldugundan 15 kutunun en fazla ~8'i dolar,
    p ~ 0,5'e cokunce 1-2 (hakem-finans I2 #3).
    """
    p, y = _as_arrays(p_up, y_up)
    bins = reliability(p, y, n_bins)
    out: Dict[str, object] = {'n': int(len(p)), 'ece': ece(p, y, n_bins), 'brier': brier(p, y),
                              'nll': nll(p, y), 'base_rate': float(y.mean()),
                              'accuracy': _accuracy(p, y), 'auc': auc(p, y),
                              'n_nonempty_bins': int(sum(1 for b in bins if b['n']))}
    if base_rate is not None:
        out['base_rate_ref'] = float(base_rate)
        out['brier_skill'] = brier_skill(p, y, base_rate)
    if ci:
        out['ece_ci'] = block_bootstrap_ci(lambda a, b: ece(a, b, n_bins), p, y)
        out['accuracy_ci'] = block_bootstrap_ci(_accuracy, p, y)
        out['auc_ci'] = block_bootstrap_ci(auc, p, y)
    if with_bins:
        out['reliability'] = bins
    return out


def informative(s: Dict[str, object], temperature_at_bound: bool = False) -> bool:
    """'Kalibre' sifati kapisi (hakem-finans I2 #1): T sinirda degil, BSS > 0, AUC GA'si 0,5'i dislar."""
    ci = s.get('auc_ci')
    bss = s.get('brier_skill')
    return bool(not temperature_at_bound and bss is not None and np.isfinite(bss) and bss > 0
                and ci is not None and ci[0] > 0.5)
