"""A3 — Iki stratejinin Sharpe oranlari farki testi (Ledoit & Wolf 2008).

Kaynak: Ledoit, O. & Wolf, M. (2008). Robust performance hypothesis testing
with the Sharpe ratio. Journal of Empirical Finance 15(5), 850-859.
(arsiv: docs/reference/papers/G_evaluation/2008_LedoitWolf_SharpeTest_WP.pdf,
§3.1 HAC, §3.2 bootstrap — PDF'ler git disi; arsiv her calisma kopyasinda yok,
uygulama Faz 9 planindaki tam metin okumasina dayaniyor)

Boyut olcumleri (H0, GARCH, esit Sharpe): tests/test_eval_stats.py ciktisi;
Faz 9 §9 A3 satiri. Blok boyu duyarliligi T=500, rho=0,9, b in {2, 5, 10}
icin ayni testte olculur.

Gosterim (makaleyle ayni): u = (mu_i, mu_n, gamma_i, gamma_n), gamma = E[r^2].
  Delta = f(u) = mu_i / sqrt(gamma_i - mu_i^2) - mu_n / sqrt(gamma_n - mu_n^2)
  s(Delta^) = sqrt(grad' Psi^ grad / T)
  y_t = (r_ti - mu_i, r_tn - mu_n, r_ti^2 - gamma_i, r_tn^2 - gamma_n)

§3.1 HAC: Psi^ prewhitened (VAR(1), Andrews & Monahan 1992) Quadratic-Spectral
cekirdekle; bant genisligi Andrews (1991) AR(1) kurali; T/(T-4) kucuk orneklem
carpani.

§3.2 Bootstrap: studentize DAIRESEL BLOK bootstrap. Ozgun veride standart hata
HAC'tan; bootstrap verisinde Gotze & Kunsch (1996) "dogal" blok tahmincisi
  Psi* = (1/l) sum_j zeta_j zeta_j',  zeta_j = b^(-1/2) sum_{t in blok j} y*_t
p = (#{ |Delta* - Delta^| / s(Delta*) >= |Delta^| / s(Delta^) } + 1) / (M + 1).

Blok boyu b: makaledeki kalibrasyon yerine SABIT b (varsayilan 5) — tasarim
varsayimi; cagiran duyarlilik icin farkli b ile tekrar kosmalidir.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ----------------------------------------------------------------------------
# Yardimcilar
# ----------------------------------------------------------------------------

def _u_hat(ri: np.ndarray, rn: np.ndarray) -> np.ndarray:
    return np.array([ri.mean(), rn.mean(), (ri ** 2).mean(), (rn ** 2).mean()])


def _delta(u: np.ndarray) -> float:
    return float(u[0] / np.sqrt(u[2] - u[0] ** 2) - u[1] / np.sqrt(u[3] - u[1] ** 2))


def _grad(u: np.ndarray) -> np.ndarray:
    vi = u[2] - u[0] ** 2
    vn = u[3] - u[1] ** 2
    return np.array([
        u[2] / vi ** 1.5,
        -u[3] / vn ** 1.5,
        -0.5 * u[0] / vi ** 1.5,
        0.5 * u[1] / vn ** 1.5,
    ])


def _y(ri: np.ndarray, rn: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.column_stack([ri - u[0], rn - u[1], ri ** 2 - u[2], rn ** 2 - u[3]])


def _qs_kernel(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    nz = x != 0
    z = 6.0 * np.pi * x[nz] / 5.0
    out[nz] = 25.0 / (12.0 * np.pi ** 2 * x[nz] ** 2) * (np.sin(z) / z - np.cos(z))
    return out


def _andrews_bandwidth_qs(v: np.ndarray) -> float:
    """Andrews (1991) AR(1) kurali, QS cekirdegi: S_T = 1.3221 (alpha(2) T)^(1/5)."""
    T, k = v.shape
    num = den = 0.0
    for a in range(k):
        x, x1 = v[1:, a], v[:-1, a]
        denom = float(np.dot(x1, x1))
        rho = float(np.dot(x, x1) / denom) if denom > 0 else 0.0
        rho = float(np.clip(rho, -0.97, 0.97))
        sig2 = float(np.mean((x - rho * x1) ** 2))
        num += 4.0 * rho ** 2 * sig2 ** 2 / (1.0 - rho) ** 8
        den += sig2 ** 2 / (1.0 - rho) ** 4
    alpha2 = num / den if den > 0 else 0.0
    return float(1.3221 * (alpha2 * T) ** 0.2)


def hac_psi(y: np.ndarray, prewhiten: bool = True) -> np.ndarray:
    """Prewhitened QS HAC kestirimi (k x k), T/(T-k) carpani dahil."""
    T, k = y.shape
    A = np.zeros((k, k))
    v = y
    if prewhiten:
        X, Y = y[:-1], y[1:]
        A = np.linalg.lstsq(X, Y, rcond=None)[0].T          # y_t = A y_{t-1} + v_t
        # Andrews & Monahan (1992): tekil degerleri 0.97'de kirp (yakin birim kok)
        U, s, Vt = np.linalg.svd(A)
        if s.max() > 0.97:
            A = U @ np.diag(np.minimum(s, 0.97)) @ Vt
        v = Y - X @ A.T
    n = len(v)
    S = _andrews_bandwidth_qs(v)
    psi_v = v.T @ v / n
    if S > 0:
        # QS cekirdegi sonlu destekli degil; agirlik |w| < 1e-8 olana ve
        # 50*S'yi gecene kadar topla (kuyruk x^-2 hizinda sonuyor).
        for j in range(1, n):
            wj = float(_qs_kernel(np.array([j / S]))[0])
            if j > 50 * S and abs(wj) < 1e-8:
                break
            g = v[j:].T @ v[:-j] / n
            psi_v += wj * (g + g.T)
    if prewhiten:
        inv = np.linalg.inv(np.eye(k) - A)
        psi = inv @ psi_v @ inv.T
    else:
        psi = psi_v
    return psi * T / (T - k)


def _clean_pair(r1, r2):
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    if r1.shape != r2.shape:
        raise ValueError("Getiri serileri ayni tarihlerde, ayni uzunlukta olmali")
    m = np.isfinite(r1) & np.isfinite(r2)
    if m.sum() < 30:
        raise ValueError(f"En az 30 eslesmis gozlem gerekli, {int(m.sum())} var")
    return r1[m], r2[m]


# ----------------------------------------------------------------------------
# Testler
# ----------------------------------------------------------------------------

def sharpe_diff_hac(r1, r2, prewhiten: bool = True) -> dict:
    """§3.1 — HAC standart hatasi ile iki yonlu test (H0: SR1 = SR2)."""
    ri, rn = _clean_pair(r1, r2)
    T = len(ri)
    u = _u_hat(ri, rn)
    delta = _delta(u)
    g = _grad(u)
    psi = hac_psi(_y(ri, rn, u), prewhiten=prewhiten)
    se = float(np.sqrt(g @ psi @ g / T))
    z = delta / se
    return {
        'delta': delta,              # donem basi Sharpe farki
        'se': se,
        'stat': float(z),
        'p_value': float(2.0 * stats.norm.sf(abs(z))),
        'n': T,
        'method': 'HAC (prewhitened QS)' if prewhiten else 'HAC (QS)',
    }


def _natural_se_batch(Yb: np.ndarray, g: np.ndarray, b: int) -> np.ndarray:
    """Gotze-Kunsch dogal standart hata, M bootstrap ornegi icin vektorel.

    Yb: (M, l*b, 4) bootstrap y* (kendi ortalamasina gore merkezlenmis).
    g : (M, 4) her ornegin gradyani.
    """
    M, L, k = Yb.shape
    l = L // b
    zeta = Yb[:, : l * b].reshape(M, l, b, k).sum(axis=2) / np.sqrt(b)   # (M, l, k)
    psi = np.einsum('mlk,mlj->mkj', zeta, zeta) / l                      # (M, k, k)
    var = np.einsum('mk,mkj,mj->m', g, psi, g) / L
    return np.sqrt(np.maximum(var, 1e-300))


def sharpe_diff_bootstrap(
    r1, r2, block_size: int = 5, n_boot: int = 4999, seed=None, prewhiten: bool = True
) -> dict:
    """§3.2 — studentize dairesel blok bootstrap (H0: SR1 = SR2, iki yonlu)."""
    ri, rn = _clean_pair(r1, r2)
    T = len(ri)
    b = int(block_size)
    if b < 1 or b > T // 2:
        raise ValueError("block_size 1 ile T/2 arasinda olmali")
    base = sharpe_diff_hac(ri, rn, prewhiten=prewhiten)
    delta, se = base['delta'], base['se']

    rng = np.random.default_rng(seed)
    Tb = (T // b) * b                       # tam bloklar: bootstrap ornegi l*b uzunlukta
    l = Tb // b
    data = np.column_stack([ri, rn])
    d_boot = np.empty(n_boot)
    chunk = max(1, int(2_000_000 // max(Tb, 1)))   # bellek: chunk x Tb x 2
    for s in range(0, n_boot, chunk):
        m = min(chunk, n_boot - s)
        starts = rng.integers(0, T, size=(m, l))
        idx = ((starts[:, :, None] + np.arange(b)[None, None, :]) % T).reshape(m, Tb)
        boot = data[idx]                    # (m, Tb, 2) — dairesel bloklar
        bi, bn = boot[..., 0], boot[..., 1]
        U = np.stack([bi.mean(1), bn.mean(1), (bi ** 2).mean(1), (bn ** 2).mean(1)], axis=1)
        vi = U[:, 2] - U[:, 0] ** 2
        vn = U[:, 3] - U[:, 1] ** 2
        d_star = U[:, 0] / np.sqrt(vi) - U[:, 1] / np.sqrt(vn)
        G = np.stack([U[:, 2] / vi ** 1.5, -U[:, 3] / vn ** 1.5,
                      -0.5 * U[:, 0] / vi ** 1.5, 0.5 * U[:, 1] / vn ** 1.5], axis=1)
        Yb = np.stack([bi - U[:, [0]], bn - U[:, [1]],
                       bi ** 2 - U[:, [2]], bn ** 2 - U[:, [3]]], axis=2)
        se_star = _natural_se_batch(Yb, G, b)
        d_boot[s:s + m] = np.abs(d_star - delta) / se_star

    d_obs = abs(delta) / se
    p = (np.sum(d_boot >= d_obs) + 1.0) / (n_boot + 1.0)
    return {
        'delta': delta,
        'se': se,
        'stat': float(delta / se),
        'p_value': float(p),
        'n': T,
        'block_size': b,
        'n_boot': int(n_boot),
        'method': 'studentized circular block bootstrap',
    }
