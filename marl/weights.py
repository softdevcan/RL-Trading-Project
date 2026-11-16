"""Logit -> hedef agirlik ve risk projeksiyonu (tasarim §3; plan §3.2.2, §3.3 K6).

Ortam (kesirli ve tam sayi modu) ve canli karar AYNI `target_weights` + `project` cifti ile hedef
uretir. Iki fonksiyon da ogrenmeyen sabit islemlerdir; merkezi bir dagitici ajan degildir (plan
§3.2.2). Agirlik vektorlerinde indeks 0 NAKIT, 1..N hisseler (panel/meta `symbols` sirasi).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

BASE_BOUND = 5.0


def logit_bound(n_agents: int) -> float:
    """Ajan logit siniri B = 5 + ln N (kullanici karari 2026-11-16, tasarim §3.1).

    Nakit logiti sabit z0 = 0. Tum ajanlar -B'deyken nakit agirligi 1/(1 + N e^-B) = 1/(1 + e^-5)
    ~ %99,33 — N'den bagimsiz. Sabit B=5 ile N=30'da tavan %83'tu (M1 olcumu, plan §3.2.7).
    """
    if n_agents < 1:
        raise ValueError(f"n_agents >= 1 olmali: {n_agents}")
    return BASE_BOUND + float(np.log(n_agents))


def target_weights(z, bound: float) -> np.ndarray:
    """u = softmax([0, clip(z, -B, B)]). Donus uzunlugu N+1 (0 = nakit)."""
    z = np.clip(np.asarray(z, dtype=np.float64).reshape(-1), -bound, bound)
    logits = np.concatenate([[0.0], z])
    e = np.exp(logits - logits.max())
    return e / e.sum()


@dataclass(frozen=True)
class Constraints:
    """Risk projeksiyonu parametreleri (K6). Model meta'sina `risk_projection` olarak yazilir.

    band: |w*_i - w~_i| < band ise islem yok (donusum bandi). 0 = kapali.
    max_position: tek hisse ust siniri (0..1]. 1 = kapali.
    max_group: grup (sektor) ust siniri (0..1]. 1 = kapali; `groups` verilmezse yok sayilir.
    """
    band: float = 0.0
    max_position: float = 1.0
    max_group: float = 1.0

    def __post_init__(self):
        if not (0.0 <= self.band < 1.0):
            raise ValueError(f"band [0, 1) olmali: {self.band}")
        for k in ('max_position', 'max_group'):
            v = getattr(self, k)
            if not (0.0 < v <= 1.0):
                raise ValueError(f"{k} (0, 1] olmali: {v}")

    def to_dict(self) -> dict:
        return asdict(self)


def project(u, w_drift, tradable, constraints: Constraints = Constraints(),
            groups: Optional[np.ndarray] = None) -> tuple[np.ndarray, dict]:
    """Ham hedef u'yu uygulanabilir hedef w*'a tasir (tasarim §3.2). Sira sabittir:

      a. Dondurma: islem goremeyen hisse (tradable=False) kaymis agirligini korur; kalan kutle nakit +
         islem gorenlere u oraninda dagitilir.
      b. Bant: |w*_i - w~_i| < band olan islem goren hisse w~_i'de kalir. Hisse toplami 1'i asarsa
         bantta kalmayan (islem goren) hisseler orantili kucultulur.
      c. Pozisyon siniri: islem goren w*_i <= max_position; fazlasi NAKDE.
      d. Grup siniri: grubun islem goren kismi orantili kucultulur; fazlasi NAKDE. Dondurulmus kisim
         sinirdan dusulur (o gun satilamaz).
    Sinirlar yalniz kucultur ve fazlayi nakde verir (yeniden dagitim yok): bu bir tasarim
    varsayimidir; hedef her zaman uzun-yalniz, toplam 1, nakit >= 0. Dondurulmus hisse siniri
    asabilir (satilamaz); `frozen_over_cap` ile raporlanir.

    Args:
        u: [N+1] ham hedef (`target_weights`).
        w_drift: [N+1] yurutme anindaki kaymis agirlik (0 = nakit).
        tradable: [N] bool, yurutme gunu hucresi gecerli mi.
        groups: [N] tam sayi grup kimligi (or. E2 sektoru); None -> grup siniri yok.
    Returns:
        (w* [N+1], rapor)
    """
    u = np.asarray(u, dtype=np.float64)
    wd = np.asarray(w_drift, dtype=np.float64)
    tr = np.asarray(tradable, dtype=bool)
    n = tr.shape[0]
    if u.shape != (n + 1,) or wd.shape != (n + 1,):
        raise ValueError(f"u ve w_drift [N+1={n + 1}] olmali: {u.shape}, {wd.shape}")
    c = constraints
    w = np.zeros(n + 1)
    stocks_wd = wd[1:]

    # a. dondurma
    frozen = ~tr
    w[1:][frozen] = stocks_wd[frozen]
    free_mass = max(0.0, 1.0 - float(stocks_wd[frozen].sum()))
    live = np.concatenate([[True], tr])
    denom = float(u[live].sum())
    if denom > 0:
        w[live] = free_mass * u[live] / denom
    else:
        w[0] = free_mass

    # b. bant
    banded = np.zeros(n, dtype=bool)
    if c.band > 0:
        banded = tr & (np.abs(w[1:] - stocks_wd) < c.band)
        w[1:][banded] = stocks_wd[banded]
        fixed = frozen | banded
        traded = ~fixed
        s_fixed = float(w[1:][fixed].sum())
        s_traded = float(w[1:][traded].sum())
        if s_fixed + s_traded > 1.0 and s_traded > 0:
            w[1:][traded] *= max(0.0, 1.0 - s_fixed) / s_traded

    # c. pozisyon siniri
    if c.max_position < 1.0:
        over = tr & (w[1:] > c.max_position)
        w[1:][over] = c.max_position

    # d. grup siniri
    if groups is not None and c.max_group < 1.0:
        g = np.asarray(groups)
        if g.shape != (n,):
            raise ValueError(f"groups [N={n}] olmali: {g.shape}")
        for gid in np.unique(g):
            m = g == gid
            t_sum = float(w[1:][m & tr].sum())
            f_sum = float(w[1:][m & frozen].sum())
            if t_sum + f_sum > c.max_group and t_sum > 0:
                w[1:][m & tr] *= max(0.0, c.max_group - f_sum) / t_sum

    w[0] = 1.0 - float(w[1:].sum())
    if w[0] < 0:  # yalniz yuvarlama: dondurulmus kutle 1'i asamaz
        if w[0] < -1e-12:
            raise AssertionError(f"projeksiyon nakdi eksiye dusurdu: {w[0]}")
        w[0] = 0.0
    report = {
        'n_frozen': int(frozen.sum()),
        'n_banded': int(banded.sum()),
        'frozen_over_cap': int((frozen & (w[1:] > c.max_position + 1e-12)).sum()),
    }
    return w, report
