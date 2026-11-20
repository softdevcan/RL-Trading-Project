"""Ikinci rejim tanimi: gizli Markov modelinin YALNIZ filtrelenmis olasiliklari (Faz 10 M6a; K10, K-M6-2).

Plan (Faz 9 M6, Faz 10 §3.1.3 S-reg): Hamilton (1989) rejim degisimli model, portfoyde Nystrup vd. (2018). Girdi gunluk
log getiri x 100 (birincil: XU100 / USDTRY_{t-1}, `evaluation.regimes.usd_index`). Model: rejime gore degisen sabit +
varyans (`statsmodels` MarkovRegression, trend='c', switching_variance=True).

Nedensellik (B0):
  - Parametreler capalarda yeniden uydurulur: capa a_i'de YALNIZ a_i'den onceki getirilerle (genisleyen pencere).
  - a_i <= t < a_{i+1} gunleri icin olasilik, a_i parametreleriyle Hamilton ILERI filtresinin t'deki degeridir
    (filtered_marginal_probabilities). Duzeltilmis (smoothed) olasilik t'den sonraki veriyi kullanir; KULLANILMAZ.
  - Durum sayisi K, ilk capada {2, 3} icinden BIC ile secilir ve sabit kalir (K-M6-2).
Kesme degismezligi: tam seride ve t'de kesilmis seride t'ye kadarki olasiliklar ayni capa takvimiyle bit-es
(`tests/test_regimes_hmm.py`).

Kanonik durum sirasi: capa basina sigma'ya gore artan (0 = en sakin). Uyumdan uyuma etiket yer degistirmesini (label
switching) boyle giderir. Ortalamanin sigma sirasiyla ters sirali olup olmadigi (sakin durum = en yuksek ortalama) her
capada olculur ve `fit_log`'a yazilir; bu bir tasarim varsayimi degil olcumdur.

Tasarim varsayimlari (M6 tasarimi §3, M6a): yeniden uyum her REFIT_EVERY = 63 seansta (P tablosuyla ayni aralik), ilk
capa cagiran tarafindan verilir (MARL icin panelin ilk seansi), ilk capadan sonra sicak baslangic (onceki parametreler).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REFIT_EVERY = 63
K_CANDIDATES = (2, 3)
MIN_FIT_OBS = 504          # ilk capadan once en az ~2 yil getiri
SCALE = 100.0              # getiri yuzde biriminde (optimizasyon kararliligi)
FIT_SEED = 0               # search_reps baslangiclari icin (deterministik uyum)
# Iki hucreli eslemenin birincil secimi (kullanici karari K-M6-5, 2026-11-20; M6 tasarimi §5). 'two_cell' K = 3'te dejenere.
PRIMARY_TWO_CELL = 'calm_vs_rest'
MAXITER = 500              # varsayilan 100'de ilk capa (2013-06-03, K=3) yakinsamiyordu (olculdu; llf farki ~0,5)


def log_returns(close: pd.Series) -> pd.Series:
    c = pd.Series(close, dtype=float).sort_index().dropna()
    if (c <= 0).any():
        raise ValueError("pozitif olmayan kapanis")
    return (np.log(c).diff().dropna() * SCALE).rename('r')


def _model(x: np.ndarray, k: int):
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    return MarkovRegression(x, k_regimes=k, trend='c', switching_variance=True)


def _state_moments(model, params: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    names = list(model.param_names)
    mu = np.array([params[names.index(f'const[{i}]')] for i in range(k)])
    sd = np.sqrt(np.array([params[names.index(f'sigma2[{i}]')] for i in range(k)]))
    return mu, sd


def fit_params(x: np.ndarray, k: int, start: Optional[np.ndarray] = None, search_reps: int = 5):
    """Uyum; (params, bic, mu, sd). Uyari ciktisi bastirilir (yakinsama uyarisi `converged` ile kaydedilir)."""
    m = _model(np.asarray(x, dtype=float), k)
    # statsmodels `search_reps` rastgele baslangiclari NumPy'in KURESEL ureticisinden ceker: tohumsuz uyum ayni veride
    # farkli sonuc verir (tekrar uretim + kesme degismezligi bozulur). Kuresel durum sabit tohumla kurulur, sonra geri yuklenir.
    state = np.random.get_state()
    np.random.seed(FIT_SEED)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if start is not None:
                res = m.fit(start_params=start, disp=False, maxiter=MAXITER)
            else:
                res = m.fit(disp=False, search_reps=search_reps, maxiter=MAXITER)
    finally:
        np.random.set_state(state)
    mu, sd = _state_moments(m, res.params, k)
    conv = bool(res.mle_retvals.get('converged', True)) if isinstance(res.mle_retvals, dict) else True
    return np.asarray(res.params), float(res.bic), mu, sd, conv


def select_k(x: np.ndarray, candidates: Sequence[int] = K_CANDIDATES) -> tuple[int, Dict[int, float]]:
    bics = {k: fit_params(x, k)[1] for k in candidates}
    return min(bics, key=bics.get), bics


def _filtered(x: np.ndarray, k: int, params: np.ndarray) -> np.ndarray:
    """Hamilton ileri filtresi [n, k]; t satiri yalniz x[:t+1]'e bagli."""
    m = _model(np.asarray(x, dtype=float), k)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = m.filter(params)
    p = np.asarray(res.filtered_marginal_probabilities)
    return p if p.shape[1] == k else p.T


@dataclass
class HMMRegimes:
    probs: pd.DataFrame                     # tarih x 'p0'..'p{K-1}' (kanonik sira: sigma artan)
    k: int
    bic_first: Dict[int, float]
    fit_log: List[dict] = field(default_factory=list)


def filtered_regimes(r: pd.Series, first_anchor: str, refit_every: int = REFIT_EVERY,
                     k: Optional[int] = None, end: Optional[str] = None) -> HMMRegimes:
    """Capa takvimi: ilk capa `first_anchor`'dan sonraki ilk seans, sonra her `refit_every` seans.

    r: tarih indeksli getiri (log_returns). end verilirse r o tarihten (dahil degil) once kesilir.
    """
    r = pd.Series(r, dtype=float).sort_index().dropna()
    if end is not None:
        r = r[r.index < pd.Timestamp(end)]
    idx = r.index
    first_pos = int(idx.searchsorted(pd.Timestamp(first_anchor)))
    if first_pos < MIN_FIT_OBS:
        raise ValueError(f"ilk capadan once {first_pos} getiri var, en az {MIN_FIT_OBS} gerekli")
    if first_pos >= len(idx):
        raise ValueError("ilk capa serinin sonundan sonra")
    x = r.values
    bic_first: Dict[int, float] = {}
    if k is None:
        k, bic_first = select_k(x[:first_pos])
    anchors = list(range(first_pos, len(idx), refit_every))
    out = np.full((len(idx), k), np.nan)
    log: List[dict] = []
    prev = None
    for j, a in enumerate(anchors):
        b = anchors[j + 1] if j + 1 < len(anchors) else len(idx)
        params, bic, mu, sd, conv = fit_params(x[:a], k, start=prev)
        if not np.all(np.isfinite(params)):          # sicak baslangic dagildiysa soguk dene
            params, bic, mu, sd, conv = fit_params(x[:a], k)
        prev = params
        order = np.argsort(sd)                       # kanonik: sigma artan
        p = _filtered(x[:b], k, params)[a:b][:, order]
        out[a:b] = p
        log.append({'anchor': str(idx[a].date()), 'n_fit': int(a), 'bic': bic, 'converged': conv,
                    'mu_sorted': [float(v) for v in mu[order]], 'sd_sorted': [float(v) for v in sd[order]],
                    'mean_desc_with_sd': bool(np.all(np.diff(mu[order]) < 0))})
    probs = pd.DataFrame(out, index=idx, columns=[f'p{i}' for i in range(k)])
    return HMMRegimes(probs=probs.iloc[first_pos:], k=k, bic_first=bic_first, fit_log=log)


def hmm_labels(h: HMMRegimes) -> pd.DataFrame:
    """Etiketler: 'state' = argmax (kanonik sira); 'calm_vs_rest' = en sakin durum olasiligi >= 0,5 ise 'calm';
    'two_cell' = en yuksek sigma durumu olasiligi >= 0,5 ise 'bear_like',
    aksi halde 'bull_like' (K-M6-1 iki hucreli okumanin ikinci tanim karsiligi; adlandirma `mean_desc_with_sd`
    olcumune dayanir, olcum tutmazsa adlar yorum tasimaz)."""
    p = h.probs
    hi = p[f'p{h.k - 1}']
    calm = p['p0']
    return pd.DataFrame({'state': p.values.argmax(axis=1), 'p_high_vol': hi, 'p_calm': calm,
                         'two_cell': np.where(hi >= 0.5, 'bear_like', 'bull_like'),
                         # Alternatif iki hucre (M6 tasarimi §3.1): en sakin durum ('calm') <-> geri kalanlar. K = 3'te en yuksek
                         # sigma durumu nadir bir kriz durumu oldugu icin 'two_cell' neredeyse hep 'bull_like' cikiyor (olculdu).
                         'calm_vs_rest': np.where(calm >= 0.5, 'calm', 'not_calm')}, index=p.index)


def load_primary_returns() -> pd.Series:
    """Birincil girdi: XU100 / USDTRY_{t-1} gunluk log getirisi x 100."""
    from data.index_fetcher import DEFAULT_INDEX_CSV, DEFAULT_USDTRY_CSV, FX_JUMP_LIMIT, load_index_close
    from evaluation.regimes import usd_index
    return log_returns(usd_index(load_index_close(DEFAULT_INDEX_CSV),
                                 load_index_close(DEFAULT_USDTRY_CSV, jump_limit=FX_JUMP_LIMIT)))
