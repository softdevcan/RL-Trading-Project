"""Gozlem tanimi `marl-obs-v1` (tasarim §5; plan §3.2.3, K18, K26, S-obs).

K2: ortam ve canli karar AYNI saf fonksiyonlari cagirir. Ortam ozellikleri tum panel icin bir kez
hesaplar (`compute_raw_features`); canli karar `build_obs_at` ile panelin t'ye kadar kesilmis halinden
yalniz son satiri uretir. Iki yolun bit-es olmasi `tests/test_marl_obs.py` ile sinanir (ayni zamanda
B0: t sonrasi satirlar degisince t gozlemi degismez).

Ozellikler (hepsi olcekten bagimsiz; fiyat seviyesi, TL cinsinden MACD ve pay cinsinden hacim YOK):
  r1, r5, r20  log getiri (Cf_t / Cf_{t-k})            range  (H - L) / C
  gap          log(O_t / Cf_{t-1})                      vol20  r1'in 20 gunluk std'si
  rsi          RSI / 100 - 0,5                          cci    CCI / 200
  adx          ADX / 100 ('wilder'; legacy ADX = 0 KULLANILMAZ, `panel.build_arrays` denetler)
  macd_rel     MACD / C  (K26: TL cinsinden MACD testte %75 doygundu)
  vlog_z       log(1 + hacim)'in 60 gunluk nedensel kayan z'si (K26 hacim adayi; sermaye artirimi
               seviye kaydirmasina dayanikli)
  Turbulans YOK (K26: 30 sembolde Mahalanobis uzakligi tanh(./2)'yi %99,7 doyuruyordu).

Normalizasyon (Y12/Y13 dersi): her ozelligin havuzlanmis ortalama/std'si YALNIZ egitim satirlarindan
(`fit_norm_stats`), model meta'sina `norm_stats` olarak yazilir; z = clip((x - mu)/sd, -5, 5).
Gecersiz hucre (eksik gun, isinma, listelenmemis) ozelligi 0 ve `feat_ok` = 0 tasir; sessiz bir
varsayilana dusulmez.

Ajan i'nin yerel gozlemi (aktor), sirayla:
  [z ozellikleri (F), feat_ok, w_i, w_nakit, ajan kimligi one-hot (N),
   (grup one-hot (G), opsiyonel), (S-obs: p_up_cal - 0,5, z(pred_logret), z(disagreement), pred_ok),
   (paylasilan ek ozellikler (E), opsiyonel: rejim/makro, ayni deger tum ajanlara)]
Not: t+1 hucresinin gecerliligi gozleme KONMAZ (t+1 acilisinda bilinir; ileriye bakma olurdu).

Kuresel durum (MAPPO elestirmeni): [w (N+1), z ozelliklerinin kesitsel ortalamasi (F),
gecerli hucre orani, ek ozellikler (E)].
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from marl.accounting import fill_prices

OBS_SPEC_VERSION = 'marl-obs-v1'
FEATURES_V1 = ('r1', 'r5', 'r20', 'range', 'gap', 'vol20', 'rsi', 'cci', 'adx', 'macd_rel', 'vlog_z')
PRED_FIELDS = ('p_up_cal', 'pred_logret', 'model_disagreement')
REQUIRED_ARRAYS = ('open', 'high', 'low', 'close', 'volume', 'rsi', 'cci', 'adx', 'macd')
VOL_WIN = 20
VLOG_WIN = 60
VLOG_MIN_OBS = 40
WARMUP_ROWS = VLOG_WIN - 1
CLIP = 5.0


@dataclass(frozen=True)
class ObsSpec:
    n_agents: int
    use_pred: bool = False
    n_groups: int = 0
    n_extras: int = 0
    version: str = OBS_SPEC_VERSION
    features: tuple = FEATURES_V1
    clip: float = CLIP

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def local_dim(self) -> int:
        return (self.n_features + 3 + self.n_agents + self.n_groups
                + (4 if self.use_pred else 0) + self.n_extras)

    @property
    def state_dim(self) -> int:
        return (self.n_agents + 1) + self.n_features + 1 + self.n_extras

    def to_dict(self) -> dict:
        d = asdict(self)
        d['features'] = list(self.features)
        d['local_dim'] = self.local_dim
        d['state_dim'] = self.state_dim
        return d


def _lag_log(lc: np.ndarray, k: int) -> np.ndarray:
    x = np.full_like(lc, np.nan)
    x[k:] = lc[k:] - lc[:-k]
    return x


def _rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    if x.shape[0] >= win:
        out[win - 1:] = sliding_window_view(x, win, axis=0).std(axis=-1)
    return out


def _rolling_z_last(x: np.ndarray, win: int, min_obs: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    if x.shape[0] < win:
        return out
    v = sliding_window_view(x, win, axis=0)          # [T-win+1, N, win]
    fin = np.isfinite(v)
    cnt = fin.sum(axis=-1)
    v0 = np.where(fin, v, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        mu = v0.sum(axis=-1) / cnt
        sd = np.sqrt((np.where(fin, (v - mu[..., None]) ** 2, 0.0)).sum(axis=-1) / cnt)
        cur = x[win - 1:]
        z = np.where(sd > 0, (cur - mu) / sd, 0.0)
    z = np.where((cnt >= min_obs) & np.isfinite(cur), z, np.nan)
    out[win - 1:] = z
    return out


def compute_raw_features(arrays: dict) -> tuple[np.ndarray, np.ndarray]:
    """[T, N] diziler -> (ham ozellik [T, N, F], gecerli [T, N]). Satir t yalniz <= t satirlarina bagli."""
    missing = [k for k in REQUIRED_ARRAYS if k not in arrays]
    if missing:
        raise ValueError(f"eksik dizi: {missing}")
    px = fill_prices(arrays['open'], arrays['close'])
    valid = px['valid']
    cf, of = px['close'], px['open']
    h = np.where(valid, np.asarray(arrays['high'], dtype=np.float64), cf)
    l = np.where(valid, np.asarray(arrays['low'], dtype=np.float64), cf)
    with np.errstate(invalid='ignore', divide='ignore'):
        lc = np.log(cf)
        r1 = _lag_log(lc, 1)
        gap = np.full_like(cf, np.nan)
        gap[1:] = np.log(of[1:] / cf[:-1])
        rng = (h - l) / cf
        vol = np.asarray(arrays['volume'], dtype=np.float64)
        lv = np.where(valid & np.isfinite(vol) & (vol >= 0), np.log1p(np.where(vol >= 0, vol, 0.0)), np.nan)
        macd_rel = np.asarray(arrays['macd'], dtype=np.float64) / cf
    f = np.stack([
        r1, _lag_log(lc, 5), _lag_log(lc, 20), rng, gap, _rolling_std(r1, VOL_WIN),
        np.asarray(arrays['rsi'], dtype=np.float64) / 100.0 - 0.5,
        np.asarray(arrays['cci'], dtype=np.float64) / 200.0,
        np.asarray(arrays['adx'], dtype=np.float64) / 100.0,
        macd_rel, _rolling_z_last(lv, VLOG_WIN, VLOG_MIN_OBS),
    ], axis=-1)
    ok = valid & np.isfinite(f).all(axis=-1)
    return f, ok


def _pooled(x: np.ndarray, m: np.ndarray) -> tuple[float, float]:
    v = x[m]
    if v.size < 2:
        raise ValueError("normalizasyon icin egitim penceresinde yeterli gecerli hucre yok")
    sd = float(v.std())
    return float(v.mean()), (sd if sd > 0 else 1.0)


def fit_norm_stats(feat: np.ndarray, ok: np.ndarray, train_rows: np.ndarray,
                   pred: Optional[dict] = None, features=FEATURES_V1) -> dict:
    """Havuzlanmis mu/sd YALNIZ egitim satirlarindan (train_rows [T] bool). JSON'a yazilabilir."""
    rows = np.asarray(train_rows, dtype=bool)
    m = ok & rows[:, None]
    stats = {'version': OBS_SPEC_VERSION, 'n_cells': int(m.sum()), 'features': {}}
    for j, name in enumerate(features):
        mu, sd = _pooled(feat[..., j], m)
        stats['features'][name] = {'mean': mu, 'std': sd}
    if pred is not None:
        stats['pred'] = {}
        for name in ('pred_logret', 'model_disagreement'):
            x = np.asarray(pred[name], dtype=np.float64)
            mm = np.isfinite(x) & rows[:, None]
            mu, sd = _pooled(x, mm)
            stats['pred'][name] = {'mean': mu, 'std': sd}
    return stats


def normalize(feat: np.ndarray, ok: np.ndarray, stats: dict, features=FEATURES_V1,
              clip: float = CLIP) -> np.ndarray:
    mu = np.array([stats['features'][n]['mean'] for n in features])
    sd = np.array([stats['features'][n]['std'] for n in features])
    with np.errstate(invalid='ignore'):
        z = np.clip((feat - mu) / sd, -clip, clip)
    return np.where(ok[..., None], z, 0.0)


def pred_block(pred: dict, stats: dict, clip: float = CLIP) -> tuple[np.ndarray, np.ndarray]:
    """S-obs blogu [..., 3] + pred_ok [...]. Eksik tahmin maskelenir; 0,5'e dusulmez (§3.1.3)."""
    p = np.asarray(pred['p_up_cal'], dtype=np.float64)
    r = np.asarray(pred['pred_logret'], dtype=np.float64)
    d = np.asarray(pred['model_disagreement'], dtype=np.float64)
    okp = np.isfinite(p) & np.isfinite(r) & np.isfinite(d)
    ps = stats['pred']
    with np.errstate(invalid='ignore'):
        zr = np.clip((r - ps['pred_logret']['mean']) / ps['pred_logret']['std'], -clip, clip)
        zd = np.clip((d - ps['model_disagreement']['mean']) / ps['model_disagreement']['std'], -clip, clip)
    blk = np.stack([p - 0.5, zr, zd], axis=-1)
    return np.where(okp[..., None], blk, 0.0), okp


def local_obs(spec: ObsSpec, z_t: np.ndarray, ok_t: np.ndarray, w: np.ndarray,
              group_onehot: Optional[np.ndarray] = None, pred_t: Optional[np.ndarray] = None,
              pred_ok_t: Optional[np.ndarray] = None, extras_t: Optional[np.ndarray] = None) -> np.ndarray:
    """Tek tarih icin tum ajanlarin yerel gozlemi [N, local_dim] (float32)."""
    n = spec.n_agents
    parts = [z_t, ok_t[:, None].astype(np.float64), w[1:, None], np.full((n, 1), w[0]), np.eye(n)]
    if spec.n_groups:
        if group_onehot is None or group_onehot.shape != (n, spec.n_groups):
            raise ValueError(f"group_onehot [N, {spec.n_groups}] gerekli")
        parts.append(group_onehot)
    if spec.use_pred:
        if pred_t is None or pred_ok_t is None:
            raise ValueError("use_pred=True: pred_t ve pred_ok_t gerekli")
        parts += [pred_t, pred_ok_t[:, None].astype(np.float64)]
    if spec.n_extras:
        if extras_t is None or np.shape(extras_t) != (spec.n_extras,):
            raise ValueError(f"extras_t [{spec.n_extras}] gerekli")
        parts.append(np.broadcast_to(np.asarray(extras_t, dtype=np.float64), (n, spec.n_extras)))
    o = np.concatenate(parts, axis=1).astype(np.float32)
    assert o.shape == (n, spec.local_dim), (o.shape, spec.local_dim)
    return o


def global_state(spec: ObsSpec, z_t: np.ndarray, ok_t: np.ndarray, w: np.ndarray,
                 extras_t: Optional[np.ndarray] = None) -> np.ndarray:
    cnt = int(ok_t.sum())
    xs = z_t[ok_t].mean(axis=0) if cnt else np.zeros(spec.n_features)
    parts = [w, xs, [cnt / spec.n_agents]]
    if spec.n_extras:
        parts.append(np.asarray(extras_t, dtype=np.float64))
    return np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1) for p in parts]).astype(np.float32)


def build_obs_at(spec: ObsSpec, arrays_upto_t: dict, stats: dict, w: np.ndarray,
                 group_onehot=None, pred_upto_t: Optional[dict] = None, extras_t=None) -> np.ndarray:
    """K2 canli giris noktasi: panelin t'ye KADAR kesilmis hali -> t gozlemi [N, local_dim].

    Ortamin onceden hesapladigi gozlemle bit-es (test). Isinma icin en az WARMUP_ROWS + 1 satir ver.
    """
    feat, ok = compute_raw_features(arrays_upto_t)
    z = normalize(feat[-1:], ok[-1:], stats, spec.features, spec.clip)[0]
    pred_t = pred_ok_t = None
    if spec.use_pred:
        last = {k: np.asarray(v)[-1:] for k, v in pred_upto_t.items()}
        blk, okp = pred_block(last, stats, spec.clip)
        pred_t, pred_ok_t = blk[0], okp[0]
    return local_obs(spec, z, ok[-1], np.asarray(w, dtype=np.float64), group_onehot, pred_t, pred_ok_t,
                     extras_t)


def clip_report(feat: np.ndarray, ok: np.ndarray, stats: dict, rows: np.ndarray,
                features=FEATURES_V1, clip: float = CLIP) -> dict:
    """Gozlem sagligi (K18): ozellik basina kirpilan oran ve sabitlik, verilen satirlarda."""
    rows = np.asarray(rows, dtype=bool)
    out = {}
    for j, n in enumerate(features):
        mu, sd = stats['features'][n]['mean'], stats['features'][n]['std']
        m = ok & rows[:, None]
        x = (feat[..., j][m] - mu) / sd
        out[n] = {'clip_frac': float((np.abs(x) >= clip).mean()) if x.size else None,
                  'constant': bool(x.size and np.ptp(x) == 0), 'n': int(x.size)}
    return out
