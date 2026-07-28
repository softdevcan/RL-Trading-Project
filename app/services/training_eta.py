"""
Egitim suresi tahmini (ETA)

Iki soruyu cevaplar:
  1. Egitim baslamadan once: "bu ayarlarla ne kadar surer?"
  2. Egitim devam ederken: "ne kadar kaldi, saat kacta biter?"

Yaklasim olcum-oncelikli: tahmin, ayni makinede tamamlanmis gercek kosumlardan
ogrenilir. Gecmis yoksa yerlesik varsayilan katsayilar kullanilir ve bu durum
`confidence='default'` ile acikca isaretlenir — ilk kosum bittiginde katsayilar
gercek olcumle degisir.

Maliyet modeli:

    toplam = hazirlik + ogrenme + degerlendirme
    ogrenme = total_timesteps x adim_maliyeti(algoritma, sembol_sayisi)

`adim_maliyeti` (algoritma, sembol_sayisi) kovasindan okunur. Kova yoksa en yakin
sembol sayisina sahip kovadan dogrusal olceklenir: env.step() sembol sayisiyla
dogrusal buyur (her sembol icin lookup + islem dongusu), politika agi girdisi de
oyle. Farkli bir sembol sayisina olcekleme yaklasiktir ve `confidence='scaled'`
ile isaretlenir.

Canli tahmin, ogrenme fazi basladiktan sonra gozlenen gercek hizi (adim/sn)
kullanir; onceki tahmin yalnizca isinma penceresinde (ilk birkac yuz adim)
gecerlidir.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from statistics import median
from typing import Any, Dict, List, Optional

from app.auth import workspace as ws

logger = logging.getLogger(__name__)

HISTORY_FILENAME = "training_eta_history.json"

# Ayni makinede tutulan kayit sayisi ustu. Eski kosumlar donanim/kod degisince
# yaniltici olur; yenilerin agirligi bu sinirla korunur.
MAX_HISTORY = 200

# Gozlenen hizin onceki tahminin yerini almasi icin gereken en az ilerleme.
# Bunun altinda SB3'un ilk rollout'u henuz dolmadigi icin hiz yaniltici olur.
WARMUP_STEPS = 200
WARMUP_FRACTION = 0.01

# Yerlesik varsayilanlar: gelistirme makinesinde GERCEK egitim rotasi uzerinden
# olculmus saniye/timestep degerleri (30 sembollu panel, SB3'un sectigi cihaz).
# Yalnizca gecmis kayit yokken kullanilir; ilk gercek kosum bunlarin yerini alir.
#
# Olcum: scripts/benchmarking/measure_training_eta_defaults.py --steps 10000,25000
# Kucuk N'de (4k/8k) ayni olcum kosumdan kosuma ~2x oynadi; sabit maliyet ve
# soguk baslangic egimi bozuyor. Buyuk N'de dort algoritma da ~0.0028'de
# birlesiyor — bu konfigurasyonda maliyeti env adimi belirliyor, algoritmanin
# gradient adimi degil. Bu yuzden degerler birbirine yakin.
_DEFAULT_STEP_COST: Dict[str, float] = {
    "ppo": 0.002917,
    "a2c": 0.002673,
    "td3": 0.002784,
    "sac": 0.003135,
}
_DEFAULT_STEP_COST_SYMBOLS = 30

# Hazirlik + sabit maliyet: import, CSV yukleme, split, env kurulumu.
# Olculen sabit terim 6.5-9.8s araliginda; ortasi alindi.
_DEFAULT_SETUP_SECONDS = 8.0
# Degerlendirme: test setinde tek gecis. 30 sembolde 1011 adim ~0.19s olculdu;
# metrik hesabi ve model kaydi icin pay birakilarak yukari yuvarlandi.
_DEFAULT_EVAL_SECONDS_PER_SYMBOL = 0.05


# ── Bicimlendirme ─────────────────────────────────────────────────────────────

def humanize(seconds: Optional[float]) -> str:
    """Saniyeyi 'dk/sn' metnine cevir. None/negatif icin '—'."""
    if seconds is None or seconds < 0:
        return "—"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} sn"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} dk {sec:02d} sn" if sec else f"{minutes} dk"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} sa {minutes:02d} dk"


def _finish_clock(eta_seconds: Optional[float]) -> Optional[str]:
    """Tahmini bitis saati (HH:MM). ETA yoksa None."""
    if eta_seconds is None or eta_seconds < 0:
        return None
    return (datetime.now() + timedelta(seconds=eta_seconds)).strftime("%H:%M")


# ── Gecmis kayitlari ──────────────────────────────────────────────────────────

def history_path(user_id: Optional[str] = None) -> str:
    return os.path.join(ws.results_dir(user_id), HISTORY_FILENAME)


def load_history(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    path = history_path(user_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"ETA gecmisi okunamadi ({exc}) — bos kabul ediliyor")
        return []


def record_run(
    *,
    algorithm: str,
    phase: int,
    n_symbols: int,
    total_timesteps: int,
    setup_seconds: float,
    learn_seconds: float,
    eval_seconds: float,
    total_seconds: float,
    device: str = "cpu",
    user_id: Optional[str] = None,
) -> None:
    """Tamamlanmis bir kosumu gecmise yaz. Hata halinde egitimi bozmaz."""
    if total_timesteps <= 0 or learn_seconds <= 0:
        return

    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "algorithm": (algorithm or "").lower(),
        "phase": phase,
        "n_symbols": n_symbols,
        "total_timesteps": total_timesteps,
        "device": device,
        "setup_seconds": round(setup_seconds, 3),
        "learn_seconds": round(learn_seconds, 3),
        "eval_seconds": round(eval_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "step_cost": learn_seconds / total_timesteps,
    }

    try:
        history = load_history(user_id)
        history.append(record)
        history = history[-MAX_HISTORY:]
        path = history_path(user_id)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(history, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        logger.info(
            f"ETA gecmisine yazildi: {record['algorithm']} {n_symbols} sembol "
            f"{total_timesteps} timestep -> {humanize(total_seconds)}"
        )
    except OSError as exc:
        logger.warning(f"ETA gecmisi yazilamadi ({exc}) — tahmin etkilenmez")


# ── On tahmin ─────────────────────────────────────────────────────────────────

def _bucket(history: List[Dict], algorithm: str, n_symbols: int) -> List[Dict]:
    return [r for r in history
            if r.get("algorithm") == algorithm and r.get("n_symbols") == n_symbols]


def _nearest_bucket(history: List[Dict], algorithm: str, n_symbols: int) -> List[Dict]:
    """Ayni algoritmanin en yakin sembol sayisina sahip kovasi."""
    same_algo = [r for r in history if r.get("algorithm") == algorithm and r.get("n_symbols")]
    if not same_algo:
        return []
    nearest = min(same_algo, key=lambda r: abs(r["n_symbols"] - n_symbols))["n_symbols"]
    return [r for r in same_algo if r["n_symbols"] == nearest]


def estimate(
    algorithm: str,
    phase: int,
    total_timesteps: int,
    n_symbols: Optional[int] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Egitim baslamadan once toplam sureyi tahmin et.

    Returns:
        total_seconds, setup_seconds, learn_seconds, eval_seconds,
        confidence ('measured' | 'scaled' | 'default'), sample_size, source (metin)
    """
    algorithm = (algorithm or "ppo").lower()
    if n_symbols is None:
        n_symbols = symbols_for_phase(phase)

    history = load_history(user_id)
    exact = _bucket(history, algorithm, n_symbols)

    if exact:
        step_cost = median(r["step_cost"] for r in exact)
        confidence = "measured"
        sample = len(exact)
        source = f"{sample} benzer kosumdan olculdu"
    else:
        near = _nearest_bucket(history, algorithm, n_symbols)
        if near:
            ref_symbols = near[0]["n_symbols"]
            step_cost = median(r["step_cost"] for r in near) * (n_symbols / ref_symbols)
            confidence = "scaled"
            sample = len(near)
            source = f"{ref_symbols} sembollu {sample} kosumdan olceklendi"
        else:
            base = _DEFAULT_STEP_COST.get(algorithm, _DEFAULT_STEP_COST["ppo"])
            step_cost = base * (n_symbols / _DEFAULT_STEP_COST_SYMBOLS)
            confidence = "default"
            sample = 0
            source = "varsayilan katsayi (bu makinede henuz kosum yok)"

    learn_seconds = step_cost * total_timesteps

    if exact or _nearest_bucket(history, algorithm, n_symbols):
        pool = exact or _nearest_bucket(history, algorithm, n_symbols)
        setup_seconds = median(r.get("setup_seconds", _DEFAULT_SETUP_SECONDS) for r in pool)
        eval_seconds = median(r.get("eval_seconds", 0.0) for r in pool)
    else:
        setup_seconds = _DEFAULT_SETUP_SECONDS
        eval_seconds = _DEFAULT_EVAL_SECONDS_PER_SYMBOL * n_symbols

    total = setup_seconds + learn_seconds + eval_seconds
    return {
        "total_seconds": total,
        "setup_seconds": setup_seconds,
        "learn_seconds": learn_seconds,
        "eval_seconds": eval_seconds,
        "step_cost": step_cost,
        "confidence": confidence,
        "sample_size": sample,
        "source": source,
        "n_symbols": n_symbols,
        "total_text": humanize(total),
    }


_symbol_count_cache: Dict[str, int] = {}


def symbols_for_phase(phase: int) -> int:
    """Egitimin gercekte kac sembol goreceginin tahmini.

    Dikkat: `get_symbols(phase)` DEGILDIR. Egitim rotasi o listeyi yalnizca veri
    *cekerken* kullanir; egitim `stock_data_with_indicators.csv`'nin tamamiyla
    yapilir (bkz. `_run_training_inner` -> `load_data` + `split_data`, sembol
    filtresi yok). Faz 1 secilse bile panel 30 sembolluyse 30 sembolle egitilir
    ve sure buna gore olceklenir.

    Sembol kolonu tek basina okunur ve dosya mtime'ina gore onbelleklenir —
    pano bunu form her degistiginde cagirir.
    """
    path = os.path.join("data", "bist", "stock_data_with_indicators.csv")
    try:
        key = f"{path}:{os.path.getmtime(path)}"
        if key not in _symbol_count_cache:
            import pandas as pd
            _symbol_count_cache.clear()
            _symbol_count_cache[key] = int(pd.read_csv(path, usecols=["symbol"])["symbol"].nunique())
        return _symbol_count_cache[key]
    except Exception as exc:
        logger.debug(f"Sembol sayisi panelden okunamadi ({exc}) — faz varsayilanina dusuluyor")
        try:
            from data.bist30_symbols import get_symbols
            return len(get_symbols(phase=phase))
        except Exception:
            return 5 if phase == 1 else 30


# ── Canli tahmin ──────────────────────────────────────────────────────────────

def live_eta(state: Dict[str, Any]) -> Dict[str, Any]:
    """Devam eden egitim icin kalan sureyi tahmin et.

    `state` icinde beklenenler (hepsi opsiyonel, eksikse tahmin duser):
        current_step, total_steps, estimate, run_start_ts, learn_start_ts, phase_name
    """
    out: Dict[str, Any] = {
        "eta_seconds": None,
        "eta_text": "—",
        "finish_at": None,
        "steps_per_sec": None,
        "estimated_total_seconds": None,
        "eta_source": "unknown",
    }

    prior = state.get("estimate") or {}
    total_steps = state.get("total_steps") or 0
    current_step = state.get("current_step") or 0
    now = time.time()

    run_start = state.get("run_start_ts")
    elapsed = (now - run_start) if run_start else None
    phase_name = state.get("phase_name") or "preparing"

    if phase_name == "completed":
        out["eta_seconds"] = 0.0
        out["eta_text"] = "tamamlandi"
        out["eta_source"] = "completed"
        return out

    learn_start = state.get("learn_start_ts")
    learn_end = state.get("learn_end_ts")
    prior_setup = prior.get("setup_seconds", 0.0)
    prior_learn = prior.get("learn_seconds", 0.0)
    prior_eval = prior.get("eval_seconds", 0.0)

    rate = None
    if learn_start and current_step > 0:
        rate = current_step / max((learn_end or now) - learn_start, 1e-6)

    def _emit(eta, source):
        out.update({
            "eta_seconds": eta,
            "eta_text": humanize(eta),
            "finish_at": _finish_clock(eta),
            "steps_per_sec": rate,
            "estimated_total_seconds": (elapsed + eta) if elapsed is not None else None,
            "eta_source": source,
        })
        return out

    # On tahminde ise yarar bir sey var mi? Yoksa ETA'yi 0 gostermek yanlis
    # olur — "bitti" demek olurdu; dogrusu "bilinmiyor" (None).
    has_prior = bool(prior.get("total_seconds") or prior_learn or prior_eval)

    if phase_name == "evaluating":
        # Ogrenme bitti; kalan yalnizca test seti gecisi. Bunu 0 gostermek
        # "bitti ama bitmedi" hissi yaratirdi.
        if not has_prior:
            return out
        spent = (now - learn_end) if learn_end else 0.0
        return _emit(max(prior_eval - spent, 0.0), "measured" if learn_end else "prior")

    if phase_name == "training":
        warmup = max(WARMUP_STEPS, int(total_steps * WARMUP_FRACTION))
        if rate and current_step >= warmup and total_steps > 0:
            # Gozlenen gercek hiz — onceki tahminden her zaman iyidir.
            # SB3 rollout yuvarlamasi yuzunden current_step toplami asabilir;
            # kalan negatife dusmesin diye tabanlanir, degerlendirme eklenir.
            remaining_learn = max(total_steps - current_step, 0) / rate
            return _emit(remaining_learn + prior_eval, "measured")

        # Isinma penceresi: SB3'un ilk rollout'u dolmadan hiz yaniltici olur.
        if not has_prior:
            return out
        learn_elapsed = (now - learn_start) if learn_start else 0.0
        return _emit(max(prior_learn - learn_elapsed, 0.0) + prior_eval, "prior")

    # phase_name == "preparing": veri yukleniyor, ogrenme henuz baslamadi.
    # Sadece hazirlik payi erir; ogrenme + degerlendirme tam durur.
    if has_prior:
        remaining_setup = max(prior_setup - (elapsed or 0.0), 0.0)
        return _emit(remaining_setup + prior_learn + prior_eval, "prior")
    return out
