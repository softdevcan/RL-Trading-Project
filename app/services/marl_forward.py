"""Faz 11 S4: M8 ileri veri gunluk isinin uygulama ici tetigi (K-F11-4; plan phase-11-integration-plan.md §3 S4).

Is `marl.forward_job.run_daily` (panel -> karar -> defter; idempotent, bosluk doldurur). Bu modul yalniz NE ZAMAN
calisacagini belirler:
  - Sunucu acilisinda (MARL_FORWARD_AUTORUN acikken) STARTUP_DELAY_SEC sonra bir kez: kacirilan gunleri doldurur.
  - Sonra her TICK_SEC'te: hafta ici ve TRT saati CUTOFF_HOUR'dan sonra, o gun icin en fazla MAX_ATTEMPTS_PER_DAY deneme
    (Is Yatirim / EVDS gecikirse gun kapisi bekletir; sonraki deneme alir). Gunun seansi deftere girince o gun biter.
  - Yonetici elle tetikleyebilir (`POST /api/marl/forward/run`, RequireAdmin).
Defter ortak arastirma alanina yazilir (`data/marl/forward/m8/ledger`, K-F11-3); kullanici calisma alanina degil, bu
yuzden is kullanici kimligi tasimaz. Sunucu kapali gunler acilista doldurulur. Docker imajinda `data/marl` yoktur:
ayar varsayilan KAPALI, `run_server.py` acar.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

from app.services import background_jobs

logger = logging.getLogger(__name__)

TRT = ZoneInfo('Europe/Istanbul')
JOB_NAME = 'marl-forward-daily'
STARTUP_DELAY_SEC = 60
TICK_SEC = 30 * 60
CUTOFF_HOUR = 19                    # forward_panel.CUTOFF_HOUR_TRT ile ayni (gun kapisi da bunu ister)
MAX_ATTEMPTS_PER_DAY = 4

_state = {'scheduler': False, 'attempts': {}, 'done': set(), 'last_report': None}
_stop = threading.Event()


def available() -> bool:
    from marl import forward_job as FJ
    return os.path.exists(os.path.join(FJ.ARCHIVE, 'manifest.json')) and os.path.exists(
        os.path.join('data', 'marl', 'frozen', 'm8-source', 'manifest.json'))


def run_now(reason: str = 'manual') -> bool:
    """Isi arka planda baslatir; zaten calisiyorsa False."""
    if background_jobs.is_running(JOB_NAME):
        return False

    def _job():
        from marl import forward_job as FJ
        rep = FJ.run_daily()
        rep['trigger'] = reason
        _state['last_report'] = rep
        day = datetime.now(TRT).date().isoformat()
        led = rep.get('ledger') or {}
        if rep.get('status') in ('yazildi', 'degisiklik_yok') and str(led.get('last_date', '')) >= day:
            _state['done'].add(day)
        logger.info("MARL ileri veri isi (%s): %s", reason, rep.get('status'))

    background_jobs.spawn_sync(_job, name=JOB_NAME)
    return True


def _due(now: datetime) -> bool:
    day = now.date().isoformat()
    if now.weekday() >= 5 or now.hour < CUTOFF_HOUR or day in _state['done']:
        return False
    return _state['attempts'].get(day, 0) < MAX_ATTEMPTS_PER_DAY


def _loop():
    if _stop.wait(STARTUP_DELAY_SEC):
        return
    run_now('startup')
    while not _stop.wait(TICK_SEC):
        now = datetime.now(TRT)
        if _due(now):
            day = now.date().isoformat()
            if run_now('schedule'):
                _state['attempts'][day] = _state['attempts'].get(day, 0) + 1


def start_scheduler() -> bool:
    if _state['scheduler']:
        return True
    if not available():
        logger.warning("MARL ileri veri isi baslatilmadi: dondurulmus arsiv/kaynak yok (data/marl/frozen)")
        return False
    _stop.clear()
    threading.Thread(target=_loop, name='marl-forward-scheduler', daemon=True).start()
    _state['scheduler'] = True
    return True


def stop_scheduler() -> None:
    _stop.set()
    _state['scheduler'] = False


def info() -> dict:
    from marl import forward_job as FJ
    out = FJ.status() if available() else {'available': False}
    out.update({'available': available(), 'scheduler': _state['scheduler'], 'running': background_jobs.is_running(JOB_NAME),
                'cutoff_hour_trt': CUTOFF_HOUR, 'attempts_today': _state['attempts'].get(datetime.now(TRT).date().isoformat(), 0)})
    return out
