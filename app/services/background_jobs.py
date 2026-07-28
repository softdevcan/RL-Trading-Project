"""
Uzun suren isleri istek yasam dongusunden ayirir.

NEDEN: FastAPI'nin `BackgroundTasks`'i isi yanit gonderildikten *sonra* ama
AYNI ASGI cagrisinin *icinde* calistirir. Bu projede iki sonucu vardi:

1. Gorevler `async def` idi ve iclerinde bloke eden CPU isi yapiyordu
   (Optuna `study.optimize`, SB3 `model.learn`). `async def` bir arka plan
   gorevi dogrudan event loop'ta beklenir — dolayisiyla tum sunucu, uzerine
   mount edilmis Dash panosu dahil, is bitene kadar donuyordu.

2. Pano backend'i AG UZERINDEN DEGIL, in-process ASGI ile cagiriyor
   (`dashboard/api_client.py` -> `httpx.ASGITransport`). ASGI cagrisi arka
   plan gorevleri bitmeden tamamlanmaz, yani istemci de bekliyordu.
   Olculdu:
     POST /api/hyperopt/start  (n_trials=1, 10k timestep) -> 32.8 sn
     POST /api/trading/train   (3000 timestep)            -> 15.0 sn
   Pano istemcisinin TIMEOUT'u 30 sn oldugundan istek zaman asimina ugruyor,
   `_post` None donuyor ve kullanici HICBIR bildirim goremiyordu.

COZUM: is, istegin yasam dongusune bagli olmayan ayri bir daemon thread'de,
kendi event loop'unda calisir. Uc hemen doner; durum ayri bir "progress"
ucundan izlenir.

DIKKAT: is artik farkli bir event loop'ta kostugu icin, ana loop'a ait
nesneler (orn. acik WebSocket baglantilari) oradan kullanilamaz. Boyle bir
gonderim hata verir; cagiran taraf bunu yutmali. Pano zaten WebSocket degil
polling kullaniyor.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# Calisan islerin kaydi — teshis icin (hangi is hala ayakta?).
_jobs: dict[str, threading.Thread] = {}
_lock = threading.Lock()


def spawn(factory: Callable[[], Awaitable[None]], *, name: str) -> threading.Thread:
    """`factory()` ile uretilen coroutine'i ayri bir thread + event loop'ta calistir.

    Args:
        factory: Cagrildiginda coroutine dondüren fonksiyon. Coroutine'in
            KENDISINI degil fabrikayi al — coroutine nesnesi baska bir
            loop'ta beklenmek uzere burada uretilmeli.
        name: Thread adi; loglarda ve `running()` ciktisinda gorunur.

    Returns:
        Baslatilmis daemon thread. Cagiran genellikle beklemez.
    """
    def _runner():
        try:
            asyncio.run(factory())
        except Exception:
            # Thread'de yakalanmayan istisna sessizce kaybolur; gorunur kil.
            logger.exception(f"Arka plan isi basarisiz: {name}")
        finally:
            with _lock:
                _jobs.pop(name, None)

    thread = threading.Thread(target=_runner, name=name, daemon=True)
    with _lock:
        _jobs[name] = thread
    thread.start()
    logger.info(f"Arka plan isi baslatildi: {name}")
    return thread


def spawn_sync(fn: Callable[[], None], *, name: str) -> threading.Thread:
    """`spawn`'in senkron fonksiyonlar icin karsiligi."""
    def _runner():
        try:
            fn()
        except Exception:
            logger.exception(f"Arka plan isi basarisiz: {name}")
        finally:
            with _lock:
                _jobs.pop(name, None)

    thread = threading.Thread(target=_runner, name=name, daemon=True)
    with _lock:
        _jobs[name] = thread
    thread.start()
    logger.info(f"Arka plan isi baslatildi: {name}")
    return thread


def is_running(name: str) -> bool:
    with _lock:
        thread = _jobs.get(name)
    return bool(thread and thread.is_alive())


def running() -> list[str]:
    """Halen calisan islerin adlari (teshis icin)."""
    with _lock:
        return [n for n, t in _jobs.items() if t.is_alive()]
