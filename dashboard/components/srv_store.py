"""Sunucu tarafı veri deposu (Faz 11 §3.19, performans).

Dash `dcc.Store` içeriği tarayıcıda durur ve onu Input/State alan HER callback'te sunucuya GERİ YÜKLENİR. Simülasyon
defteri 2026 penceresinde ~1,1 MB idi: bir güne tıklamak ya da aralığı değiştirmek her seferinde 1 MB'ı tarayıcıdan
yukarı gönderiyordu (istek gövdesi sıkıştırılmaz; ev bağlantısında saniyeye yakın).

Burada büyük değer süreç içinde tutulur, tarayıcıya yalnız küçük bir referans gider:
    ref = {"token": "...", **yeniden kurma parametreleri}
Jeton bulunamazsa (sunucu yeniden başladı, ya da LRU'dan düştü) çağıran parametrelerle yeniden kurar; sayfa bozulmaz.
Tek uvicorn işçisi (Dockerfile) -> depo tutarlı. Değer PAYLAŞILIR: çağıran değiştirmez.
"""
from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from typing import Any, Callable

_LOCK = threading.Lock()
_DATA: OrderedDict = OrderedDict()
_MAX = 48            # ~ eşzamanlı sekme × (defter + 2 görünüm); 2026 defteri bellekte birkaç MB


def put(value: Any, **params) -> dict:
    """Değeri sakla; tarayıcıya gidecek referansı döndür."""
    token = uuid.uuid4().hex
    with _LOCK:
        _DATA[token] = value
        while len(_DATA) > _MAX:
            _DATA.popitem(last=False)
    return {"token": token, **params}


def get(ref: dict | None, rebuild: Callable[[dict], Any] | None = None) -> Any:
    """Referanstan değer; yoksa `rebuild(ref)` ile yeniden kurar ve aynı jetona yazar (None dönerse None)."""
    if not ref or not isinstance(ref, dict) or "token" not in ref:
        return None
    with _LOCK:
        if ref["token"] in _DATA:
            _DATA.move_to_end(ref["token"])
            return _DATA[ref["token"]]
    if rebuild is None:
        return None
    value = rebuild(ref)
    if value is not None:
        with _LOCK:
            _DATA[ref["token"]] = value
            while len(_DATA) > _MAX:
                _DATA.popitem(last=False)
    return value


def clear() -> None:
    with _LOCK:
        _DATA.clear()
