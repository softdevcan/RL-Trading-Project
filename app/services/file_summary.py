"""Dosyaya bağlı özet önbelleği (Faz 11 §3.19, performans).

Veri sayfası, hiperoptimizasyon ve model uçları her istekte büyük CSV'leri (raw_stock_data, stock_data_with_indicators;
on binlerce satır) baştan okuyup yalnız birkaç alan (son tarih, semboller, satır sayısı, kat kapsamı) döndürüyordu:
istek başına 200–450 ms. Burada yalnız o KÜÇÜK özet tutulur, DataFrame tutulmaz (bellek ve paylaşılan nesneyi
değiştirme riski yok).

Anahtar = (mutlak yol, mtime_ns, boyut, ad): dosya yeniden yazılınca anahtar değişir, eski özet bir daha okunmaz.
Bayat veri gösterme yolu yoktur; `clear()` yalnız belleği boşaltır (veri güncellemesinden sonra çağrılır).
Özet salt okunur kabul edilir: çağıran değiştirecekse kopyalar (`get` derin kopya döndürür).
"""
from __future__ import annotations

import copy
import os
import threading
from typing import Any, Callable

_LOCK = threading.Lock()
_CACHE: dict[tuple, Any] = {}
_MAX = 64


def _key(path: str, name: str) -> tuple | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (os.path.abspath(path), st.st_mtime_ns, st.st_size, name)


def get(path: str, name: str, build: Callable[[], Any]) -> Any:
    """`path` değişmediyse önceki `build()` sonucunu döndürür; değiştiyse yeniden kurar. Dosya yoksa önbelleksiz kurar
    (build kendi FileNotFoundError'ını fırlatabilir)."""
    k = _key(path, name)
    if k is None:
        return build()
    with _LOCK:
        if k in _CACHE:
            return copy.deepcopy(_CACHE[k])
    val = build()
    with _LOCK:
        for old in [o for o in _CACHE if o[0] == k[0] and o[3] == name]:      # aynı dosyanın eski sürümü
            del _CACHE[old]
        if len(_CACHE) >= _MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[k] = val
    return copy.deepcopy(val)


def clear() -> None:
    with _LOCK:
        _CACHE.clear()
