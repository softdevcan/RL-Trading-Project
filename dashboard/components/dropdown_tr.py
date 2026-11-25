"""dcc.Dropdown iç metinleri Türkçe (Faz 11 §3.16).

Dash 4 `dcc.Dropdown` arama kutusu, "Select All", "N selected", "No options found" gibi metinleri İngilizce çiziyor;
`labels` sözlüğüyle değişiyor. Her çağrıya tek tek yazmak yerine pano yüklenirken varsayılan bir kez verilir
(`dashboard/app.py` sayfalardan ÖNCE `install()` çağırır). Çağrıda açıkça `labels=` verilirse o kazanır.
"""
from __future__ import annotations

import functools

from dash import dcc

# {num_selected}: Dash'in sayı yer tutucusu (selected_count); tarayıcıda denenerek bulundu ({num}, {count} çalışmıyor).
LABELS_TR = {
    "select_all": "Tümünü seç",
    "deselect_all": "Seçimi kaldır",
    "selected_count": "{num_selected} seçili",
    "search": "Ara",
    "clear_search": "Aramayı temizle",
    "clear_selection": "Seçimi temizle",
    "no_options_found": "Sonuç yok",
}

_installed = False


def install() -> None:
    """dcc.Dropdown.__init__'i `labels` varsayılanıyla sarar (bir kez)."""
    global _installed
    if _installed:
        return
    orig = dcc.Dropdown.__init__

    @functools.wraps(orig)
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("labels", dict(LABELS_TR))
        orig(self, *args, **kwargs)

    dcc.Dropdown.__init__ = __init__
    _installed = True
