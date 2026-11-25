"""Pano genelinde Türkçe biçim (Faz 11 §3.15): para, yüzde, saat, iş durumu.

Tek kaynak; sayfalar kendi `f"₺{x:,.2f}"` biçimini yazmaz. Türkçe yazım: binlik nokta, ondalık virgül, gerçek eksi
(U+2212), para birimi sonda ("100.000,00 TL"; simülasyon sayfasıyla aynı).
"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

MINUS = "−"
TRT = ZoneInfo("Europe/Istanbul")


def num(x: float, d: int = 2) -> str:
    """Türkçe sayı, işaretsiz mutlak değer: 1234567.8 -> '1.234.567,80'."""
    return f"{abs(float(x)):,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def money(v, d: int = 2) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    return ("" if x >= 0 else MINUS) + num(x, d) + " TL"


def signed_money(v, d: int = 2) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    return ("+" if x >= 0 else MINUS) + num(x, d) + " TL"


def signed_pct(v, d: int = 2) -> str:
    """Yüzde PUANI girer (5.2 -> '+%5,20')."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    return ("+" if x >= 0 else MINUS) + "%" + num(x, d)


def dt_trt(value: str | None, naive_is_utc: bool = True) -> str:
    """ISO damga -> '25.11.2026 19:16 TRT'. Kimlik kayıtları naive UTC tutulur (app/auth/models.utcnow)."""
    if not value:
        return "—"
    try:
        d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc) if naive_is_utc else d.astimezone()
    return d.astimezone(TRT).strftime("%d.%m.%Y %H:%M") + " TRT"


# Optuna / eğitim iş durumu -> (Türkçe etiket, dbc rengi). API "complete" ve "completed" ikisini de döndürebiliyor.
STATUS = {
    "complete": ("Tamamlandı", "success"), "completed": ("Tamamlandı", "success"),
    "running": ("Sürüyor", "info"), "pruned": ("Budandı", "warning"),
    "fail": ("Hata", "danger"), "failed": ("Hata", "danger"), "error": ("Hata", "danger"),
    "waiting": ("Bekliyor", "secondary"), "queued": ("Sırada", "secondary"),
}


def status(s) -> tuple[str, str]:
    key = str(s or "").lower()
    return STATUS.get(key, (key.capitalize() or "—", "secondary"))
