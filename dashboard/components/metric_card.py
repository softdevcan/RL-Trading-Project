"""KPI metrik karti (Faz 8, C.3).

Degisen kural: **renk yalnizca anlam tasir.**

Faz 8 oncesi ana sayfadaki bes kartin besi farkli renkteydi (yesil, mavi,
kirmizi, sari, mor) — hicbiri bir sey ifade etmiyordu, hiyerarsi yerine
gorsel gurultu uretiyordu. Artik deger varsayilan olarak `--text`; renk
yalnizca sayi bir YON tasidiginda devreye girer (kar/zarar, drawdown).

`tone` bunun icin: "auto" degeri isaretine bakar, "profit"/"loss" zorlar,
"neutral" renklendirmez. Eski `color` parametresi cagri yerlerini bozmamak
icin duruyor; verilirse tone'u ezer.
"""

from dash import html
import dash_bootstrap_components as dbc

from dashboard.theme import GREEN, RED, TEXT, TEXT_MUTED

_TONES = {"neutral": TEXT, "profit": GREEN, "loss": RED}


def tone_color(tone: str) -> str:
    """Ton adindan renk. Bilinmeyen ton notr sayilir."""
    return _TONES.get(tone, TEXT)


def tone_for_number(value, invert: bool = False) -> str:
    """Sayidan ton cikar: pozitif kar, negatif zarar, sifir/None notr.

    `invert=True` olcunun kucugu iyi oldugunda kullanilir (or. drawdown).
    Bicimlenmis metinden degil SAYIDAN karar verir — "394.3%" gibi isaretsiz
    bir metinde yon kaybolur.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "neutral"
    if number == 0:
        return "neutral"
    positive = number > 0
    if invert:
        positive = not positive
    return "profit" if positive else "loss"


def _resolve_color(value, tone: str, color: str | None) -> str:
    if color:                      # geriye donuk: acikca verilen renk kazanir
        return color
    if tone == "auto":
        return _auto_tone(value)
    return _TONES.get(tone, TEXT)


def _auto_tone(value) -> str:
    """Metinden yon cikar: "+%12.4" yesil, "-3.1%" kirmizi, digerleri notr.

    Sayilar callback'ten bicimlenmis metin olarak geliyor; sayisal tipe
    guvenemiyoruz. Isaret yoksa renklendirme yapilmaz — tahmin etmektense
    notr birakmak dogru.
    """
    text = str(value).strip()
    if text.startswith(("+", "↑")):
        return GREEN
    if text.startswith(("-", "−", "↓")):
        return RED
    return TEXT


def create_metric_card(
    title: str,
    value: str = "—",
    subtitle: str = "",
    color: str | None = None,
    icon: str = "bi bi-bar-chart",
    card_id=None,
    tone: str = "neutral",
) -> dbc.Card:
    """KPI karti dondurur.

    Parameters
    ----------
    title:    Ust etiket (11px uppercase — `.section-title`)
    value:    Buyuk deger metni
    subtitle: Degerin altindaki kucuk aciklama
    color:    Geriye donuk kacis kapisi. Verilirse `tone` yok sayilir.
    icon:     Bootstrap ikon sinifi. Her zaman notr (`--muted`) cizilir —
              ikon dekoratiftir, anlam tasimaz.
    card_id:  Deger elemani icin id (callback hedefi)
    tone:     "neutral" (varsayilan) | "auto" | "profit" | "loss"
    """
    value_props = {
        "className": "metric-value",
        "style": {"color": _resolve_color(value, tone, color)},
    }
    if card_id:
        value_props["id"] = card_id

    children = [
        html.Div(
            [
                html.Small(title, className="section-title", style={"marginBottom": 0}),
                html.I(className=icon, style={"color": TEXT_MUTED, "fontSize": "14px"}),
            ],
            style={"display": "flex", "justifyContent": "space-between",
                   "alignItems": "center", "marginBottom": "10px"},
        ),
        html.Div(value, **value_props),
    ]
    if subtitle:
        children.append(
            html.Small(subtitle, style={"color": TEXT_MUTED, "fontSize": "12px",
                                        "display": "block", "marginTop": "6px"})
        )

    return dbc.Card(dbc.CardBody(children), className="h-100")
