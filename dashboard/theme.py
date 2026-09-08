"""Tema koprusu: DOM icin CSS degiskenleri, Plotly icin gercek hex.

Faz 8 — iki katman, iki kural
-----------------------------
1. **DOM katmani.** Asagidaki sabitler artik hex degil, `var(--token)` dizesi.
   Dash inline stiline (`style={"color": TEXT}`) girdiginde tarayici cozer ve
   tema degistiginde kendiliginden guncellenir. Sayfa kodunun degismesi
   gerekmez; f-string kullanimlari (`f"1px solid {BORDER}"`) da calisir.

2. **Plotly katmani.** Grafikler `var()` kabul etmez — figure JSON'i sunucuda
   uretilir ve CSS baglami yoktur. Bu yuzden `plot_palette()` gercek hex
   dondurur; hangi paletin secilecegini `current_theme()` cerezden okur.

Yeni renk eklerken: once `static/tokens.css`, sonra buraya. Kontrast esigini
`tests/test_theme_contrast.py` dogrular — tahminle ekleme.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# ── DOM katmani — inline style icin ─────────────────────────────────────────
# Not: bunlar HEX DEGIL. Plotly'ye verilirsen sessizce siyah cizer;
# grafik icin plot_palette() kullan.
BG = "var(--rlt-bg)"
CARD = "var(--rlt-surface)"
CARD2 = "var(--rlt-surface-2)"
BORDER = "var(--rlt-border)"
TEXT = "var(--rlt-text)"
TEXT_MUTED = "var(--rlt-muted)"

GREEN = "var(--rlt-profit)"
RED = "var(--rlt-loss)"
BLUE = "var(--rlt-primary)"
YELLOW = "var(--rlt-warn)"
PURPLE = "var(--rlt-accent)"
CYAN = "var(--rlt-info)"
ORANGE = "var(--rlt-orange)"
GOLD = "var(--rlt-gold)"

# Dolgu (rozet/dugme zemini) ve uzerine yazilacak renk.
PRIMARY_FILL = "var(--rlt-primary-fill)"
ON_FILL = "var(--rlt-on-fill)"

# Algoritma rozetleri.
#
# Neden inline stil DEGIL de sinif: dbc.Badge kendi `bg-secondary` sinifini
# basiyor ve Bootstrap'in utility kurallari `!important` tasiyor — inline
# backgroundColor bu yuzden sessizce eziliyordu. Renk artik custom.css'te
# `.algo-badge.algo-ppo` gibi sinif eslesmesiyle veriliyor.
ALGO_BADGE_CLASSES = {"PPO": "algo-ppo", "A2C": "algo-a2c",
                      "SAC": "algo-sac", "TD3": "algo-td3"}


def algo_badge_class(algorithm: str) -> str:
    """Algoritma rozeti icin className. Taninmayan algoritma notr kalir."""
    key = (algorithm or "").strip().upper()
    return "algo-badge " + ALGO_BADGE_CLASSES.get(key, "")


# Geriye donuk: bazi yerler hala sozluk bekliyor olabilir.
ALGO_COLORS = {
    "PPO": "var(--rlt-primary-fill)",
    "A2C": "var(--rlt-profit-fill)",
    "SAC": "var(--rlt-accent)",
    "TD3": "var(--rlt-orange)",
}

# Kenar cubugu / icerik yerlesimi
SIDEBAR_WIDTH = "220px"

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": SIDEBAR_WIDTH,
    "padding": "0",
    "backgroundColor": CARD,
    "borderRight": f"1px solid {BORDER}",
    "overflowY": "auto",
    "zIndex": 1000,
}

# Ust cubuk (Faz 8/G) yalnizca icerik alanini kaplar: kenar cubugu tam boy
# kalir, marka tek yerde durur.
TOPBAR_HEIGHT = "54px"

TOPBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": SIDEBAR_WIDTH,
    "right": 0,
    "height": TOPBAR_HEIGHT,
    "zIndex": 900,
}

CONTENT_STYLE = {
    "marginLeft": SIDEBAR_WIDTH,
    # Ust cubuk `fixed`, yani akistan cikmis: icerik onun altindan baslasin
    # diye ust bosluk cubugun yuksekligi kadar artirilir.
    "padding": f"calc({TOPBAR_HEIGHT} + 24px) 24px 24px",
    "backgroundColor": BG,
    "minHeight": "100vh",
}


# ── Plotly katmani — gercek hex ─────────────────────────────────────────────
# Degerler static/tokens.css ile birebir ayni olmali; test bunu dogrular.
PLOT = {
    "light": {
        "bg": "#ffffff",
        "grid": "#e2e8f0",
        "line": "#8595a9",
        "text": "#0f172a",
        "muted": "#556275",
        "hover_bg": "#f1f5f9",
        "blue": "#2563eb",
        "green": "#15803d",
        "red": "#b91c1c",
        "yellow": "#b45309",
        "purple": "#7e22ce",
        "cyan": "#0e7490",
        "orange": "#c2410c",
        "gold": "#946005",
    },
    "dark": {
        "bg": "#1e293b",
        "grid": "#334155",
        "line": "#64748b",
        "text": "#e2e8f0",
        "muted": "#a8b6c9",
        "hover_bg": "#334155",
        "blue": "#93c5fd",
        "green": "#4ade80",
        "red": "#fca5a5",
        "yellow": "#fbbf24",
        "purple": "#d8b4fe",
        "cyan": "#22d3ee",
        "orange": "#fb923c",
        "gold": "#fcd34d",
    },
}

# Grafiklerde algoritma serisi renkleri (hex — plot_palette ile ayni tema).
ALGO_PLOT_KEYS = {"PPO": "blue", "A2C": "green", "SAC": "purple", "TD3": "orange"}

def current_theme() -> str:
    """Bu istek icin gecerli tema: "light" veya "dark".

    Kaynak sirasi:
      1. rlt_theme_r  — istemcinin cozdugu deger. "system" seciliyken
         gercek sonucu yalnizca tarayici bilir, sunucuya bu cerezle gelir.
      2. rlt_theme    — tercih dogrudan light/dark ise onu kullan.
      3. "light"      — cerez yok (ilk ziyaret): tokens.css tabani da aydinlik.

    Dash callback'leri Flask istek baglaminda calisir; baglam yoksa (test,
    dogrudan cagri) varsayilana duser.
    """
    try:
        from flask import has_request_context, request

        if not has_request_context():
            return "light"

        from app.core.config import get_settings

        s = get_settings()
        resolved = request.cookies.get(s.THEME_RESOLVED_COOKIE_NAME, "")
        if resolved in ("light", "dark"):
            return resolved
        pref = request.cookies.get(s.THEME_COOKIE_NAME, "")
        if pref in ("light", "dark"):
            return pref
    except Exception:  # pragma: no cover - baglam disi cagri
        pass
    return "light"


def plot_palette(theme: str | None = None) -> dict:
    """Grafikler icin hex palet. `theme` verilmezse istekten cozulur."""
    return PLOT[theme if theme in PLOT else current_theme()]


def algo_plot_colors(theme: str | None = None) -> dict:
    """Algoritma adi -> hex (grafik serileri icin)."""
    p = plot_palette(theme)
    return {algo: p[key] for algo, key in ALGO_PLOT_KEYS.items()}


def plot_rgba(key: str, alpha: float, theme: str | None = None) -> str:
    """Palet renginin yari saydam hali — alan dolgulari icin.

    Plotly `fillcolor` yari saydamlik ister; sabit rgba yazmak temayi kirar
    (aydinlikta koyu mavi dolgu, koyuda acik mavi dolgu gerekiyor).
    """
    h = plot_palette(theme)[key].lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def plot_template(theme: str | None = None) -> dict:
    """Aktif temanin Plotly layout sozlugu."""
    p = plot_palette(theme)
    axis = {
        "gridcolor": p["grid"],
        "linecolor": p["line"],
        "zerolinecolor": p["grid"],
        "tickcolor": p["muted"],
        "tickfont": {"color": p["muted"]},
        "title": {"font": {"color": p["muted"]}},
    }
    return {
        "paper_bgcolor": p["bg"],
        "plot_bgcolor": p["bg"],
        "font": {"color": p["text"], "family": "Inter, system-ui, sans-serif"},
        "title": {"font": {"color": p["text"]}},
        "xaxis": dict(axis),
        "yaxis": dict(axis),
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": p["grid"],
            "font": {"color": p["muted"]},
        },
        "hoverlabel": {
            "bgcolor": p["hover_bg"],
            "bordercolor": p["line"],
            "font": {"color": p["text"]},
        },
        "margin": {"l": 50, "r": 20, "t": 40, "b": 50},
        "colorway": [p["blue"], p["green"], p["purple"], p["orange"],
                     p["cyan"], p["yellow"], p["red"]],
    }


def apply_theme_template(fig, theme: str | None = None):
    """Aktif temanin sablonunu figure'e uygula ve dondur."""
    fig.update_layout(**plot_template(theme))
    return fig


# Faz 8 oncesi ad. 63 cagri yerinin tek seferde degismesi gerekmesin diye
# duruyor; yeni kod apply_theme_template kullanmali.
def apply_dark_template(fig):
    """Geriye donuk takma ad — artik aktif temayi uygular, her zaman koyuyu degil."""
    return apply_theme_template(fig)


# DARK_TEMPLATE'i sozluk olarak import eden yerler icin (home.py gibi):
# artik cagrilabilir degil ama okundugunda aktif temayi verir.
class _TemplateProxy(dict):
    """`DARK_TEMPLATE["layout"]` erisimini aktif temaya baglar."""

    def __getitem__(self, key):
        if key == "layout":
            return plot_template()
        return super().__getitem__(key)


DARK_TEMPLATE = _TemplateProxy()


def empty_figure(message: str = "Veri yok"):
    """Ortasinda mesaj olan bos figure."""
    import plotly.graph_objects as go

    p = plot_palette()
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font={"color": p["muted"], "size": 14},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    apply_theme_template(fig)
    return fig
