"""Hesabim sayfasi — gorunum tercihi ve hesap bilgileri (Faz 8, B.3).

Her rol erisir, viewer dahil: tema okuma yetkisiyle ilgisi olmayan kisisel
bir tercihtir. Kullanici yonetimi (/dash/users) admin sayfasidir ve ayri kalir.

Tema secimi SUNUCU TURU GEREKTIRMEZ: dugmelere `data-theme-set` konur,
assets/theme-toggle.js tiklamayi yakalar, damgayi ve cerezi gunceller,
sonra PATCH /auth/preferences ile hesaba yazar. Bu yuzden burada tema icin
Dash callback'i yok — anlik tepki icin kasitli bir tercih.
"""

from dash import html
import dash_bootstrap_components as dbc

from dashboard.auth_context import current_user, display_name
from dashboard.components.page_header import create_page_header
from dashboard.theme import BORDER, CARD2, TEXT, TEXT_MUTED

ROLE_LABELS = {"admin": "Yonetici", "user": "Kullanici", "viewer": "Izleyici"}

ROLE_HELP = {
    "admin": "Tum yetkiler ve kullanici yonetimi.",
    "user": "Kendi calisma alaninda egitim, tahmin ve karar.",
    "viewer": "Yalnizca okuma.",
}

THEME_OPTIONS = [
    ("light", "Aydinlik", "bi-sun", "Her zaman acik zemin."),
    ("dark", "Koyu", "bi-moon-stars", "Her zaman koyu zemin."),
    ("system", "Sistem", "bi-circle-half", "Isletim sisteminizi izler."),
]


def _theme_preview(kind: str):
    """Secenegin altindaki kucuk onizleme seridi.

    Renkler kasitli olarak TOKENLARDAN BAGIMSIZ: iki tema da ayni anda
    gorunmeli ki secim yapmadan once karsilastirilabilsin. Her seridin
    metin/zemin cifti kendi paletinden gelir.
    """
    palettes = {
        "light": {"bg": "#f6f8fb", "surface": "#ffffff", "border": "#e2e8f0",
                  "text": "#0f172a", "muted": "#556275"},
        "dark": {"bg": "#0f172a", "surface": "#1e293b", "border": "#334155",
                 "text": "#e2e8f0", "muted": "#a8b6c9"},
    }

    def strip(p):
        return html.Div(
            [
                html.Div(style={"height": "6px", "width": "60%", "borderRadius": "3px",
                                "backgroundColor": p["text"], "marginBottom": "4px"}),
                html.Div(style={"height": "5px", "width": "85%", "borderRadius": "3px",
                                "backgroundColor": p["muted"]}),
            ],
            style={
                "backgroundColor": p["surface"],
                "border": f"1px solid {p['border']}",
                "borderRadius": "4px",
                "padding": "8px",
            },
        )

    if kind == "system":
        # Sistem: iki tarafi yan yana goster — secenegin ne yaptigi boyle anlasilir
        return html.Div(
            [
                html.Div(strip(palettes["light"]),
                         style={"backgroundColor": palettes["light"]["bg"],
                                "padding": "6px", "flex": "1"}),
                html.Div(strip(palettes["dark"]),
                         style={"backgroundColor": palettes["dark"]["bg"],
                                "padding": "6px", "flex": "1"}),
            ],
            style={"display": "flex", "borderRadius": "6px", "overflow": "hidden",
                   "border": f"1px solid {BORDER}"},
        )

    p = palettes[kind]
    return html.Div(
        strip(p),
        style={"backgroundColor": p["bg"], "padding": "6px", "borderRadius": "6px",
               "border": f"1px solid {BORDER}"},
    )


def _theme_card():
    options = []
    for value, label, icon, hint in THEME_OPTIONS:
        options.append(
            dbc.Col(
                html.Div(
                    [
                        html.Button(
                            [
                                html.I(className=f"bi {icon} me-2"),
                                html.Span(label),
                            ],
                            id=f"theme-option-{value}",
                            className="btn theme-option",
                            **{"data-theme-set": value, "aria-pressed": "false"},
                            type="button",
                        ),
                        _theme_preview(value),
                        html.Small(hint, style={"color": TEXT_MUTED, "fontSize": "12px",
                                                "display": "block", "marginTop": "6px"}),
                    ],
                    style={"display": "flex", "flexDirection": "column", "gap": "8px"},
                ),
                md=4, sm=12, className="mb-3",
            )
        )

    return dbc.Card(
        [
            dbc.CardHeader(html.Span("Gorunum", className="card-title-sm")),
            dbc.CardBody(
                [
                    html.P(
                        "Tercih hesabiniza kaydedilir — baska bir makineden "
                        "girdiginizde de gecerli olur.",
                        style={"color": TEXT_MUTED, "marginBottom": "16px"},
                    ),
                    dbc.Row(options, className="g-3"),
                ]
            ),
        ]
    )


def _info_row(label: str, value, last: bool = False):
    return html.Div(
        [
            html.Span(label, style={"color": TEXT_MUTED, "fontSize": "13px"}),
            html.Span(value, style={"color": TEXT, "fontSize": "13px", "fontWeight": "500"}),
        ],
        style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "padding": "10px 0",
            "borderBottom": "none" if last else f"1px solid {BORDER}",
        },
    )


def _account_card():
    user = current_user() or {}
    role = user.get("role", "user")

    if not user:
        # AUTH_ENABLED=False ile calisirken oturum yok
        body = html.P(
            "Kimlik dogrulama kapali; hesap bilgisi yok. Gorunum tercihi "
            "yalnizca bu tarayicida saklanir.",
            style={"color": TEXT_MUTED, "margin": 0},
        )
    else:
        body = html.Div(
            [
                _info_row("Ad Soyad", display_name()),
                _info_row("E-posta", user.get("email", "—")),
                _info_row(
                    "Rol",
                    html.Span(
                        [
                            html.Span(
                                ROLE_LABELS.get(role, role),
                                style={"backgroundColor": CARD2, "color": TEXT,
                                       "fontSize": "11px", "padding": "3px 9px",
                                       "borderRadius": "10px", "fontWeight": "600"},
                            ),
                            html.Small(
                                ROLE_HELP.get(role, ""),
                                style={"color": TEXT_MUTED, "marginLeft": "8px"},
                            ),
                        ]
                    ),
                    last=True,
                ),
            ]
        )

    return dbc.Card(
        [
            dbc.CardHeader(html.Span("Hesap", className="card-title-sm")),
            dbc.CardBody(body),
        ]
    )


def _security_card():
    return dbc.Card(
        [
            dbc.CardHeader(html.Span("Guvenlik", className="card-title-sm")),
            dbc.CardBody(
                [
                    html.P(
                        "Parolanizi degistirdiginizde diger tum oturumlariniz kapanir.",
                        style={"color": TEXT_MUTED, "marginBottom": "14px"},
                    ),
                    html.A(
                        [html.I(className="bi bi-key me-2"), "Parolayi degistir"],
                        href="/change-password",
                        className="btn btn-outline-secondary",
                    ),
                ]
            ),
        ]
    )


# ── Layout ────────────────────────────────────────────────────────────────

def layout():
    return html.Div(
        [
            create_page_header(
                "Hesabim",
                "Gorunum tercihi ve hesap bilgileri",
            ),
            dbc.Row(
                [
                    dbc.Col(_theme_card(), lg=7, className="mb-4"),
                    dbc.Col(
                        html.Div(
                            [_account_card(), html.Div(_security_card(), className="mt-4")]
                        ),
                        lg=5, className="mb-4",
                    ),
                ]
            ),
        ]
    )


def register_callbacks(app):
    """Tema secimi clientside calisiyor (bkz. modul dokumani) — sunucu
    callback'i gerekmiyor. Fonksiyon, app.py'nin tek tip cagrisi icin var."""
    return
