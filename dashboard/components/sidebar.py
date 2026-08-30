"""Sol kenar cubugu (Faz 8, C.8 + B.4).

Menu uc gruba ayrildi; duz bir 8 maddelik liste yerine ne aradigini bilen
kullanicinin dogrudan bulabilecegi bir yapi. Alt kisimda kullanici rozeti,
gorunum anahtari ve cikis.
"""

from dash import html
import dash_bootstrap_components as dbc

from dashboard.auth_context import current_user, display_name, is_admin
from dashboard.theme import BORDER, CARD, CARD2, TEXT, TEXT_MUTED, SIDEBAR_STYLE

# Gruplar sirayla cizilir; admin grubu yalnizca admin icin eklenir.
NAV_GROUPS = [
    ("Analiz", [
        {"label": "Dashboard", "icon": "bi bi-speedometer2", "href": "/dash/"},
        {"label": "Modeller", "icon": "bi bi-diagram-3", "href": "/dash/models"},
        {"label": "Akademik", "icon": "bi bi-journal-bookmark", "href": "/dash/academic"},
    ]),
    ("Islem", [
        {"label": "Trading", "icon": "bi bi-graph-up-arrow", "href": "/dash/daily-trading"},
        {"label": "Tahmin", "icon": "bi bi-lightning-charge", "href": "/dash/prediction"},
    ]),
    ("Sistem", [
        {"label": "Egitim", "icon": "bi bi-cpu", "href": "/dash/training"},
        {"label": "Veri", "icon": "bi bi-database", "href": "/dash/data"},
        {"label": "HiperParam", "icon": "bi bi-sliders", "href": "/dash/hyperopt"},
    ]),
]

ADMIN_GROUP = ("Yonetim", [
    {"label": "Kullanicilar", "icon": "bi bi-people", "href": "/dash/users"},
])

ROLE_LABELS = {"admin": "Yonetici", "user": "Kullanici", "viewer": "Izleyici"}


def _nav_link(item: dict) -> dbc.NavLink:
    return dbc.NavLink(
        [html.I(className=f"{item['icon']} me-2"), item["label"]],
        href=item["href"],
        active="exact",
        className="sidebar-link",
        style={"padding": "9px 17px", "margin": "1px 8px"},
    )


def _theme_button() -> html.Button:
    """Uc durumu sirayla dolasan gorunum anahtari.

    Etiket ve ikon assets/theme-toggle.js tarafindan acilista ve her
    tiklamada guncellenir; buradaki degerler yalnizca ilk cizimde
    (script calismadan once) gorunen yer tutuculardir.
    """
    return html.Button(
        [
            html.I(className="bi bi-circle-half"),
            html.Span("Sistem", className="theme-label ms-2"),
        ],
        id="theme-toggle",
        type="button",
        title="Gorunum tercihini degistir",
        **{"data-theme-set": "__cycle__"},
    )


def _user_footer():
    """Kullanici rozeti + gorunum anahtari + cikis.

    Auth kapaliyken hesap satiri gosterilmez ama gorunum anahtari kalir —
    tema tercihi kimlik dogrulamadan bagimsiz calisir (cerezde saklanir).
    """
    user = current_user()

    rows = []
    if user:
        role = user.get("role", "user")
        rows.append(
            html.A(
                [
                    html.I(className="bi bi-person-circle me-2", style={"color": TEXT_MUTED}),
                    html.Span(
                        display_name(),
                        style={
                            "color": TEXT, "fontSize": "13px", "fontWeight": "600",
                            "overflow": "hidden", "textOverflow": "ellipsis",
                            "whiteSpace": "nowrap", "maxWidth": "120px",
                            "display": "inline-block", "verticalAlign": "middle",
                        },
                    ),
                ],
                href="/dash/account",
                title=f"{user.get('email', '')} — Hesabim",
                style={"textDecoration": "none", "display": "flex",
                       "alignItems": "center", "marginBottom": "8px"},
            )
        )
        rows.append(
            html.Div(
                [
                    html.Span(
                        ROLE_LABELS.get(role, role),
                        style={
                            "backgroundColor": CARD2, "color": TEXT,
                            "fontSize": "10px", "padding": "2px 8px",
                            "borderRadius": "10px", "fontWeight": "600",
                        },
                    ),
                    html.A(
                        [html.I(className="bi bi-box-arrow-right me-1"), "Cikis"],
                        href="/logout",
                        style={"color": TEXT_MUTED, "fontSize": "11px",
                               "textDecoration": "none"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center",
                       "justifyContent": "space-between", "marginBottom": "10px"},
            )
        )

    rows.append(_theme_button())

    return html.Div(
        rows,
        style={
            "position": "absolute", "bottom": "0", "left": "0", "right": "0",
            "padding": "12px 16px", "borderTop": f"1px solid {BORDER}",
            "backgroundColor": CARD,
        },
    )


def create_sidebar():
    """Kenar cubugunu dondur.

    Not: Kullanici rozeti ve admin grubu role gore degistiginden layout,
    `app.py` icinde her sayfa yuklemesinde yeniden uretilir (callable layout).
    """
    groups = list(NAV_GROUPS) + ([ADMIN_GROUP] if is_admin() else [])

    nav_children = []
    for title, items in groups:
        nav_children.append(html.Div(title, className="sidebar-group"))
        nav_children.extend(_nav_link(item) for item in items)

    return html.Div(
        [
            # Marka
            html.Div(
                [
                    html.Div(
                        [
                            html.I(className="bi bi-robot me-2",
                                   style={"color": "var(--primary)", "fontSize": "19px"}),
                            html.Span("RL Trading",
                                      style={"color": TEXT, "fontWeight": "700",
                                             "fontSize": "16px"}),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    html.Small("by softdevcan",
                               style={"color": TEXT_MUTED, "fontSize": "11px"}),
                ],
                style={
                    "padding": "20px 20px 16px",
                    "borderBottom": f"1px solid {BORDER}",
                    "marginBottom": "4px",
                },
            ),
            dbc.Nav(nav_children, vertical=True, pills=True),
            _user_footer(),
        ],
        style={**SIDEBAR_STYLE, "paddingBottom": "120px"},
        id="sidebar",
    )
