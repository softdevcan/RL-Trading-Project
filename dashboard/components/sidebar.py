"""Left sidebar navigation component."""

from dash import html
import dash_bootstrap_components as dbc

from dashboard.auth_context import current_user, display_name, is_admin
from dashboard.theme import CARD, CARD2, BORDER, TEXT, TEXT_MUTED, BLUE, GREEN, SIDEBAR_STYLE

NAV_ITEMS = [
    {"label": "Dashboard",   "icon": "bi bi-speedometer2",  "href": "/dash/"},
    {"label": "Egitim",      "icon": "bi bi-cpu",            "href": "/dash/training"},
    {"label": "Veri",        "icon": "bi bi-database",       "href": "/dash/data"},
    {"label": "Modeller",    "icon": "bi bi-diagram-3",      "href": "/dash/models"},
    {"label": "Trading",     "icon": "bi bi-graph-up-arrow", "href": "/dash/daily-trading"},
    {"label": "Tahmin",      "icon": "bi bi-lightning-charge","href": "/dash/prediction"},
    {"label": "Akademik",    "icon": "bi bi-journal-bookmark","href": "/dash/academic"},
    {"label": "HiperParam",  "icon": "bi bi-sliders",        "href": "/dash/hyperopt"},
]

# Yalnizca admin rolu gorur
ADMIN_NAV_ITEMS = [
    {"label": "Kullanicilar", "icon": "bi bi-people", "href": "/dash/users"},
]

ROLE_LABELS = {"admin": "Yonetici", "user": "Kullanici", "viewer": "Izleyici"}
ROLE_COLORS = {"admin": GREEN, "user": BLUE, "viewer": TEXT_MUTED}


def _user_footer():
    """Aktif kullanici rozeti + cikis. Auth kapaliysa gosterilmez."""
    user = current_user()
    if not user:
        return None

    role = user.get("role", "user")
    return html.Div(
        [
            html.Div(
                [
                    html.I(className="bi bi-person-circle me-2", style={"color": TEXT_MUTED}),
                    html.Span(
                        display_name(),
                        style={
                            "color": TEXT, "fontSize": "13px", "fontWeight": "600",
                            "overflow": "hidden", "textOverflow": "ellipsis",
                            "whiteSpace": "nowrap", "maxWidth": "130px", "display": "inline-block",
                            "verticalAlign": "middle",
                        },
                        title=user.get("email", ""),
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "6px"},
            ),
            html.Div(
                [
                    html.Span(
                        ROLE_LABELS.get(role, role),
                        style={
                            "backgroundColor": CARD2, "color": ROLE_COLORS.get(role, TEXT_MUTED),
                            "fontSize": "10px", "padding": "2px 8px", "borderRadius": "10px",
                            "fontWeight": "600",
                        },
                    ),
                    html.A(
                        [html.I(className="bi bi-box-arrow-right me-1"), "Cikis"],
                        href="/logout",
                        style={"color": TEXT_MUTED, "fontSize": "11px", "textDecoration": "none"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"},
            ),
        ],
        style={
            "position": "absolute", "bottom": "0", "left": "0", "right": "0",
            "padding": "12px 16px", "borderTop": f"1px solid {BORDER}",
            "backgroundColor": CARD,
        },
    )


def create_sidebar():
    """Return the sidebar navigation element.

    Not: Kullanici rozeti istege gore degistiginden layout, `app.py` icinde
    her sayfa yuklemesinde yeniden uretilir (callable layout).
    """
    nav_links = []
    items = NAV_ITEMS + (ADMIN_NAV_ITEMS if is_admin() else [])
    for item in items:
        nav_links.append(
            dbc.NavLink(
                [
                    html.I(className=f"{item['icon']} me-2"),
                    item["label"],
                ],
                href=item["href"],
                active="exact",
                className="sidebar-link",
                style={
                    "color": TEXT_MUTED,
                    "padding": "10px 20px",
                    "borderRadius": "6px",
                    "margin": "2px 8px",
                    "transition": "all 0.15s",
                    "fontSize": "14px",
                    "fontWeight": "500",
                },
            )
        )

    return html.Div(
        [
            # Brand header
            html.Div(
                [
                    html.Div(
                        [
                            html.I(className="bi bi-robot me-2", style={"color": BLUE, "fontSize": "20px"}),
                            html.Span("RL Trading", style={"color": TEXT, "fontWeight": "700", "fontSize": "16px"}),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    html.Small("BIST-30 System", style={"color": TEXT_MUTED, "fontSize": "11px"}),
                ],
                style={
                    "padding": "20px 20px 16px",
                    "borderBottom": f"1px solid {BORDER}",
                    "marginBottom": "8px",
                },
            ),
            # Navigation
            dbc.Nav(nav_links, vertical=True, pills=True),
            # Aktif kullanici + cikis (auth aciksa)
            _user_footer(),
        ],
        style={**SIDEBAR_STYLE, "paddingBottom": "70px"},
        id="sidebar",
    )
