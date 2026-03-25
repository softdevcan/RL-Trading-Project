"""Left sidebar navigation component."""

from dash import html
import dash_bootstrap_components as dbc

from dashboard.theme import CARD, BORDER, TEXT, TEXT_MUTED, BLUE, SIDEBAR_STYLE

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


def create_sidebar():
    """Return the sidebar navigation element."""
    nav_links = []
    for item in NAV_ITEMS:
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
        ],
        style=SIDEBAR_STYLE,
        id="sidebar",
    )
