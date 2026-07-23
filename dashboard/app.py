"""
Dash application factory for RL Trading Dashboard.

Mounted under /dash/ via Starlette WSGIMiddleware.

WSGI path note
--------------
Starlette's Mount("/dash", ...) strips the "/dash" prefix from PATH_INFO
before forwarding to Flask.  PrefixMiddleware restores it so that Dash's
routes (registered at /dash/*) match correctly.
"""

import os
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

from dashboard.theme import BG, CARD, CONTENT_STYLE
from dashboard.components.sidebar import create_sidebar


class PrefixMiddleware:
    """
    Restore the mount prefix that Starlette strips from PATH_INFO.

    Example: Starlette receives GET /dash/training
             Strips /dash  →  PATH_INFO = /training
             PrefixMiddleware prepends /dash  →  PATH_INFO = /dash/training
             Dash (routes_pathname_prefix=/dash/) matches /dash/training ✓
    """

    def __init__(self, wsgi_app, prefix: str):
        self.wsgi_app = wsgi_app
        self.prefix = prefix.rstrip("/")  # e.g. "/dash"

    def __call__(self, environ, start_response):
        environ["PATH_INFO"] = self.prefix + environ.get("PATH_INFO", "/")
        environ["SCRIPT_NAME"] = ""
        return self.wsgi_app(environ, start_response)


def create_dash_app(prefix: str = "/dash/") -> Dash:
    """
    Instantiate and configure the Dash application.

    Parameters
    ----------
    prefix : URL prefix used by both the browser and Dash's router,
             e.g. "/dash/"
    """
    # Lazy-import pages to avoid circular imports at module load time
    from dashboard.pages import (
        home, training, data as data_page, models,
        daily_trading, prediction, academic, hyperopt, users,
    )
    from dashboard.auth_context import is_admin

    dash_app = Dash(
        __name__,
        # Tells the JS client where to send API calls (includes the mount prefix)
        requests_pathname_prefix=prefix,
        # Flask registers its own routes at the same prefix; PrefixMiddleware
        # restores the path so they match after Starlette strips it.
        routes_pathname_prefix=prefix,
        external_stylesheets=[
            dbc.themes.DARKLY,
            dbc.icons.BOOTSTRAP,
        ],
        suppress_callback_exceptions=True,
        assets_folder=os.path.join(os.path.dirname(__file__), "assets"),
        title="RL Trading",
        update_title=None,
    )

    # ── Main layout ──────────────────────────────────────────────────────
    # Callable layout: her sayfa yuklemesinde yeniden uretilir; boylece
    # kenar cubugu aktif kullaniciyi ve rolune gore menuyu gosterebilir.
    def serve_layout():
        return html.Div(
            [
                dcc.Location(id="url", refresh=False),
                create_sidebar(),
                html.Div(id="page-content", style=CONTENT_STYLE),
            ],
            style={"backgroundColor": BG, "minHeight": "100vh"},
        )

    dash_app.layout = serve_layout

    # ── Page routing callback ─────────────────────────────────────────────
    from dash import Input, Output

    @dash_app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def display_page(pathname: str):
        """Route URL pathname to the correct page layout."""
        # Normalise: strip prefix, keep the rest as the local path
        p = prefix.rstrip("/")
        path = pathname[len(p):] if pathname and pathname.startswith(p) else pathname
        path = path or "/"

        if path in ("/", ""):
            return home.layout()
        elif path == "/training":
            return training.layout()
        elif path == "/data":
            return data_page.layout()
        elif path == "/models":
            return models.layout()
        elif path == "/daily-trading":
            return daily_trading.layout()
        elif path == "/prediction":
            return prediction.layout()
        elif path == "/academic":
            return academic.layout()
        elif path == "/hyperopt":
            return hyperopt.layout()
        elif path == "/users":
            # Kullanici yonetimi yalnizca admin. URL'i elle yazan kullanici
            # icin de kapali — API tarafi ayrica RequireAdmin ile korunur.
            if not is_admin():
                return html.Div(
                    [
                        html.H3("Yetkisiz erisim", style={"color": "#ef4444"}),
                        html.P("Bu sayfa yalnizca yoneticiler icindir."),
                    ],
                    style={"padding": "40px"},
                )
            return users.layout()
        return html.Div(
            [
                html.H3("404 – Sayfa bulunamadi", style={"color": "#ef4444"}),
                html.P(f"Yol: {pathname}"),
            ],
            style={"padding": "40px"},
        )

    # ── Register page-specific callbacks ─────────────────────────────────
    home.register_callbacks(dash_app)
    training.register_callbacks(dash_app)
    data_page.register_callbacks(dash_app)
    models.register_callbacks(dash_app)
    daily_trading.register_callbacks(dash_app)
    prediction.register_callbacks(dash_app)
    academic.register_callbacks(dash_app)
    hyperopt.register_callbacks(dash_app)
    users.register_callbacks(dash_app)

    return dash_app
