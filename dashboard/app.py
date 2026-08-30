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

from dashboard.theme import BG, CONTENT_STYLE, RED, TEXT_MUTED
from dashboard.components.sidebar import create_sidebar


def _index_template() -> str:
    """Dash'in HTML iskeleti + FOUC engelleyici (Faz 8, B.5).

    <head> icindeki script SENKRON calisir: <body> boyanmadan once tema
    damgasini koyar. Olmazsa sayfa once aydinlik acilip koyuya atliyor.

    Ayni script `rlt_theme_r` cerezini de tazeler — "system" seciliyken
    sunucunun Plotly figurunu dogru palette uretmesi buna bagli.
    """
    from app.core.config import get_settings

    s = get_settings()
    script = """
    <script>
    (function () {
      try {
        var read = function (n) {
          var m = document.cookie.match(new RegExp("(?:^|; )" + n + "=([^;]*)"));
          return m ? decodeURIComponent(m[1]) : "";
        };
        var pref = read("__THEME__");
        if (["light", "dark", "system"].indexOf(pref) === -1) pref = "system";
        if (pref === "system") {
          document.documentElement.removeAttribute("data-theme");
        } else {
          document.documentElement.setAttribute("data-theme", pref);
        }
        var resolved = pref !== "system" ? pref
          : (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
             ? "dark" : "light");
        document.cookie = "__RESOLVED__=" + resolved + ";path=/;max-age=31536000;samesite=lax";
      } catch (e) { /* tema cozulemedi: tokens.css tabani (aydinlik) gecerli */ }
    })();
    </script>
"""
    script = script.replace("__THEME__", s.THEME_COOKIE_NAME)
    script = script.replace("__RESOLVED__", s.THEME_RESOLVED_COOKIE_NAME)

    return """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
""" + script + """        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""


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
        daily_trading, prediction, academic, hyperopt, users, account,
    )
    from dashboard.auth_context import is_admin

    dash_app = Dash(
        __name__,
        # Tells the JS client where to send API calls (includes the mount prefix)
        requests_pathname_prefix=prefix,
        # Flask registers its own routes at the same prefix; PrefixMiddleware
        # restores the path so they match after Starlette strips it.
        routes_pathname_prefix=prefix,
        # Notr taban: renk kararini tamamen static/tokens.css veriyor.
        # Faz 8 oncesi DARKLY (derlenmis koyu) vardi ve aydinlik temayi
        # imkansiz kiliyordu.
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            dbc.icons.BOOTSTRAP,
        ],
        suppress_callback_exceptions=True,
        assets_folder=os.path.join(os.path.dirname(__file__), "assets"),
        title="RL Trading",
        update_title=None,
    )

    # FOUC engeli: <body> boyanmadan once temayi damgala (Faz 8, B.5)
    dash_app.index_string = _index_template()

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
        elif path == "/account":
            return account.layout()
        elif path == "/users":
            # Kullanici yonetimi yalnizca admin. URL'i elle yazan kullanici
            # icin de kapali — API tarafi ayrica RequireAdmin ile korunur.
            if not is_admin():
                return html.Div(
                    [
                        html.H3("Yetkisiz erisim", style={"color": RED}),
                        html.P("Bu sayfa yalnizca yoneticiler icindir.", style={"color": TEXT_MUTED}),
                    ],
                    style={"padding": "40px"},
                )
            return users.layout()
        return html.Div(
            [
                html.H3("404 – Sayfa bulunamadi", style={"color": RED}),
                html.P(f"Yol: {pathname}", style={"color": TEXT_MUTED}),
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
    account.register_callbacks(dash_app)

    return dash_app
