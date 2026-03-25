"""
Training page – start model training, poll status with dcc.Interval.

API endpoints used:
  POST /api/trading/train
  GET  /api/trading/train/status
  GET  /api/hyperopt/studies    (for study dropdown)
"""

from dash import html, dcc
from dash import Input, Output, State
import dash_bootstrap_components as dbc

from dashboard.theme import CARD, CARD2, TEXT, TEXT_MUTED, GREEN, BLUE, ORANGE, RED, YELLOW, empty_figure
import dashboard.api_client as api

ALGORITHMS = ["PPO", "A2C", "SAC", "TD3"]
PHASES = [1, 2]


# ── Layout ────────────────────────────────────────────────────────────────────

def layout():
    return html.Div([
        # Polling interval (starts disabled)
        dcc.Interval(id="training-poll", interval=3_000, disabled=True, n_intervals=0),
        dcc.Store(id="training-store", data={}),

        # Header
        html.H4("Model Egitimi", style={"color": TEXT, "marginBottom": "4px"}),
        html.P("Yeni RL modeli egit ve durumunu izle", style={"color": TEXT_MUTED, "marginBottom": "24px"}),

        dbc.Row([
            # ── Left: Training form ─────────────────────────────────────────
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Egitim Parametreleri", style={"color": TEXT, "fontWeight": "600"})),
                    dbc.CardBody([
                        # Algorithm
                        html.Label("Algoritma", className="section-title"),
                        dcc.Dropdown(
                            id="training-algo",
                            options=[{"label": a, "value": a} for a in ALGORITHMS],
                            value="PPO",
                            clearable=False,
                            style={"marginBottom": "16px"},
                        ),
                        # Phase
                        html.Label("Faz", className="section-title"),
                        dbc.RadioItems(
                            id="training-phase",
                            options=[{"label": f"Faz {p}", "value": p} for p in PHASES],
                            value=1,
                            inline=True,
                            className="mb-3",
                        ),
                        # Timesteps
                        html.Label("Timestep Sayisi", className="section-title"),
                        dbc.Input(
                            id="training-timesteps",
                            type="number",
                            value=50_000,
                            min=1_000,
                            max=2_000_000,
                            step=1_000,
                            className="mb-3",
                        ),
                        # Initial balance
                        html.Label("Baslangic Bakiyesi (₺)", className="section-title"),
                        dbc.Input(
                            id="training-balance",
                            type="number",
                            value=100_000,
                            min=10_000,
                            className="mb-3",
                        ),
                        # Hyperopt study
                        html.Label("Hiper Parametre Calismasi", className="section-title"),
                        dcc.Dropdown(
                            id="training-study",
                            options=[{"label": "Varsayilan parametreler", "value": ""}],
                            value="",
                            clearable=False,
                            placeholder="Calisma sec (opsiyonel)...",
                            style={"marginBottom": "24px"},
                        ),
                        # Start button
                        dbc.Button(
                            [html.I(className="bi bi-play-circle me-2"), "Egitimi Baslat"],
                            id="training-start-btn",
                            color="success",
                            className="w-100",
                            size="lg",
                        ),
                    ]),
                ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
            ], md=4, className="mb-4"),

            # ── Right: Status ───────────────────────────────────────────────
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Egitim Durumu", style={"color": TEXT, "fontWeight": "600"})),
                    dbc.CardBody([
                        html.Div(id="training-status-content", children=_idle_status()),
                    ]),
                ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
            ], md=8, className="mb-4"),
        ]),
    ])


def _idle_status():
    return html.Div([
        html.Div(
            html.I(className="bi bi-hourglass text-muted", style={"fontSize": "48px"}),
            className="text-center py-4",
        ),
        html.P("Egitim baslatilmadi.", style={"color": TEXT_MUTED, "textAlign": "center"}),
    ])


def _running_status(status):
    step = status.get("current_step", 0)
    total = status.get("total_steps", 1) or 1
    pct = min(int(step / total * 100), 100)
    elapsed = status.get("elapsed", "—")
    metrics = status.get("metrics", {}) or {}

    return html.Div([
        dbc.Alert(
            [dbc.Spinner(size="sm", color="primary", className="me-2"), "Egitim devam ediyor..."],
            color="primary", className="mb-3",
        ),
        html.Div([
            dbc.Row([
                dbc.Col(html.Small("Adim", className="section-title"), width=4),
                dbc.Col(html.Span(f"{step:,} / {total:,}", style={"color": TEXT, "fontWeight": "600"}), width=8),
            ], className="mb-2"),
            dbc.Progress(value=pct, label=f"{pct}%", color="primary", className="mb-3"),
            dbc.Row([
                dbc.Col(html.Small("Sure", className="section-title"), width=4),
                dbc.Col(html.Span(str(elapsed), style={"color": TEXT_MUTED}), width=8),
            ], className="mb-2"),
        ]),
        # Live metrics
        html.Div(_render_live_metrics(metrics), id="training-live-metrics"),
    ])


def _completed_status(status):
    metrics = status.get("metrics", {}) or {}
    return html.Div([
        dbc.Alert(
            [html.I(className="bi bi-check-circle-fill me-2"), "Egitim tamamlandi!"],
            color="success", className="mb-3",
        ),
        _render_live_metrics(metrics),
        dbc.Button(
            [html.I(className="bi bi-arrow-clockwise me-2"), "Yeni Egitim"],
            id="training-reset-btn",
            color="secondary",
            outline=True,
            className="mt-3 w-100",
        ),
    ])


def _error_status(status):
    return dbc.Alert(
        [html.I(className="bi bi-x-circle-fill me-2"), f"Hata: {status.get('error', 'Bilinmeyen hata')}"],
        color="danger",
    )


def _render_live_metrics(metrics):
    if not metrics:
        return html.Span()
    items = []
    for k, v in list(metrics.items())[:6]:
        label = k.replace("_", " ").title()
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        items.append(
            dbc.Row([
                dbc.Col(html.Small(label, style={"color": TEXT_MUTED}), width=6),
                dbc.Col(html.Span(val, style={"color": TEXT, "fontSize": "13px", "fontWeight": "600"}), width=6),
            ], className="py-1 border-bottom")
        )
    return html.Div(items)


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("training-study", "options"),
        Input("training-algo", "value"),
    )
    def update_study_options(algo):
        opts = [{"label": "Varsayilan parametreler", "value": ""}]
        studies = api.get_hyperopt_studies()
        for s in studies:
            s_algo = s.get("algorithm", "").upper()
            if s_algo == (algo or "").upper():
                sid = s.get("id", s.get("study_name", ""))
                label = f"{s.get('study_name', sid)} (en iyi: {s.get('best_value', 0):.3f})"
                opts.append({"label": label, "value": str(sid)})
        return opts

    @app.callback(
        [Output("training-poll", "disabled"), Output("training-store", "data")],
        Input("training-start-btn", "n_clicks"),
        [
            State("training-algo", "value"),
            State("training-phase", "value"),
            State("training-timesteps", "value"),
            State("training-balance", "value"),
            State("training-study", "value"),
        ],
        prevent_initial_call=True,
    )
    def start_training(n, algo, phase, timesteps, balance, study):
        if not n:
            return True, {}
        payload = {
            "algorithm": algo,
            "phase": int(phase or 1),
            "total_timesteps": int(timesteps or 50_000),
            "initial_balance": float(balance or 100_000),
        }
        if study:
            payload["hyperopt_study_id"] = study
        result = api.start_training(payload)
        return False, result or {}

    @app.callback(
        [Output("training-status-content", "children"), Output("training-poll", "disabled", allow_duplicate=True)],
        Input("training-poll", "n_intervals"),
        prevent_initial_call=True,
    )
    def poll_training(n):
        status = api.get_training_status()
        state = status.get("state", status.get("status", "idle"))

        if state == "running":
            return _running_status(status), False
        elif state == "completed":
            return _completed_status(status), True
        elif state == "error":
            return _error_status(status), True
        return _idle_status(), True
