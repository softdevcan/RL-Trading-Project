"""
Hyperparameter Optimization page – start studies, poll progress, modal details.

API endpoints used:
  POST /api/hyperopt/start
  GET  /api/hyperopt/studies
  GET  /api/hyperopt/studies/{id}
  GET  /api/hyperopt/studies/{id}/progress
  GET  /api/hyperopt/search-spaces/{algorithm}
"""

import json
from datetime import date, timedelta

from dash import html, dcc
from dash import ALL as _ALL
from dash import Input, Output, State
import dash_bootstrap_components as dbc

from dashboard.theme import (
    CARD, CARD2, TEXT, TEXT_MUTED, BORDER, GREEN, RED, BLUE, PURPLE, ORANGE, YELLOW,
    ALGO_COLORS, empty_figure, apply_dark_template,
)
import dashboard.api_client as api

# Backend /config/* ulasilamadiginda kullanilan emniyet listeleri.
_FALLBACK_ALGORITHMS = [
    {"value": "ppo", "label": "PPO"},
    {"value": "a2c", "label": "A2C"},
    {"value": "td3", "label": "TD3"},
    {"value": "sac", "label": "SAC"},
]
_FALLBACK_PHASES = [
    {"value": 1, "label": "Faz 1"},
    {"value": 2, "label": "Faz 2"},
]
_FALLBACK_REWARD_TYPES = [
    {"value": "psr", "label": "PSR"},
    {"value": "simple", "label": "Simple"},
]


def _algo_options():
    items = api.get_config_algorithms() or _FALLBACK_ALGORITHMS
    return [{"label": it.get("label", it.get("value")), "value": it.get("value")} for it in items]


def _phase_options():
    items = api.get_config_phases() or _FALLBACK_PHASES
    return [{"label": it.get("label", f"Faz {it.get('value')}"), "value": it.get("value")} for it in items]


def _reward_options():
    items = api.get_config_reward_types() or _FALLBACK_REWARD_TYPES
    return [{"label": it.get("label", it.get("value", "").upper()), "value": it.get("value")} for it in items]


# ── Layout ────────────────────────────────────────────────────────────────────

def layout():
    today = date.today()
    default_start = today - timedelta(days=365)

    return html.Div([
        dcc.Interval(id="hyperopt-poll", interval=3_000, disabled=True, n_intervals=0),
        dcc.Store(id="hyperopt-modal-study-id", data=None),

        html.H4("Hiper Parametre Optimizasyonu", style={"color": TEXT, "marginBottom": "4px"}),
        html.P("Optuna ile otomatik hiper parametre arama", style={"color": TEXT_MUTED, "marginBottom": "24px"}),

        dbc.Row([
            # ── Form panel ──────────────────────────────────────────────────
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Optimizasyon Baslat", style={"color": TEXT, "fontWeight": "600"})),
                    dbc.CardBody([
                        # Algorithm
                        html.Label("Algoritma", className="section-title"),
                        dcc.Dropdown(
                            id="hyperopt-algo",
                            options=_algo_options(),
                            value="ppo", clearable=False,
                            style={"marginBottom": "16px", "color": CARD},
                        ),
                        # Phase
                        html.Label("Faz", className="section-title"),
                        dbc.RadioItems(
                            id="hyperopt-phase",
                            options=_phase_options(),
                            value=1, inline=True, className="mb-3",
                        ),
                        # Reward type
                        html.Label("Ödül Tipi", className="section-title"),
                        dcc.Dropdown(
                            id="hyperopt-reward",
                            options=_reward_options(),
                            value="psr", clearable=False,
                            style={"marginBottom": "16px", "color": CARD},
                        ),
                        # n_trials
                        html.Label("Deneme Sayisi (n_trials)", className="section-title"),
                        dbc.Input(id="hyperopt-trials", type="number", value=20, min=1, max=500,
                                  className="mb-3"),
                        # timesteps
                        html.Label("Timestep / Trial", className="section-title"),
                        dbc.Input(id="hyperopt-timesteps", type="number", value=10_000, min=1_000,
                                  className="mb-3"),
                        # Date range
                        html.Label("Egitim Tarihleri", className="section-title"),
                        dbc.Row([
                            dbc.Col(dcc.DatePickerSingle(id="hyperopt-start-date", date=str(default_start),
                                                          display_format="YYYY-MM-DD"), width=6),
                            dbc.Col(dcc.DatePickerSingle(id="hyperopt-end-date", date=str(today),
                                                          display_format="YYYY-MM-DD"), width=6),
                        ], className="mb-3"),
                        # Search space info
                        html.Div(id="hyperopt-space-info", className="mb-3"),
                        # Start button
                        dbc.Button(
                            [html.I(className="bi bi-play-circle me-2"), "Optimizasyonu Baslat"],
                            id="hyperopt-start-btn",
                            color="success",
                            className="w-100",
                        ),
                        html.Div(id="hyperopt-start-result", className="mt-3"),
                    ]),
                ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
            ], md=4, className="mb-4"),

            # ── Studies grid ────────────────────────────────────────────────
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            [html.I(className="bi bi-arrow-clockwise me-1"), "Yenile"],
                            id="hyperopt-refresh-btn",
                            color="secondary", outline=True, size="sm",
                        ),
                    ], className="mb-3 d-flex justify-content-end"),
                ]),
                html.Div(id="hyperopt-studies-grid"),
            ], md=8, className="mb-4"),
        ]),

        # ── Detail modal ─────────────────────────────────────────────────────
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(html.Span(id="hyperopt-modal-title")), close_button=True),
            dbc.ModalBody([
                dbc.Row([
                    dbc.Col([
                        html.Div(id="hyperopt-modal-info"),
                    ], md=6),
                    dbc.Col([
                        html.H6("En Iyi Parametreler", style={"color": TEXT_MUTED}),
                        html.Pre(id="hyperopt-modal-params",
                                  style={"backgroundColor": CARD2, "padding": "12px",
                                         "borderRadius": "6px", "color": TEXT, "fontSize": "12px",
                                         "overflowX": "auto"}),
                    ], md=6),
                ], className="mb-3"),
                html.H6("Trial Sonuclari", style={"color": TEXT_MUTED, "marginTop": "16px"}),
                html.Div(id="hyperopt-modal-trials"),
            ]),
            dbc.ModalFooter(dbc.Button("Kapat", id="hyperopt-modal-close", color="secondary")),
        ], id="hyperopt-modal", size="xl", is_open=False),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("hyperopt-space-info", "children"),
        Input("hyperopt-algo", "value"),
        prevent_initial_call=False,
    )
    def load_search_space(algo):
        if not algo:
            return html.Span()
        spaces = api.get_search_spaces(algo) or {}
        if not spaces:
            return html.Small(f"{algo} arama uzayi bilgisi yok.", style={"color": TEXT_MUTED})
        n = len(spaces.get("parameters", spaces)) if isinstance(spaces, dict) else 0
        return html.Small(f"{algo} icin {n} hiper parametre aranacak.", style={"color": TEXT_MUTED})

    @app.callback(
        [Output("hyperopt-poll", "disabled"), Output("hyperopt-start-result", "children")],
        Input("hyperopt-start-btn", "n_clicks"),
        [
            State("hyperopt-algo", "value"),
            State("hyperopt-phase", "value"),
            State("hyperopt-reward", "value"),
            State("hyperopt-trials", "value"),
            State("hyperopt-timesteps", "value"),
            State("hyperopt-start-date", "date"),
            State("hyperopt-end-date", "date"),
        ],
        prevent_initial_call=True,
    )
    def start_optimization(n, algo, phase, reward, trials, timesteps, start_date, end_date):
        if not n:
            return True, html.Span()
        payload = {
            "algorithm": algo,
            "phase": int(phase or 1),
            "reward_type": reward,
            "n_trials": int(trials or 20),
            "timesteps_per_trial": int(timesteps or 10_000),
            "start_date": start_date,
            "end_date": end_date,
        }
        result = api.start_hyperopt(payload)
        if result:
            return False, dbc.Alert(
                [html.I(className="bi bi-check-circle me-2"), f"Optimizasyon baslatildi: {result.get('study_id', result.get('id', 'OK'))}"],
                color="success", dismissable=True,
            )
        return True, dbc.Alert("Optimizasyon baslatılamadı.", color="danger", dismissable=True)

    @app.callback(
        Output("hyperopt-studies-grid", "children"),
        [Input("hyperopt-refresh-btn", "n_clicks"), Input("hyperopt-poll", "n_intervals")],
        prevent_initial_call=False,
    )
    def refresh_studies(n_ref, n_poll):
        studies = api.get_hyperopt_studies()
        if not studies:
            return html.P("Kayitli optimizasyon calismasi yok.", style={"color": TEXT_MUTED})
        return _render_studies_grid(studies)

    @app.callback(
        [
            Output("hyperopt-modal", "is_open"),
            Output("hyperopt-modal-title", "children"),
            Output("hyperopt-modal-info", "children"),
            Output("hyperopt-modal-params", "children"),
            Output("hyperopt-modal-trials", "children"),
            Output("hyperopt-modal-study-id", "data"),
        ],
        [Input({"type": "hyperopt-study-card", "index": _ALL}, "n_clicks"),
         Input("hyperopt-modal-close", "n_clicks")],
        prevent_initial_call=True,
    )
    def open_close_modal(card_clicks, close_n):
        from dash import ctx
        if ctx.triggered_id == "hyperopt-modal-close" or not any(card_clicks or []):
            return False, "", html.Span(), "", html.Span(), None

        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return False, "", html.Span(), "", html.Span(), None

        study_id = triggered.get("index")
        study = api.get_hyperopt_study(str(study_id)) or {}

        title = study.get("study_name", study.get("name", str(study_id)))
        info_div = _render_modal_info(study)
        params_str = json.dumps(study.get("best_params", {}), indent=2, ensure_ascii=False)
        trials_div = _render_trials_table(study.get("trials", []))

        return True, title, info_div, params_str, trials_div, study_id

    @app.callback(
        Output("hyperopt-poll", "disabled", allow_duplicate=True),
        Input("hyperopt-poll", "n_intervals"),
        prevent_initial_call=True,
    )
    def auto_stop_poll(n):
        # Stop after 100 polls (5 minutes)
        return (n or 0) >= 100


# ── Helpers ───────────────────────────────────────────────────────────────────

def _render_studies_grid(studies):
    cols = []
    for s in studies:
        sid = s.get("id", s.get("study_name", "unknown"))
        algo = str(s.get("algorithm", "—")).upper()
        status = str(s.get("status", s.get("state", "unknown"))).lower()
        best = s.get("best_value", s.get("best_sharpe", 0)) or 0
        n_trials = s.get("n_trials", s.get("completed_trials", 0)) or 0
        total_trials = s.get("total_trials", s.get("n_trials_target", n_trials)) or n_trials
        progress = int(n_trials / total_trials * 100) if total_trials else 0

        status_color = "success" if status == "complete" else "warning" if status == "running" else "secondary"
        algo_color = ALGO_COLORS.get(algo, BLUE)

        cols.append(dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dbc.Badge(algo, style={"backgroundColor": algo_color}, pill=True), width="auto"),
                        dbc.Col(dbc.Badge(status.upper(), color=status_color, pill=True), width="auto", className="ms-auto"),
                    ], className="mb-2"),
                    html.Div(str(s.get("study_name", sid))[:30],
                             style={"color": TEXT, "fontWeight": "600", "fontSize": "13px", "marginBottom": "8px"}),
                    html.Small(f"Deneme: {n_trials}/{total_trials}", style={"color": TEXT_MUTED}),
                    dbc.Progress(value=progress, color="primary", style={"height": "4px"}, className="my-2"),
                    dbc.Row([
                        dbc.Col(html.Small("En Iyi", style={"color": TEXT_MUTED}), width=5),
                        dbc.Col(html.Span(f"{best:.4f}", style={"color": GREEN, "fontWeight": "700", "fontSize": "14px"}), width=7),
                    ]),
                    html.Div(
                        dbc.Button("Detay", id={"type": "hyperopt-study-card", "index": str(sid)},
                                   size="sm", color="primary", outline=True, className="w-100 mt-2"),
                    ),
                ])
            ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
            md=4, sm=6, xs=12, className="mb-3",
        ))
    return dbc.Row(cols)


def _render_modal_info(study):
    fields = [
        ("Calisma Adi", study.get("study_name", "—")),
        ("Algoritma", str(study.get("algorithm", "—")).upper()),
        ("Durum", str(study.get("status", study.get("state", "—"))).upper()),
        ("Tamamlanan", str(study.get("completed_trials", study.get("n_trials", "—")))),
        ("En Iyi Deger", f"{study.get('best_value', 0):.4f}" if study.get("best_value") else "—"),
        ("Odul Tipi", study.get("reward_type", "—")),
        ("Baslangic", str(study.get("created_at", study.get("start_time", "—")))[:16]),
    ]
    rows = []
    for label, val in fields:
        rows.append(dbc.Row([
            dbc.Col(html.Small(label, style={"color": TEXT_MUTED}), width=6),
            dbc.Col(html.Span(str(val), style={"color": TEXT, "fontWeight": "600", "fontSize": "13px"}), width=6),
        ], className="py-1 border-bottom"))
    return html.Div(rows)


def _render_trials_table(trials):
    if not trials:
        return html.P("Trial verisi yok.", style={"color": TEXT_MUTED})

    header = dbc.Row([
        dbc.Col(html.Small("#", className="section-title"), width=1),
        dbc.Col(html.Small("Deger", className="section-title"), width=2),
        dbc.Col(html.Small("Durum", className="section-title"), width=2),
        dbc.Col(html.Small("Sure (s)", className="section-title"), width=2),
        dbc.Col(html.Small("Parametreler", className="section-title"), width=5),
    ])
    rows = [header]
    for t in trials[:15]:
        val = t.get("value", t.get("score", 0)) or 0
        status = str(t.get("state", t.get("status", "—"))).lower()
        status_color = "success" if status == "complete" else "danger" if "fail" in status else "secondary"
        duration = t.get("duration", t.get("elapsed", "—"))
        params_str = json.dumps(t.get("params", {}), ensure_ascii=False)[:60] + "..."

        rows.append(dbc.Row([
            dbc.Col(html.Small(str(t.get("number", t.get("id", "—"))), style={"color": TEXT_MUTED}), width=1),
            dbc.Col(html.Small(f"{val:.4f}", style={"color": GREEN if val > 0 else RED}), width=2),
            dbc.Col(dbc.Badge(status.upper(), color=status_color, pill=True, style={"fontSize": "10px"}), width=2),
            dbc.Col(html.Small(str(duration)[:8] if duration else "—", style={"color": TEXT_MUTED}), width=2),
            dbc.Col(html.Small(params_str, style={"color": TEXT_MUTED, "fontSize": "11px"}), width=5),
        ], className="py-1 border-bottom"))
    return html.Div(rows)
