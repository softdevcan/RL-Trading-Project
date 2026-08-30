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

from dashboard.theme import BORDER, CARD2, TEXT, TEXT_MUTED, GREEN, BLUE, YELLOW
from dashboard.components.page_header import create_page_header
from dashboard.components.state_block import create_state_block
import dashboard.api_client as api

# Backend /config/* endpoint'leri ulasilamadiginda kullanilan emniyet listeleri.
# Normalde layout() acilisi /config/algorithms ve /config/phases'ten dinamik yukler.
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


def _algo_options():
    items = api.get_config_algorithms() or _FALLBACK_ALGORITHMS
    return [{"label": it.get("label", it.get("value")), "value": it.get("value")} for it in items]


def _phase_options():
    items = api.get_config_phases() or _FALLBACK_PHASES
    return [{"label": it.get("label", f"Faz {it.get('value')}"), "value": it.get("value")} for it in items]


# ── Layout ────────────────────────────────────────────────────────────────────

def layout():
    return html.Div([
        # Polling interval (starts disabled)
        dcc.Interval(id="training-poll", interval=3_000, disabled=True, n_intervals=0),
        dcc.Store(id="training-store", data={}),

        # Header
        create_page_header("Model Egitimi",
                           "Yeni RL modeli egit ve durumunu izle"),

        dbc.Row([
            # ── Left: Training form ─────────────────────────────────────────
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Egitim Parametreleri", className="card-title-sm")),
                    dbc.CardBody([
                        # Algorithm
                        html.Label("Algoritma", className="section-title"),
                        dcc.Dropdown(
                            id="training-algo",
                            options=_algo_options(),
                            value="ppo",
                            clearable=False,
                            style={"marginBottom": "16px"},
                        ),
                        # Phase
                        html.Label("Faz", className="section-title"),
                        dbc.RadioItems(
                            id="training-phase",
                            options=_phase_options(),
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
                        # Tahmini sure — form degistikce guncellenir
                        html.Div(id="training-estimate", className="mb-3"),
                        # Start button
                        dbc.Button(
                            [html.I(className="bi bi-play-circle me-2"), "Egitimi Baslat"],
                            id="training-start-btn",
                            color="success",
                            className="w-100",
                            size="lg",
                        ),
                    ]),
                ]),
            ], md=4, className="mb-4"),

            # ── Right: Status ───────────────────────────────────────────────
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Egitim Durumu", className="card-title-sm")),
                    dbc.CardBody([
                        html.Div(id="training-status-content", children=_idle_status()),
                    ]),
                ]),
            ], md=8, className="mb-4"),
        ]),
    ])


def _idle_status():
    # Kum saati "bir sey oluyor" izlenimi veriyordu; burada henuz hicbir sey
    # baslamamis durumda. Bos durum blogu (C.7) dogru olan.
    return create_state_block(
        "empty",
        "Egitim baslatilmadi.",
        hint="Soldaki formu doldurup egitimi baslatin.",
    )


# Backend `phase_name` degerlerinin panoda gosterilen karsiliklari.
_PHASE_LABELS = {
    "preparing": "Veri hazirlaniyor",
    "training": "Model egitiliyor",
    "evaluating": "Test setinde degerlendiriliyor",
    "completed": "Tamamlandi",
}


def _eta_row(status):
    """Kalan sure + tahmini bitis saati. ETA yoksa satiri hic basma."""
    eta_text = status.get("eta_text")
    if not eta_text or eta_text == "—":
        return None

    finish_at = status.get("finish_at")
    source = status.get("eta_source")
    # 'prior' = henuz gozlem yok, on tahminden; kullaniciya bunu durustce soyle.
    note = " (on tahmin)" if source == "prior" else ""
    finish_txt = f" · bitis ~{finish_at}" if finish_at else ""

    return dbc.Row([
        dbc.Col(html.Small("Kalan", className="section-title"), width=4),
        dbc.Col(html.Span(
            f"~{eta_text}{finish_txt}{note}",
            className="card-title-sm",
        ), width=8),
    ], className="mb-2")


def _with_warnings(block, status):
    """Backend uyarilarini durum blogunun basina koy.

    Sembol hizalamasi gibi durumlar sessiz kalirsa kullanici 30 sembol
    sandigi bir modeli 5 sembolle egitmis olur — koşum "basarili" gorunur,
    kapsam farkli olur. Uyari yoksa blok oldugu gibi doner.
    """
    messages = status.get("warnings") or []
    if not messages:
        return block
    return html.Div(
        [dbc.Alert([html.I(className="bi bi-exclamation-triangle me-2"), m],
                   color="warning", className="py-2")
         for m in messages] + [block]
    )


def _running_status(status):
    step = status.get("current_step", 0)
    total = status.get("total_steps", 1) or 1
    pct = min(int(step / total * 100), 100)
    elapsed = status.get("elapsed_text") or status.get("elapsed", "—")
    metrics = status.get("metrics", {}) or {}
    phase_label = _PHASE_LABELS.get(status.get("phase_name"), "Egitim devam ediyor")
    rate = status.get("steps_per_sec")

    rows = [
        dbc.Row([
            dbc.Col(html.Small("Adim", className="section-title"), width=4),
            dbc.Col(html.Span(f"{step:,} / {total:,}", className="card-title-sm"), width=8),
        ], className="mb-2"),
        dbc.Progress(value=pct, label=f"{pct}%", color="primary", className="mb-3"),
        dbc.Row([
            dbc.Col(html.Small("Gecen", className="section-title"), width=4),
            dbc.Col(html.Span(str(elapsed), style={"color": TEXT_MUTED}), width=8),
        ], className="mb-2"),
    ]

    eta_row = _eta_row(status)
    if eta_row is not None:
        rows.append(eta_row)

    if rate:
        rows.append(dbc.Row([
            dbc.Col(html.Small("Hiz", className="section-title"), width=4),
            dbc.Col(html.Span(f"{rate:,.0f} adim/sn", style={"color": TEXT_MUTED}), width=8),
        ], className="mb-2"))

    return html.Div([
        dbc.Alert(
            [dbc.Spinner(size="sm", color="primary", spinner_class_name="me-2"),
             f"{phase_label}..."],
            color="primary", className="mb-3",
        ),
        html.Div(rows),
        # Live metrics
        html.Div(_render_live_metrics(metrics), id="training-live-metrics"),
    ])


def _completed_status(status):
    metrics = status.get("metrics", {}) or {}
    # Gerceklesen sure metriklerden gelir (training_time_minutes), yoksa durumdan.
    minutes = metrics.get("training_time_minutes")
    if isinstance(minutes, (int, float)):
        took = f" ({minutes:.1f} dk)"
    else:
        took = f" ({status['elapsed_text']})" if status.get("elapsed_text") else ""

    return html.Div([
        dbc.Alert(
            [html.I(className="bi bi-check-circle-fill me-2"), f"Egitim tamamlandi!{took}"],
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
                sid = s.get("study_id") or s.get("study_name", "")
                best = s.get("best_value")
                best_txt = f"{best:.3f}" if isinstance(best, (int, float)) else "—"
                label = f"{s.get('study_name', sid)} (en iyi: {best_txt})"
                opts.append({"label": label, "value": str(sid)})
        return opts

    @app.callback(
        Output("training-estimate", "children"),
        [
            Input("training-algo", "value"),
            Input("training-phase", "value"),
            Input("training-timesteps", "value"),
        ],
    )
    def update_estimate(algo, phase, timesteps):
        """Form degistikce tahmini sureyi goster (egitim baslamadan once)."""
        if not timesteps or int(timesteps) <= 0:
            return html.Span()

        est = api.get_training_estimate(algo or "ppo", int(phase or 1), int(timesteps))
        if not est or not est.get("total_text"):
            return html.Span()

        confidence = est.get("confidence")
        # Yerlesik katsayidan gelen tahmin kaba olabilir; kullanici bunu bilsin.
        color, icon = {
            "measured": (GREEN, "bi-check-circle"),
            "scaled": (BLUE, "bi-arrows-angle-expand"),
        }.get(confidence, (YELLOW, "bi-exclamation-circle"))

        return html.Div([
            html.Div([
                html.I(className=f"bi {icon} me-2", style={"color": color}),
                html.Span("Tahmini sure: ", style={"color": TEXT_MUTED, "fontSize": "13px"}),
                html.Span(f"~{est['total_text']}", className="card-title-sm"),
            ]),
            html.Small(
                f"{est.get('source', '')} · {est.get('n_symbols', '?')} sembol",
                style={"color": TEXT_MUTED, "fontSize": "11px"},
            ),
        ], style={
            "padding": "10px 12px",
            "borderRadius": "6px",
            "border": f"1px solid {BORDER}",
            "backgroundColor": CARD2,
        })

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
            payload["hyperparameter_study"] = study
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
            return _with_warnings(_running_status(status), status), False
        elif state == "completed":
            return _with_warnings(_completed_status(status), status), True
        elif state == "error":
            return _error_status(status), True
        return _idle_status(), True
