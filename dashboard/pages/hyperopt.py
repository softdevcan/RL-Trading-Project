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

    # CSV mevcut ise tarih/sembol sınırlarını oradan dolduralım; yoksa
    # makul fallback (son 1 yıl, sembol seçici boş).
    data_range = api.get_hyperopt_data_range() or {"available": False, "symbols": []}
    cached_available = bool(data_range.get("available"))
    csv_min = data_range.get("min_date") or str(default_start)
    csv_max = data_range.get("max_date") or str(today)
    csv_symbols = data_range.get("symbols", []) or []

    # Default penceresi: train = ilk %80, val = son %20
    csv_min_d = date.fromisoformat(csv_min)
    csv_max_d = date.fromisoformat(csv_max)
    total_days = (csv_max_d - csv_min_d).days
    split_d = csv_min_d + timedelta(days=int(total_days * 0.8))

    csv_status_msg = (
        f"CSV: {csv_min} → {csv_max} • {data_range.get('total_rows', 0)} satır • "
        f"{len(csv_symbols)} sembol"
        if cached_available
        else "CSV yok — 'yfinance'tan tazele' otomatik açık."
    )

    return html.Div([
        dcc.Interval(id="hyperopt-poll", interval=3_000, disabled=True, n_intervals=0),
        dcc.Store(id="hyperopt-modal-study-id", data=None),
        dcc.Store(id="hyperopt-data-range", data=data_range),

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
                        # timesteps (schema: ge=10_000, le=1_000_000)
                        html.Label("Timestep / Trial (min 10000)", className="section-title"),
                        dbc.Input(id="hyperopt-timesteps", type="number", value=10_000,
                                  min=10_000, max=1_000_000, step=1_000,
                                  className="mb-3"),

                        # Data source
                        html.Label("Veri Kaynağı", className="section-title"),
                        dbc.Checklist(
                            id="hyperopt-refresh-data",
                            options=[{"label": " yfinance'tan tazele (CSV ezilir)", "value": "refresh"}],
                            value=[] if cached_available else ["refresh"],
                            switch=True,
                            className="mb-2",
                        ),
                        html.Small(csv_status_msg, id="hyperopt-csv-status",
                                   style={"color": TEXT_MUTED, "display": "block", "marginBottom": "12px"}),

                        # Symbols
                        html.Label("Semboller", className="section-title"),
                        dcc.Dropdown(
                            id="hyperopt-symbols",
                            options=[{"label": s, "value": s} for s in csv_symbols],
                            value=csv_symbols,
                            multi=True,
                            placeholder="Boş = PHASE1_SYMBOLS (default)",
                            style={"marginBottom": "12px", "color": CARD},
                        ),

                        # Train date range (val = train_end +1 → csv_max)
                        html.Label("Eğitim Aralığı (val otomatik son %20)", className="section-title"),
                        dbc.Row([
                            dbc.Col(dcc.DatePickerSingle(
                                id="hyperopt-train-start",
                                date=str(csv_min_d), min_date_allowed=csv_min, max_date_allowed=csv_max,
                                display_format="YYYY-MM-DD"), width=6),
                            dbc.Col(dcc.DatePickerSingle(
                                id="hyperopt-train-end",
                                date=str(split_d), min_date_allowed=csv_min, max_date_allowed=csv_max,
                                display_format="YYYY-MM-DD"), width=6),
                        ], className="mb-2"),
                        html.Small(id="hyperopt-val-preview",
                                   style={"color": TEXT_MUTED, "display": "block", "marginBottom": "12px"}),

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
        Output("hyperopt-val-preview", "children"),
        [Input("hyperopt-train-end", "date"),
         Input("hyperopt-data-range", "data")],
    )
    def update_val_preview(train_end, data_range):
        """Val = train_end+1 → csv_max. Görsel feedback."""
        if not train_end:
            return ""
        try:
            te = date.fromisoformat(str(train_end)[:10])
            csv_max = (data_range or {}).get("max_date")
            if not csv_max:
                return f"Val: {te + timedelta(days=1)} → (CSV yok)"
            vmax = date.fromisoformat(csv_max)
            vmin = te + timedelta(days=1)
            if vmin >= vmax:
                return html.Span(
                    f"⚠ train_end ({te}) çok geç — val aralığı boş",
                    style={"color": RED}
                )
            return f"Val: {vmin} → {vmax} ({(vmax - vmin).days} gün)"
        except Exception:
            return ""

    @app.callback(
        [Output("hyperopt-poll", "disabled"), Output("hyperopt-start-result", "children")],
        Input("hyperopt-start-btn", "n_clicks"),
        [
            State("hyperopt-algo", "value"),
            State("hyperopt-phase", "value"),
            State("hyperopt-reward", "value"),
            State("hyperopt-trials", "value"),
            State("hyperopt-timesteps", "value"),
            State("hyperopt-train-start", "date"),
            State("hyperopt-train-end", "date"),
            State("hyperopt-refresh-data", "value"),
            State("hyperopt-symbols", "value"),
            State("hyperopt-data-range", "data"),
        ],
        prevent_initial_call=True,
    )
    def start_optimization(n, algo, phase, reward, trials, timesteps,
                           train_start, train_end, refresh, symbols, data_range):
        if not n:
            return True, html.Span()

        if not train_start or not train_end:
            return True, dbc.Alert("Eğitim tarihleri eksik.", color="danger", dismissable=True)

        # val = train_end+1 → csv_max
        try:
            te = date.fromisoformat(str(train_end)[:10])
        except Exception:
            return True, dbc.Alert("Train end tarihi geçersiz.", color="danger", dismissable=True)
        val_start_d = te + timedelta(days=1)
        csv_max = (data_range or {}).get("max_date") or str(date.today())
        try:
            val_end_d = date.fromisoformat(csv_max)
        except Exception:
            val_end_d = date.today()
        if val_start_d >= val_end_d:
            return True, dbc.Alert(
                "Val aralığı boş — train_end'i öne çekin veya veri çekin.",
                color="danger", dismissable=True,
            )

        payload = {
            "algorithm": algo,
            "phase": int(phase or 1),
            "reward_type": reward or "psr",
            "n_trials": int(trials or 20),
            "total_timesteps": int(timesteps or 10_000),
            "train_start": str(train_start)[:10],
            "train_end": str(train_end)[:10],
            "val_start": str(val_start_d),
            "val_end": str(val_end_d),
            "use_cached_data": "refresh" not in (refresh or []),
            "stock_symbols": symbols or None,
        }
        result = api.start_hyperopt(payload)
        if result:
            sid = result.get('study_id', result.get('id', 'OK'))
            info = (
                f"Train {payload['train_start']}→{payload['train_end']} • "
                f"Val {payload['val_start']}→{payload['val_end']} • "
                f"Kaynak: {'CSV' if payload['use_cached_data'] else 'yfinance'}"
            )
            return False, dbc.Alert(
                [html.I(className="bi bi-check-circle me-2"),
                 f"Optimizasyon baslatildi: {sid}", html.Br(), html.Small(info)],
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
        detail = api.get_hyperopt_study(str(study_id)) or {}

        # Backend StudyDetailResponse: { study: {...}, trials: [...], mean_value, ... }
        # Fall back to flat shape if a caller ever returns one.
        study_obj = detail.get("study") if isinstance(detail.get("study"), dict) else detail
        trials = detail.get("trials", []) or []
        stats = {
            "mean_value": detail.get("mean_value"),
            "median_value": detail.get("median_value"),
            "std_value": detail.get("std_value"),
            "min_value": detail.get("min_value"),
            "max_value": detail.get("max_value"),
        }

        title = study_obj.get("study_name", study_obj.get("name", str(study_id)))
        info_div = _render_modal_info(study_obj, stats, total_trials=len(trials))
        params_str = json.dumps(study_obj.get("best_params") or {}, indent=2, ensure_ascii=False) or "{}"
        trials_div = _render_trials_table(trials)

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
        sid = s.get("study_id") or s.get("id") or s.get("study_name") or "unknown"
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


def _render_modal_info(study, stats=None, total_trials=None):
    stats = stats or {}
    best_val = study.get("best_value")
    fields = [
        ("Calisma Adi", study.get("study_name", "—")),
        ("Algoritma", str(study.get("algorithm", "—")).upper()),
        ("Durum", str(study.get("status", study.get("state", "—"))).upper()),
        ("Faz", str(study.get("phase", "—"))),
        ("Odul Tipi", str(study.get("reward_type", "—")).upper()),
        ("Tamamlanan", f"{study.get('trials_completed', 0)}/{study.get('n_trials', total_trials or '—')}"),
        ("Pruned/Failed", f"{study.get('trials_pruned', 0)}/{study.get('trials_failed', 0)}"),
        ("En Iyi Deger", f"{best_val:.4f}" if isinstance(best_val, (int, float)) else "—"),
        ("En Iyi Trial #", str(study.get("best_trial")) if study.get("best_trial") is not None else "—"),
        ("Train Aralığı", f"{study.get('train_start', '—')} → {study.get('train_end', '—')}"),
        ("Val Aralığı", f"{study.get('val_start', '—')} → {study.get('val_end', '—')}"),
        ("Olusturuldu", str(study.get("created_at", "—"))[:19]),
    ]
    # Stats only when present (running studies have None)
    if stats.get("mean_value") is not None:
        fields.extend([
            ("Ort. Deger", f"{stats['mean_value']:.4f}"),
            ("Medyan", f"{stats.get('median_value', 0):.4f}"),
            ("Std", f"{stats.get('std_value', 0):.4f}"),
            ("Min / Max", f"{stats.get('min_value', 0):.4f} / {stats.get('max_value', 0):.4f}"),
        ])
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
    # Schema: trial_number, state (TrialState enum string), value, params, duration_seconds
    for t in trials[:25]:
        val = t.get("value") if t.get("value") is not None else 0
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = 0.0
        status = str(t.get("state", t.get("status", "—"))).lower()
        status_color = "success" if status == "complete" else "warning" if status == "pruned" else "danger" if "fail" in status else "secondary"
        duration = t.get("duration_seconds", t.get("duration", t.get("elapsed")))
        duration_str = f"{float(duration):.1f}" if duration not in (None, "—", "") else "—"
        try:
            params_str = json.dumps(t.get("params", {}), ensure_ascii=False)
            if len(params_str) > 80:
                params_str = params_str[:77] + "..."
        except Exception:
            params_str = str(t.get("params", ""))[:80]

        trial_num = t.get("trial_number", t.get("number", t.get("id", "—")))

        rows.append(dbc.Row([
            dbc.Col(html.Small(str(trial_num), style={"color": TEXT_MUTED}), width=1),
            dbc.Col(html.Small(f"{val:.4f}", style={"color": GREEN if val > 0 else RED}), width=2),
            dbc.Col(dbc.Badge(status.upper(), color=status_color, pill=True, style={"fontSize": "10px"}), width=2),
            dbc.Col(html.Small(duration_str, style={"color": TEXT_MUTED}), width=2),
            dbc.Col(html.Small(params_str, style={"color": TEXT_MUTED, "fontSize": "11px"}), width=5),
        ], className="py-1 border-bottom"))
    return html.Div(rows)
