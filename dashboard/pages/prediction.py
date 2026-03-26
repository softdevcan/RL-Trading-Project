"""
Price Prediction page – gold price cards, symbol prediction, charts.

API endpoints used:
  GET  /api/prediction/symbols
  POST /api/prediction/train
  POST /api/prediction/predict
  POST /api/prediction/evaluate
  GET  /api/prediction/performance/{symbol}
  GET  /api/prediction/predictions/{symbol}
  GET  /api/prediction/chart-data/{symbol}
  GET  /api/prediction/gold/prices
  GET  /api/prediction/gold/history
"""

from typing import Any
from dash import html, dcc
from dash import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from dashboard.theme import (
    CARD, CARD2, TEXT, TEXT_MUTED, BORDER, GREEN, RED, BLUE, GOLD, PURPLE, CYAN, YELLOW,
    DARK_TEMPLATE, empty_figure, apply_dark_template,
)
import dashboard.api_client as api

HORIZONS = [
    {"label": "Gunluk (1 gun sonrasi)",  "value": "daily"},
    {"label": "Haftalik (5 gun sonrasi)", "value": "weekly"},
]


# ── Layout ────────────────────────────────────────────────────────────────────

def layout():
    return html.Div([
        dcc.Interval(id="pred-refresh", interval=60_000, n_intervals=0),
        dcc.Store(id="pred-result-store", data={}),

        html.H4("Fiyat Tahmini", style={"color": TEXT, "marginBottom": "4px"}),
        html.P("Hisse ve altin fiyati tahmin modelleri", style={"color": TEXT_MUTED, "marginBottom": "24px"}),

        # ── Gold section ──────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Altin Fiyatlari", style={"color": TEXT, "fontWeight": "600"})),
                    dbc.CardBody([
                        dbc.Row(id="pred-gold-cards", className="g-2"),
                    ]),
                ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
            ], md=5, className="mb-4"),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Altin Fiyat Gecmisi", style={"color": TEXT, "fontWeight": "600"})),
                    dbc.CardBody(dcc.Graph(id="pred-gold-chart", figure=empty_figure(),
                                          config={"displayModeBar": False})),
                ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
            ], md=7, className="mb-4"),
        ]),

        # ── Prediction section ────────────────────────────────────────────────
        dbc.Row([
            # Left: controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Tahmin Ayarlari", style={"color": TEXT, "fontWeight": "600"})),
                    dbc.CardBody([
                        html.Label("Sembol", className="section-title"),
                        dcc.Dropdown(id="pred-symbol", options=[], value=None,
                                     placeholder="Sembol sec...", clearable=False,
                                     style={"marginBottom": "16px"}),

                        html.Label("Tahmin Ufku", className="section-title"),
                        dcc.Dropdown(
                            id="pred-horizon",
                            options=HORIZONS,
                            value="daily", clearable=False,
                            style={"marginBottom": "24px"},
                        ),

                        dbc.Button([html.I(className="bi bi-cpu me-2"), "Model Egit"],
                                   id="pred-train-btn", color="primary", className="w-100 mb-2"),
                        dbc.Button([html.I(className="bi bi-lightning me-2"), "Tahmin Yap"],
                                   id="pred-predict-btn", color="success", outline=True, className="w-100 mb-2"),
                        dbc.Button([html.I(className="bi bi-graph-up me-2"), "Degerlendir"],
                                   id="pred-evaluate-btn", color="info", outline=True, className="w-100"),

                        html.Div(id="pred-action-result", className="mt-3"),
                    ]),
                ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
            ], md=3, className="mb-4"),

            # Right: results
            dbc.Col([
                # Prediction result card
                html.Div(id="pred-result-card", className="mb-4"),

                # Performance metrics
                html.Div(id="pred-performance-section"),
            ], md=9, className="mb-4"),
        ]),

        # ── Charts row ────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Tahmin vs Gercek", style={"color": TEXT, "fontWeight": "600"})),
                    dbc.CardBody(dcc.Graph(id="pred-vs-actual-chart", figure=empty_figure(),
                                          config={"displayModeBar": False})),
                ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
            ], md=7, className="mb-4"),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Dogruluk Trendi", style={"color": TEXT, "fontWeight": "600"})),
                    dbc.CardBody(dcc.Graph(id="pred-accuracy-chart", figure=empty_figure(),
                                          config={"displayModeBar": False})),
                ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
            ], md=5, className="mb-4"),
        ]),

        # History table
        dbc.Card([
            dbc.CardHeader(html.Span("Tahmin Gecmisi", style={"color": TEXT, "fontWeight": "600"})),
            dbc.CardBody(html.Div(id="pred-history-table")),
        ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}", "marginBottom": "24px"}),

        # ── Trained models list ───────────────────────────────────────────────
        dbc.Card([
            dbc.CardHeader(html.Span("Egitilmis Modeller", style={"color": TEXT, "fontWeight": "600"})),
            dbc.CardBody(html.Div(id="pred-models-list",
                                  children=html.P("Yukleniyor...", style={"color": TEXT_MUTED}))),
        ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        [Output("pred-symbol", "options"), Output("pred-gold-cards", "children"),
         Output("pred-gold-chart", "figure"), Output("pred-models-list", "children")],
        Input("pred-refresh", "n_intervals"),
        prevent_initial_call=False,
    )
    def refresh_gold_and_symbols(n):
        # Symbols — kullanıcı dostu label'lar
        from data.bist30_symbols import STOCK_INFO, ASSET_INFO
        _info = {**STOCK_INFO, **ASSET_INFO}
        syms = api.get_prediction_symbols() or []
        opts = [
            {"label": f"{_info[s]['name']} ({s})" if s in _info else s, "value": s}
            for s in syms
        ]

        # Gold prices (isolated — failure should not block other outputs)
        try:
            gold = api.get_gold_prices()
            cards = _render_gold_cards(gold)
        except Exception:
            cards = [dbc.Col(html.P("Altin verisi alinamadi.", style={"color": TEXT_MUTED}))]

        # Gold history chart (isolated)
        try:
            history = api.get_gold_history()
            gold_fig = _build_gold_chart(history)
        except Exception:
            gold_fig = empty_figure("Veri alinamadi")

        # Trained models list (isolated)
        try:
            models_ui = _render_models_list(api.get_prediction_models() or [])
        except Exception:
            models_ui = html.P("Model listesi alinamadi.", style={"color": TEXT_MUTED})

        return opts, cards, gold_fig, models_ui

    @app.callback(
        [Output("pred-action-result", "children"),
         Output("pred-models-list", "children", allow_duplicate=True)],
        Input("pred-train-btn", "n_clicks"),
        [State("pred-symbol", "value"), State("pred-horizon", "value")],
        prevent_initial_call=True,
    )
    def train_prediction(n, symbol, horizon):
        models_ui = _render_models_list(api.get_prediction_models())
        if not n or not symbol:
            return "", models_ui
        result = api.train_prediction({"symbol": symbol, "horizon": horizon or "daily"})
        if not result or result.get("detail"):
            err = result.get("detail", "Bilinmeyen hata") if result else "Sunucu yanit vermedi"
            return (dbc.Alert([html.I(className="bi bi-x-circle me-2"), f"Egitim basarisiz: {err}"],
                              color="danger", dismissable=True), models_ui)

        test_m = result.get("test_metrics", {})
        train_m = result.get("train_metrics", {})
        rows = [
            dbc.Row([
                dbc.Col(html.Small("Sembol",      style={"color": TEXT_MUTED}), width=5),
                dbc.Col(html.Strong(result.get("symbol", symbol), style={"color": TEXT}), width=7),
            ], className="mb-1"),
            dbc.Row([
                dbc.Col(html.Small("Horizon",     style={"color": TEXT_MUTED}), width=5),
                dbc.Col(html.Span(result.get("horizon", horizon), style={"color": TEXT}), width=7),
            ], className="mb-1"),
            dbc.Row([
                dbc.Col(html.Small("Egitim seti", style={"color": TEXT_MUTED}), width=5),
                dbc.Col(html.Span(f"{result.get('n_train', '?')} satir", style={"color": TEXT}), width=7),
            ], className="mb-1"),
            dbc.Row([
                dbc.Col(html.Small("Test seti",   style={"color": TEXT_MUTED}), width=5),
                dbc.Col(html.Span(f"{result.get('n_test', '?')} satir", style={"color": TEXT}), width=7),
            ], className="mb-1"),
            dbc.Row([
                dbc.Col(html.Small("Train MAPE",  style={"color": TEXT_MUTED}), width=5),
                dbc.Col(html.Span(f"{train_m.get('mape', 0):.2f}%", style={"color": TEXT}), width=7),
            ], className="mb-1"),
            dbc.Row([
                dbc.Col(html.Small("Test MAPE",   style={"color": TEXT_MUTED}), width=5),
                dbc.Col(html.Strong(
                    f"{test_m.get('mape', 0):.2f}%",
                    style={"color": "#f0c27a" if test_m.get("mape", 99) > 5 else "#5cb85c"}
                ), width=7),
            ], className="mb-1"),
            dbc.Row([
                dbc.Col(html.Small("Yon Acc.",    style={"color": TEXT_MUTED}), width=5),
                dbc.Col(html.Strong(
                    f"{test_m.get('direction_accuracy', 0):.1f}%",
                    style={"color": "#5cb85c" if test_m.get("direction_accuracy", 0) >= 55 else "#e07b54"}
                ), width=7),
            ], className="mb-1"),
        ]
        alert = dbc.Alert([
            html.Div([html.I(className="bi bi-check-circle me-2"),
                      html.Strong("Model egitildi")], className="mb-2"),
            html.Hr(style={"borderColor": "rgba(255,255,255,0.2)", "margin": "6px 0"}),
            *rows,
        ], color="success", dismissable=True)
        return alert, _render_models_list(api.get_prediction_models())

    @app.callback(
        [
            Output("pred-result-card", "children"),
            Output("pred-vs-actual-chart", "figure"),
            Output("pred-accuracy-chart", "figure"),
            Output("pred-history-table", "children"),
            Output("pred-performance-section", "children"),
            Output("pred-result-store", "data"),
        ],
        Input("pred-predict-btn", "n_clicks"),
        [State("pred-symbol", "value"), State("pred-horizon", "value")],
        prevent_initial_call=True,
    )
    def make_prediction(n, symbol, horizon):
        blank = (html.Span(), empty_figure(), empty_figure(), html.Span(), html.Span(), {})
        if not n or not symbol:
            return blank

        resp = api.make_prediction({"symbols": [symbol], "horizon": horizon or "daily"})
        predictions = resp.get("predictions", []) if isinstance(resp, dict) else []
        if not predictions:
            err = resp.get("detail", "Model egitilmemis olabilir.") if isinstance(resp, dict) else "Tahmin alinamadi."
            return (dbc.Alert([html.I(className="bi bi-x-circle me-2"), err], color="danger"),) + blank[1:]

        result = predictions[0]  # ilk (tek) tahmin

        # Chart data
        chart_data = api.get_prediction_chart_data(symbol) or {}
        perf = api.get_prediction_performance(symbol) or {}
        hist = api.get_prediction_history(symbol) or {}

        result_card = _render_result_card(result, symbol)
        vs_chart = _build_vs_actual_chart(chart_data)
        acc_chart = _build_accuracy_chart(chart_data)
        hist_table = _render_history_table(hist)
        perf_section = _render_performance(perf)

        return result_card, vs_chart, acc_chart, hist_table, perf_section, result

    @app.callback(
        Output("pred-performance-section", "children", allow_duplicate=True),
        Input("pred-evaluate-btn", "n_clicks"),
        [State("pred-symbol", "value"), State("pred-horizon", "value")],
        prevent_initial_call=True,
    )
    def evaluate_model(n, symbol, horizon):
        if not n or not symbol:
            return html.Span()
        result = api.evaluate_prediction({"symbol": symbol, "horizon": horizon or "daily"})
        return _render_performance(result or {})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _render_models_list(models: list) -> Any:
    if not models:
        return html.P("Henuz egitilmis model yok. Sembol secip 'Model Egit' butonuna basin.",
                      style={"color": TEXT_MUTED, "fontSize": "13px"})

    rows = []
    for m in sorted(models, key=lambda x: (x.get("symbol", ""), x.get("horizon", ""))):
        sym      = m.get("symbol", "?")
        horizon  = m.get("horizon", "?")
        saved_at = m.get("saved_at", "")[:10] if m.get("saved_at") else "—"
        n_feat   = len(m.get("feature_cols", []))
        h_badge  = dbc.Badge("Gunluk" if horizon == "daily" else "Haftalik",
                             color="primary" if horizon == "daily" else "info",
                             pill=True, className="me-2")
        rows.append(
            dbc.Row([
                dbc.Col(html.Span(sym, style={"color": TEXT, "fontWeight": "600"}), width=3),
                dbc.Col(h_badge, width=2),
                dbc.Col(html.Small(f"{n_feat} ozellik", style={"color": TEXT_MUTED}), width=2),
                dbc.Col(html.Small(f"Kaydedildi: {saved_at}", style={"color": TEXT_MUTED}), width=5),
            ], className="mb-2 align-items-center")
        )

    return html.Div([
        html.P(f"{len(models)} egitilmis model",
               style={"color": TEXT_MUTED, "fontSize": "12px", "marginBottom": "12px"}),
        *rows,
    ])


def _render_gold_cards(gold):
    if not gold:
        return [dbc.Col(html.P("Altin verisi yok.", style={"color": TEXT_MUTED}))]

    items = [
        ("TRY/gram", gold.get("try_per_gram", gold.get("price_try", "—")), GOLD),
        ("USD/oz", gold.get("usd_per_oz", gold.get("price_usd", "—")), GREEN),
        ("Degisim %", gold.get("change_pct", gold.get("daily_change", "—")), BLUE),
    ]
    cols = []
    for label, val, color in items:
        if isinstance(val, float):
            val = f"{val:,.2f}"
        cols.append(dbc.Col(
            dbc.Card(dbc.CardBody([
                html.Small(label, style={"color": TEXT_MUTED, "fontSize": "11px", "textTransform": "uppercase"}),
                html.Div(str(val), style={"color": color, "fontWeight": "700", "fontSize": "22px"}),
            ]), style={"backgroundColor": CARD2, "border": f"1px solid {CARD2}"}),
        ))
    return cols


def _build_gold_chart(history):
    fig = go.Figure()
    records = []
    if isinstance(history, dict):
        records = history.get("history", history.get("data", []))
    elif isinstance(history, list):
        records = history

    if records:
        dates = [r.get("date", str(i)) for i, r in enumerate(records)]
        prices = [r.get("price", r.get("try_per_gram", r.get("value", 0))) for r in records]
        fig.add_trace(go.Scatter(
            x=dates, y=prices, mode="lines",
            fill="tozeroy",
            fillcolor=f"rgba(245,158,11,0.15)",
            line={"color": GOLD, "width": 2},
            name="Altin (TRY/gr)",
            hovertemplate="<b>%{x}</b><br>₺%{y:,.2f}<extra></extra>",
        ))
    else:
        return empty_figure("Altin gecmisi yok")

    apply_dark_template(fig)
    fig.update_layout(showlegend=False, height=220, margin={"l": 50, "r": 10, "t": 10, "b": 40})
    return fig


def _render_result_card(result, symbol):
    # SinglePrediction field adları: predicted_close, predicted_direction, confidence,
    # current_close, predicted_change_pct, prediction_date
    price      = result.get("predicted_close", 0) or 0
    current    = result.get("current_close", 0) or 0
    change_pct = result.get("predicted_change_pct", 0) or 0
    direction  = result.get("predicted_direction", "NEUTRAL").upper()
    confidence = result.get("confidence", 0) or 0
    pred_date  = result.get("prediction_date", "—")
    horizon    = result.get("horizon", "daily")

    dir_color = GREEN if "UP" in direction else RED if "DOWN" in direction else YELLOW
    dir_icon  = ("bi bi-arrow-up-circle-fill" if "UP" in direction
                 else "bi bi-arrow-down-circle-fill" if "DOWN" in direction
                 else "bi bi-dash-circle")
    conf_pct  = int(confidence * 100) if confidence <= 1 else int(confidence)

    currency_symbol = "$" if symbol in ("GC=F", "GOLD_GRAM_USD") else "₺"
    horizon_label = "1 gun sonrasi" if horizon == "daily" else "5 gun sonrasi"

    # Model kalite bilgisi (meta'dan gelir; tahmin endpoint test_mape döndürmüyor,
    # bu yüzden api ile model listesinden çekiyoruz)
    model_quality_parts = []
    try:
        all_models = api.get_prediction_models() or []
        meta = next((m for m in all_models
                     if m.get("symbol") == symbol and m.get("horizon") == horizon), None)
        if meta:
            mape = meta.get("test_mape")
            dir_acc = meta.get("test_direction_accuracy")
            if mape is not None:
                q_color = GREEN if mape < 3 else (YELLOW if mape < 7 else RED)
                model_quality_parts.append(
                    html.Span(f"MAPE {mape:.1f}%", style={"color": q_color, "marginRight": "10px",
                                                           "fontSize": "12px", "fontWeight": "600"}))
            if dir_acc is not None:
                q_color = GREEN if dir_acc >= 55 else RED
                model_quality_parts.append(
                    html.Span(f"Yon Acc {dir_acc:.0f}%", style={"color": q_color,
                                                                  "fontSize": "12px", "fontWeight": "600"}))
    except Exception:
        pass

    return dbc.Card([
        dbc.CardHeader(
            dbc.Row([
                dbc.Col(html.Span(f"{symbol}  ·  {pred_date} tahmini ({horizon_label})",
                                  style={"color": TEXT_MUTED, "fontSize": "13px"})),
                dbc.Col(html.Span(model_quality_parts),
                        width="auto", className="ms-auto d-flex align-items-center"),
            ], className="align-items-center"),
        ),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Small("Mevcut Fiyat", style={"color": TEXT_MUTED}),
                    html.H5(f"{currency_symbol}{current:,.4f}", style={"color": TEXT}),
                    html.Small("Tahmini Fiyat", style={"color": TEXT_MUTED, "marginTop": "8px", "display": "block"}),
                    html.H3(f"{currency_symbol}{price:,.4f}", style={"color": TEXT, "fontWeight": "700"}),
                    dbc.Badge(
                        f"{'▲' if change_pct >= 0 else '▼'} {abs(change_pct):.2f}%",
                        color="success" if change_pct >= 0 else "danger",
                        className="mt-1",
                    ),
                ], md=5),
                dbc.Col([
                    html.Div([
                        html.I(className=f"{dir_icon} me-2",
                               style={"color": dir_color, "fontSize": "24px"}),
                        html.Span(direction,
                                  style={"color": dir_color, "fontWeight": "700", "fontSize": "18px"}),
                    ], className="d-flex align-items-center mb-3"),
                    html.Small("Model Guveni", style={"color": TEXT_MUTED}),
                    dbc.Progress(
                        value=conf_pct,
                        label=f"{conf_pct}%",
                        color="success" if conf_pct >= 70 else "warning" if conf_pct >= 50 else "danger",
                        style={"height": "12px", "marginTop": "4px"},
                    ),
                    html.Small(
                        "Guven: ensemble model uzlasmasina gore hesaplanir.",
                        style={"color": TEXT_MUTED, "fontSize": "10px", "marginTop": "6px",
                               "display": "block"},
                    ),

                    # Ensemble agreement
                    *(_render_ensemble_agreement(result) if result.get("ensemble_agreement") is not None else []),
                ], md=7),
            ]),

            # Model predictions breakdown
            *(_render_model_predictions(result) if result.get("model_predictions") else []),
        ])
    ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"})


def _render_ensemble_agreement(result):
    """Ensemble model uzlasma gosterimi."""
    agreement = result.get("ensemble_agreement", 0)
    n_models = result.get("n_models", 0)
    agreement_pct = int(agreement * 100) if agreement <= 1 else int(agreement)

    return [
        html.Hr(style={"borderColor": "rgba(255,255,255,0.1)", "margin": "10px 0"}),
        html.Small("Ensemble Uzlasma", style={"color": TEXT_MUTED}),
        dbc.Progress(
            value=agreement_pct,
            label=f"{agreement_pct}% ({n_models} model)",
            color="success" if agreement_pct >= 80 else "warning" if agreement_pct >= 60 else "danger",
            style={"height": "10px", "marginTop": "4px"},
        ),
    ]


def _render_model_predictions(result):
    """Her modelin tahminini goster."""
    model_preds = result.get("model_predictions", {})
    if not model_preds:
        return []

    current = result.get("current_close", 0)
    items = []
    for model_name, pred_price in sorted(model_preds.items()):
        change = (pred_price - current) / current * 100 if current > 0 else 0
        dir_color = GREEN if change > 0 else RED
        items.append(
            dbc.Col(
                html.Div([
                    html.Small(model_name.upper(), style={"color": TEXT_MUTED, "fontSize": "10px"}),
                    html.Div(f"{pred_price:,.2f}", style={"color": TEXT, "fontWeight": "600", "fontSize": "13px"}),
                    html.Small(f"{'▲' if change >= 0 else '▼'}{abs(change):.1f}%",
                               style={"color": dir_color, "fontSize": "10px"}),
                ], style={"textAlign": "center"}),
                width=True,
            )
        )

    return [
        html.Hr(style={"borderColor": "rgba(255,255,255,0.1)", "margin": "12px 0"}),
        html.Small("Model Tahminleri", style={"color": TEXT_MUTED, "display": "block", "marginBottom": "8px"}),
        dbc.Row(items, className="g-2"),
    ]


def _build_vs_actual_chart(chart_data):
    fig = go.Figure()
    actual = chart_data.get("actual", [])
    predicted = chart_data.get("predicted", [])
    dates = chart_data.get("dates", list(range(max(len(actual), len(predicted)))))

    if actual:
        fig.add_trace(go.Scatter(x=dates[:len(actual)], y=actual, mode="lines",
                                  name="Gercek", line={"color": TEXT_MUTED, "width": 1.5}))
    if predicted:
        fig.add_trace(go.Scatter(x=dates[:len(predicted)], y=predicted, mode="lines",
                                  name="Tahmin", line={"color": BLUE, "width": 2, "dash": "dot"}))

    if not actual and not predicted:
        return empty_figure("Veri yok")

    apply_dark_template(fig)
    fig.update_layout(height=300, legend={"orientation": "h", "y": 1.1})
    return fig


def _build_accuracy_chart(chart_data):
    """Dual y-axis: accuracy trend + MAE."""
    acc = chart_data.get("accuracy_history", chart_data.get("accuracy", []))
    mae = chart_data.get("mae_history", chart_data.get("mae", []))

    if not acc and not mae:
        return empty_figure("Dogruluk verisi yok")

    fig = go.Figure()
    if acc:
        fig.add_trace(go.Scatter(y=acc, mode="lines+markers",
                                  name="Dogruluk", line={"color": GREEN}, yaxis="y1"))
    if mae:
        fig.add_trace(go.Scatter(y=mae, mode="lines+markers",
                                  name="MAE", line={"color": RED, "dash": "dot"}, yaxis="y2"))

    apply_dark_template(fig)
    fig.update_layout(
        height=300,
        yaxis={"title": "Dogruluk"},
        yaxis2={"title": "MAE", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "y": 1.1},
    )
    return fig


def _render_performance(perf):
    if not perf:
        return html.Span()
    metrics = [
        ("MAPE", perf.get("mape")),
        ("MAE", perf.get("mae")),
        ("RMSE", perf.get("rmse")),
        ("Yon Dogrulugu", perf.get("direction_accuracy")),
        ("Tahmin Sayisi", perf.get("n_predictions")),
    ]
    items = []
    for label, val in metrics:
        if val is None:
            continue
        fmt = f"{val:.4f}" if isinstance(val, float) else str(val)
        items.append(dbc.Row([
            dbc.Col(html.Small(label, style={"color": TEXT_MUTED}), width=6),
            dbc.Col(html.Span(fmt, style={"color": TEXT, "fontWeight": "600", "fontSize": "13px"}), width=6),
        ], className="py-1 border-bottom"))
    return dbc.Card([
        dbc.CardHeader(html.Span("Performans Metrikleri", style={"color": TEXT, "fontWeight": "600"})),
        dbc.CardBody(html.Div(items) if items else html.P("Metrik yok.", style={"color": TEXT_MUTED})),
    ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"})


def _render_history_table(hist):
    records = hist.get("predictions", []) if isinstance(hist, dict) else []
    if not records:
        return html.P("Tahmin gecmisi yok.", style={"color": TEXT_MUTED})

    header = dbc.Row([
        dbc.Col(html.Small("Tarih", className="section-title"), width=3),
        dbc.Col(html.Small("Tahmin", className="section-title"), width=3),
        dbc.Col(html.Small("Gercek", className="section-title"), width=3),
        dbc.Col(html.Small("Hata %", className="section-title"), width=3),
    ])
    rows = [header]
    for r in records[:15]:
        pred = r.get("predicted", 0) or 0
        actual = r.get("actual", 0) or 0
        err = abs(pred - actual) / actual * 100 if actual else 0
        rows.append(dbc.Row([
            dbc.Col(html.Small(str(r.get("date", "—"))[:10], style={"color": TEXT_MUTED}), width=3),
            dbc.Col(html.Small(f"₺{pred:,.2f}", style={"color": BLUE}), width=3),
            dbc.Col(html.Small(f"₺{actual:,.2f}", style={"color": TEXT}), width=3),
            dbc.Col(html.Small(f"{err:.1f}%", style={"color": RED if err > 5 else GREEN}), width=3),
        ], className="py-1 border-bottom"))
    return html.Div(rows)
