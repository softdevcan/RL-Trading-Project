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

from dash import html, dcc
from dash import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from dashboard.theme import (
    CARD, CARD2, TEXT, TEXT_MUTED, BORDER, GREEN, RED, BLUE, GOLD, PURPLE, CYAN, YELLOW,
    DARK_TEMPLATE, empty_figure, apply_dark_template,
)
import dashboard.api_client as api

HORIZONS = [1, 3, 5, 7, 14, 30]


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

                        html.Label("Tahmin Ufku (gun)", className="section-title"),
                        dcc.Dropdown(
                            id="pred-horizon",
                            options=[{"label": f"{h} gun", "value": h} for h in HORIZONS],
                            value=5, clearable=False,
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
        ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        [Output("pred-symbol", "options"), Output("pred-gold-cards", "children"),
         Output("pred-gold-chart", "figure")],
        Input("pred-refresh", "n_intervals"),
        prevent_initial_call=False,
    )
    def refresh_gold_and_symbols(n):
        # Symbols
        syms = api.get_prediction_symbols()
        opts = [{"label": s, "value": s} for s in syms] if syms else []

        # Gold prices
        gold = api.get_gold_prices()
        cards = _render_gold_cards(gold)

        # Gold history chart
        history = api.get_gold_history()
        gold_fig = _build_gold_chart(history)

        return opts, cards, gold_fig

    @app.callback(
        Output("pred-action-result", "children"),
        Input("pred-train-btn", "n_clicks"),
        [State("pred-symbol", "value"), State("pred-horizon", "value")],
        prevent_initial_call=True,
    )
    def train_prediction(n, symbol, horizon):
        if not n or not symbol:
            return ""
        result = api.train_prediction({"symbol": symbol, "horizon": int(horizon or 5)})
        if result:
            return dbc.Alert(
                [html.I(className="bi bi-check-circle me-2"), f"{symbol} modeli egitildi."],
                color="success", dismissable=True,
            )
        return dbc.Alert("Egitim basarisiz.", color="danger", dismissable=True)

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

        result = api.make_prediction({"symbol": symbol, "horizon": int(horizon or 5)})
        if not result:
            return (dbc.Alert("Tahmin alinamadi.", color="danger"),) + blank[1:]

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
        result = api.evaluate_prediction({"symbol": symbol, "horizon": int(horizon or 5)})
        return _render_performance(result or {})


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    price = result.get("predicted_price", result.get("price", 0)) or 0
    direction = result.get("direction", result.get("trend", "neutral")).upper()
    confidence = result.get("confidence", result.get("confidence_score", 0)) or 0

    dir_color = GREEN if "UP" in direction or "YUKARI" in direction else RED if "DOWN" in direction or "ASAGI" in direction else YELLOW
    dir_icon = "bi bi-arrow-up-circle-fill" if "UP" in direction else "bi bi-arrow-down-circle-fill" if "DOWN" in direction else "bi bi-dash-circle"

    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6(f"{symbol} Tahmini", style={"color": TEXT_MUTED}),
                    html.H3(f"₺{price:,.2f}", style={"color": TEXT, "fontWeight": "700"}),
                ], md=5),
                dbc.Col([
                    html.Div([
                        html.I(className=f"{dir_icon} me-2", style={"color": dir_color, "fontSize": "24px"}),
                        html.Span(direction, style={"color": dir_color, "fontWeight": "700", "fontSize": "18px"}),
                    ], className="d-flex align-items-center mb-2"),
                    html.Small("Guven", style={"color": TEXT_MUTED}),
                    dbc.Progress(
                        value=int(confidence * 100) if confidence <= 1 else int(confidence),
                        label=f"{int(confidence * 100) if confidence <= 1 else int(confidence)}%",
                        color="success" if confidence > 0.7 else "warning",
                        style={"height": "8px"},
                    ),
                ], md=7),
            ]),
        ])
    ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"})


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
        ("MAE", perf.get("mae")), ("RMSE", perf.get("rmse")),
        ("R²", perf.get("r2")), ("Yon Dogrulugu", perf.get("direction_accuracy")),
        ("Hit Rate", perf.get("hit_rate")),
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
