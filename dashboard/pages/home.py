"""
Dashboard home page – system overview, portfolio chart, algorithm comparison.

API endpoints used:
  GET /api/trading/health
  GET /api/trading/models
  GET /api/trading/models/{name}/metrics
  GET /api/trading/portfolio-history
"""

from dash import html, dcc
from dash import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from dashboard.theme import (
    CARD, CARD2, TEXT, TEXT_MUTED, BORDER,
    GREEN, RED, BLUE, PURPLE, ORANGE, YELLOW, CYAN, GOLD,
    DARK_TEMPLATE, ALGO_COLORS, empty_figure, apply_dark_template,
)
from dashboard.components.metric_card import create_metric_card
import dashboard.api_client as api


# ── Layout ───────────────────────────────────────────────────────────────────

def layout():
    return html.Div([
        # Auto-refresh interval (every 30 s)
        dcc.Interval(id="home-interval", interval=30_000, n_intervals=0),

        # Page header
        dbc.Row([
            dbc.Col([
                html.H4("Dashboard", style={"color": TEXT, "marginBottom": "4px"}),
                html.P("Sistem ozeti ve portfoy performansi", style={"color": TEXT_MUTED, "margin": 0}),
            ], width="auto"),
            dbc.Col([
                html.Div(id="home-status-badge"),
            ], width="auto", className="ms-auto d-flex align-items-center"),
        ], className="mb-4 align-items-center"),

        # Metric cards row
        dbc.Row([
            dbc.Col(create_metric_card("Toplam Getiri", "—", "", GREEN, "bi bi-graph-up", card_id="home-metric-return"), md=2, sm=4, xs=6, className="mb-3"),
            dbc.Col(create_metric_card("Sharpe Orani", "—", "", BLUE, "bi bi-calculator", card_id="home-metric-sharpe"), md=2, sm=4, xs=6, className="mb-3"),
            dbc.Col(create_metric_card("Max Drawdown", "—", "", RED, "bi bi-arrow-down-circle", card_id="home-metric-drawdown"), md=2, sm=4, xs=6, className="mb-3"),
            dbc.Col(create_metric_card("Portfoy Degeri", "—", "", YELLOW, "bi bi-wallet2", card_id="home-metric-portfolio"), md=3, sm=4, xs=6, className="mb-3"),
            dbc.Col(create_metric_card("Toplam Islem", "—", "", PURPLE, "bi bi-arrow-left-right", card_id="home-metric-trades"), md=3, sm=4, xs=6, className="mb-3"),
        ], className="mb-4"),

        # Charts row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Portfoy Performansi", style={"color": TEXT, "fontWeight": "600"})),
                    dbc.CardBody(dcc.Graph(id="home-portfolio-chart", figure=empty_figure(), config={"displayModeBar": False})),
                ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
            ], md=8, className="mb-4"),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Algoritma Karsilastirma", style={"color": TEXT, "fontWeight": "600"})),
                    dbc.CardBody(dcc.Graph(id="home-algo-chart", figure=empty_figure(), config={"displayModeBar": False})),
                ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
            ], md=4, className="mb-4"),
        ]),

        # Models row: RL models + Prediction models side by side
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("RL Modeller", style={"color": TEXT, "fontWeight": "600"})),
                    dbc.CardBody(html.Div(id="home-models-list", children=html.P("Yukleniyor...", style={"color": TEXT_MUTED}))),
                ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
            ], md=6, className="mb-4"),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Tahmin Modelleri (XGBoost)", style={"color": TEXT, "fontWeight": "600"})),
                    dbc.CardBody(html.Div(id="home-pred-models-list", children=html.P("Yukleniyor...", style={"color": TEXT_MUTED}))),
                ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
            ], md=6, className="mb-4"),
        ]),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("home-status-badge", "children"),
        Input("home-interval", "n_intervals"),
    )
    def update_status(n):
        data = api.get_health()
        status = data.get("status", "unknown")
        color = "success" if status == "healthy" else "danger"
        return dbc.Badge(
            [html.I(className=f"bi bi-circle-fill me-1", style={"fontSize": "8px"}), status.upper()],
            color=color, pill=True,
        )

    @app.callback(
        [
            Output("home-metric-return", "children"),
            Output("home-metric-sharpe", "children"),
            Output("home-metric-drawdown", "children"),
            Output("home-metric-portfolio", "children"),
            Output("home-metric-trades", "children"),
            Output("home-portfolio-chart", "figure"),
            Output("home-algo-chart", "figure"),
            Output("home-models-list", "children"),
            Output("home-pred-models-list", "children"),
        ],
        Input("home-interval", "n_intervals"),
    )
    def update_home(n):
        models = api.get_models()
        history = api.get_portfolio_history()

        # ── Aggregate metrics from best model ──────────────────────────────
        total_return = sharpe = drawdown = portfolio = trades = "—"
        if models:
            best = None
            best_sharpe = -999
            for m in models:
                metrics = api.get_model_metrics(m.get("name", ""))
                sr = metrics.get("sharpe_ratio", -999) if metrics else -999
                if sr > best_sharpe:
                    best_sharpe = sr
                    best = metrics
            if best:
                ret = best.get("total_return", 0) or 0
                total_return = f"{ret:.1%}"
                sharpe = f"{best.get('sharpe_ratio', 0):.2f}"
                dd = best.get("max_drawdown", 0) or 0
                drawdown = f"{dd:.1%}"
                pv = best.get("final_portfolio_value", 0) or 0
                portfolio = f"₺{pv:,.0f}"
                trades = str(best.get("total_trades", 0) or 0)

        metric_return = _metric_val(total_return, GREEN)
        metric_sharpe = _metric_val(sharpe, BLUE)
        metric_drawdown = _metric_val(drawdown, RED)
        metric_portfolio = _metric_val(portfolio, YELLOW)
        metric_trades = _metric_val(trades, PURPLE)

        # ── Portfolio performance chart ────────────────────────────────────
        portfolio_fig = _build_portfolio_chart(history)

        # ── Algorithm comparison chart ─────────────────────────────────────
        algo_fig = _build_algo_chart(models)

        # ── RL Models list ─────────────────────────────────────────────────
        models_list = _build_models_list(models)

        # ── Prediction models list ──────────────────────────────────────────
        try:
            pred_models = api.get_prediction_models() or []
        except Exception:
            pred_models = []
        pred_models_list = _build_pred_models_list(pred_models)

        return (
            metric_return, metric_sharpe, metric_drawdown,
            metric_portfolio, metric_trades,
            portfolio_fig, algo_fig, models_list, pred_models_list,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _metric_val(value, color):
    return html.Span(value, style={"color": color, "fontSize": "26px", "fontWeight": "700"})


def _build_portfolio_chart(history):
    fig = go.Figure()
    if history:
        records = history.get("history", []) if isinstance(history, dict) else []
        if records:
            dates = [r.get("date", "") for r in records]
            values = [r.get("value", 0) for r in records]
            fig.add_trace(go.Scatter(
                x=dates, y=values,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(59,130,246,0.15)",
                line={"color": BLUE, "width": 2},
                name="Portfoy",
                hovertemplate="<b>%{x}</b><br>₺%{y:,.0f}<extra></extra>",
            ))
    else:
        fig = empty_figure("Portfoy gecmisi yok")

    apply_dark_template(fig)
    fig.update_layout(showlegend=False, height=280, margin={"l": 50, "r": 10, "t": 10, "b": 40})
    return fig


def _build_algo_chart(models):
    fig = go.Figure()
    if not models:
        return empty_figure("Model yok")

    names, sharpes = [], []
    for m in models:
        metrics = api.get_model_metrics(m.get("name", ""))
        if metrics:
            algo = m.get("algorithm", "UNK").upper()
            names.append(algo)
            sharpes.append(metrics.get("sharpe_ratio", 0) or 0)

    if not names:
        return empty_figure("Metrik yok")

    colors = [ALGO_COLORS.get(n, BLUE) for n in names]
    fig.add_trace(go.Bar(
        x=names, y=sharpes,
        marker_color=colors,
        name="Sharpe",
        hovertemplate="<b>%{x}</b><br>Sharpe: %{y:.2f}<extra></extra>",
    ))
    apply_dark_template(fig)
    fig.update_layout(showlegend=False, height=280, margin={"l": 40, "r": 10, "t": 10, "b": 40})
    return fig


def _build_pred_models_list(models):
    """XGBoost prediction modelleri listesi."""
    if not models:
        return html.P("Egitilmis tahmin modeli yok. Tahmin sayfasindan model egitebilirsiniz.",
                      style={"color": TEXT_MUTED, "fontSize": "13px"})

    rows = []
    for m in sorted(models, key=lambda x: (x.get("symbol", ""), x.get("horizon", ""))):
        sym     = m.get("symbol", "?")
        horizon = m.get("horizon", "?")
        saved   = (m.get("saved_at", "") or "")[:10] or "—"
        n_feat  = len(m.get("feature_cols", []))
        mape    = m.get("test_mape")
        dir_acc = m.get("test_direction_accuracy")

        h_badge = dbc.Badge(
            "Gunluk" if horizon == "daily" else "Haftalik",
            color="primary" if horizon == "daily" else "info",
            pill=True, className="me-2",
        )
        perf_parts = []
        if mape is not None:
            clr = GREEN if mape < 3 else (YELLOW if mape < 7 else RED)
            perf_parts.append(html.Small(f"MAPE {mape:.1f}%", style={"color": clr, "marginRight": "8px"}))
        if dir_acc is not None:
            clr = GREEN if dir_acc >= 55 else RED
            perf_parts.append(html.Small(f"Yon {dir_acc:.0f}%", style={"color": clr}))
        if not perf_parts:
            perf_parts = [html.Small(f"{n_feat} ozellik", style={"color": TEXT_MUTED})]

        rows.append(
            dbc.Row([
                dbc.Col(html.Span(sym, style={"color": GOLD, "fontWeight": "600", "fontSize": "13px"}), width=3),
                dbc.Col(h_badge, width="auto"),
                dbc.Col(html.Span(perf_parts), width=4),
                dbc.Col(html.Small(saved, style={"color": TEXT_MUTED}), width="auto", className="ms-auto"),
            ], className="align-items-center py-2", style={"borderBottom": f"1px solid {CARD2}"})
        )
    return html.Div([
        html.P(f"{len(models)} model egitildi",
               style={"color": TEXT_MUTED, "fontSize": "12px", "marginBottom": "8px"}),
        *rows,
    ])


def _build_models_list(models):
    if not models:
        return html.P("Egitilmis model bulunamadi.", style={"color": TEXT_MUTED})

    rows = []
    for m in models:
        name = m.get("name", "—")
        algo = m.get("algorithm", "—").upper()
        phase = m.get("phase", "—")
        created = m.get("created_at", m.get("timestamp", "—"))
        color = ALGO_COLORS.get(algo, BLUE)
        rows.append(
            dbc.Row([
                dbc.Col(dbc.Badge(algo, style={"backgroundColor": color}, pill=True), width="auto"),
                dbc.Col(html.Span(name, style={"color": TEXT, "fontSize": "13px"}), width=5),
                dbc.Col(html.Small(f"Faz {phase}", style={"color": TEXT_MUTED}), width="auto"),
                dbc.Col(html.Small(str(created)[:10], style={"color": TEXT_MUTED}), width="auto", className="ms-auto"),
            ], className="align-items-center py-2", style={"borderBottom": f"1px solid {CARD2}"})
        )
    return html.Div(rows)
