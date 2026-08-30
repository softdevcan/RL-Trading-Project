"""
Academic Analysis page – report generation, best models, comparison charts.

API endpoints used:
  GET /api/trading/analysis/generate-report
  GET /api/trading/analysis/model-comparison
  GET /api/trading/analysis/best-models
"""

from dash import html, dcc
from dash import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from dashboard.theme import (
    TEXT, TEXT_MUTED, GREEN, RED, BLUE, PURPLE, algo_badge_class, empty_figure,
    apply_theme_template, plot_palette, algo_plot_colors,
)
from dashboard.components.page_header import create_page_header
from dashboard.components.state_block import create_state_block
import dashboard.api_client as api


# ── Layout ────────────────────────────────────────────────────────────────────

def layout():
    return html.Div([
        dcc.Interval(id="academic-poll", interval=5_000, disabled=True, n_intervals=0),
        dcc.Store(id="academic-report-store", data={}),

        create_page_header("Akademik Analiz",
                           "Model performans raporu ve akademik metrikler"),

        # Generate report button + status
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    [html.I(className="bi bi-file-earmark-bar-graph me-2"), "Rapor Olustur"],
                    id="academic-report-btn",
                    color="primary",
                    className="me-3",
                ),
                dbc.Button(
                    [html.I(className="bi bi-arrow-clockwise me-2"), "Yenile"],
                    id="academic-refresh-btn",
                    color="secondary",
                    outline=True,
                ),
            ], width="auto"),
            dbc.Col(html.Div(id="academic-report-status"), width="auto", className="d-flex align-items-center"),
        ], className="mb-4 align-items-center"),

        # Best model cards
        dbc.Row(id="academic-best-cards", className="mb-4"),

        # Comparison table
        dbc.Card([
            dbc.CardHeader(html.Span("Model Karsilastirma Tablosu", className="card-title-sm")),
            dbc.CardBody(html.Div(id="academic-comparison-table")),
        ], className="mb-4"),

        # Charts – 4 charts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Portfoy Karsilastirma", className="card-title-sm")),
                    dbc.CardBody(dcc.Graph(id="academic-portfolio-chart", figure=empty_figure(),
                                          config={"displayModeBar": False})),
                ]),
            ], md=6, className="mb-4"),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Risk-Getiri Dagilimi", className="card-title-sm")),
                    dbc.CardBody(dcc.Graph(id="academic-scatter-chart", figure=empty_figure(),
                                          config={"displayModeBar": False})),
                ]),
            ], md=6, className="mb-4"),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Metrik Karsilastirma", className="card-title-sm")),
                    dbc.CardBody(dcc.Graph(id="academic-metrics-chart", figure=empty_figure(),
                                          config={"displayModeBar": False})),
                ]),
            ], md=6, className="mb-4"),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Kazanma Orani", className="card-title-sm")),
                    dbc.CardBody(dcc.Graph(id="academic-winrate-chart", figure=empty_figure(),
                                          config={"displayModeBar": False})),
                ]),
            ], md=6, className="mb-4"),
        ]),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        [Output("academic-poll", "disabled"), Output("academic-report-status", "children")],
        Input("academic-report-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def start_report(n):
        if not n:
            return True, html.Span()
        result = api.generate_report()
        if result:
            return False, dbc.Spinner(size="sm", color="primary")
        return True, dbc.Alert("Rapor baslatılamadı.", color="danger")

    @app.callback(
        [
            Output("academic-best-cards", "children"),
            Output("academic-comparison-table", "children"),
            Output("academic-portfolio-chart", "figure"),
            Output("academic-scatter-chart", "figure"),
            Output("academic-metrics-chart", "figure"),
            Output("academic-winrate-chart", "figure"),
        ],
        [Input("academic-refresh-btn", "n_clicks"), Input("academic-poll", "n_intervals")],
        prevent_initial_call=False,
    )
    def refresh_academic(n_ref, n_poll):
        best = api.get_best_models() or {}
        comparison = api.get_model_comparison() or {}

        best_cards = _render_best_cards(best)
        cmp_table = _render_comparison_table(comparison)
        port_fig = _build_portfolio_chart(comparison)
        scatter_fig = _build_scatter_chart(comparison)
        metrics_fig = _build_metrics_chart(comparison)
        winrate_fig = _build_winrate_chart(comparison)

        return best_cards, cmp_table, port_fig, scatter_fig, metrics_fig, winrate_fig

    @app.callback(
        Output("academic-poll", "disabled", allow_duplicate=True),
        Input("academic-poll", "n_intervals"),
        prevent_initial_call=True,
    )
    def stop_poll_on_data(n):
        # Stop polling after 6 ticks (30 s)
        if n and n >= 6:
            return True
        return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _render_best_cards(best):
    """4 best model cards: Sharpe, Return, Drawdown, WinRate."""
    categories = [
        ("En Iyi Sharpe", "sharpe", BLUE, "bi bi-calculator"),
        ("En Iyi Getiri", "return", GREEN, "bi bi-graph-up"),
        ("Min Drawdown", "drawdown", RED, "bi bi-shield-check"),
        ("En Iyi WinRate", "winrate", PURPLE, "bi bi-trophy"),
    ]
    cols = []
    for label, key, color, icon in categories:
        model_info = best.get(key, best.get(f"best_{key}", {})) or {}
        name = model_info.get("name", model_info.get("model_name", "—"))
        value = model_info.get("value", model_info.get(key, "—"))
        if isinstance(value, float):
            value = f"{value:.3f}"
        cols.append(dbc.Col(
            dbc.Card(dbc.CardBody([
                html.Div([
                    html.I(className=f"{icon} me-2", style={"color": color}),
                    html.Small(label, style={"color": TEXT_MUTED, "fontSize": "11px", "textTransform": "uppercase"}),
                ], className="d-flex align-items-center mb-2"),
                html.Div(str(value), style={"color": color, "fontWeight": "700", "fontSize": "24px"}),
                html.Small(name, style={"color": TEXT_MUTED, "fontSize": "12px"}),
            ])),
            md=3, sm=6, xs=12, className="mb-3",
        ))
    return cols


def _render_comparison_table(comparison):
    models = comparison.get("models", comparison.get("results", [])) if comparison else []
    if not models:
        return create_state_block("empty", "Karsilastirma verisi yok.")

    header = dbc.Row([
        dbc.Col(html.Small("Model", className="section-title"), width=3),
        dbc.Col(html.Small("Algo", className="section-title"), width=1),
        dbc.Col(html.Small("Getiri", className="section-title"), width=2),
        dbc.Col(html.Small("Sharpe", className="section-title"), width=2),
        dbc.Col(html.Small("Max DD", className="section-title"), width=2),
        dbc.Col(html.Small("WinRate", className="section-title"), width=2),
    ])
    rows = [header]
    for m in models:
        algo = m.get("algorithm", "—").upper()
        badge_class = algo_badge_class(algo)
        ret = m.get("total_return", 0) or 0
        rows.append(dbc.Row([
            dbc.Col(html.Small(str(m.get("name", "—"))[:25], style={"color": TEXT}), width=3),
            dbc.Col(dbc.Badge(algo, pill=True, className=badge_class), width=1),
            dbc.Col(html.Small(f"{ret:.1%}" if isinstance(ret, float) else str(ret), style={"color": GREEN if (ret or 0) > 0 else RED}), width=2),
            dbc.Col(html.Small(f"{m.get('sharpe_ratio', 0):.2f}" if m.get('sharpe_ratio') else "—", style={"color": BLUE}), width=2),
            dbc.Col(html.Small(f"{m.get('max_drawdown', 0):.1%}" if m.get('max_drawdown') else "—", style={"color": RED}), width=2),
            dbc.Col(html.Small(f"{m.get('win_rate', 0):.1%}" if m.get('win_rate') else "—", style={"color": PURPLE}), width=2),
        ], className="py-2 border-bottom"))
    return html.Div(rows)


def _build_portfolio_chart(comparison):
    """Line chart per model using equity curves."""
    fig = go.Figure()
    models = comparison.get("models", []) if comparison else []
    P = plot_palette()
    colors = [P["blue"], P["green"], P["purple"], P["orange"], P["cyan"], P["yellow"]]
    for i, m in enumerate(models):
        curve = m.get("equity_curve", m.get("portfolio_history", []))
        if curve:
            fig.add_trace(go.Scatter(
                y=curve, mode="lines",
                name=str(m.get("name", f"Model {i}"))[:15],
                line={"color": colors[i % len(colors)], "width": 1.5},
            ))
    if not models:
        return empty_figure("Model verisi yok")
    apply_theme_template(fig)
    fig.update_layout(height=300, legend={"orientation": "h", "y": 1.1})
    return fig


def _build_scatter_chart(comparison):
    """Risk (Max DD) vs Return scatter."""
    fig = go.Figure()
    models = comparison.get("models", []) if comparison else []
    P = plot_palette()
    algo_hex = algo_plot_colors()
    colors = [algo_hex.get(m.get("algorithm", "").upper(), P["blue"]) for m in models]

    xs = [abs(m.get("max_drawdown", 0) or 0) for m in models]
    ys = [m.get("total_return", 0) or 0 for m in models]
    names = [str(m.get("name", ""))[:12] for m in models]

    if not models:
        return empty_figure("Veri yok")

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        text=names, textposition="top center",
        marker={"color": colors, "size": 12},
        hovertemplate="<b>%{text}</b><br>Max DD: %{x:.1%}<br>Getiri: %{y:.1%}<extra></extra>",
    ))
    apply_theme_template(fig)
    fig.update_layout(height=300, xaxis_title="Max Drawdown", yaxis_title="Toplam Getiri")
    return fig


def _build_metrics_chart(comparison):
    """Grouped bar: Sharpe, Sortino, Calmar."""
    models = comparison.get("models", []) if comparison else []
    if not models:
        return empty_figure("Veri yok")

    names = [str(m.get("name", ""))[:15] for m in models]
    sharpes = [m.get("sharpe_ratio") or 0 for m in models]
    sortinos = [m.get("sortino_ratio") or 0 for m in models]
    calmars = [m.get("calmar_ratio") or 0 for m in models]

    P = plot_palette()
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Sharpe", x=names, y=sharpes, marker_color=P["blue"]))
    fig.add_trace(go.Bar(name="Sortino", x=names, y=sortinos, marker_color=P["green"]))
    fig.add_trace(go.Bar(name="Calmar", x=names, y=calmars, marker_color=P["purple"]))
    fig.update_layout(barmode="group", height=300)
    apply_theme_template(fig)
    return fig


def _build_winrate_chart(comparison):
    """Bar chart for win rate with dual y-axis for trade count."""
    models = comparison.get("models", []) if comparison else []
    if not models:
        return empty_figure("Veri yok")

    names = [str(m.get("name", ""))[:15] for m in models]
    winrates = [(m.get("win_rate") or 0) * 100 for m in models]
    trades = [m.get("total_trades") or 0 for m in models]

    P = plot_palette()
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Kazanma %", x=names, y=winrates,
                          marker_color=P["green"], yaxis="y1"))
    fig.add_trace(go.Scatter(name="Islem Sayisi", x=names, y=trades,
                              mode="lines+markers", marker={"color": P["orange"]},
                              yaxis="y2"))
    apply_theme_template(fig)
    fig.update_layout(
        height=300,
        yaxis={"title": "Kazanma %"},
        yaxis2={"title": "Islem Sayisi", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "y": 1.1},
    )
    return fig
