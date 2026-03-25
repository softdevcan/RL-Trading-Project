"""
Daily Trading page – model selection, risk mode, portfolio inputs, decisions.

API endpoints used:
  GET  /api/trading/models
  POST /api/trading/daily-decision
  POST /api/trading/apply-decision
  GET  /api/trading/latest-portfolio
  GET  /api/trading/portfolio-history
"""

import json
from datetime import date

from dash import html, dcc
from dash import Input, Output, State
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable
import plotly.graph_objects as go

from dashboard.theme import (
    CARD, CARD2, TEXT, TEXT_MUTED, BORDER, GREEN, RED, BLUE, YELLOW, PURPLE,
    DARK_TEMPLATE, empty_figure, apply_dark_template,
)
import dashboard.api_client as api

BIST30_SYMBOLS = [
    "AKBNK","ARCLK","ASELS","BIMAS","EKGYO","EREGL","FROTO","GARAN",
    "HALKB","ISCTR","KCHOL","KOZAL","KRDMD","MGROS","ODAS","PETKM",
    "PGSUS","SAHOL","SASA","SISE","SKBNK","TAVHL","TCELL","THYAO",
    "TKFEN","TOASO","TSKB","TUPRS","VAKBN","YKBNK",
]
DEFAULT_SYMBOLS = BIST30_SYMBOLS[:5]


# ── Layout ────────────────────────────────────────────────────────────────────

def layout():
    return html.Div([
        dcc.Store(id="dt-decision-store", data={}),

        html.H4("Gunluk Trading", style={"color": TEXT, "marginBottom": "4px"}),
        html.P("Model karari al ve portfoy uygula", style={"color": TEXT_MUTED, "marginBottom": "24px"}),

        dbc.Row([
            # ── Left panel: settings ─────────────────────────────────────────
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Ayarlar", style={"color": TEXT, "fontWeight": "600"})),
                    dbc.CardBody([
                        # Model selection
                        html.Label("Model", className="section-title"),
                        dcc.Dropdown(id="dt-model-select", options=[], value=None,
                                     placeholder="Model sec...", clearable=False,
                                     style={"marginBottom": "16px"}),

                        # Risk mode
                        html.Label("Risk Modu", className="section-title"),
                        dbc.RadioItems(
                            id="dt-risk-mode",
                            options=[
                                {"label": "Dusuk Risk", "value": "conservative"},
                                {"label": "Orta Risk", "value": "balanced"},
                                {"label": "Yuksek Risk", "value": "aggressive"},
                            ],
                            value="balanced",
                            inline=True,
                            className="mb-3",
                        ),

                        # Date
                        html.Label("Tarih", className="section-title"),
                        dcc.DatePickerSingle(
                            id="dt-date",
                            date=str(date.today()),
                            display_format="YYYY-MM-DD",
                            className="mb-3",
                            style={"width": "100%"},
                        ),

                        # Max shares slider
                        html.Label("Max Hisse Adedi", className="section-title"),
                        dcc.Slider(
                            id="dt-max-shares",
                            min=1, max=10, step=1, value=5,
                            marks={i: str(i) for i in range(1, 11)},
                            className="mb-3",
                        ),

                        html.Hr(style={"borderColor": CARD2}),

                        # Portfolio inputs
                        html.Label("Mevcut Portfoy", className="section-title"),
                        html.Label("Bakiye (₺)", className="section-title"),
                        dbc.Input(id="dt-balance", type="number", value=100_000,
                                  min=0, className="mb-2"),
                        *[
                            html.Div([
                                html.Label(f"Hisse {i+1}", className="section-title"),
                                dbc.Row([
                                    dbc.Col(dcc.Dropdown(
                                        id=f"dt-sym-{i}",
                                        options=[{"label": s, "value": s} for s in BIST30_SYMBOLS],
                                        value=DEFAULT_SYMBOLS[i] if i < len(DEFAULT_SYMBOLS) else None,
                                        clearable=True,
                                    ), width=7),
                                    dbc.Col(dbc.Input(id=f"dt-qty-{i}", type="number", value=0, min=0,
                                                      placeholder="Adet"), width=5),
                                ], className="mb-2"),
                            ])
                            for i in range(5)
                        ],

                        html.Hr(style={"borderColor": CARD2}),

                        # Action buttons
                        dbc.Button(
                            [html.I(className="bi bi-calculator me-2"), "Karar Al"],
                            id="dt-decide-btn",
                            color="primary",
                            className="w-100 mb-2",
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-check-circle me-2"), "Uygula"],
                            id="dt-apply-btn",
                            color="success",
                            outline=True,
                            className="w-100 mb-2",
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-download me-2"), "Disa Aktar (JSON)"],
                            id="dt-export-btn",
                            color="secondary",
                            outline=True,
                            className="w-100",
                        ),
                        dcc.Download(id="dt-download"),
                        html.Div(id="dt-action-result", className="mt-3"),
                    ]),
                ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
            ], md=4, className="mb-4"),

            # ── Right panel: results ─────────────────────────────────────────
            dbc.Col([
                # Summary cards
                dbc.Row([
                    dbc.Col(html.Div(id="dt-summary-cards"), className="mb-4"),
                ]),

                # Decision table
                dbc.Card([
                    dbc.CardHeader(html.Span("Trading Kararlari", style={"color": TEXT, "fontWeight": "600"})),
                    dbc.CardBody(html.Div(id="dt-decision-table")),
                ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}", "marginBottom": "24px"}),

                # Portfolio chart
                dbc.Card([
                    dbc.CardHeader(html.Span("Portfoy Gecmisi", style={"color": TEXT, "fontWeight": "600"})),
                    dbc.CardBody(dcc.Graph(id="dt-portfolio-chart", figure=empty_figure(),
                                          config={"displayModeBar": False})),
                ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
            ], md=8, className="mb-4"),
        ]),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("dt-model-select", "options"),
        Input("dt-decide-btn", "id"),  # fire once on load
        prevent_initial_call=False,
    )
    def load_models(_):
        models = api.get_models()
        return [{"label": f"[{m.get('algorithm','').upper()}] {m.get('name','')}", "value": m.get("name", "")} for m in models]

    @app.callback(
        [
            Output("dt-decision-table", "children"),
            Output("dt-summary-cards", "children"),
            Output("dt-portfolio-chart", "figure"),
            Output("dt-decision-store", "data"),
        ],
        Input("dt-decide-btn", "n_clicks"),
        [
            State("dt-model-select", "value"),
            State("dt-risk-mode", "value"),
            State("dt-date", "date"),
            State("dt-max-shares", "value"),
            State("dt-balance", "value"),
            State("dt-sym-0", "value"), State("dt-qty-0", "value"),
            State("dt-sym-1", "value"), State("dt-qty-1", "value"),
            State("dt-sym-2", "value"), State("dt-qty-2", "value"),
            State("dt-sym-3", "value"), State("dt-qty-3", "value"),
            State("dt-sym-4", "value"), State("dt-qty-4", "value"),
        ],
        prevent_initial_call=True,
    )
    def get_decision(n, model, risk, dt, max_shares, balance,
                     s0, q0, s1, q1, s2, q2, s3, q3, s4, q4):
        if not n or not model:
            return html.P("Model secin ve 'Karar Al' butonuna basin.", style={"color": TEXT_MUTED}), html.Span(), empty_figure(), {}

        # Build portfolio snapshot
        holdings = {}
        for sym, qty in [(s0, q0), (s1, q1), (s2, q2), (s3, q3), (s4, q4)]:
            if sym and qty:
                holdings[sym] = int(qty)

        payload = {
            "model_name": model,
            "risk_mode": risk or "balanced",
            "date": str(dt) if dt else str(date.today()),
            "max_shares_per_stock": int(max_shares or 5),
            "portfolio": {
                "balance": float(balance or 100_000),
                "holdings": holdings,
            },
        }

        result = api.get_daily_decision(payload)
        if not result:
            return dbc.Alert("Karar alinamadi.", color="danger"), html.Span(), empty_figure(), {}

        table = _render_decision_table(result)
        cards = _render_summary_cards(result)
        history = api.get_portfolio_history()
        chart = _build_portfolio_chart(history)
        return table, cards, chart, result

    @app.callback(
        [Output("dt-action-result", "children", allow_duplicate=True)],
        Input("dt-apply-btn", "n_clicks"),
        State("dt-decision-store", "data"),
        prevent_initial_call=True,
    )
    def apply_decision(n, decision):
        if not n or not decision:
            return [html.Span()]
        result = api.apply_decision(decision)
        if result:
            return [dbc.Alert(
                [html.I(className="bi bi-check-circle me-2"), "Karar basariyla uygulandı."],
                color="success", dismissable=True,
            )]
        return [dbc.Alert("Uygulama basarisiz.", color="danger", dismissable=True)]

    @app.callback(
        Output("dt-download", "data"),
        Input("dt-export-btn", "n_clicks"),
        State("dt-decision-store", "data"),
        prevent_initial_call=True,
    )
    def export_decision(n, decision):
        if not n or not decision:
            return None
        return dcc.send_string(json.dumps(decision, indent=2, ensure_ascii=False), filename="trading_decision.json")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _render_decision_table(result):
    decisions = result.get("decisions", result.get("trades", []))
    if not decisions:
        return html.P("Karar yok.", style={"color": TEXT_MUTED})

    header = dbc.Row([
        dbc.Col(html.Small("Sembol", className="section-title"), width=3),
        dbc.Col(html.Small("Islem", className="section-title"), width=2),
        dbc.Col(html.Small("Adet", className="section-title"), width=2),
        dbc.Col(html.Small("Fiyat", className="section-title"), width=2),
        dbc.Col(html.Small("Toplam", className="section-title"), width=3),
    ])
    rows = [header]
    for d in decisions:
        action = str(d.get("action", d.get("type", "—"))).upper()
        color = GREEN if "BUY" in action or "AL" in action else RED if "SELL" in action or "SAT" in action else TEXT_MUTED
        rows.append(dbc.Row([
            dbc.Col(html.Span(d.get("symbol", "—"), style={"color": TEXT, "fontWeight": "600"}), width=3),
            dbc.Col(dbc.Badge(action, style={"backgroundColor": color}, pill=True), width=2),
            dbc.Col(html.Span(str(d.get("quantity", d.get("shares", "—"))), style={"color": TEXT}), width=2),
            dbc.Col(html.Span(f"₺{d.get('price', 0):,.2f}", style={"color": TEXT}), width=2),
            dbc.Col(html.Span(f"₺{d.get('total', d.get('amount', 0)):,.2f}", style={"color": TEXT}), width=3),
        ], className="py-2 border-bottom"))
    return html.Div(rows)


def _render_summary_cards(result):
    total_buy = result.get("total_buy_amount", 0) or 0
    total_sell = result.get("total_sell_amount", 0) or 0
    net = result.get("net_change", total_buy - total_sell)
    n_decisions = len(result.get("decisions", result.get("trades", [])))

    return dbc.Row([
        dbc.Col(_mini_card("Al Toplam", f"₺{total_buy:,.0f}", GREEN), xs=6, md=3, className="mb-3"),
        dbc.Col(_mini_card("Sat Toplam", f"₺{total_sell:,.0f}", RED), xs=6, md=3, className="mb-3"),
        dbc.Col(_mini_card("Net Degisim", f"₺{net:,.0f}", BLUE), xs=6, md=3, className="mb-3"),
        dbc.Col(_mini_card("Islem Sayisi", str(n_decisions), PURPLE), xs=6, md=3, className="mb-3"),
    ])


def _mini_card(title, value, color):
    return dbc.Card(dbc.CardBody([
        html.Small(title, style={"color": TEXT_MUTED, "fontSize": "11px", "textTransform": "uppercase"}),
        html.Div(value, style={"color": color, "fontWeight": "700", "fontSize": "20px"}),
    ]), style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"})


def _build_portfolio_chart(history):
    fig = go.Figure()
    records = []
    if isinstance(history, dict):
        records = history.get("history", [])
    elif isinstance(history, list):
        records = history

    if records:
        dates = [r.get("date", str(i)) for i, r in enumerate(records)]
        values = [r.get("value", r.get("portfolio_value", 0)) for r in records]
        fig.add_trace(go.Scatter(
            x=dates, y=values, mode="lines",
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.12)",
            line={"color": BLUE, "width": 2},
            hovertemplate="<b>%{x}</b><br>₺%{y:,.0f}<extra></extra>",
        ))
    else:
        return empty_figure("Portfoy gecmisi yok")

    apply_dark_template(fig)
    fig.update_layout(showlegend=False, height=280)
    return fig
