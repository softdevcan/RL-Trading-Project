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

import dash
from dash import html, dcc
from dash import Input, Output, State
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable
import plotly.graph_objects as go

from dashboard.theme import (
    TEXT, TEXT_MUTED, GREEN, RED, BLUE, PURPLE, empty_figure, apply_theme_template,
    plot_palette, plot_rgba,
)
from dashboard.components.page_header import create_page_header
from dashboard.components.state_block import create_state_block
import dashboard.api_client as api

BIST30_SYMBOLS = [
    "AKBNK.IS","ARCLK.IS","ASELS.IS","BIMAS.IS","EKGYO.IS","EREGL.IS","FROTO.IS","GARAN.IS",
    "HALKB.IS","ISCTR.IS","KCHOL.IS","KOZAL.IS","KRDMD.IS","MGROS.IS","ODAS.IS","PETKM.IS",
    "PGSUS.IS","SAHOL.IS","SASA.IS","SISE.IS","SKBNK.IS","TAVHL.IS","TCELL.IS","THYAO.IS",
    "TKFEN.IS","TOASO.IS","TSKB.IS","TUPRS.IS","VAKBN.IS","YKBNK.IS",
]
DEFAULT_SYMBOLS = BIST30_SYMBOLS[:5]


# ── Layout ────────────────────────────────────────────────────────────────────

def layout():
    return html.Div([
        dcc.Store(id="dt-decision-store", data={}),

        create_page_header("Gunluk Trading",
                           "Model karari al ve portfoy uygula"),

        dbc.Row([
            # ── Left panel: settings ─────────────────────────────────────────
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Ayarlar", className="card-title-sm")),
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
                                {"label": "Orta Risk", "value": "moderate"},
                                {"label": "Yuksek Risk", "value": "aggressive"},
                            ],
                            value="moderate",
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

                        html.Hr(),

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

                        html.Hr(),

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
                ]),
            ], md=4, className="mb-4"),

            # ── Right panel: results ─────────────────────────────────────────
            dbc.Col([
                # History selector — gecmis kararlari gormek icin tarih dropdown.
                dbc.Card([
                    dbc.CardHeader(html.Span("Gecmis Kararlar", className="card-title-sm")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(
                                id="dt-history-select",
                                options=[],
                                value=None,
                                placeholder="Onceki bir karari yukle...",
                                clearable=True,
                                                            ), width=9),
                            dbc.Col(dbc.Button(
                                [html.I(className="bi bi-arrow-clockwise me-2"), "Yenile"],
                                id="dt-history-refresh",
                                color="secondary",
                                outline=True,
                                className="w-100",
                            ), width=3),
                        ]),
                    ]),
                ], className="mb-3"),

                # Summary cards
                dbc.Row([
                    dbc.Col(html.Div(id="dt-summary-cards"), className="mb-4"),
                ]),

                # Decision table
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("Trading Kararlari", className="card-title-sm"),
                        html.Span(id="dt-decision-date-badge", style={"color": TEXT_MUTED, "marginLeft": "12px", "fontSize": "13px"}),
                    ]),
                    dbc.CardBody(html.Div(id="dt-decision-table")),
                ], className="mb-4"),

                # Portfolio chart
                dbc.Card([
                    dbc.CardHeader(html.Span("Portfoy Gecmisi", className="card-title-sm")),
                    dbc.CardBody(dcc.Graph(id="dt-portfolio-chart", figure=empty_figure(),
                                          config={"displayModeBar": False})),
                ]),
            ], md=8, className="mb-4"),
        ]),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        [
            Output("dt-model-select", "options"),
            Output("dt-history-select", "options", allow_duplicate=True),
        ],
        Input("dt-decide-btn", "id"),  # fire once on load
        prevent_initial_call="initial_duplicate",
    )
    def load_models(_):
        models = api.get_models()
        model_opts = [
            {"label": f"[{m.get('algorithm','').upper()}] {m.get('name','')}", "value": m.get("name", "")}
            for m in models
        ]
        # Populate history dropdown at page load so the user can pick a past
        # decision without having to click "Karar Al" first.
        hist = api.get_decisions_history() or {"dates": []}
        hist_opts = [{"label": d, "value": d} for d in (hist.get("dates") or [])]
        return model_opts, hist_opts

    @app.callback(
        [
            Output("dt-decision-table", "children"),
            Output("dt-summary-cards", "children"),
            Output("dt-portfolio-chart", "figure"),
            Output("dt-decision-store", "data"),
            Output("dt-decision-date-badge", "children"),
            Output("dt-history-select", "options"),
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
    def get_decision(decide_n, model, risk, dt, max_shares, balance,
                     s0, q0, s1, q1, s2, q2, s3, q3, s4, q4):
        if not decide_n or not model:
            empty_msg = html.P("Model secin ve 'Karar Al' butonuna basin.", style={"color": TEXT_MUTED})
            return empty_msg, html.Span(), empty_figure(), {}, html.Span(), dash.no_update

        holdings = {}
        for sym, qty in [(s0, q0), (s1, q1), (s2, q2), (s3, q3), (s4, q4)]:
            if sym and qty:
                normalized = sym if "." in sym else f"{sym}.IS"
                holdings[normalized] = int(qty)

        payload = {
            "model_name": model,
            "risk_mode": risk or "moderate",
            "date": str(dt) if dt else str(date.today()),
            "max_shares_per_trade": int(max_shares or 5),
            "balance": float(balance or 100_000),
            "shares": holdings,
        }

        result = api.get_daily_decision(payload)
        if not result:
            return (
                dbc.Alert("Karar alinamadi.", color="danger"),
                html.Span(), empty_figure(), {},
                html.Span(), dash.no_update,
            )

        table = _render_decision_table(result)
        cards = _render_summary_cards(result)
        chart = _build_portfolio_chart(api.get_portfolio_history())
        # Yeni karar kaydedildi — dropdown listesini yenile ki hemen gorunsun.
        refreshed = api.get_decisions_history() or {"dates": [], "decisions": {}}
        history_options = [{"label": d, "value": d} for d in (refreshed.get("dates") or [])]
        badge = f"({result.get('date', payload['date'])})"
        return table, cards, chart, result, badge, history_options

    # Gecmis karar yuklemesi ayri callback — boylece options refresh edilince
    # dropdown.value=None firing'i karar tablosunu silmez.
    @app.callback(
        [
            Output("dt-decision-table", "children", allow_duplicate=True),
            Output("dt-summary-cards", "children", allow_duplicate=True),
            Output("dt-decision-store", "data", allow_duplicate=True),
            Output("dt-decision-date-badge", "children", allow_duplicate=True),
        ],
        Input("dt-history-select", "value"),
        prevent_initial_call=True,
    )
    def load_history(hist_date):
        if not hist_date:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        history_doc = api.get_decisions_history() or {"decisions": {}}
        entry = (history_doc.get("decisions") or {}).get(hist_date)
        if not entry:
            return (
                dbc.Alert("Secili tarih icin kayit bulunamadi.", color="warning"),
                html.Span(), {}, html.Span(),
            )
        result = {
            "date": hist_date,
            "decisions": entry.get("decisions", []),
            "portfolio_before": entry.get("portfolio_before", {}),
            "portfolio_after": entry.get("portfolio_after", {}),
            "summary": entry.get("summary", {}),
        }
        table = _render_decision_table(result)
        cards = _render_summary_cards(result)
        badge = f"({hist_date} - arsivden)"
        return table, cards, result, badge

    # Yenile butonu sadece dropdown options'u tazeler — kararlari etkilemez.
    @app.callback(
        Output("dt-history-select", "options", allow_duplicate=True),
        Input("dt-history-refresh", "n_clicks"),
        prevent_initial_call=True,
    )
    def refresh_history_options(_n):
        hist = api.get_decisions_history() or {"dates": []}
        return [{"label": d, "value": d} for d in (hist.get("dates") or [])]

    @app.callback(
        [Output("dt-action-result", "children", allow_duplicate=True)],
        Input("dt-apply-btn", "n_clicks"),
        State("dt-decision-store", "data"),
        prevent_initial_call=True,
    )
    def apply_decision(n, decision):
        if not n or not decision:
            return [html.Span()]
        decision_date = decision.get("date") if isinstance(decision, dict) else None
        if not decision_date:
            return [dbc.Alert("Once 'Karar Al' ile bir karar uretilmeli.", color="warning", dismissable=True)]
        result = api.apply_decision(decision_date)
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
        return create_state_block("empty", "Karar yok.")

    header = dbc.Row([
        dbc.Col(html.Small("Sembol", className="section-title"), width=2),
        dbc.Col(html.Small("Islem", className="section-title"), width=2),
        dbc.Col(html.Small("Adet", className="section-title"), width=1),
        dbc.Col(html.Small("Fiyat", className="section-title"), width=2),
        dbc.Col(html.Small("Toplam", className="section-title"), width=2),
        dbc.Col(html.Small("Sebep", className="section-title"), width=3),
    ])
    rows = [header]
    for d in decisions:
        action = str(d.get("action", d.get("type", "—"))).upper()
        is_buy = "BUY" in action or "AL" == action
        is_sell = "SELL" in action or "SAT" == action
        color = GREEN if is_buy else RED if is_sell else TEXT_MUTED
        # Backend dondurur: cost (BUY), revenue (SELL); 0 ise diger taraf.
        total_val = d.get("total", d.get("amount"))
        if total_val is None:
            total_val = d.get("cost", 0) if is_buy else d.get("revenue", 0) if is_sell else 0
        shares_val = d.get("quantity", d.get("shares", 0))
        rows.append(dbc.Row([
            dbc.Col(html.Span(d.get("symbol", "—"), className="card-title-sm"), width=2),
            dbc.Col(dbc.Badge(action, style={"backgroundColor": color}, pill=True), width=2),
            dbc.Col(html.Span(str(shares_val), style={"color": TEXT}), width=1),
            dbc.Col(html.Span(f"₺{d.get('price', 0):,.2f}", style={"color": TEXT}), width=2),
            dbc.Col(html.Span(f"₺{total_val:,.2f}", style={"color": TEXT}), width=2),
            dbc.Col(html.Span(d.get("reason", ""), style={"color": TEXT_MUTED, "fontSize": "12px"}), width=3),
        ], className="py-2 border-bottom"))
    return html.Div(rows)


def _render_summary_cards(result):
    """Backend response field names: decisions[].cost / .revenue / .executed; summary.daily_return_pct."""
    decisions = result.get("decisions", result.get("trades", []))
    total_buy = 0.0
    total_sell = 0.0
    n_executed = 0
    for d in decisions:
        if not d.get("executed", False):
            continue
        action = str(d.get("action", "")).upper()
        if "BUY" in action or action == "AL":
            total_buy += float(d.get("cost", d.get("total", 0)) or 0)
            n_executed += 1
        elif "SELL" in action or action == "SAT":
            total_sell += float(d.get("revenue", d.get("total", 0)) or 0)
            n_executed += 1

    # API zaten net_change vs. donerse onlari kullan (custom override icin).
    total_buy = result.get("total_buy_amount") or total_buy
    total_sell = result.get("total_sell_amount") or total_sell
    net = result.get("net_change")
    if net is None:
        net = total_sell - total_buy  # nakit akisi: satistan giren - alima cikan

    # Negatif = nakit cikisi (alim agirligi), pozitif = nakit girisi (satis
    # agirligi). Renkle ayirt et, sifirken notr goster.
    if net > 0:
        net_color, net_prefix = GREEN, "+"
    elif net < 0:
        net_color, net_prefix = RED, ""
    else:
        net_color, net_prefix = BLUE, ""

    return dbc.Row([
        dbc.Col(_mini_card("Al Toplam", f"₺{total_buy:,.2f}", GREEN), xs=6, md=3, className="mb-3"),
        dbc.Col(_mini_card("Sat Toplam", f"₺{total_sell:,.2f}", RED), xs=6, md=3, className="mb-3"),
        dbc.Col(_mini_card("Net Nakit Akisi", f"{net_prefix}₺{net:,.2f}", net_color), xs=6, md=3, className="mb-3"),
        dbc.Col(_mini_card("Islem Sayisi", str(n_executed), PURPLE), xs=6, md=3, className="mb-3"),
    ])


def _mini_card(title, value, color):
    return dbc.Card(dbc.CardBody([
        html.Small(title, style={"color": TEXT_MUTED, "fontSize": "11px", "textTransform": "uppercase"}),
        html.Div(value, style={"color": color, "fontWeight": "700", "fontSize": "20px"}),
    ]))


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
            fillcolor=plot_rgba("blue", 0.12),
            line={"color": plot_palette()["blue"], "width": 2},
            hovertemplate="<b>%{x}</b><br>₺%{y:,.0f}<extra></extra>",
        ))
    else:
        return empty_figure("Portfoy gecmisi yok")

    apply_theme_template(fig)
    fig.update_layout(showlegend=False, height=280)
    return fig
