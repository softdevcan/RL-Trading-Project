"""
Models comparison page – multi-model selection, metrics table, charts, modal.

API endpoints used:
  GET /api/trading/models
  GET /api/trading/models/{name}/metrics
"""

from dash import html, dcc, no_update
from dash import Input, Output, State
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable
import plotly.graph_objects as go

from dashboard.theme import (
    TEXT, TEXT_MUTED, GREEN, RED, empty_figure, apply_theme_template, plot_palette,
    plot_rgba,
)
from dashboard.components.page_header import create_page_header
from dashboard.components.state_block import create_state_block
from dashboard.components.table import (
    TABLE_STYLES, numeric_columns, tone_rules_formatted,
)
import dashboard.api_client as api


# ── Layout ────────────────────────────────────────────────────────────────────

def layout():
    return html.Div([
        dcc.Store(id="models-store", data=[]),
        # Silme sonrasi listeyi tazeler
        dcc.Store(id="models-delete-tick", data=0),

        create_page_header("Model Karsilastirma",
                           "Birden fazla modeli karsilastir ve detaylari incele"),

        # Model selection
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Model Sec", className="section-title"),
                        dbc.Checklist(id="models-checklist", options=[], value=[], inline=False),
                    ], md=4),
                    dbc.Col([
                        dbc.Button([html.I(className="bi bi-check-all me-1"), "Tumu Sec"],
                                   id="models-select-all", size="sm", color="primary", outline=True, className="me-2"),
                        dbc.Button([html.I(className="bi bi-x-lg me-1"), "Tumu Kaldir"],
                                   id="models-deselect-all", size="sm", color="secondary", outline=True),
                    ], md=4, className="d-flex align-items-end"),
                    dbc.Col([
                        dbc.Button([html.I(className="bi bi-trash me-1"), "Secilenleri sil"],
                                   id="models-delete-btn", size="sm", color="danger",
                                   outline=True, disabled=True),
                    ], md=2, className="d-flex align-items-end"),
                    dbc.Col([
                        dbc.Button([html.I(className="bi bi-arrow-clockwise me-2"), "Yenile"],
                                   id="models-refresh-btn", color="secondary", outline=True, className="w-100"),
                    ], md=2, className="d-flex align-items-end ms-auto"),
                ], className="g-3"),
                html.Div(id="models-delete-alert", className="mt-3"),
            ])
        ], className="mb-4"),

        # Silme onayi — geri alinamaz bir islem, tek tikla olmamali
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Modelleri sil")),
                dbc.ModalBody(html.Div(id="models-delete-body")),
                dbc.ModalFooter([
                    dbc.Button("Vazgec", id="models-delete-cancel",
                               color="secondary", outline=True, className="me-2"),
                    dbc.Button([html.I(className="bi bi-trash me-1"), "Sil"],
                               id="models-delete-confirm", color="danger"),
                ]),
            ],
            id="models-delete-modal",
            is_open=False,
            centered=True,
        ),

        # Comparison table
        dbc.Card([
            dbc.CardHeader(html.Span("Karsilastirma Tablosu", className="card-title-sm")),
            dbc.CardBody([
                DataTable(
                    id="models-table",
                    columns=[
                        {"name": "Model", "id": "name"},
                        {"name": "Algo", "id": "algorithm"},
                        {"name": "Faz", "id": "phase"},
                        {"name": "Toplam Getiri", "id": "total_return"},
                        {"name": "Sharpe", "id": "sharpe_ratio"},
                        {"name": "Sortino", "id": "sortino_ratio"},
                        {"name": "Calmar", "id": "calmar_ratio"},
                        {"name": "Max DD", "id": "max_drawdown"},
                        {"name": "Kazanma %", "id": "win_rate"},
                    ],
                    data=[],
                    page_size=10,
                    sort_action="native",
                    **TABLE_STYLES,
                    # Sayilar saga dayali; basamak sayisi degisen degerler
                    # sola dayaliyken alt alta hizalanmiyordu (C.6).
                    style_cell_conditional=numeric_columns(
                        "total_return", "sharpe_ratio", "sortino_ratio",
                        "calmar_ratio", "max_drawdown", "win_rate",
                    ),
                    # Renk yalnizca yon tasiyan sutunlarda
                    style_data_conditional=(
                        tone_rules_formatted("total_return")
                        + tone_rules_formatted("max_drawdown")
                    ),
                    row_selectable="single",
                ),
            ]),
        ], className="mb-4"),

        # Charts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Sharpe / Sortino / Calmar", className="card-title-sm")),
                    dbc.CardBody(dcc.Graph(id="models-bar-chart", figure=empty_figure(), config={"displayModeBar": False})),
                ]),
            ], md=12, className="mb-4"),
        ]),

        # Detail modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(html.Span(id="models-modal-title")), close_button=True),
            dbc.ModalBody([
                dbc.Row([
                    dbc.Col(dcc.Graph(id="models-modal-portfolio-chart", figure=empty_figure(), config={"displayModeBar": False}), md=8),
                    dbc.Col(html.Div(id="models-modal-metrics"), md=4),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id="models-modal-activity-chart", figure=empty_figure(), config={"displayModeBar": False}),
                    ], md=12),
                ]),
                html.Div(id="models-modal-trades-table"),
            ]),
            dbc.ModalFooter(dbc.Button("Kapat", id="models-modal-close", color="secondary")),
        ], id="models-modal", size="xl", is_open=False),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        [Output("models-checklist", "options"), Output("models-store", "data")],
        [Input("models-refresh-btn", "n_clicks"),
         Input("models-delete-tick", "data")],
        prevent_initial_call=False,
    )
    def refresh_models(n, _tick):
        models = api.get_models()
        opts = [{"label": _model_label(m), "value": m.get("name", "")} for m in models]
        return opts, models

    @app.callback(
        Output("models-checklist", "value"),
        [Input("models-select-all", "n_clicks"), Input("models-deselect-all", "n_clicks")],
        State("models-checklist", "options"),
        prevent_initial_call=True,
    )
    def toggle_all(sel, desel, opts):
        from dash import ctx
        if ctx.triggered_id == "models-select-all":
            return [o["value"] for o in (opts or [])]
        return []

    @app.callback(
        Output("models-delete-btn", "disabled"),
        Input("models-checklist", "value"),
    )
    def toggle_delete_enabled(selected):
        return not selected

    @app.callback(
        [Output("models-delete-modal", "is_open"),
         Output("models-delete-body", "children")],
        [Input("models-delete-btn", "n_clicks"),
         Input("models-delete-cancel", "n_clicks"),
         Input("models-delete-confirm", "n_clicks")],
        State("models-checklist", "value"),
        prevent_initial_call=True,
    )
    def toggle_delete_modal(open_n, cancel_n, confirm_n, selected):
        from dash import ctx
        if ctx.triggered_id != "models-delete-btn":
            return False, no_update
        names = selected or []
        body = html.Div([
            html.P(f"{len(names)} model kalici olarak silinecek. Bu islem geri alinamaz.",
                   style={"color": TEXT}),
            html.Ul([html.Li(n, style={"color": TEXT_MUTED, "fontSize": "13px"})
                     for n in names]),
            html.Small(
                "Model dosyasi ve varsa metrik JSON'u silinir. Ortak (kullanici "
                "oncesi) dizindeki modeller salt-okunurdur ve atlanir.",
                style={"color": TEXT_MUTED},
            ),
        ])
        return True, body

    @app.callback(
        [Output("models-delete-alert", "children"),
         Output("models-delete-tick", "data"),
         Output("models-checklist", "value", allow_duplicate=True)],
        Input("models-delete-confirm", "n_clicks"),
        [State("models-checklist", "value"), State("models-delete-tick", "data")],
        prevent_initial_call=True,
    )
    def delete_selected(n_clicks, selected, tick):
        if not n_clicks or not selected:
            return no_update, no_update, no_update

        ok, failed = [], []
        for name in selected:
            result = api.delete_model(name)
            if result.get("ok"):
                ok.append(name)
            else:
                detail = (result.get("body") or {}).get("detail") or f"HTTP {result.get('status')}"
                failed.append(f"{name}: {detail}")

        blocks = []
        if ok:
            blocks.append(dbc.Alert(f"{len(ok)} model silindi.", color="success",
                                    className="py-2 mb-2", duration=4000))
        if failed:
            # Her basarisiz satir ayri gosterilir: 403 (ortak dizin) ile 404
            # (baska bir yerde zaten silinmis) farkli sebepler.
            blocks.append(dbc.Alert(
                [html.Div("Silinemeyenler:", style={"fontWeight": "600"})]
                + [html.Div(f, style={"fontSize": "12px"}) for f in failed],
                color="danger", className="py-2 mb-0",
            ))

        return html.Div(blocks), (tick or 0) + 1, []

    @app.callback(
        [Output("models-table", "data"), Output("models-bar-chart", "figure")],
        Input("models-checklist", "value"),
        State("models-store", "data"),
        prevent_initial_call=False,
    )
    def update_comparison(selected, models):
        if not selected:
            return [], empty_figure("Model secilmedi")

        rows = []
        names, sharpes, sortinos, calmars = [], [], [], []

        for m in (models or []):
            name = m.get("name", "")
            if name not in selected:
                continue
            metrics = api.get_model_metrics(name) or {}
            row = {
                "name": name,
                "algorithm": m.get("algorithm", "—").upper(),
                "phase": str(m.get("phase", "—")),
                "total_return": _fmt_pct(api.model_return(metrics)),
                "sharpe_ratio": _fmt2(metrics.get("sharpe_ratio")),
                "sortino_ratio": _fmt2(metrics.get("sortino_ratio")),
                "calmar_ratio": _fmt2(metrics.get("calmar_ratio")),
                "max_drawdown": _fmt_pct(metrics.get("max_drawdown")),
                "win_rate": _fmt_pct(metrics.get("win_rate")),
            }
            rows.append(row)
            names.append(name[:20])
            sharpes.append(metrics.get("sharpe_ratio") or 0)
            sortinos.append(metrics.get("sortino_ratio") or 0)
            calmars.append(metrics.get("calmar_ratio") or 0)

        # Grouped bar chart
        fig = go.Figure()
        P = plot_palette()
        fig.add_trace(go.Bar(name="Sharpe", x=names, y=sharpes, marker_color=P["blue"]))
        fig.add_trace(go.Bar(name="Sortino", x=names, y=sortinos, marker_color=P["green"]))
        fig.add_trace(go.Bar(name="Calmar", x=names, y=calmars, marker_color=P["purple"]))
        fig.update_layout(barmode="group", height=300)
        apply_theme_template(fig)

        return rows, fig

    @app.callback(
        [
            Output("models-modal", "is_open"),
            Output("models-modal-title", "children"),
            Output("models-modal-portfolio-chart", "figure"),
            Output("models-modal-activity-chart", "figure"),
            Output("models-modal-metrics", "children"),
            Output("models-modal-trades-table", "children"),
        ],
        [Input("models-table", "selected_rows"), Input("models-modal-close", "n_clicks")],
        [State("models-table", "data"), State("models-modal", "is_open")],
        prevent_initial_call=True,
    )
    def open_modal(selected_rows, close_n, table_data, is_open):
        from dash import ctx
        if ctx.triggered_id == "models-modal-close" or not selected_rows:
            return False, "", empty_figure(), empty_figure(), html.Span(), html.Span()

        row = (table_data or [])[selected_rows[0]]
        name = row.get("name", "")
        metrics = api.get_model_metrics(name) or {}

        # Portfolio chart (simulated from metrics)
        port_fig = _build_modal_portfolio_chart(metrics)
        act_fig = _build_activity_chart(metrics)
        metrics_div = _render_modal_metrics(metrics)
        trades_table = _render_trades_table(metrics)

        return True, name, port_fig, act_fig, metrics_div, trades_table

    @app.callback(
        Output("models-modal", "is_open", allow_duplicate=True),
        Input("models-modal-close", "n_clicks"),
        prevent_initial_call=True,
    )
    def close_modal(n):
        return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _model_label(m):
    name = m.get("name", "unknown")
    algo = m.get("algorithm", "").upper()
    return f"[{algo}] {name}"


def _fmt_pct(v):
    if v is None:
        return "—"
    try:
        return f"{float(v):.1%}"
    except Exception:
        return str(v)


def _fmt2(v):
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)


def _build_modal_portfolio_chart(metrics):
    history = metrics.get("portfolio_history") or metrics.get("equity_curve") or []
    fig = go.Figure()
    if history:
        values = [h.get("value", h) if isinstance(h, dict) else h for h in history]
        fig.add_trace(go.Scatter(
            y=values,
            mode="lines",
            fill="tozeroy",
            fillcolor=plot_rgba("blue", 0.15),
            line={"color": plot_palette()["blue"], "width": 2},
            name="Portfoy",
        ))
    else:
        return empty_figure("Portfoy gecmisi yok")
    apply_theme_template(fig)
    fig.update_layout(height=300, showlegend=False)
    return fig


def _build_activity_chart(metrics):
    buy = metrics.get("buy_count") or metrics.get("total_buys") or 0
    sell = metrics.get("sell_count") or metrics.get("total_sells") or 0
    hold = metrics.get("hold_count") or metrics.get("total_holds") or 0

    P = plot_palette()
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Al", x=["Al", "Sat", "Bekle"], y=[buy, sell, hold],
                         marker_color=[P["green"], P["red"], P["yellow"]]))
    apply_theme_template(fig)
    fig.update_layout(height=250, showlegend=False, title_text="Islem Aktivitesi")
    return fig


def _render_modal_metrics(metrics):
    # Getiri alan adi kaynaga gore degisir; tek ada normalize et (api.model_return).
    metrics = {**(metrics or {}), "total_return": api.model_return(metrics)}
    keys = [
        ("Toplam Getiri", "total_return", True),
        ("Sharpe", "sharpe_ratio", False),
        ("Sortino", "sortino_ratio", False),
        ("Calmar", "calmar_ratio", False),
        ("Max Drawdown", "max_drawdown", True),
        ("Kazanma Orani", "win_rate", True),
        ("Toplam Islem", "total_trades", False),
    ]
    rows = []
    for label, key, is_pct in keys:
        v = metrics.get(key)
        val = _fmt_pct(v) if is_pct and v is not None else _fmt2(v) if v is not None else "—"
        rows.append(dbc.Row([
            dbc.Col(html.Small(label, style={"color": TEXT_MUTED}), width=7),
            dbc.Col(html.Span(val, style={"color": TEXT, "fontWeight": "600", "fontSize": "13px"}), width=5),
        ], className="py-1 border-bottom"))
    return html.Div(rows)


def _render_trades_table(metrics):
    trades = metrics.get("trades") or metrics.get("trade_history") or []
    if not trades:
        return create_state_block("empty", "Islem gecmisi yok.")

    header = dbc.Row([
        dbc.Col(html.Small("Tarih", className="section-title"), width=3),
        dbc.Col(html.Small("Sembol", className="section-title"), width=3),
        dbc.Col(html.Small("Islem", className="section-title"), width=2),
        dbc.Col(html.Small("Fiyat", className="section-title"), width=2),
        dbc.Col(html.Small("Adet", className="section-title"), width=2),
    ])
    rows = [header]
    for t in trades[:20]:
        action = t.get("action", t.get("type", "—")).upper()
        color = GREEN if action in ("BUY", "AL") else RED if action in ("SELL", "SAT") else TEXT_MUTED
        rows.append(dbc.Row([
            dbc.Col(html.Small(str(t.get("date", "—"))[:10], style={"color": TEXT_MUTED}), width=3),
            dbc.Col(html.Small(t.get("symbol", "—"), style={"color": TEXT}), width=3),
            dbc.Col(dbc.Badge(action, style={"backgroundColor": color}, pill=True), width=2),
            dbc.Col(html.Small(f"₺{t.get('price', 0):,.2f}", style={"color": TEXT}), width=2),
            dbc.Col(html.Small(str(t.get("quantity", t.get("shares", "—"))), style={"color": TEXT}), width=2),
        ], className="py-1 border-bottom"))
    return html.Div(rows, className="mt-3")
