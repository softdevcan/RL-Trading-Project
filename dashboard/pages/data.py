"""
Data Statistics page – data generation, dataset list, BIST-30 stocks.

API endpoints used:
  GET /api/trading/data/info
  GET /api/trading/data/list
  POST /api/trading/data/generate
"""

from datetime import date, timedelta

from dash import html, dcc
from dash import Input, Output, State
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable

from dashboard.theme import CARD, CARD2, TEXT, TEXT_MUTED, BORDER, GREEN, BLUE, empty_figure
import dashboard.api_client as api

# BIST-30 symbols reference (static)
BIST30 = [
    "AKBNK","ARCLK","ASELS","BIMAS","EKGYO","EREGL","FROTO","GARAN",
    "HALKB","ISCTR","KCHOL","KOZAL","KRDMD","MGROS","ODAS","PETKM",
    "PGSUS","SAHOL","SASA","SISE","SKBNK","TAVHL","TCELL","THYAO",
    "TKFEN","TOASO","TSKB","TUPRS","VAKBN","YKBNK",
]


# ── Layout ────────────────────────────────────────────────────────────────────

def layout():
    today = date.today()
    three_years_ago = today - timedelta(days=3 * 365)

    return html.Div([
        # Header
        html.H4("Veri Istatistikleri", style={"color": TEXT, "marginBottom": "4px"}),
        html.P("Veri setleri ve BIST-30 hisse bilgileri", style={"color": TEXT_MUTED, "marginBottom": "24px"}),

        # Controls row
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Tarih Araligi", className="section-title"),
                        dcc.DatePickerRange(
                            id="data-date-range",
                            start_date=str(three_years_ago),
                            end_date=str(today),
                            display_format="YYYY-MM-DD",
                            style={"width": "100%"},
                        ),
                    ], md=5),
                    dbc.Col([
                        html.Label("Hizli Secim", className="section-title"),
                        dbc.ButtonGroup([
                            dbc.Button("1Y", id="data-btn-1y", size="sm", outline=True, color="primary"),
                            dbc.Button("3Y", id="data-btn-3y", size="sm", outline=True, color="primary"),
                            dbc.Button("5Y", id="data-btn-5y", size="sm", outline=True, color="primary"),
                        ]),
                    ], md=3, className="d-flex flex-column justify-content-end"),
                    dbc.Col([
                        html.Label(" ", className="section-title"),
                        dbc.Button(
                            [html.I(className="bi bi-cloud-download me-2"), "Veri Uret"],
                            id="data-generate-btn",
                            color="success",
                            className="w-100",
                        ),
                    ], md=2, className="d-flex flex-column justify-content-end"),
                    dbc.Col([
                        html.Label(" ", className="section-title"),
                        dbc.Button(
                            [html.I(className="bi bi-arrow-clockwise me-2"), "Yenile"],
                            id="data-refresh-btn",
                            color="secondary",
                            outline=True,
                            className="w-100",
                        ),
                    ], md=2, className="d-flex flex-column justify-content-end"),
                ], className="align-items-end g-3"),
                html.Div(id="data-generate-result", className="mt-3"),
            ])
        ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}", "marginBottom": "24px"}),

        # Dataset info
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Veri Seti Ozeti", style={"color": TEXT, "fontWeight": "600"})),
                    dbc.CardBody(html.Div(id="data-info-panel")),
                ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
            ], md=4, className="mb-4"),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Mevcut Veri Setleri", style={"color": TEXT, "fontWeight": "600"})),
                    dbc.CardBody(html.Div(id="data-list-panel")),
                ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
            ], md=8, className="mb-4"),
        ]),

        # BIST-30 table
        dbc.Card([
            dbc.CardHeader(html.Span("BIST-30 Hisse Listesi", style={"color": TEXT, "fontWeight": "600"})),
            dbc.CardBody(_bist30_table()),
        ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        [Output("data-date-range", "start_date"), Output("data-date-range", "end_date")],
        [
            Input("data-btn-1y", "n_clicks"),
            Input("data-btn-3y", "n_clicks"),
            Input("data-btn-5y", "n_clicks"),
        ],
        prevent_initial_call=True,
    )
    def quick_select(y1, y3, y5):
        from dash import ctx
        today = date.today()
        btn = ctx.triggered_id
        if btn == "data-btn-1y":
            start = today - timedelta(days=365)
        elif btn == "data-btn-3y":
            start = today - timedelta(days=3 * 365)
        else:
            start = today - timedelta(days=5 * 365)
        return str(start), str(today)

    @app.callback(
        Output("data-generate-result", "children"),
        Input("data-generate-btn", "n_clicks"),
        [State("data-date-range", "start_date"), State("data-date-range", "end_date")],
        prevent_initial_call=True,
    )
    def generate_data(n, start, end):
        if not n:
            return ""
        result = api.generate_data({"start_date": start, "end_date": end})
        if result:
            return dbc.Alert(
                [html.I(className="bi bi-check-circle me-2"), f"Veri basariyla uretildi: {result.get('message', 'Tamam')}"],
                color="success", dismissable=True,
            )
        return dbc.Alert(
            [html.I(className="bi bi-exclamation-triangle me-2"), "Veri uretimi basarisiz."],
            color="danger", dismissable=True,
        )

    @app.callback(
        [Output("data-info-panel", "children"), Output("data-list-panel", "children")],
        Input("data-refresh-btn", "n_clicks"),
        prevent_initial_call=False,
    )
    def refresh_data(n):
        info = api.get_data_info()
        datasets = api.get_data_list()
        return _render_info(info), _render_dataset_list(datasets)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _render_info(info):
    if not info:
        return html.P("Veri bilgisi alinamadi.", style={"color": TEXT_MUTED})

    items = [
        ("Hisse Sayisi", str(info.get("symbol_count", info.get("num_symbols", "—")))),
        ("Satir Sayisi", f"{info.get('row_count', info.get('total_rows', 0)):,}" if info.get("row_count") or info.get("total_rows") else "—"),
        ("Baslangic", str(info.get("start_date", "—"))[:10]),
        ("Bitis", str(info.get("end_date", "—"))[:10]),
        ("Ozellik Sayisi", str(info.get("feature_count", info.get("num_features", "—")))),
    ]
    rows = []
    for label, value in items:
        rows.append(
            dbc.Row([
                dbc.Col(html.Small(label, style={"color": TEXT_MUTED}), width=6),
                dbc.Col(html.Span(value, style={"color": TEXT, "fontWeight": "600", "fontSize": "13px"}), width=6),
            ], className="py-1 border-bottom", style={"borderColor": CARD2 + " !important"})
        )
    return html.Div(rows)


def _render_dataset_list(datasets):
    if not datasets:
        return html.P("Kayitli veri seti yok.", style={"color": TEXT_MUTED})

    rows = []
    for ds in datasets:
        name = ds.get("name", ds.get("filename", "—"))
        size = ds.get("size", "—")
        modified = str(ds.get("modified", ds.get("created", "—")))[:10]
        rows.append(
            dbc.Row([
                dbc.Col(html.Span(name, style={"color": TEXT, "fontSize": "13px"}), width=6),
                dbc.Col(html.Small(str(size), style={"color": TEXT_MUTED}), width=3),
                dbc.Col(html.Small(modified, style={"color": TEXT_MUTED}), width=3),
            ], className="py-2 border-bottom", style={"borderColor": CARD2 + " !important"})
        )
    return html.Div([
        dbc.Row([
            dbc.Col(html.Small("Dosya", className="section-title"), width=6),
            dbc.Col(html.Small("Boyut", className="section-title"), width=3),
            dbc.Col(html.Small("Tarih", className="section-title"), width=3),
        ]),
        html.Div(rows),
    ])


def _bist30_table():
    """Build a simple 5-column grid of BIST-30 tickers."""
    cols = 5
    rows = []
    for i in range(0, len(BIST30), cols):
        chunk = BIST30[i:i + cols]
        rows.append(
            dbc.Row([
                dbc.Col(
                    dbc.Badge(sym, color="primary", pill=True, style={"fontSize": "12px"}),
                    width=12 // cols, className="mb-2"
                )
                for sym in chunk
            ])
        )
    return html.Div(rows)
