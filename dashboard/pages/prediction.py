"""
Tahmin sayfasi — iki sekme: "Model Egit" ve "Tahmin Yap".

Tasarim ilkeleri:
  - Egitim ve tahmin akisi ayri sekmelerde, kullanici akisi netlestir.
  - Gold/FX sembolleri icin veri kaynagi (borsapy/yfinance) ozellikle secilir;
    her sembol+kaynak kombinasyonu ayri bir model olarak egitilir ve listelenir.
  - Tahmin sekmesinde sadece egitilmis modeller secilebilir; veri-model uyumsuzlugu olamaz.

API endpoint'leri:
  GET  /api/prediction/symbols              -> [{symbol, name, category, sources}]
  GET  /api/prediction/models               -> [{symbol, source, horizon, name, ...metrikler}]
  POST /api/prediction/train                 (body: {symbol, horizon, source})
  POST /api/prediction/predict               (body: {targets: [{symbol, source}], horizon})
  GET  /api/prediction/predictions/{symbol}
  GET  /api/prediction/chart-data/{symbol}
  GET  /api/prediction/gold/prices, /history
"""

from typing import Any, Dict, List, Optional

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, ctx, dcc, html

from dashboard.components.page_header import create_page_header
from dashboard.components.state_block import create_state_block
import dashboard.api_client as api
from dashboard.theme import (
    BLUE, CARD2, GOLD, GREEN, RED, TEXT, TEXT_MUTED, YELLOW, apply_theme_template,
    empty_figure, plot_palette, plot_rgba,
)
from prediction.feature_groups import (
    DEFAULT_TARGET_TYPE, TARGET_TYPES, default_groups, groups_by_category,
)

HORIZONS = [
    {"label": "Gunluk (1 gun sonrasi)",  "value": "daily"},
    {"label": "Haftalik (5 gun sonrasi)", "value": "weekly"},
]

# Kategori etiket renkleri (sidebar'larda ayraç olarak kullanilir)
CATEGORY_LABEL = {
    "bist30": "BIST-30 Hisse",
    "gold":   "Altin / Sentetik",
    "fx":     "Doviz",
}


# ─── Layout ───────────────────────────────────────────────────────────────────

def layout():
    return html.Div([
        dcc.Interval(id="pred-refresh", interval=60_000, n_intervals=0),
        dcc.Store(id="pred-symbols-store", data=[]),
        dcc.Store(id="pred-models-store", data=[]),

        create_page_header("Fiyat Tahmini",
                           "Model egit, egitilmis modellerle tahmin uret."),

        # Altin ozeti — kompakt seritler
        _gold_summary_section(),

        # Iki sekmeli ana icerik
        dbc.Tabs(
            id="pred-tabs",
            active_tab="tab-train",
            className="mb-3",
            children=[
                dbc.Tab(_train_tab(),   tab_id="tab-train",   label="Model Egit"),
                dbc.Tab(_predict_tab(), tab_id="tab-predict", label="Tahmin Yap"),
            ],
        ),
    ])


def _gold_summary_section():
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.Span("Altin Fiyatlari",
                                  className="card-title-sm"),
                        md=2, className="d-flex align-items-center"),
                dbc.Col(dbc.Row(id="pred-gold-cards", className="g-2"), md=10),
            ], className="align-items-center"),
        ], style={"padding": "12px 16px"}),
    ], className="mb-3")


# ─── Tab: Model Egit ──────────────────────────────────────────────────────────

def _train_tab():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Yeni Model Egit",
                                             className="card-title-sm")),
                    dbc.CardBody([
                        html.Label("Kategori", className="section-title"),
                        dcc.Dropdown(
                            id="train-category",
                            options=[
                                {"label": "BIST-30 Hisseler", "value": "bist30"},
                                {"label": "Altin / Sentetik",  "value": "gold"},
                                {"label": "Doviz",             "value": "fx"},
                            ],
                            value="gold", clearable=False,
                            style={"marginBottom": "12px"},
                        ),

                        html.Label("Sembol", className="section-title"),
                        dcc.Dropdown(id="train-symbol", options=[], value=None,
                                     placeholder="Sembol sec...", clearable=False,
                                     style={"marginBottom": "12px"}),

                        html.Div(id="train-source-wrapper", children=[
                            html.Label("Veri Kaynagi", className="section-title"),
                            dcc.RadioItems(
                                id="train-source", options=[], value=None,
                                inputStyle={"marginRight": "6px"},
                                labelStyle={"marginRight": "16px", "color": TEXT,
                                            "fontSize": "13px"},
                                style={"marginBottom": "12px"},
                            ),
                            html.Small(
                                id="train-source-hint",
                                style={"color": TEXT_MUTED, "fontSize": "11px",
                                       "display": "block", "marginBottom": "12px"},
                            ),
                        ]),

                        html.Label("Tahmin Ufku", className="section-title"),
                        dcc.Dropdown(
                            id="train-horizon", options=HORIZONS,
                            value="daily", clearable=False,
                            style={"marginBottom": "20px"},
                        ),

                        dbc.Button(
                            [html.I(className="bi bi-cpu me-2"), "Modeli Egit"],
                            id="train-btn", color="primary", className="w-100",
                        ),
                        html.Div(id="train-result", className="mt-3"),
                    ]),
                ]),

                # Feature grup secim paneli
                _feature_groups_card(),

                # Hedef tipi
                _target_type_card(),
            ], md=4),

            dbc.Col([
                # Interaktif fiyat grafigi
                dbc.Card([
                    dbc.CardHeader(dbc.Row([
                        dbc.Col(html.Span(id="train-chart-title",
                                          children="Fiyat Grafigi",
                                          className="card-title-sm")),
                        dbc.Col(dcc.Dropdown(
                            id="train-chart-range",
                            options=[
                                {"label": "30 gun",  "value": 30},
                                {"label": "90 gun",  "value": 90},
                                {"label": "180 gun", "value": 180},
                                {"label": "1 yil",   "value": 365},
                                {"label": "3 yil",   "value": 1095},
                            ],
                            value=180, clearable=False,
                            style={"width": "120px"},
                        ), width="auto"),
                    ], className="align-items-center justify-content-between")),
                    dbc.CardBody(dcc.Graph(
                        id="train-price-chart",
                        figure=empty_figure("Sembol secince fiyat grafigi gelir"),
                        config={"displayModeBar": False},
                    )),
                ], className="mb-3"),

                dbc.Card([
                    dbc.CardHeader(html.Span("Egitilmis Modeller",
                                             className="card-title-sm")),
                    dbc.CardBody(html.Div(
                        id="train-models-table",
                        children=create_state_block("loading"),
                    )),
                ]),
            ], md=8),
        ]),
    ])


# Kategori basligi etiketleri
_CATEGORY_TITLES = {
    "technical":   "Teknik Gostergeler",
    "macro":       "Makro / Capraz Varlik",
    "alternative": "Alternatif Veri",
}
_CATEGORY_ORDER = ["technical", "macro", "alternative"]


def _feature_groups_card():
    """Feature gruplari icin kategorize checkbox paneli."""
    grouped = groups_by_category()
    sections = []
    for cat in _CATEGORY_ORDER:
        items = grouped.get(cat) or []
        if not items:
            continue
        opts = [
            {"label": meta["label"], "value": meta["id"]}
            for meta in items
        ]
        defaults = [meta["id"] for meta in items if meta["default"]]
        sections.append(html.Div([
            html.Small(_CATEGORY_TITLES.get(cat, cat),
                       className="section-title",
                       style={"color": TEXT_MUTED, "textTransform": "uppercase",
                              "fontSize": "10px", "fontWeight": "700",
                              "letterSpacing": "0.5px", "marginBottom": "6px",
                              "display": "block"}),
            dcc.Checklist(
                id={"type": "feature-group-cat", "category": cat},
                options=opts,
                value=defaults,
                inputStyle={"marginRight": "6px"},
                labelStyle={"display": "block", "color": TEXT,
                            "fontSize": "12px", "marginBottom": "4px"},
            ),
        ], style={"marginBottom": "12px"}))

    return dbc.Card([
        dbc.CardHeader(dbc.Row([
            dbc.Col(html.Span("Ozellik Gruplari",
                              className="card-title-sm")),
            dbc.Col(dbc.ButtonGroup([
                dbc.Button("Hepsi", id="features-select-all",
                           size="sm", color="secondary", outline=True),
                dbc.Button("Hicbiri", id="features-select-none",
                           size="sm", color="secondary", outline=True),
                dbc.Button("Varsayilan", id="features-select-default",
                           size="sm", color="secondary", outline=True),
            ], size="sm"), width="auto"),
        ], className="align-items-center justify-content-between")),
        dbc.CardBody(sections),
    ], className="mt-3")


def _target_type_card():
    """Hedef tipi (log_return / abs_price) ve ICEEMDAN switch'i."""
    target_opts = [
        {"label": html.Span([
            html.Span(meta["label"],
                      style={"color": TEXT, "fontWeight": "600", "fontSize": "13px"}),
            html.Br(),
            html.Small(meta["description"],
                       style={"color": TEXT_MUTED, "fontSize": "10px"}),
        ]), "value": tid}
        for tid, meta in TARGET_TYPES.items()
    ]
    return dbc.Card([
        dbc.CardHeader(html.Span("Hedef Tipi",
                                 className="card-title-sm")),
        dbc.CardBody([
            dcc.RadioItems(
                id="train-target-type",
                options=target_opts,
                value=DEFAULT_TARGET_TYPE,
                inputStyle={"marginRight": "8px"},
                labelStyle={"display": "block", "marginBottom": "10px",
                            "alignItems": "flex-start"},
            ),
        ]),
    ], className="mt-3")


# ─── Tab: Tahmin Yap ──────────────────────────────────────────────────────────

def _predict_tab():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Tahmin Ayarlari",
                                             className="card-title-sm")),
                    dbc.CardBody([
                        html.Label("Egitilmis Model", className="section-title"),
                        dcc.Dropdown(
                            id="predict-model", options=[], value=None,
                            placeholder="Egitilmis bir model sec...",
                            clearable=False, style={"marginBottom": "16px"},
                        ),
                        html.Small(
                            "Yalnizca egitilmis modeller listelenir. "
                            "Veri ayni kaynaktan cekilir, tahmin tutarliligi garantidir.",
                            style={"color": TEXT_MUTED, "fontSize": "11px",
                                   "display": "block", "marginBottom": "16px"},
                        ),

                        dbc.Button(
                            [html.I(className="bi bi-lightning me-2"), "Tahmin Yap"],
                            id="predict-btn", color="success", className="w-100 mb-2",
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-graph-up me-2"), "Bekleyenleri Degerlendir"],
                            id="predict-evaluate-btn", color="info",
                            outline=True, className="w-100",
                        ),
                        html.Div(id="predict-action-result", className="mt-3"),
                    ]),
                ]),

                dbc.Card([
                    dbc.CardHeader(html.Span("Altin Fiyat Gecmisi",
                                             className="card-title-sm")),
                    dbc.CardBody(dcc.Graph(id="pred-gold-chart", figure=empty_figure(),
                                           config={"displayModeBar": False})),
                ], className="mt-3"),
            ], md=4),

            dbc.Col([
                html.Div(id="predict-result-card", className="mb-3"),
                html.Div(id="predict-performance-section", className="mb-3"),

                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader(html.Span("Tahmin vs Gercek",
                                                 className="card-title-sm")),
                        dbc.CardBody(dcc.Graph(id="predict-vs-actual-chart",
                                               figure=empty_figure(),
                                               config={"displayModeBar": False})),
                    ]),
                            md=7, className="mb-3"),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader(html.Span("Dogruluk Trendi",
                                                 className="card-title-sm")),
                        dbc.CardBody(dcc.Graph(id="predict-accuracy-chart",
                                               figure=empty_figure(),
                                               config={"displayModeBar": False})),
                    ]),
                            md=5, className="mb-3"),
                ]),

                dbc.Card([
                    dbc.CardHeader(html.Span("Tahmin Gecmisi",
                                             className="card-title-sm")),
                    dbc.CardBody(html.Div(
                        id="predict-history-table",
                        children=create_state_block("empty", "Henuz tahmin yok."),
                    )),
                ]),
            ], md=8),
        ]),
    ])


# ─── Callbacks ────────────────────────────────────────────────────────────────

def register_callbacks(app):

    # Periyodik veri tazeleme: gold + sembol listesi + egitilmis modeller
    @app.callback(
        [Output("pred-gold-cards", "children"),
         Output("pred-gold-chart", "figure"),
         Output("pred-symbols-store", "data"),
         Output("pred-models-store", "data")],
        Input("pred-refresh", "n_intervals"),
        prevent_initial_call=False,
    )
    def refresh_data(_n):
        try:
            cards = _render_gold_cards(api.get_gold_prices() or {})
        except Exception:
            cards = [dbc.Col(html.P("Altin verisi alinamadi.", style={"color": TEXT_MUTED}))]
        try:
            gold_fig = _build_gold_chart(api.get_gold_history())
        except Exception:
            gold_fig = empty_figure("Veri alinamadi")

        symbols = api.get_prediction_symbols() or []
        models = api.get_prediction_models() or []
        return cards, gold_fig, symbols, models

    # ── Egitim sekmesi: kategori secimine gore sembol listesi
    @app.callback(
        [Output("train-symbol", "options"),
         Output("train-symbol", "value")],
        [Input("train-category", "value"),
         Input("pred-symbols-store", "data")],
    )
    def filter_symbols(category, symbols):
        symbols = symbols or []
        filtered = [s for s in symbols if s.get("category") == category]
        opts = [{"label": f"{s.get('name', s['symbol'])} ({s['symbol']})",
                 "value": s["symbol"]} for s in filtered]
        value = opts[0]["value"] if opts else None
        return opts, value

    # ── Egitim sekmesi: sembol degisince source secenekleri
    @app.callback(
        [Output("train-source", "options"),
         Output("train-source", "value"),
         Output("train-source-wrapper", "style"),
         Output("train-source-hint", "children")],
        [Input("train-symbol", "value"),
         Input("pred-symbols-store", "data")],
    )
    def update_sources(symbol, symbols):
        if not symbol or not symbols:
            return [], None, {"display": "none"}, ""

        entry = next((s for s in symbols if s["symbol"] == symbol), None)
        sources = (entry or {}).get("sources") or []

        # Tek source varsa (hisse senedi: yfinance) UI'da gostermeye gerek yok
        if len(sources) <= 1:
            return [], (sources[0] if sources else None), {"display": "none"}, ""

        opts = [{"label": s.upper(), "value": s} for s in sources]
        hint = (
            "borsapy: TR yerel kaynak (gunluk guncel). "
            "yfinance: GC=F * USDTRY hesaplanmis seri (uzun gecmis)."
        )
        return opts, sources[0], {"display": "block"}, hint

    # ── Egitim sekmesi: butonla egit, durumu goster
    @app.callback(
        [Output("train-result", "children"),
         Output("pred-models-store", "data", allow_duplicate=True)],
        Input("train-btn", "n_clicks"),
        [State("train-symbol", "value"),
         State("train-horizon", "value"),
         State("train-source", "value"),
         State({"type": "feature-group-cat", "category": "technical"}, "value"),
         State({"type": "feature-group-cat", "category": "macro"}, "value"),
         State({"type": "feature-group-cat", "category": "alternative"}, "value"),
         State("train-target-type", "value")],
        prevent_initial_call=True,
    )
    def start_training(n, symbol, horizon, source,
                       tech_groups, macro_groups, alt_groups, target_type):
        models = api.get_prediction_models() or []
        if not n or not symbol:
            return "", models

        payload: Dict[str, Any] = {"symbol": symbol, "horizon": horizon or "daily"}
        if source:
            payload["source"] = source

        # Feature config: kategorilerden secilen tum gruplar
        enabled = list(set((tech_groups or []) + (macro_groups or []) + (alt_groups or [])))
        # iceemdan checklist'inde varsa use_iceemdan flag'i true (ayni grup id)
        use_iceemdan = "iceemdan" in enabled
        payload["feature_config"] = {
            "enabled_groups": enabled if enabled else None,
            "target_type": target_type or DEFAULT_TARGET_TYPE,
            "use_iceemdan": use_iceemdan,
        }

        result = api.train_prediction(payload)
        if not result or result.get("detail"):
            err = result.get("detail", "Bilinmeyen hata") if result else "Sunucu yanit vermedi"
            return (dbc.Alert([html.I(className="bi bi-x-circle me-2"),
                               f"Egitim baslatilamadi: {err}"],
                              color="danger", dismissable=True), models)

        already_running = "zaten" in (result.get("message") or "")
        icon = "bi-hourglass-split" if already_running else "bi-play-circle"
        title = "Egitim zaten surmekte" if already_running else "Egitim arka planda baslatildi"
        src_label = f" | Kaynak: {result.get('source') or '—'}" if result.get('source') else ""
        body = [
            html.Div([html.I(className=f"bi {icon} me-2"), html.Strong(title)],
                     className="mb-2"),
            html.Small(
                f"Sembol: {result.get('symbol', symbol)} | "
                f"Horizon: {result.get('horizon', horizon)}{src_label} | "
                f"Durum: {result.get('state', 'running')}",
                style={"color": TEXT_MUTED},
            ),
            html.Br(),
            html.Small(
                "Bittiginde sag tarafta listede gorunur (otomatik yenilenir ~60 sn).",
                style={"color": TEXT_MUTED},
            ),
        ]
        return dbc.Alert(body, color="info", dismissable=True), models

    # ── Egitim sekmesi: feature secim hizli butonlari (Hepsi/Hicbiri/Varsayilan)
    @app.callback(
        Output({"type": "feature-group-cat", "category": ALL}, "value"),
        [Input("features-select-all", "n_clicks"),
         Input("features-select-none", "n_clicks"),
         Input("features-select-default", "n_clicks")],
        State({"type": "feature-group-cat", "category": ALL}, "id"),
        prevent_initial_call=True,
    )
    def feature_quick_select(_a, _b, _c, ids):
        trig = ctx.triggered_id
        result = []
        defaults = set(default_groups())
        for cid in ids:
            cat = cid["category"]
            cat_groups = [meta["id"] for meta in groups_by_category().get(cat, [])]
            if trig == "features-select-all":
                result.append(cat_groups)
            elif trig == "features-select-none":
                result.append([])
            else:  # default
                result.append([g for g in cat_groups if g in defaults])
        return result

    # ── Egitim sekmesi: sembol/kaynak/range -> interaktif fiyat grafigi
    @app.callback(
        [Output("train-price-chart", "figure"),
         Output("train-chart-title", "children")],
        [Input("train-symbol", "value"),
         Input("train-source", "value"),
         Input("train-chart-range", "value")],
    )
    def update_price_chart(symbol, source, days):
        if not symbol:
            return empty_figure("Sembol secince fiyat grafigi gelir"), "Fiyat Grafigi"
        try:
            data = api.get_price_history(symbol, source=source, days=days or 180) or {}
        except Exception as exc:
            return empty_figure(f"Veri alinamadi: {exc}"), f"{symbol}"
        if not data or not data.get("dates"):
            return empty_figure("Fiyat verisi bulunamadi"), f"{symbol}"
        title = _format_chart_title(symbol, source, data.get("source"))
        return _build_price_chart(data), title

    # ── Egitim sekmesi: egitilmis modeller tablosu
    @app.callback(
        Output("train-models-table", "children"),
        Input("pred-models-store", "data"),
    )
    def render_models_table(models):
        return _render_models_table(models or [])

    # ── Tahmin sekmesi: model dropdown'i
    @app.callback(
        [Output("predict-model", "options"),
         Output("predict-model", "value")],
        Input("pred-models-store", "data"),
        State("predict-model", "value"),
    )
    def fill_predict_models(models, current):
        models = models or []
        opts = [{"label": _format_model_label(m),
                 "value": _model_value(m)} for m in models]
        # Mevcut secim hala gecerliyse koru
        valid = {o["value"] for o in opts}
        value = current if current in valid else (opts[0]["value"] if opts else None)
        return opts, value

    # ── Tahmin sekmesi: tahmin yap
    @app.callback(
        [Output("predict-result-card", "children"),
         Output("predict-vs-actual-chart", "figure"),
         Output("predict-accuracy-chart", "figure"),
         Output("predict-history-table", "children"),
         Output("predict-performance-section", "children"),
         Output("predict-action-result", "children")],
        Input("predict-btn", "n_clicks"),
        [State("predict-model", "value"),
         State("pred-models-store", "data")],
        prevent_initial_call=True,
    )
    def make_prediction(n, model_value, models):
        blank = (html.Div(), empty_figure(), empty_figure(),
                 create_state_block("empty", "Henuz tahmin yok."),
                 html.Div(), "")
        if not n or not model_value:
            return blank

        symbol, source, horizon = _parse_model_value(model_value)
        target = {"symbol": symbol}
        if source:
            target["source"] = source
        payload = {"targets": [target], "horizon": horizon or "daily"}

        resp = api.make_prediction(payload)
        predictions = resp.get("predictions", []) if isinstance(resp, dict) else []
        if not predictions:
            err = (resp.get("detail") if isinstance(resp, dict) else None) or "Tahmin alinamadi."
            alert = dbc.Alert([html.I(className="bi bi-x-circle me-2"), err],
                              color="danger", dismissable=True)
            return (html.Div(), empty_figure(), empty_figure(),
                    create_state_block("empty", "Henuz tahmin yok."),
                    html.Div(), alert)

        result = predictions[0]
        meta = next((m for m in (models or [])
                     if m.get("symbol") == symbol and m.get("source") == source
                     and m.get("horizon") == horizon), None)

        chart_data = api.get_prediction_chart_data(symbol) or {}
        perf = api.get_prediction_performance(symbol) or {}
        hist = api.get_prediction_history(symbol) or {}

        return (
            _render_result_card(result, symbol, source, meta),
            _build_vs_actual_chart(chart_data),
            _build_accuracy_chart(chart_data),
            _render_history_table(hist),
            _render_performance(perf),
            "",
        )

    # ── Tahmin sekmesi: bekleyenleri degerlendir
    @app.callback(
        Output("predict-action-result", "children", allow_duplicate=True),
        Input("predict-evaluate-btn", "n_clicks"),
        State("predict-model", "value"),
        prevent_initial_call=True,
    )
    def evaluate_pending(n, model_value):
        if not n:
            return ""
        symbols = None
        if model_value:
            symbol, _src, _h = _parse_model_value(model_value)
            symbols = [symbol]
        result = api.evaluate_prediction({"symbols": symbols}) if symbols else api.evaluate_prediction({})
        msg = (result or {}).get("message", "Degerlendirme tamamlandi")
        return dbc.Alert([html.I(className="bi bi-check2-circle me-2"), msg],
                         color="success", dismissable=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _model_value(m: Dict) -> str:
    """Dropdown value: 'symbol|source|horizon' (source bos olabilir)."""
    return f"{m.get('symbol', '')}|{m.get('source') or ''}|{m.get('horizon', 'daily')}"


def _parse_model_value(v: str):
    parts = (v or "").split("|")
    sym = parts[0] if len(parts) > 0 else ""
    src = (parts[1] or None) if len(parts) > 1 else None
    hor = parts[2] if len(parts) > 2 else "daily"
    return sym, src, hor


def _format_model_label(m: Dict) -> str:
    sym = m.get("symbol", "?")
    name = m.get("name") or sym
    src = m.get("source")
    hor = m.get("horizon", "daily")
    hor_label = "Gunluk" if hor == "daily" else "Haftalik"
    src_label = f" · {src}" if src else ""
    metrics = []
    if m.get("test_mape") is not None:
        metrics.append(f"MAPE {m['test_mape']:.1f}%")
    if m.get("test_direction_accuracy") is not None:
        metrics.append(f"Yon {m['test_direction_accuracy']:.0f}%")
    metric_str = f" — {' / '.join(metrics)}" if metrics else ""
    return f"{name} ({sym}){src_label} · {hor_label}{metric_str}"


def _render_models_table(models: List[Dict]):
    if not models:
        return html.P(
            "Henuz egitilmis model yok. Sol panelden bir sembol ve kaynak secip "
            "'Modeli Egit' butonuna basin.",
            style={"color": TEXT_MUTED, "fontSize": "13px"},
        )

    header = dbc.Row([
        dbc.Col(html.Small("Sembol", className="section-title"), width=3),
        dbc.Col(html.Small("Kaynak", className="section-title"), width=2),
        dbc.Col(html.Small("Ufuk",   className="section-title"), width=2),
        dbc.Col(html.Small("MAPE",   className="section-title"), width=1),
        dbc.Col(html.Small("Yon",    className="section-title"), width=1),
        dbc.Col(html.Small("Ozellik", className="section-title"), width=1),
        dbc.Col(html.Small("Egitildi", className="section-title"), width=2),
    ], className="border-bottom pb-2 mb-2")

    rows = [header]
    for m in sorted(models, key=lambda x: (x.get("symbol", ""),
                                            x.get("source") or "",
                                            x.get("horizon", ""))):
        sym = m.get("symbol", "?")
        name = m.get("name") or sym
        src = m.get("source") or "—"
        hor = m.get("horizon", "?")
        mape = m.get("test_mape")
        diracc = m.get("test_direction_accuracy")
        n_feat = m.get("n_features", 0)
        saved = (m.get("saved_at") or "—")[:10]

        mape_color = (GREEN if (mape is not None and mape < 3)
                      else YELLOW if (mape is not None and mape < 7)
                      else RED if mape is not None else TEXT_MUTED)
        dir_color = GREEN if (diracc is not None and diracc >= 55) else RED if diracc is not None else TEXT_MUTED
        h_badge = dbc.Badge("Gunluk" if hor == "daily" else "Haftalik",
                            color="primary" if hor == "daily" else "info", pill=True)
        src_badge = dbc.Badge(src, color="secondary", pill=True) if src != "—" else html.Small("—", style={"color": TEXT_MUTED})

        rows.append(dbc.Row([
            dbc.Col([
                html.Div(name, style={"color": TEXT, "fontWeight": "600", "fontSize": "13px"}),
                html.Small(sym, style={"color": TEXT_MUTED, "fontSize": "11px"}),
            ], width=3),
            dbc.Col(src_badge, width=2, className="d-flex align-items-center"),
            dbc.Col(h_badge, width=2, className="d-flex align-items-center"),
            dbc.Col(html.Small(f"{mape:.2f}%" if mape is not None else "—",
                               style={"color": mape_color, "fontWeight": "600"}), width=1),
            dbc.Col(html.Small(f"{diracc:.0f}%" if diracc is not None else "—",
                               style={"color": dir_color, "fontWeight": "600"}), width=1),
            dbc.Col(html.Small(str(n_feat), style={"color": TEXT_MUTED}), width=1),
            dbc.Col(html.Small(saved, style={"color": TEXT_MUTED, "fontSize": "11px"}), width=2),
        ], className="py-2 border-bottom align-items-center"))

    return html.Div([
        html.P(f"{len(models)} egitilmis model",
               style={"color": TEXT_MUTED, "fontSize": "12px", "marginBottom": "8px"}),
        html.Div(rows),
    ])


def _render_gold_cards(gold):
    if not gold:
        return [dbc.Col(create_state_block("empty", "Altin verisi yok."))]

    items = [
        ("TRY/gram", gold.get("try_per_gram", gold.get("price_try", "—")), GOLD),
        ("USD/oz",   gold.get("usd_per_oz",   gold.get("price_usd", "—")), GREEN),
        ("Degisim %", gold.get("change_pct",   gold.get("daily_change", "—")), BLUE),
    ]
    cols = []
    for label, val, color in items:
        if isinstance(val, float):
            val = f"{val:,.2f}"
        cols.append(dbc.Col(html.Div([
            html.Small(label, style={"color": TEXT_MUTED, "fontSize": "10px",
                                      "textTransform": "uppercase"}),
            html.Div(str(val), style={"color": color, "fontWeight": "700",
                                       "fontSize": "18px"}),
        ], style={"backgroundColor": CARD2, "padding": "8px 12px",
                  "borderRadius": "6px"}), width="auto"))
    return cols


def _format_chart_title(symbol: str, requested_source: Optional[str],
                        resolved_source: Optional[str]) -> str:
    src = requested_source or resolved_source
    if src:
        return f"{symbol}  ·  Kaynak: {src}"
    return f"{symbol}"


def _build_price_chart(data: Dict) -> go.Figure:
    """OHLC + MA20/MA50 + Bollinger Bands + Volume (subplot)."""
    from plotly.subplots import make_subplots

    dates = data.get("dates") or []
    if not dates:
        return empty_figure("Veri yok")

    o = data.get("open") or []
    h = data.get("high") or []
    l = data.get("low") or []
    c = data.get("close") or []
    v = data.get("volume") or []
    ma20 = data.get("ma20") or []
    ma50 = data.get("ma50") or []
    bbu = data.get("bb_upper") or []
    bbl = data.get("bb_lower") or []

    P = plot_palette()
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.04,
    )

    # Bollinger band (alt+ust dolgu)
    if bbu and bbl:
        fig.add_trace(go.Scatter(
            x=dates, y=bbu, mode="lines", name="BB ust",
            line={"color": "rgba(148,163,184,0.0)"}, showlegend=False,
            hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=bbl, mode="lines", name="BB alt",
            line={"color": "rgba(148,163,184,0.0)"}, fill="tonexty",
            fillcolor=plot_rgba("purple", 0.10), showlegend=False,
            hoverinfo="skip",
        ), row=1, col=1)

    # Mum
    if o and h and l and c:
        fig.add_trace(go.Candlestick(
            x=dates, open=o, high=h, low=l, close=c,
            name="OHLC",
            increasing={"line": {"color": P["green"]}, "fillcolor": P["green"]},
            decreasing={"line": {"color": P["red"]}, "fillcolor": P["red"]},
        ), row=1, col=1)
    elif c:
        fig.add_trace(go.Scatter(
            x=dates, y=c, mode="lines", name="Kapanis",
            line={"color": P["blue"], "width": 2},
        ), row=1, col=1)

    # MA'lar
    if ma20:
        fig.add_trace(go.Scatter(
            x=dates, y=ma20, mode="lines", name="MA20",
            line={"color": P["yellow"], "width": 1.2},
        ), row=1, col=1)
    if ma50:
        fig.add_trace(go.Scatter(
            x=dates, y=ma50, mode="lines", name="MA50",
            line={"color": P["orange"], "width": 1.2},
        ), row=1, col=1)

    # Volume bar
    if v and any(x for x in v if x):
        colors = []
        for i in range(len(v)):
            if i == 0 or not c or c[i] is None or c[i - 1] is None:
                colors.append(P["muted"])
            else:
                colors.append(P["green"] if c[i] >= c[i - 1] else P["red"])
        fig.add_trace(go.Bar(
            x=dates, y=v, name="Hacim",
            marker={"color": colors, "opacity": 0.6},
            showlegend=False,
        ), row=2, col=1)

    apply_theme_template(fig)
    fig.update_layout(
        height=460,
        margin={"l": 50, "r": 20, "t": 10, "b": 40},
        legend={"orientation": "h", "y": 1.05, "x": 0,
                "bgcolor": "rgba(0,0,0,0)"},
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Fiyat", row=1, col=1)
    fig.update_yaxes(title_text="Hacim", row=2, col=1)
    return fig


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
            fill="tozeroy", fillcolor=plot_rgba("gold", 0.15),
            line={"color": plot_palette()["gold"], "width": 2}, name="Altin (TRY/gr)",
            hovertemplate="<b>%{x}</b><br>₺%{y:,.2f}<extra></extra>",
        ))
    else:
        return empty_figure("Altin gecmisi yok")

    apply_theme_template(fig)
    fig.update_layout(showlegend=False, height=200,
                      margin={"l": 50, "r": 10, "t": 10, "b": 40})
    return fig


def _render_result_card(result, symbol, source, meta):
    price      = result.get("predicted_close", 0) or 0
    current    = result.get("current_close", 0) or 0
    change_pct = result.get("predicted_change_pct", 0) or 0
    direction  = (result.get("predicted_direction", "NEUTRAL") or "NEUTRAL").upper()
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

    quality_parts = []
    if meta:
        mape = meta.get("test_mape")
        diracc = meta.get("test_direction_accuracy")
        if mape is not None:
            q_color = GREEN if mape < 3 else (YELLOW if mape < 7 else RED)
            quality_parts.append(html.Span(f"MAPE {mape:.1f}%",
                                            style={"color": q_color,
                                                   "marginRight": "10px",
                                                   "fontSize": "12px",
                                                   "fontWeight": "600"}))
        if diracc is not None:
            q_color = GREEN if diracc >= 55 else RED
            quality_parts.append(html.Span(f"Yon Acc {diracc:.0f}%",
                                            style={"color": q_color,
                                                   "fontSize": "12px",
                                                   "fontWeight": "600"}))

    src_str = f" · Kaynak: {source}" if source else ""

    return dbc.Card([
        dbc.CardHeader(
            dbc.Row([
                dbc.Col(html.Span(f"{symbol}{src_str}  ·  {pred_date} tahmini ({horizon_label})",
                                  style={"color": TEXT_MUTED, "fontSize": "13px"})),
                dbc.Col(html.Span(quality_parts), width="auto",
                        className="ms-auto d-flex align-items-center"),
            ], className="align-items-center"),
        ),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Small("Mevcut Fiyat", style={"color": TEXT_MUTED}),
                    html.H5(f"{currency_symbol}{current:,.4f}", style={"color": TEXT}),
                    html.Small("Tahmini Fiyat",
                               style={"color": TEXT_MUTED, "marginTop": "8px",
                                      "display": "block"}),
                    html.H3(f"{currency_symbol}{price:,.4f}",
                            style={"color": TEXT, "fontWeight": "700"}),
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
                        html.Span(direction, style={"color": dir_color,
                                                     "fontWeight": "700",
                                                     "fontSize": "18px"}),
                    ], className="d-flex align-items-center mb-3"),
                    html.Small("Model Guveni", style={"color": TEXT_MUTED}),
                    dbc.Progress(
                        value=conf_pct, label=f"{conf_pct}%",
                        color="success" if conf_pct >= 70
                              else "warning" if conf_pct >= 50 else "danger",
                        style={"height": "12px", "marginTop": "4px"},
                    ),
                    *(_render_ensemble_agreement(result)
                      if result.get("ensemble_agreement") is not None else []),
                ], md=7),
            ]),
            *(_render_model_predictions(result) if result.get("model_predictions") else []),
        ])
    ])


def _render_ensemble_agreement(result):
    agreement = result.get("ensemble_agreement", 0)
    n_models = result.get("n_models", 0)
    pct = int(agreement * 100) if agreement <= 1 else int(agreement)
    return [
        html.Hr(style={"borderColor": "rgba(255,255,255,0.1)", "margin": "10px 0"}),
        html.Small("Ensemble Uzlasma", style={"color": TEXT_MUTED}),
        dbc.Progress(
            value=pct, label=f"{pct}% ({n_models} model)",
            color="success" if pct >= 80 else "warning" if pct >= 60 else "danger",
            style={"height": "10px", "marginTop": "4px"},
        ),
    ]


def _render_model_predictions(result):
    model_preds = result.get("model_predictions", {})
    if not model_preds:
        return []

    current = result.get("current_close", 0)
    items = []
    for model_name, pred_price in sorted(model_preds.items()):
        change = (pred_price - current) / current * 100 if current > 0 else 0
        dir_color = GREEN if change > 0 else RED
        items.append(dbc.Col(html.Div([
            html.Small(model_name.upper(), style={"color": TEXT_MUTED, "fontSize": "10px"}),
            html.Div(f"{pred_price:,.2f}", style={"color": TEXT, "fontWeight": "600",
                                                    "fontSize": "13px"}),
            html.Small(f"{'▲' if change >= 0 else '▼'}{abs(change):.1f}%",
                       style={"color": dir_color, "fontSize": "10px"}),
        ], style={"textAlign": "center"}), width=True))

    return [
        html.Hr(style={"borderColor": "rgba(255,255,255,0.1)", "margin": "12px 0"}),
        html.Small("Model Tahminleri",
                   style={"color": TEXT_MUTED, "display": "block", "marginBottom": "8px"}),
        dbc.Row(items, className="g-2"),
    ]


def _build_vs_actual_chart(chart_data):
    P = plot_palette()
    fig = go.Figure()
    actual = chart_data.get("actual", [])
    predicted = chart_data.get("predicted", [])
    dates = chart_data.get("dates", list(range(max(len(actual), len(predicted)))))

    if actual:
        fig.add_trace(go.Scatter(x=dates[:len(actual)], y=actual, mode="lines",
                                  name="Gercek", line={"color": P["muted"], "width": 1.5}))
    if predicted:
        fig.add_trace(go.Scatter(x=dates[:len(predicted)], y=predicted, mode="lines",
                                  name="Tahmin", line={"color": P["blue"], "width": 2,
                                                        "dash": "dot"}))
    if not actual and not predicted:
        return empty_figure("Veri yok")

    apply_theme_template(fig)
    fig.update_layout(height=280, legend={"orientation": "h", "y": 1.1})
    return fig


def _build_accuracy_chart(chart_data):
    acc = chart_data.get("accuracy_history", chart_data.get("accuracy", []))
    mae = chart_data.get("mae_history", chart_data.get("mae", []))

    if not acc and not mae:
        return empty_figure("Dogruluk verisi yok")

    P = plot_palette()
    fig = go.Figure()
    if acc:
        fig.add_trace(go.Scatter(y=acc, mode="lines+markers", name="Dogruluk",
                                  line={"color": P["green"]}, yaxis="y1"))
    if mae:
        fig.add_trace(go.Scatter(y=mae, mode="lines+markers", name="MAE",
                                  line={"color": P["red"], "dash": "dot"}, yaxis="y2"))

    apply_theme_template(fig)
    fig.update_layout(height=280, yaxis={"title": "Dogruluk"},
                      yaxis2={"title": "MAE", "overlaying": "y", "side": "right"},
                      legend={"orientation": "h", "y": 1.1})
    return fig


def _render_performance(perf):
    if not perf:
        return html.Div()
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
        items.append(dbc.Col(html.Div([
            html.Small(label, style={"color": TEXT_MUTED, "fontSize": "11px",
                                      "textTransform": "uppercase"}),
            html.Div(fmt, style={"color": TEXT, "fontWeight": "600", "fontSize": "14px"}),
        ], style={"backgroundColor": CARD2, "padding": "8px 12px",
                  "borderRadius": "6px"}), md=True))

    if not items:
        return html.Div()
    return dbc.Card([
        dbc.CardHeader(html.Span("Performans Metrikleri",
                                 className="card-title-sm")),
        dbc.CardBody(dbc.Row(items, className="g-2")),
    ])


def _render_history_table(hist):
    records = hist.get("history", hist.get("predictions", [])) if isinstance(hist, dict) else []
    if not records:
        return create_state_block("empty", "Tahmin gecmisi yok.")

    header = dbc.Row([
        dbc.Col(html.Small("Tarih", className="section-title"), width=3),
        dbc.Col(html.Small("Tahmin", className="section-title"), width=3),
        dbc.Col(html.Small("Gercek", className="section-title"), width=3),
        dbc.Col(html.Small("Hata %", className="section-title"), width=3),
    ], className="border-bottom pb-2 mb-2")

    rows = [header]
    for r in records[:15]:
        pred = r.get("predicted_close", r.get("predicted", 0)) or 0
        actual = r.get("actual_close", r.get("actual", 0)) or 0
        err = abs(pred - actual) / actual * 100 if actual else 0
        rows.append(dbc.Row([
            dbc.Col(html.Small(str(r.get("prediction_date", r.get("date", "—")))[:10],
                               style={"color": TEXT_MUTED}), width=3),
            dbc.Col(html.Small(f"₺{pred:,.2f}", style={"color": BLUE}), width=3),
            dbc.Col(html.Small(f"₺{actual:,.2f}" if actual else "—",
                               style={"color": TEXT}), width=3),
            dbc.Col(html.Small(f"{err:.1f}%" if actual else "—",
                               style={"color": RED if err > 5 else GREEN}), width=3),
        ], className="py-1 border-bottom"))
    return html.Div(rows)
