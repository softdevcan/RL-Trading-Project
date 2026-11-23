"""
Sanal para simulasyonu (Faz 11 S5; plan phase-11-integration-plan.md §3 S5, K-F11-1/2/6).

Dondurulmus M8 politikalari (6 kol x 12 tohum) ve kural portfoyleri ayni defterde, ayni pencere ve ayni maliyetle
(20 bps) yan yana. Varsayilan gorunum (K-F11-1): MARL iletisimsiz (12 tohum ortalamasi +- std) + 1/N + min-varyans.
Ileri veri gorunumu ARA GORUNUMDUR: tez sonucu degil; ileri test 126. seansta on kayitli kurala gore bir kez okunur
(phase-11-m8-forward-annex.md). Urun portfoyu (tam sayi lot, kullanici sermayesi) ileri teste girmez.

API: GET /api/marl/forward/status | /view | /product ; POST /api/marl/forward/run (yonetici).
Sayfa acilista durumu sorar ve is calisiyorsa yoklamayi acar (dashboard/CLAUDE.md uzun isler kurali).
"""

from dash import html, dcc, no_update
from dash import Input, Output, State
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable
import plotly.graph_objects as go

from dashboard.theme import TEXT_MUTED, empty_figure, apply_theme_template, plot_palette, plot_rgba
from dashboard.components.page_header import create_page_header
from dashboard.components.metric_card import create_metric_card
from dashboard.components.table import TABLE_STYLES, numeric_columns, tone_rules_formatted
from dashboard.auth_context import is_admin
import dashboard.api_client as api

ARM_OPTIONS = [
    {"label": "MARL iletisimsiz (12 tohum)", "value": "M-none"},
    {"label": "SARL esmimari (12 tohum)", "value": "S-esm"},
    {"label": "SARL agirlik (12 tohum)", "value": "S-agr"},
    {"label": "MARL graf dikkati (12 tohum)", "value": "M-gat"},
    {"label": "MARL kendine dikkat (12 tohum)", "value": "M-self"},
    {"label": "Bos kol, negatif kontrol (12 tohum)", "value": "null"},
    {"label": "1/N esit agirlik", "value": "rule/ew"},
    {"label": "Al-tut", "value": "rule/bh"},
    {"label": "Min-varyans (aylik)", "value": "rule/minvar"},
    {"label": "HRP (aylik)", "value": "rule/hrp"},
]
DEFAULT_KEYS = ["M-none", "rule/ew", "rule/minvar"]          # K-F11-1
PRODUCT_OPTIONS = [{"label": o["label"], "value": o["value"]} for o in ARM_OPTIONS if o["value"].startswith("rule/")] + [
    {"label": f"{arm} tohum {s}", "value": f"policy/{arm}/s{s}"}
    for arm in ("M-none", "S-esm", "S-agr", "M-gat", "M-self") for s in range(201, 213)]
LINE_COLORS = ("blue", "orange", "green", "purple", "cyan", "red", "gold", "yellow")

NOTE_FORWARD = ("Ara gorunum — tez sonucu degildir. Ileri veri testi 2026-11-23'ten itibaren TAM 126 seans dolunca on "
                "kayitli kurala gore bir kez okunur (phase-11-m8-forward-annex.md). Buradaki sayilara gore hicbir "
                "kural, kol ya da pencere degistirilmez.")
NOTE_TEST = ("2026 test penceresi (2026-01-02..10-30): M8 degerlendirmesinde zaten okunmus gecmis pencere; ayni defterle "
             "yeniden oynatildi (S3, 72/72 bit-es).")


def layout():
    status = api.get_marl_forward_status()
    running = bool(status.get("running"))
    n = int(status.get("n_sessions") or 0)
    default_window = "forward" if n > 0 else "test2026"
    admin_btn = dbc.Button([html.I(className="bi bi-play-circle me-2"), "Simdi calistir"],
                           id="sim-run-btn", color="primary", size="sm") if is_admin() else html.Span(id="sim-run-btn")
    return html.Div([
        dcc.Interval(id="sim-poll", interval=5_000, disabled=not running, n_intervals=0),
        create_page_header("Sanal Para Simulasyonu",
                           "Dondurulmus M8 politikalari ve kural portfoyleri; ayni defter, ayni pencere, 20 bps",
                           actions=admin_btn),
        html.Div(id="sim-run-msg", className="mb-2"),
        dbc.Row(id="sim-status-cards", children=_status_cards(status), className="mb-3"),
        dbc.Card([
            dbc.CardHeader(html.Span("Karsilastirma", className="card-title-sm")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Pencere", className="section-title"),
                        dbc.RadioItems(id="sim-window", inline=True, value=default_window, options=[
                            {"label": "Ileri veri (2026-11-23 →)", "value": "forward"},
                            {"label": "2026 test penceresi (gecmis)", "value": "test2026"}]),
                    ], md=5),
                    dbc.Col([
                        html.Label("Stratejiler", className="section-title"),
                        dcc.Dropdown(id="sim-keys", options=ARM_OPTIONS, value=DEFAULT_KEYS, multi=True,
                                     clearable=False),
                    ], md=7),
                ], className="mb-2"),
                html.Div(id="sim-note", className="small mb-2", style={"color": TEXT_MUTED}),
                dcc.Loading(dcc.Graph(id="sim-chart", figure=empty_figure(), config={"displayModeBar": False})),
                html.Div(id="sim-table"),
            ]),
        ], className="mb-4"),
        dbc.Card([
            dbc.CardHeader(html.Span("Urun portfoyu (tam sayi lot)", className="card-title-sm")),
            dbc.CardBody([
                html.P("Secilen stratejinin kararlari sizin sermayenizle tam sayi lot olarak uygulanir ve kesirli "
                       "defterle karsilastirilir. Urun portfoyu ileri teste girmez ve kayit tutmaz.",
                       className="small", style={"color": TEXT_MUTED}),
                dbc.Row([
                    dbc.Col(dcc.Dropdown(id="sim-product-sid", options=PRODUCT_OPTIONS, value="rule/ew",
                                         clearable=False), md=5),
                    dbc.Col(dbc.Input(id="sim-product-capital", type="number", value=100_000, min=1_000,
                                      step=1_000), md=3),
                    dbc.Col(dbc.Button("Hesapla", id="sim-product-btn", color="secondary", outline=True), md=2),
                ], className="mb-2"),
                dcc.Loading(html.Div(id="sim-product-out")),
            ]),
        ], className="mb-4"),
        dbc.Card([
            dbc.CardHeader(html.Span("Eski tek ajan modelleri", className="card-title-sm")),
            dbc.CardBody([
                html.P("Faz 1-9'un tek ajanli modelleri farkli bir ortam, gozlem ve muhasebeyle egitildi; bu "
                       "defterlerle dogrudan kiyaslanmaz.", className="small", style={"color": TEXT_MUTED}),
                dbc.Button("Modeller", href="/dash/models", color="secondary", outline=True, size="sm",
                           className="me-2"),
                dbc.Button("Gunluk karar ve kagit portfoy", href="/dash/daily-trading", color="secondary",
                           outline=True, size="sm"),
            ]),
        ]),
    ])


def _status_cards(s: dict) -> list:
    if not s or s.get("available") is False:
        return [dbc.Col(dbc.Alert("Dondurulmus M8 arsivi bu sunucuda yok (data/marl/frozen).", color="secondary"))]
    n, need = int(s.get("n_sessions") or 0), int(s.get("read_sessions") or 126)
    last = s.get("last_run") or {}
    cov = (last.get("margin_coverage") or {}).get("horizon_end") or "—"
    run_txt = "calisiyor" if s.get("running") else (last.get("status") or "—")
    return [
        dbc.Col(create_metric_card("Ileri seans", f"{n} / {need}", f"ilk gun {s.get('forward_start')}",
                                   icon="bi bi-calendar-check"), md=3),
        dbc.Col(create_metric_card("Son defter gunu", str(s.get("last_date") or "—"),
                                   f"{s.get('n_strategies') or 0} strateji", icon="bi bi-journal-text"), md=3),
        dbc.Col(create_metric_card("Son calisma", run_txt, str(last.get("finished") or ""),
                                   icon="bi bi-arrow-repeat"), md=3),
        dbc.Col(create_metric_card("Marj denetimi gecerli", cov, "aylik kaynak denetimi",
                                   icon="bi bi-shield-check"), md=3),
    ]


def _pct(x):
    return "—" if x is None else f"{x * 100:+.2f}%"


def _num(x, d=2):
    return "—" if x is None else f"{x:+.{d}f}"


def _pm(m, s, fmt):
    if m is None:
        return "—"
    return fmt(m) + ("" if s is None else f" ± {abs(s) * (100 if fmt is _pct else 1):.2f}" + ("%" if fmt is _pct else ""))


def _tl(x):
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _holdings_table(r: dict):
    """Urun portfoyunun son gun kapanisindaki lot tablosu (hisse, adet, fiyat, deger, agirlik, son gun islemi)."""
    rows_in = r.get("holdings") or []
    if not rows_in:
        return None
    rows = [{"symbol": h["symbol"], "lots": f"{h['lots']:,}".replace(",", "."), "close": _tl(h["close"]),
             "value": _tl(h["market_value"]), "w": f"{h['weight'] * 100:.2f}%",
             "wf": f"{h['weight_frac'] * 100:.2f}%",
             "trade": "—" if h["traded_lots"] == 0 else f"{h['traded_lots']:+,}".replace(",", ".")}
            for h in rows_in]
    total = sum(h["market_value"] for h in rows_in) + r["cash_integer_tl"]
    rows.append({"symbol": "Nakit", "lots": "", "close": "", "value": _tl(r["cash_integer_tl"]),
                 "w": f"{r['last_cash_integer'] * 100:.2f}%", "wf": f"{r['cash_weight_frac'] * 100:.2f}%", "trade": ""})
    rows.append({"symbol": "Toplam", "lots": "", "close": "", "value": _tl(total), "w": "100.00%", "wf": "100.00%",
                 "trade": ""})
    table = DataTable(
        columns=[{"name": "Hisse", "id": "symbol"}, {"name": "Lot", "id": "lots"},
                 {"name": "Kapanis (TL)", "id": "close"}, {"name": "Piyasa degeri (TL)", "id": "value"},
                 {"name": "Agirlik (lot)", "id": "w"}, {"name": "Kesirli agirlik", "id": "wf"},
                 {"name": "Son gun islem (lot)", "id": "trade"}],
        data=rows, sort_action="none", **TABLE_STYLES,
        style_cell_conditional=numeric_columns("lots", "close", "value", "w", "wf", "trade"),
        style_data_conditional=tone_rules_formatted("trade") + [
            {"if": {"filter_query": '{symbol} = "Toplam" || {symbol} = "Nakit"'}, "fontWeight": "600"}],
    )
    foot = html.P(f"{r['holdings_date']} kapanisi itibariyla · {r['label']} · 1 lot = 1 pay · fiyatlar duzeltilmis "
                  "kapanis (yfinance) · son gun islem: o gunun acilisinda alinan (+) / satilan (−) adet",
                  className="small mt-1 mb-3", style={"color": TEXT_MUTED})
    return html.Div([html.H6("Eldeki hisseler", className="mt-2"), table, foot])


def register_callbacks(app):
    @app.callback(
        Output("sim-chart", "figure"), Output("sim-table", "children"), Output("sim-note", "children"),
        Input("sim-window", "value"), Input("sim-keys", "value"),
    )
    def update_view(window, keys):
        note = NOTE_FORWARD if window == "forward" else NOTE_TEST
        if not keys:
            return empty_figure("Strateji secin"), None, note
        v = api.get_marl_forward_view(window, keys)
        if not v or not v.get("dates"):
            msg = ("Henuz ileri veri seansi yok (ilk gun 2026-11-23, 19:00 TRT sonrasi islenir)"
                   if window == "forward" else "Veri alinamadi")
            return empty_figure(msg), None, note
        p = plot_palette()
        fig = go.Figure()
        for i, (key, line) in enumerate(v["lines"].items()):
            c = LINE_COLORS[i % len(LINE_COLORS)]
            if line["n"] > 1:
                up = [m + s for m, s in zip(line["mean"], line["std"])]
                lo = [m - s for m, s in zip(line["mean"], line["std"])]
                fig.add_trace(go.Scatter(x=v["dates"], y=up, line={"width": 0}, showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=v["dates"], y=lo, line={"width": 0}, fill="tonexty",
                                         fillcolor=plot_rgba(c, 0.15), showlegend=False, hoverinfo="skip"))
            name = line["label"] + (f" (ort. ± std, N={line['n']})" if line["n"] > 1 else "")
            fig.add_trace(go.Scatter(x=v["dates"], y=line["mean"], name=name, line={"color": p[c], "width": 2}))
        fig.update_layout(height=380, margin={"l": 50, "r": 20, "t": 20, "b": 40}, yaxis_title="Deger (TL)",
                          legend={"orientation": "h", "y": -0.18})
        apply_theme_template(fig)
        rows = [{"label": r["label"], "n": r["n"],
                 "ret": _pm(r["cum_return"], r.get("cum_return_std"), _pct),
                 "sharpe": _pm(r["excess_sharpe"], r.get("excess_sharpe_std"), _num),
                 "dd": _pct(r["max_drawdown"]), "turn": f"{(r['turnover'] or 0) * 100:.1f}%",
                 "cash": f"{(r['cash'] or 0) * 100:.1f}%"} for r in v["table"]]
        table = DataTable(
            columns=[{"name": "Strateji", "id": "label"}, {"name": "N", "id": "n"},
                     {"name": "Toplam getiri", "id": "ret"}, {"name": "Fazla getiri Sharpe", "id": "sharpe"},
                     {"name": "En buyuk dusus", "id": "dd"}, {"name": "Gunluk devir", "id": "turn"},
                     {"name": "Nakit", "id": "cash"}],
            data=rows, **TABLE_STYLES,
            style_cell_conditional=numeric_columns("n", "ret", "sharpe", "dd", "turn", "cash"),
            style_data_conditional=tone_rules_formatted("ret"),
        )
        foot = html.P(f"{v['n_sessions']} seans · maliyet {v['cost_bps_roundtrip']:.0f} bps gidis-donus · "
                      f"baslangic {v['capital']:,.0f} TL (kesirli defterde sermaye getiriyi degistirmez) · "
                      "kollar 12 tohumun ortalamasi ± std", className="small mt-2", style={"color": TEXT_MUTED})
        return fig, html.Div([table, foot]), note

    @app.callback(
        Output("sim-product-out", "children"),
        Input("sim-product-btn", "n_clicks"),
        State("sim-product-sid", "value"), State("sim-product-capital", "value"), State("sim-window", "value"),
        prevent_initial_call=True,
    )
    def update_product(_, sid, capital, window):
        if not sid or not capital or capital <= 0:
            return dbc.Alert("Strateji ve pozitif sermaye girin.", color="warning", className="py-2 mb-0")
        r = api.get_marl_product(sid, float(capital), window)
        if not r:
            return dbc.Alert("Hesaplanamadi.", color="danger", className="py-2 mb-0")
        if not r.get("dates"):
            return html.P(r.get("note") or "Bu pencerede veri yok.", className="small")
        p = plot_palette()
        fig = go.Figure([
            go.Scatter(x=r["dates"], y=r["value_fractional"], name="Kesirli (dondurulmus defter)",
                       line={"color": p["blue"], "width": 2}),
            go.Scatter(x=r["dates"], y=r["value_integer"], name="Tam sayi lot (urun)",
                       line={"color": p["orange"], "width": 2, "dash": "dot"}),
        ])
        fig.update_layout(height=300, margin={"l": 50, "r": 20, "t": 20, "b": 40}, yaxis_title="Deger (TL)",
                          legend={"orientation": "h", "y": -0.25})
        apply_theme_template(fig)
        cards = dbc.Row([
            dbc.Col(create_metric_card("Son deger farki", _pct(r["final_diff_pct"]), "tam sayi / kesirli - 1",
                                       tone="auto"), md=3),
            dbc.Col(create_metric_card("Izleme hatasi", f"{r['tracking_error_ann'] * 100:.2f}%", "yillik"), md=3),
            dbc.Col(create_metric_card("Ort. agirlik acigi", f"{r['mean_weight_gap'] * 100:.2f}%", "yuvarlama"), md=3),
            dbc.Col(create_metric_card("Emir sayisi", str(r["n_orders"]), "tam sayi lot"), md=3),
        ], className="mb-2")
        return html.Div([cards, _holdings_table(r), dcc.Graph(figure=fig, config={"displayModeBar": False})])

    @app.callback(
        Output("sim-run-msg", "children"), Output("sim-poll", "disabled", allow_duplicate=True),
        Input("sim-run-btn", "n_clicks"), prevent_initial_call=True,
    )
    def run_now(n):
        if not n:
            return no_update, no_update
        r = api.run_marl_forward()
        if not r:
            return dbc.Alert("Is baslatilamadi.", color="danger", className="py-2 mb-0"), no_update
        msg = "Is baslatildi." if r.get("started") else "Is zaten calisiyor."
        return dbc.Alert(msg, color="info", className="py-2 mb-0"), False

    @app.callback(
        Output("sim-status-cards", "children"), Output("sim-poll", "disabled"),
        Input("sim-poll", "n_intervals"), prevent_initial_call=True,
    )
    def poll(_):
        s = api.get_marl_forward_status()
        return _status_cards(s), not bool(s.get("running"))
