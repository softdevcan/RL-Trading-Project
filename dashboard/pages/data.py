"""
Veri Yönetimi sayfası.

Veri kaynaklarını seçip incremental veya tam güncelleme yapılır.
Her kaynak için son tarih ve eksik gün sayısı gösterilir.
BIST güncellemesinde hangi hisselerin indirileceği seçilebilir.

API endpoint'leri:
  GET  /api/trading/data/status
  POST /api/trading/data/update
  GET  /api/trading/data/list
  POST /api/trading/data/generate  (legacy)
"""

from datetime import date, timedelta

from dash import html, dcc, ctx
from dash import Input, Output, State
import dash_bootstrap_components as dbc

from dashboard.theme import CARD, CARD2, TEXT, TEXT_MUTED
import dashboard.api_client as api

# BIST-30 sembolleri ve kısa adlar
BIST30_SYMBOLS = [
    ("AKBNK.IS", "Akbank"),
    ("ARCLK.IS", "Arçelik"),
    ("ASELS.IS", "Aselsan"),
    ("BIMAS.IS", "BIM"),
    ("EKGYO.IS", "Emlak GYO"),
    ("ENKAI.IS", "Enka İnşaat"),
    ("EREGL.IS", "Ereğli"),
    ("FROTO.IS", "Ford Otosan"),
    ("GARAN.IS", "Garanti"),
    ("HALKB.IS", "Halkbank"),
    ("ISCTR.IS", "İş Bankası"),
    ("KCHOL.IS", "Koç Holding"),
    ("KOZAL.IS", "Koza Altın"),
    ("KRDMD.IS", "Kardemir"),
    ("MGROS.IS", "Migros"),
    ("ODAS.IS",  "Odaş Elektrik"),
    ("PETKM.IS", "Petkim"),
    ("PGSUS.IS", "Pegasus"),
    ("SAHOL.IS", "Sabancı"),
    ("SASA.IS",  "Sasa Polyester"),
    ("SISE.IS",  "Şişe Cam"),
    ("SKBNK.IS", "Şekerbank"),
    ("TAVHL.IS", "TAV Havalimanı"),
    ("TCELL.IS", "Turkcell"),
    ("THYAO.IS", "Türk Hava Yolları"),
    ("TKFEN.IS", "Tekfen"),
    ("TOASO.IS", "Tofaş"),
    ("TSKB.IS",  "TSKB"),
    ("TUPRS.IS", "Tüpraş"),
    ("VAKBN.IS", "Vakıfbank"),
    ("YKBNK.IS", "Yapı Kredi"),
]

PHASE1_SYMBOLS = ["AKBNK.IS", "THYAO.IS", "TUPRS.IS", "BIMAS.IS", "ASELS.IS"]
ALL_BIST_VALUES = [s for s, _ in BIST30_SYMBOLS]

ALL_SOURCES = [
    {"value": "bist_stocks", "label": "BIST-30 Hisseleri"},
    {"value": "gold",        "label": "Altın & Döviz (GC=F · USD/TRY · Gram Altın)"},
    {"value": "macro",       "label": "Makro Veri (EUR/TRY · BIST-100 · Enflasyon)"},
]

SOURCE_VALUES = [s["value"] for s in ALL_SOURCES]


# ── Layout ─────────────────────────────────────────────────────────────────────

def layout():
    today = date.today()
    five_years_ago = today - timedelta(days=5 * 365)

    return html.Div([
        html.H4("Veri Yönetimi", style={"color": TEXT, "marginBottom": "4px"}),
        html.P(
            "Veri kaynaklarını seçin, eksik günleri güncelleyin veya tümünü yeniden indirin.",
            style={"color": TEXT_MUTED, "marginBottom": "24px"},
        ),

        dcc.Store(id="data-status-store"),

        # ── Kaynak seçim + durum kartı ─────────────────────────────────────
        dbc.Card([
            dbc.CardHeader(
                dbc.Row([
                    dbc.Col(
                        html.Span(
                            "Veri Kaynakları",
                            style={"color": TEXT, "fontWeight": "600"},
                        ),
                        className="d-flex align-items-center",
                    ),
                    dbc.Col(
                        dbc.Button(
                            [html.I(className="bi bi-arrow-clockwise me-1"), "Durumu Yenile"],
                            id="data-status-refresh-btn",
                            size="sm",
                            color="secondary",
                            outline=True,
                        ),
                        width="auto",
                    ),
                ], className="align-items-center"),
            ),
            dbc.CardBody([
                dbc.Row([
                    # Sol: kaynak toggle'ları
                    dbc.Col([
                        dbc.Checklist(
                            id="data-source-checklist",
                            options=[
                                {"label": s["label"], "value": s["value"]}
                                for s in ALL_SOURCES
                            ],
                            value=SOURCE_VALUES,
                            switch=True,
                            inputStyle={"marginRight": "8px"},
                        ),
                    ], md=5),
                    # Sağ: durum badge'leri
                    dbc.Col(
                        html.Div(id="data-source-status-badges"),
                        md=7,
                    ),
                ], className="mb-3"),

                # BIST hisse seçim paneli (bist_stocks seçili olduğunda açılır)
                dbc.Collapse(
                    html.Div([
                        dbc.Row([
                            dbc.Col(
                                html.Span(
                                    "İndirilecek BIST Hisseleri",
                                    className="section-title",
                                ),
                            ),
                            dbc.Col(
                                dbc.Row([
                                    dbc.Col(
                                        html.Small(
                                            id="data-bist-count",
                                            style={"color": TEXT_MUTED},
                                        ),
                                        className="d-flex align-items-center",
                                    ),
                                    dbc.Col(
                                        dbc.ButtonGroup([
                                            dbc.Button(
                                                "Tümü",
                                                id="data-bist-select-all",
                                                size="sm", outline=True, color="secondary",
                                            ),
                                            dbc.Button(
                                                "Phase-1",
                                                id="data-bist-phase1",
                                                size="sm", outline=True, color="secondary",
                                            ),
                                            dbc.Button(
                                                "Temizle",
                                                id="data-bist-clear",
                                                size="sm", outline=True, color="danger",
                                            ),
                                        ]),
                                        width="auto",
                                    ),
                                ], className="align-items-center justify-content-end g-2"),
                                width="auto",
                            ),
                        ], className="align-items-center mb-2"),
                        # Chip-style checklist
                        dbc.Checklist(
                            id="data-bist-symbols",
                            options=[
                                {"label": sym.replace(".IS", ""), "value": sym}
                                for sym, _ in BIST30_SYMBOLS
                            ],
                            value=ALL_BIST_VALUES,
                            inline=True,
                            className="bist-chip-list",
                        ),
                    ], style={
                        "backgroundColor": CARD2,
                        "borderRadius": "6px",
                        "padding": "14px 16px",
                        "marginBottom": "4px",
                    }),
                    id="data-bist-collapse",
                    is_open=True,
                ),

                html.Hr(style={"borderColor": CARD2}),

                # Aksiyon butonları + tarih aralığı
                dbc.Row([
                    dbc.Col([
                        html.Label(
                            "Tam İndirme İçin Tarih Aralığı",
                            style={"color": TEXT_MUTED, "fontWeight": "600",
                                   "textTransform": "uppercase",
                                   "fontSize": "11px", "letterSpacing": "0.05em"},
                        ),
                        dcc.DatePickerRange(
                            id="data-date-range",
                            start_date=str(five_years_ago),
                            end_date=str(today),
                            display_format="YYYY-MM-DD",
                            style={"width": "100%"},
                            className="dark-datepicker",
                        ),
                        dbc.ButtonGroup([
                            dbc.Button("1Y", id="data-btn-1y", size="sm", outline=True, color="primary"),
                            dbc.Button("3Y", id="data-btn-3y", size="sm", outline=True, color="primary"),
                            dbc.Button("5Y", id="data-btn-5y", size="sm", outline=True, color="primary"),
                        ], className="mt-2"),
                    ], md=5),

                    dbc.Col([
                        html.Label("\u200b", style={"display": "block"}),
                        dbc.Button(
                            [html.I(className="bi bi-cloud-download me-2"),
                             "Güncelle (sadece eksik)"],
                            id="data-incremental-btn",
                            color="success",
                            className="w-100 mb-2",
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-arrow-repeat me-2"),
                             "Yeniden İndir (tam)"],
                            id="data-full-btn",
                            color="warning",
                            outline=True,
                            className="w-100",
                        ),
                    ], md=4, className="d-flex flex-column justify-content-end"),

                    dbc.Col([
                        html.Label("\u200b", style={"display": "block"}),
                        dbc.Button(
                            [html.I(className="bi bi-database me-2"),
                             "Veri Üret (eski)"],
                            id="data-generate-btn",
                            color="secondary",
                            outline=True,
                            className="w-100",
                        ),
                    ], md=3, className="d-flex flex-column justify-content-end"),
                ], className="align-items-end g-3"),

                html.Div(id="data-action-result", className="mt-3"),
            ]),
        ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}", "marginBottom": "24px"}),

        # ── Veri özeti + dosya listesi ───────────────────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span(
                        "Veri Kaynağı Özeti",
                        style={"color": TEXT, "fontWeight": "600"},
                    )),
                    dbc.CardBody(html.Div(id="data-info-panel")),
                ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
            ], md=5, className="mb-4"),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span(
                        "Mevcut Dosyalar",
                        style={"color": TEXT, "fontWeight": "600"},
                    )),
                    dbc.CardBody(html.Div(id="data-list-panel")),
                ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
            ], md=7, className="mb-4"),
        ]),

        # ── BIST-30 referans listesi ─────────────────────────────────────────
        dbc.Card([
            dbc.CardHeader(html.Span(
                "BIST-30 Hisse Listesi",
                style={"color": TEXT, "fontWeight": "600"},
            )),
            dbc.CardBody(_bist30_badges()),
        ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    # Tarih hızlı seçim
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
        today = date.today()
        btn = ctx.triggered_id
        if btn == "data-btn-1y":
            start = today - timedelta(days=365)
        elif btn == "data-btn-3y":
            start = today - timedelta(days=3 * 365)
        else:
            start = today - timedelta(days=5 * 365)
        return str(start), str(today)

    # BIST collapse açma/kapama
    @app.callback(
        Output("data-bist-collapse", "is_open"),
        Input("data-source-checklist", "value"),
        prevent_initial_call=False,
    )
    def toggle_bist_collapse(selected):
        return "bist_stocks" in (selected or [])

    # BIST hisse sayacı
    @app.callback(
        Output("data-bist-count", "children"),
        Input("data-bist-symbols", "value"),
        prevent_initial_call=False,
    )
    def update_bist_count(selected):
        n = len(selected) if selected else 0
        total = len(BIST30_SYMBOLS)
        return f"{n} / {total} seçili"

    # BIST hızlı seçim butonları
    @app.callback(
        Output("data-bist-symbols", "value"),
        [
            Input("data-bist-select-all", "n_clicks"),
            Input("data-bist-phase1", "n_clicks"),
            Input("data-bist-clear", "n_clicks"),
        ],
        prevent_initial_call=True,
    )
    def bist_quick_select(all_n, phase1_n, clear_n):
        btn = ctx.triggered_id
        if btn == "data-bist-select-all":
            return ALL_BIST_VALUES
        if btn == "data-bist-phase1":
            return PHASE1_SYMBOLS
        return []

    # Durum yenileme
    @app.callback(
        [
            Output("data-status-store", "data"),
            Output("data-source-status-badges", "children"),
            Output("data-source-checklist", "value"),
            Output("data-list-panel", "children"),
        ],
        Input("data-status-refresh-btn", "n_clicks"),
        prevent_initial_call=False,
    )
    def refresh_status(_n):
        status_resp = api.get_data_status()
        source_status = status_resp.get("status", {}) if status_resp else {}
        datasets = api.get_data_list()

        default_checked = [
            s["value"]
            for s in ALL_SOURCES
            if source_status.get(s["value"], {}).get("exists", False)
            and not source_status.get(s["value"], {}).get("error")
        ]

        return (
            source_status,
            _render_status_badges(source_status),
            default_checked if default_checked else SOURCE_VALUES,
            _render_dataset_list(datasets),
        )

    # Veri özeti paneli — status store değişince güncellenir, kaynak seçiminden bağımsız
    @app.callback(
        Output("data-info-panel", "children"),
        Input("data-status-store", "data"),
        prevent_initial_call=False,
    )
    def update_info_panel(source_status):
        if not source_status:
            return html.P(
                "Durumu yenilemek için 'Durumu Yenile' butonuna basın.",
                style={"color": TEXT_MUTED, "fontSize": "13px"},
            )
        return _render_source_details(source_status)

    # Incremental güncelleme
    @app.callback(
        Output("data-action-result", "children"),
        Input("data-incremental-btn", "n_clicks"),
        [
            State("data-source-checklist", "value"),
            State("data-bist-symbols", "value"),
        ],
        prevent_initial_call=True,
    )
    def incremental_update(n, selected, bist_symbols):
        if not n:
            return ""
        if not selected:
            return dbc.Alert(
                "Lütfen en az bir kaynak seçin.", color="warning", dismissable=True,
            )
        payload = {"sources": selected, "mode": "incremental"}
        if "bist_stocks" in selected and bist_symbols:
            payload["symbols"] = bist_symbols
        result = api.update_data(payload)
        return _render_update_result(result)

    # Tam indirme
    @app.callback(
        Output("data-action-result", "children", allow_duplicate=True),
        Input("data-full-btn", "n_clicks"),
        [
            State("data-source-checklist", "value"),
            State("data-bist-symbols", "value"),
            State("data-date-range", "start_date"),
        ],
        prevent_initial_call=True,
    )
    def full_download(n, selected, bist_symbols, start):
        if not n:
            return ""
        if not selected:
            return dbc.Alert(
                "Lütfen en az bir kaynak seçin.", color="warning", dismissable=True,
            )
        payload = {"sources": selected, "mode": "full", "start_date": start}
        if "bist_stocks" in selected and bist_symbols:
            payload["symbols"] = bist_symbols
        result = api.update_data(payload)
        return _render_update_result(result)

    # Legacy veri üret
    @app.callback(
        Output("data-action-result", "children", allow_duplicate=True),
        Input("data-generate-btn", "n_clicks"),
        [State("data-date-range", "start_date"), State("data-date-range", "end_date")],
        prevent_initial_call=True,
    )
    def generate_data(n, start, end):
        if not n:
            return ""
        result = api.generate_data({"start_date": start, "end_date": end})
        if result and result.get("status") == "success":
            return dbc.Alert(
                [html.I(className="bi bi-check-circle me-2"),
                 f"Veri başarıyla üretildi: {result.get('total_rows', 0):,} satır"],
                color="success", dismissable=True,
            )
        return dbc.Alert(
            [html.I(className="bi bi-exclamation-triangle me-2"),
             f"Veri üretimi başarısız: {result}"],
            color="danger", dismissable=True,
        )


# ── Render yardımcıları ────────────────────────────────────────────────────────

def _render_status_badges(source_status: dict) -> html.Div:
    """Her kaynak için durum badge'i döndür (checklist satırlarıyla hizalı)."""
    if not source_status:
        return html.P(
            "Durum alınamadı.",
            style={"color": TEXT_MUTED, "fontSize": "13px"},
        )

    rows = []
    for src in ALL_SOURCES:
        key = src["value"]
        info = source_status.get(key, {})
        exists = info.get("exists", False)
        last_date = info.get("last_date")
        missing = info.get("missing_days")
        error = info.get("error")

        if error:
            badge = dbc.Badge(f"Hata: {str(error)[:40]}", color="danger", pill=True)
        elif not exists or last_date is None:
            badge = dbc.Badge("Veri yok", color="secondary", pill=True)
        elif missing == 0:
            badge = dbc.Badge("Güncel ✓", color="success", pill=True)
        else:
            badge = dbc.Badge(
                f"Son: {last_date}  ·  {missing} iş günü eksik",
                color="warning", pill=True,
            )

        rows.append(html.Div(badge, style={"marginBottom": "14px", "paddingTop": "2px"}))

    return html.Div(rows)


def _render_source_details(source_status: dict) -> html.Div:
    """Tüm kaynakların özet bilgilerini göster (kaynak seçiminden bağımsız)."""
    cards = []

    for src in ALL_SOURCES:
        key = src["value"]
        info = source_status.get(key, {})

        if not info:
            continue

        exists = info.get("exists", False)
        last_date = info.get("last_date", "—")
        missing = info.get("missing_days")
        symbols = info.get("symbols", [])
        error = info.get("error")
        file_name = info.get("file", "—")

        if error:
            status_badge = dbc.Badge(f"Hata: {str(error)[:40]}", color="danger", pill=True)
        elif not exists:
            status_badge = dbc.Badge("Veri yok", color="secondary", pill=True)
        elif missing == 0:
            status_badge = dbc.Badge("Güncel ✓", color="success", pill=True)
        else:
            status_badge = dbc.Badge(
                f"{missing} iş günü eksik", color="warning", pill=True,
            )

        rows = [
            ("Durum", status_badge),
            ("Dosya", html.Small(file_name, style={"color": TEXT, "fontSize": "12px"})),
            ("Son Tarih", last_date or "—"),
        ]
        if key == "bist_stocks" and symbols:
            rows.append(("Hisseler", f"{len(symbols)} adet"))
            rows.append(("Semboller", html.Small(
                ", ".join(s.replace(".IS", "") for s in symbols[:8])
                + ("…" if len(symbols) > 8 else ""),
                style={"color": TEXT, "fontSize": "11px"},
            )))
        elif key == "gold" and symbols:
            rows.append(("Seriler", ", ".join(symbols)))
        elif key == "macro" and symbols:
            rows.append(("Sütunlar", f"{len(symbols)} adet"))

        detail_rows = []
        for label, value in rows:
            detail_rows.append(
                dbc.Row([
                    dbc.Col(
                        html.Small(label, style={"color": TEXT_MUTED}), width=5,
                    ),
                    dbc.Col(
                        html.Span(value, style={"color": TEXT, "fontSize": "13px"})
                        if isinstance(value, str)
                        else value,
                        width=7,
                    ),
                ], className="py-1 border-bottom",
                   style={"borderColor": CARD2 + " !important"})
            )

        cards.append(html.Div([
            html.Small(
                src["label"],
                style={"color": TEXT, "fontWeight": "600", "display": "block",
                       "marginBottom": "8px", "marginTop": "12px"},
            ),
            html.Div(detail_rows),
        ]))

    return html.Div(cards)


def _render_update_result(result: dict) -> html.Div:
    if not result or result.get("status") != "success":
        err = result.get("detail", str(result)) if result else "Sunucu yanıt vermedi"
        return dbc.Alert(
            [html.I(className="bi bi-exclamation-triangle me-2"),
             f"Güncelleme başarısız: {err}"],
            color="danger", dismissable=True,
        )

    results = result.get("results", {})
    lines = []
    for src_key, res in results.items():
        src_label = next(
            (s["label"] for s in ALL_SOURCES if s["value"] == src_key), src_key
        )
        if "error" in res:
            lines.append(html.Li(
                [html.Strong(src_label), f": Hata — {res['error']}"],
                style={"color": "#f87171"},
            ))
        elif res.get("mode") == "skip":
            lines.append(html.Li([html.Strong(src_label), ": Zaten güncel."]))
        else:
            new_rows = res.get("new_rows", res.get("rows", "?"))
            total = res.get("total_rows", "?")
            fetch_from = res.get("fetch_from", "")
            period = f" ({fetch_from} → bugün)" if fetch_from else ""
            if isinstance(new_rows, int) and isinstance(total, int):
                msg = f": +{new_rows:,} yeni satır{period}  ·  toplam {total:,} satır"
            else:
                msg = ": tamamlandı"
            lines.append(html.Li([html.Strong(src_label), msg]))

    return dbc.Alert(
        [
            html.I(className="bi bi-check-circle me-2"),
            html.Strong("Güncelleme tamamlandı"),
            html.Ul(lines, className="mb-0 mt-1"),
        ],
        color="success", dismissable=True,
    )


def _render_dataset_list(datasets: list) -> html.Div:
    if not datasets:
        return html.P("Kayıtlı veri seti yok.", style={"color": TEXT_MUTED})

    rows = []
    for ds in datasets:
        name = ds.get("name", ds.get("filename", "—"))
        size = ds.get("size_mb", ds.get("size", "—"))
        size_str = f"{size} MB" if isinstance(size, (int, float)) else str(size)
        modified = str(
            ds.get("modified_at", ds.get("modified", ds.get("created", "—")))
        )[:10]
        rows.append(
            dbc.Row([
                dbc.Col(html.Span(name, style={"color": TEXT, "fontSize": "12px"}), width=6),
                dbc.Col(html.Small(size_str, style={"color": TEXT_MUTED}), width=3),
                dbc.Col(html.Small(modified, style={"color": TEXT_MUTED}), width=3),
            ], className="py-2 border-bottom",
               style={"borderColor": CARD2 + " !important"})
        )
    return html.Div([
        dbc.Row([
            dbc.Col(html.Small("Dosya", style={"color": TEXT_MUTED, "fontWeight": "600",
                                               "textTransform": "uppercase",
                                               "fontSize": "11px"}), width=6),
            dbc.Col(html.Small("Boyut", style={"color": TEXT_MUTED, "fontWeight": "600",
                                               "textTransform": "uppercase",
                                               "fontSize": "11px"}), width=3),
            dbc.Col(html.Small("Tarih", style={"color": TEXT_MUTED, "fontWeight": "600",
                                               "textTransform": "uppercase",
                                               "fontSize": "11px"}), width=3),
        ]),
        html.Div(rows),
    ])


def _bist30_badges() -> html.Div:
    cols = 6
    rows = []
    for i in range(0, len(BIST30_SYMBOLS), cols):
        chunk = BIST30_SYMBOLS[i:i + cols]
        rows.append(
            dbc.Row([
                dbc.Col(
                    dbc.Badge(
                        sym, color="primary", pill=True,
                        style={"fontSize": "11px"},
                    ),
                    width=12 // cols,
                    className="mb-2",
                )
                for sym, _ in chunk
            ])
        )
    return html.Div(rows)
