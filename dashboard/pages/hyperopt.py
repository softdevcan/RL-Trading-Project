"""
Hyperparameter Optimization page – start studies, poll progress, modal details.

API endpoints used:
  POST /api/hyperopt/start
  GET  /api/hyperopt/studies
  GET  /api/hyperopt/studies/{id}
  GET  /api/hyperopt/studies/{id}/progress
  GET  /api/hyperopt/search-spaces/{algorithm}
"""

import json
from datetime import date, timedelta

from dash import html, dcc, no_update
from dash import ALL as _ALL
from dash import Input, Output, State
import dash_bootstrap_components as dbc

from dashboard.theme import (
    CARD2, TEXT, TEXT_MUTED, BORDER, GREEN, RED, BLUE, ORANGE, algo_badge_class,
)
from dashboard.components.page_header import create_page_header
from dashboard.components.state_block import create_state_block
import dashboard.api_client as api

# Backend /config/* ulasilamadiginda kullanilan emniyet listeleri.
_FALLBACK_ALGORITHMS = [
    {"value": "ppo", "label": "PPO"},
    {"value": "a2c", "label": "A2C"},
    {"value": "td3", "label": "TD3"},
    {"value": "sac", "label": "SAC"},
]
_FALLBACK_PHASES = [
    {"value": 1, "label": "Faz 1"},
    {"value": 2, "label": "Faz 2"},
]
_FALLBACK_REWARD_TYPES = [
    {"value": "psr", "label": "PSR"},
    {"value": "simple", "label": "Simple"},
]


def _algo_options():
    items = api.get_config_algorithms() or _FALLBACK_ALGORITHMS
    return [{"label": it.get("label", it.get("value")), "value": it.get("value")} for it in items]


def _phase_options():
    items = api.get_config_phases() or _FALLBACK_PHASES
    return [{"label": it.get("label", f"Faz {it.get('value')}"), "value": it.get("value")} for it in items]


def _reward_options():
    items = api.get_config_reward_types() or _FALLBACK_REWARD_TYPES
    return [{"label": it.get("label", it.get("value", "").upper()), "value": it.get("value")} for it in items]


# ── Layout ────────────────────────────────────────────────────────────────────

def layout():
    today = date.today()
    default_start = today - timedelta(days=365)

    # CSV mevcut ise tarih/sembol sınırlarını oradan dolduralım; yoksa
    # makul fallback (son 1 yıl, sembol seçici boş).
    data_range = api.get_hyperopt_data_range() or {"available": False, "symbols": []}
    cached_available = bool(data_range.get("available"))
    csv_min = data_range.get("min_date") or str(default_start)
    csv_max = data_range.get("max_date") or str(today)
    csv_symbols = data_range.get("symbols", []) or []

    # Default penceresi: train = ilk %80, val = son %20
    csv_min_d = date.fromisoformat(csv_min)
    csv_max_d = date.fromisoformat(csv_max)
    total_days = (csv_max_d - csv_min_d).days
    split_d = csv_min_d + timedelta(days=int(total_days * 0.8))

    csv_status_msg = (
        f"CSV: {csv_min} → {csv_max} • {data_range.get('total_rows', 0)} satır • "
        f"{len(csv_symbols)} sembol"
        if cached_available
        else "CSV yok — 'yfinance'tan tazele' otomatik açık."
    )

    # Sayfa her gezinmede yeniden uretiliyor; calisan study'nin kimligi
    # yalnizca `dcc.Store`'da tutulsaydi baska bir sayfaya gidip donen
    # kullanici surmekte olan optimizasyonun ilerlemesini bir daha
    # goremezdi (egitim sayfasindaki ayni kusur). Acilista listeden
    # calisan kosum bulunur ve yoklama ona baglanir.
    running = next(
        (st for st in (api.get_hyperopt_studies() or [])
         if str(st.get("status", st.get("state", ""))).lower() == "running"),
        None,
    )
    active_study = (running or {}).get("study_id") or (running or {}).get("study_name")

    return html.Div([
        dcc.Interval(id="hyperopt-poll", interval=3_000,
                     disabled=active_study is None, n_intervals=0),
        # Calisan optimizasyonun study_id'si — /progress bununla sorgulanir
        dcc.Store(id="hyperopt-active-study", data=active_study),
        dcc.Store(id="hyperopt-modal-study-id", data=None),
        dcc.Store(id="hyperopt-data-range", data=data_range),

        create_page_header("Hiper Parametre Optimizasyonu",
                           "Optuna ile otomatik hiper parametre arama"),

        dbc.Row([
            # ── Form panel ──────────────────────────────────────────────────
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Span("Optimizasyon Baslat", className="card-title-sm")),
                    dbc.CardBody([
                        # Algorithm
                        html.Label("Algoritma", className="section-title"),
                        dcc.Dropdown(
                            id="hyperopt-algo",
                            options=_algo_options(),
                            value="ppo", clearable=False,
                            style={"marginBottom": "16px"},
                        ),
                        # Phase
                        html.Label("Faz", className="section-title"),
                        dbc.RadioItems(
                            id="hyperopt-phase",
                            options=_phase_options(),
                            value=1, inline=True, className="mb-3",
                        ),
                        # Reward type
                        html.Label("Ödül Tipi", className="section-title"),
                        dcc.Dropdown(
                            id="hyperopt-reward",
                            options=_reward_options(),
                            value="psr", clearable=False,
                            style={"marginBottom": "16px"},
                        ),
                        # n_trials
                        html.Label("Deneme Sayisi (n_trials)", className="section-title"),
                        dbc.Input(id="hyperopt-trials", type="number", value=20, min=1, max=500,
                                  className="mb-3"),
                        # timesteps (schema: ge=10_000, le=1_000_000)
                        html.Label("Timestep / Trial (min 10000)", className="section-title"),
                        dbc.Input(id="hyperopt-timesteps", type="number", value=10_000,
                                  min=10_000, max=1_000_000, step=1_000,
                                  className="mb-3"),

                        # Data source
                        html.Label("Veri Kaynağı", className="section-title"),
                        dbc.Checklist(
                            id="hyperopt-refresh-data",
                            options=[{"label": " yfinance'tan tazele (CSV ezilir)", "value": "refresh"}],
                            value=[] if cached_available else ["refresh"],
                            switch=True,
                            className="mb-2",
                        ),
                        html.Small(csv_status_msg, id="hyperopt-csv-status",
                                   style={"color": TEXT_MUTED, "display": "block", "marginBottom": "12px"}),

                        # Symbols
                        html.Label("Semboller", className="section-title"),
                        dcc.Dropdown(
                            id="hyperopt-symbols",
                            options=[{"label": s, "value": s} for s in csv_symbols],
                            value=csv_symbols,
                            multi=True,
                            placeholder="Boş = PHASE1_SYMBOLS (default)",
                            style={"marginBottom": "12px"},
                        ),

                        # Train date range (val = train_end +1 → csv_max)
                        html.Label("Eğitim Aralığı (val otomatik son %20)", className="section-title"),
                        dbc.Row([
                            dbc.Col(dcc.DatePickerSingle(
                                id="hyperopt-train-start",
                                date=str(csv_min_d), min_date_allowed=csv_min, max_date_allowed=csv_max,
                                display_format="YYYY-MM-DD"), width=6),
                            dbc.Col(dcc.DatePickerSingle(
                                id="hyperopt-train-end",
                                date=str(split_d), min_date_allowed=csv_min, max_date_allowed=csv_max,
                                display_format="YYYY-MM-DD"), width=6),
                        ], className="mb-2"),
                        html.Small(id="hyperopt-val-preview",
                                   style={"color": TEXT_MUTED, "display": "block", "marginBottom": "12px"}),

                        # Search space info
                        html.Div(id="hyperopt-space-info", className="mb-3"),
                        # Start button
                        dbc.Button(
                            [html.I(className="bi bi-play-circle me-2"), "Optimizasyonu Baslat"],
                            id="hyperopt-start-btn",
                            color="success",
                            className="w-100",
                        ),
                        html.Div(id="hyperopt-start-result", className="mt-3"),
                        # Calisan optimizasyonun canli durumu (/progress ucundan)
                        html.Div(id="hyperopt-progress-panel", className="mt-3"),
                    ]),
                ]),
            ], md=4, className="mb-4"),

            # ── Studies grid ────────────────────────────────────────────────
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            [html.I(className="bi bi-arrow-clockwise me-1"), "Yenile"],
                            id="hyperopt-refresh-btn",
                            color="secondary", outline=True, size="sm",
                        ),
                    ], className="mb-3 d-flex justify-content-end"),
                ]),
                html.Div(id="hyperopt-delete-alert"),
                html.Div(id="hyperopt-studies-grid"),
            ], md=8, className="mb-4"),
        ]),

        # Silme sonrasi listeyi tazeler
        dcc.Store(id="hyperopt-delete-tick", data=0),
        dcc.Store(id="hyperopt-delete-target", data=None),

        # Silme onayi — Optuna deposundan kalici olarak siler
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Calismayi sil")),
                dbc.ModalBody(html.Div(id="hyperopt-delete-body")),
                dbc.ModalFooter([
                    dbc.Button("Vazgec", id="hyperopt-delete-cancel",
                               color="secondary", outline=True, className="me-2"),
                    dbc.Button([html.I(className="bi bi-trash me-1"), "Sil"],
                               id="hyperopt-delete-confirm", color="danger"),
                ]),
            ],
            id="hyperopt-delete-modal",
            is_open=False,
            centered=True,
        ),

        # ── Detail modal ─────────────────────────────────────────────────────
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(html.Span(id="hyperopt-modal-title")), close_button=True),
            dbc.ModalBody([
                dbc.Row([
                    dbc.Col([
                        html.Div(id="hyperopt-modal-info"),
                    ], md=6),
                    dbc.Col([
                        html.H6("En Iyi Parametreler", style={"color": TEXT_MUTED}),
                        html.Pre(id="hyperopt-modal-params",
                                  style={"backgroundColor": CARD2, "padding": "12px",
                                         "borderRadius": "6px", "color": TEXT, "fontSize": "12px",
                                         "overflowX": "auto"}),
                    ], md=6),
                ], className="mb-3"),
                html.H6("Trial Sonuclari", style={"color": TEXT_MUTED, "marginTop": "16px"}),
                html.Div(id="hyperopt-modal-trials"),
            ]),
            dbc.ModalFooter(dbc.Button("Kapat", id="hyperopt-modal-close", color="secondary")),
        ], id="hyperopt-modal", size="xl", is_open=False),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("hyperopt-space-info", "children"),
        Input("hyperopt-algo", "value"),
        prevent_initial_call=False,
    )
    def load_search_space(algo):
        if not algo:
            return html.Span()
        spaces = api.get_search_spaces(algo) or {}
        if not spaces:
            return html.Small(f"{algo} arama uzayi bilgisi yok.", style={"color": TEXT_MUTED})
        n = len(spaces.get("parameters", spaces)) if isinstance(spaces, dict) else 0
        return html.Small(f"{algo} icin {n} hiper parametre aranacak.", style={"color": TEXT_MUTED})

    @app.callback(
        Output("hyperopt-val-preview", "children"),
        [Input("hyperopt-train-end", "date"),
         Input("hyperopt-data-range", "data")],
    )
    def update_val_preview(train_end, data_range):
        """Val = train_end+1 → csv_max. Görsel feedback."""
        if not train_end:
            return ""
        try:
            te = date.fromisoformat(str(train_end)[:10])
            csv_max = (data_range or {}).get("max_date")
            if not csv_max:
                return f"Val: {te + timedelta(days=1)} → (CSV yok)"
            vmax = date.fromisoformat(csv_max)
            vmin = te + timedelta(days=1)
            if vmin >= vmax:
                return html.Span(
                    f"⚠ train_end ({te}) çok geç — val aralığı boş",
                    style={"color": RED}
                )
            return f"Val: {vmin} → {vmax} ({(vmax - vmin).days} gün)"
        except Exception:
            return ""

    @app.callback(
        [Output("hyperopt-poll", "disabled"),
         Output("hyperopt-start-result", "children"),
         Output("hyperopt-active-study", "data")],
        Input("hyperopt-start-btn", "n_clicks"),
        [
            State("hyperopt-algo", "value"),
            State("hyperopt-phase", "value"),
            State("hyperopt-reward", "value"),
            State("hyperopt-trials", "value"),
            State("hyperopt-timesteps", "value"),
            State("hyperopt-train-start", "date"),
            State("hyperopt-train-end", "date"),
            State("hyperopt-refresh-data", "value"),
            State("hyperopt-symbols", "value"),
            State("hyperopt-data-range", "data"),
        ],
        prevent_initial_call=True,
    )
    def start_optimization(n, algo, phase, reward, trials, timesteps,
                           train_start, train_end, refresh, symbols, data_range):
        if not n:
            return True, html.Span(), None

        if not train_start or not train_end:
            return True, dbc.Alert("Eğitim tarihleri eksik.", color="danger", dismissable=True), None

        # val = train_end+1 → csv_max
        try:
            te = date.fromisoformat(str(train_end)[:10])
        except Exception:
            return True, dbc.Alert("Train end tarihi geçersiz.", color="danger", dismissable=True), None
        val_start_d = te + timedelta(days=1)
        csv_max = (data_range or {}).get("max_date") or str(date.today())
        try:
            val_end_d = date.fromisoformat(csv_max)
        except Exception:
            val_end_d = date.today()
        if val_start_d >= val_end_d:
            return True, dbc.Alert(
                "Val aralığı boş — train_end'i öne çekin veya veri çekin.",
                color="danger", dismissable=True,
            ), None

        payload = {
            "algorithm": algo,
            "phase": int(phase or 1),
            "reward_type": reward or "psr",
            "n_trials": int(trials or 20),
            "total_timesteps": int(timesteps or 10_000),
            "train_start": str(train_start)[:10],
            "train_end": str(train_end)[:10],
            "val_start": str(val_start_d),
            "val_end": str(val_end_d),
            "use_cached_data": "refresh" not in (refresh or []),
            "stock_symbols": symbols or None,
        }
        result = api.start_hyperopt(payload)
        if result and result.get("study_id"):
            sid = result["study_id"]
            info = (
                f"Train {payload['train_start']}→{payload['train_end']} • "
                f"Val {payload['val_start']}→{payload['val_end']} • "
                f"Kaynak: {'CSV' if payload['use_cached_data'] else 'yfinance'}"
            )
            est = result.get("estimated_duration_minutes")
            est_txt = f" • tahmini ~{est:.0f} dk" if isinstance(est, (int, float)) else ""
            return False, dbc.Alert(
                [html.I(className="bi bi-check-circle me-2"),
                 f"Optimizasyon başlatıldı ({payload['n_trials']} trial{est_txt})",
                 html.Br(), html.Small(info),
                 html.Br(), html.Small(f"study: {sid}", style={"opacity": 0.7})],
                color="success", dismissable=True,
            ), sid

        # Nedeni burada bilinmiyor (api_client hatayi loglar). Kullaniciya en
        # sik iki sebebi soyle: dogrulama hatasi ve backend hatasi.
        return True, dbc.Alert(
            [html.I(className="bi bi-x-circle me-2"),
             "Optimizasyon başlatılamadı.", html.Br(),
             html.Small("Parametreleri kontrol edin (timestep sayısı en az 10.000 olmalı). "
                        "Ayrıntı için sunucu loglarına bakın.")],
            color="danger", dismissable=True,
        ), None

    @app.callback(
        Output("hyperopt-studies-grid", "children"),
        [Input("hyperopt-refresh-btn", "n_clicks"), Input("hyperopt-poll", "n_intervals"),
         Input("hyperopt-delete-tick", "data")],
        prevent_initial_call=False,
    )
    def refresh_studies(n_ref, n_poll, _tick):
        studies = api.get_hyperopt_studies()
        if not studies:
            return create_state_block("empty", "Kayitli optimizasyon calismasi yok.")
        return _render_studies_grid(studies)

    @app.callback(
        [Output("hyperopt-delete-modal", "is_open"),
         Output("hyperopt-delete-body", "children"),
         Output("hyperopt-delete-target", "data")],
        [Input({"type": "hyperopt-study-delete", "index": _ALL}, "n_clicks"),
         Input("hyperopt-delete-cancel", "n_clicks"),
         Input("hyperopt-delete-confirm", "n_clicks")],
        prevent_initial_call=True,
    )
    def toggle_delete_modal(del_clicks, cancel_n, confirm_n):
        from dash import ctx
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or not any(del_clicks or []):
            # Vazgec / Sil -> kapat. Hedefi TEMIZLEME: silme callback'i onu
            # ayni turda State olarak okuyor.
            return False, no_update, no_update

        study_id = str(triggered.get("index"))
        body = html.Div([
            html.P("Bu optimizasyon kaydi kalici olarak silinecek. "
                   "Deneme gecmisi ve en iyi parametreler de gider.",
                   style={"color": TEXT}),
            html.Div(study_id, style={"color": TEXT_MUTED, "fontSize": "12px",
                                      "wordBreak": "break-all"}),
            html.Small(
                "Optuna deposu tum kullanicilar arasinda ORTAK — silinen kayit "
                "herkesten silinir.",
                style={"color": TEXT_MUTED},
            ),
        ])
        return True, body, study_id

    @app.callback(
        [Output("hyperopt-delete-alert", "children"),
         Output("hyperopt-delete-tick", "data")],
        Input("hyperopt-delete-confirm", "n_clicks"),
        [State("hyperopt-delete-target", "data"), State("hyperopt-delete-tick", "data")],
        prevent_initial_call=True,
    )
    def delete_study(n_clicks, study_id, tick):
        if not n_clicks or not study_id:
            return no_update, no_update

        result = api.delete_hyperopt_study(str(study_id))
        if result.get("ok"):
            return (
                dbc.Alert("Calisma silindi.", color="success",
                          className="py-2 mb-3", duration=4000),
                (tick or 0) + 1,
            )

        detail = (result.get("body") or {}).get("detail") or f"HTTP {result.get('status')}"
        return dbc.Alert(str(detail), color="danger", className="py-2 mb-3"), no_update

    @app.callback(
        [
            Output("hyperopt-modal", "is_open"),
            Output("hyperopt-modal-title", "children"),
            Output("hyperopt-modal-info", "children"),
            Output("hyperopt-modal-params", "children"),
            Output("hyperopt-modal-trials", "children"),
            Output("hyperopt-modal-study-id", "data"),
        ],
        [Input({"type": "hyperopt-study-card", "index": _ALL}, "n_clicks"),
         Input("hyperopt-modal-close", "n_clicks")],
        prevent_initial_call=True,
    )
    def open_close_modal(card_clicks, close_n):
        from dash import ctx
        if ctx.triggered_id == "hyperopt-modal-close" or not any(card_clicks or []):
            return False, "", html.Span(), "", html.Span(), None

        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            return False, "", html.Span(), "", html.Span(), None

        study_id = triggered.get("index")
        detail = api.get_hyperopt_study(str(study_id)) or {}

        # Backend StudyDetailResponse: { study: {...}, trials: [...], mean_value, ... }
        # Fall back to flat shape if a caller ever returns one.
        study_obj = detail.get("study") if isinstance(detail.get("study"), dict) else detail
        trials = detail.get("trials", []) or []
        stats = {
            "mean_value": detail.get("mean_value"),
            "median_value": detail.get("median_value"),
            "std_value": detail.get("std_value"),
            "min_value": detail.get("min_value"),
            "max_value": detail.get("max_value"),
        }

        title = study_obj.get("study_name", study_obj.get("name", str(study_id)))
        info_div = _render_modal_info(study_obj, stats, total_trials=len(trials))
        params_str = json.dumps(study_obj.get("best_params") or {}, indent=2, ensure_ascii=False) or "{}"
        trials_div = _render_trials_table(trials)

        return True, title, info_div, params_str, trials_div, study_id

    @app.callback(
        [Output("hyperopt-progress-panel", "children"),
         Output("hyperopt-poll", "disabled", allow_duplicate=True)],
        Input("hyperopt-poll", "n_intervals"),
        State("hyperopt-active-study", "data"),
        prevent_initial_call=True,
    )
    def poll_progress(n, study_id):
        """Calisan optimizasyonun canli durumu.

        Eskiden burada sabit "100 poll sonra dur" (5 dakika) vardi ve calisma
        durumu hic gosterilmiyordu: 20 trial ~11 dakika surdugu icin polling
        is bitmeden kesiliyor, kullanici basladi mi bitti mi anlayamiyordu.
        Artik polling isin DURUMUNA gore durur.
        """
        if not study_id:
            return html.Span(), True

        pr = api.get_hyperopt_progress(str(study_id))
        if not pr:
            # Study ilk saniyelerde henuz gorunmeyebilir — polling'i surdur.
            return _render_progress_placeholder(), False

        status = (pr.get("status") or "").lower()
        bitti = status in ("completed", "failed", "cancelled")
        return _render_progress(pr), bitti


# ── Helpers ───────────────────────────────────────────────────────────────────

_STATUS_LABEL = {
    "pending": ("Sıraya alındı", BLUE),
    "running": ("Çalışıyor", BLUE),
    "completed": ("Tamamlandı", GREEN),
    "failed": ("Başarısız", RED),
    "cancelled": ("İptal edildi", ORANGE),
}


def _render_progress_placeholder():
    return dbc.Alert(
        [dbc.Spinner(size="sm", color="primary", spinner_class_name="me-2"),
         "Başlatılıyor..."],
        color="primary", className="mb-0",
    )


def _render_progress(pr):
    """Canli optimizasyon durumu — /hyperopt/studies/{id}/progress ciktisi."""
    status = (pr.get("status") or "").lower()
    label, color = _STATUS_LABEL.get(status, (status or "?", TEXT_MUTED))
    done = pr.get("trials_completed", 0) or 0
    total = pr.get("trials_total", 0) or 0
    pct = pr.get("progress_percentage", 0) or 0
    best = pr.get("current_best_value")
    eta = pr.get("estimated_time_remaining_minutes")
    last = pr.get("last_trial") or {}

    satirlar = [
        dbc.Row([
            dbc.Col(html.Small("Durum", className="section-title"), width=5),
            dbc.Col(html.Span(label, style={"color": color, "fontWeight": "600"}), width=7),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(html.Small("Trial", className="section-title"), width=5),
            dbc.Col(html.Span(f"{done} / {total}", className="card-title-sm"), width=7),
        ], className="mb-2"),
        dbc.Progress(value=pct, label=f"{pct:.0f}%",
                     color="success" if status == "completed" else "primary",
                     className="mb-3", animated=status == "running", striped=status == "running"),
    ]

    if isinstance(eta, (int, float)) and status == "running":
        satirlar.append(dbc.Row([
            dbc.Col(html.Small("Kalan", className="section-title"), width=5),
            dbc.Col(html.Span(f"~{eta:.0f} dk", style={"color": TEXT}), width=7),
        ], className="mb-2"))

    if isinstance(best, (int, float)):
        satirlar.append(dbc.Row([
            dbc.Col(html.Small("En iyi değer", className="section-title"), width=5),
            dbc.Col(html.Span(f"{best:.4f}", style={"color": GREEN, "fontWeight": "600"}), width=7),
        ], className="mb-2"))

    if last.get("trial_number") is not None:
        val = last.get("value")
        val_txt = f"{val:.4f}" if isinstance(val, (int, float)) else "—"
        satirlar.append(dbc.Row([
            dbc.Col(html.Small("Son trial", className="section-title"), width=5),
            dbc.Col(html.Span(f"#{last['trial_number']} · {last.get('state', '')} · {val_txt}",
                              style={"color": TEXT_MUTED, "fontSize": "12px"}), width=7),
        ], className="mb-2"))

    return html.Div(satirlar, style={
        "padding": "12px",
        "borderRadius": "6px",
        "border": f"1px solid {BORDER}",
        "backgroundColor": CARD2,
    })


def _render_studies_grid(studies):
    cols = []
    for s in studies:
        sid = s.get("study_id") or s.get("id") or s.get("study_name") or "unknown"
        algo = str(s.get("algorithm", "—")).upper()
        status = str(s.get("status", s.get("state", "unknown"))).lower()
        best = s.get("best_value", s.get("best_sharpe", 0)) or 0
        n_trials = s.get("n_trials", s.get("completed_trials", 0)) or 0
        total_trials = s.get("total_trials", s.get("n_trials_target", n_trials)) or n_trials
        progress = int(n_trials / total_trials * 100) if total_trials else 0

        status_color = "success" if status == "complete" else "warning" if status == "running" else "secondary"
        badge_class = algo_badge_class(algo)

        cols.append(dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dbc.Badge(algo, pill=True, className=badge_class), width="auto"),
                        dbc.Col(dbc.Badge(status.upper(), color=status_color, pill=True), width="auto", className="ms-auto"),
                    ], className="mb-2"),
                    html.Div(str(s.get("study_name", sid))[:30],
                             style={"color": TEXT, "fontWeight": "600", "fontSize": "13px", "marginBottom": "8px"}),
                    html.Small(f"Deneme: {n_trials}/{total_trials}", style={"color": TEXT_MUTED}),
                    dbc.Progress(value=progress, color="primary", style={"height": "4px"}, className="my-2"),
                    dbc.Row([
                        dbc.Col(html.Small("En Iyi", style={"color": TEXT_MUTED}), width=5),
                        dbc.Col(html.Span(f"{best:.4f}", style={"color": GREEN, "fontWeight": "700", "fontSize": "14px"}), width=7),
                    ]),
                    html.Div(
                        [
                            dbc.Button("Detay",
                                       id={"type": "hyperopt-study-card", "index": str(sid)},
                                       size="sm", color="primary", outline=True,
                                       className="flex-grow-1"),
                            dbc.Button(html.I(className="bi bi-trash"),
                                       id={"type": "hyperopt-study-delete", "index": str(sid)},
                                       size="sm", color="danger", outline=True,
                                       title="Bu calismayi sil",
                                       # Calisan kosum once iptal edilmeli; uc de
                                       # 409 ile reddediyor, dugme onu tekrar etmesin.
                                       disabled=(status == "running")),
                        ],
                        className="d-flex gap-2 mt-2",
                    ),
                ])
            ]),
            md=4, sm=6, xs=12, className="mb-3",
        ))
    return dbc.Row(cols)


def _render_modal_info(study, stats=None, total_trials=None):
    stats = stats or {}
    best_val = study.get("best_value")
    fields = [
        ("Calisma Adi", study.get("study_name", "—")),
        ("Algoritma", str(study.get("algorithm", "—")).upper()),
        ("Durum", str(study.get("status", study.get("state", "—"))).upper()),
        ("Faz", str(study.get("phase", "—"))),
        ("Odul Tipi", str(study.get("reward_type", "—")).upper()),
        ("Tamamlanan", f"{study.get('trials_completed', 0)}/{study.get('n_trials', total_trials or '—')}"),
        ("Pruned/Failed", f"{study.get('trials_pruned', 0)}/{study.get('trials_failed', 0)}"),
        ("En Iyi Deger", f"{best_val:.4f}" if isinstance(best_val, (int, float)) else "—"),
        ("En Iyi Trial #", str(study.get("best_trial")) if study.get("best_trial") is not None else "—"),
        ("Train Aralığı", f"{study.get('train_start', '—')} → {study.get('train_end', '—')}"),
        ("Val Aralığı", f"{study.get('val_start', '—')} → {study.get('val_end', '—')}"),
        ("Olusturuldu", str(study.get("created_at", "—"))[:19]),
    ]
    # Stats only when present (running studies have None)
    if stats.get("mean_value") is not None:
        fields.extend([
            ("Ort. Deger", f"{stats['mean_value']:.4f}"),
            ("Medyan", f"{stats.get('median_value', 0):.4f}"),
            ("Std", f"{stats.get('std_value', 0):.4f}"),
            ("Min / Max", f"{stats.get('min_value', 0):.4f} / {stats.get('max_value', 0):.4f}"),
        ])
    rows = []
    for label, val in fields:
        rows.append(dbc.Row([
            dbc.Col(html.Small(label, style={"color": TEXT_MUTED}), width=6),
            dbc.Col(html.Span(str(val), style={"color": TEXT, "fontWeight": "600", "fontSize": "13px"}), width=6),
        ], className="py-1 border-bottom"))
    return html.Div(rows)


def _render_trials_table(trials):
    if not trials:
        return create_state_block("empty", "Trial verisi yok.")

    header = dbc.Row([
        dbc.Col(html.Small("#", className="section-title"), width=1),
        dbc.Col(html.Small("Deger", className="section-title"), width=2),
        dbc.Col(html.Small("Durum", className="section-title"), width=2),
        dbc.Col(html.Small("Sure (s)", className="section-title"), width=2),
        dbc.Col(html.Small("Parametreler", className="section-title"), width=5),
    ])
    rows = [header]
    # Schema: trial_number, state (TrialState enum string), value, params, duration_seconds
    for t in trials[:25]:
        val = t.get("value") if t.get("value") is not None else 0
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = 0.0
        status = str(t.get("state", t.get("status", "—"))).lower()
        status_color = "success" if status == "complete" else "warning" if status == "pruned" else "danger" if "fail" in status else "secondary"
        duration = t.get("duration_seconds", t.get("duration", t.get("elapsed")))
        duration_str = f"{float(duration):.1f}" if duration not in (None, "—", "") else "—"
        try:
            params_str = json.dumps(t.get("params", {}), ensure_ascii=False)
            if len(params_str) > 80:
                params_str = params_str[:77] + "..."
        except Exception:
            params_str = str(t.get("params", ""))[:80]

        trial_num = t.get("trial_number", t.get("number", t.get("id", "—")))

        rows.append(dbc.Row([
            dbc.Col(html.Small(str(trial_num), style={"color": TEXT_MUTED}), width=1),
            dbc.Col(html.Small(f"{val:.4f}", style={"color": GREEN if val > 0 else RED}), width=2),
            dbc.Col(dbc.Badge(status.upper(), color=status_color, pill=True, style={"fontSize": "10px"}), width=2),
            dbc.Col(html.Small(duration_str, style={"color": TEXT_MUTED}), width=2),
            dbc.Col(html.Small(params_str, style={"color": TEXT_MUTED, "fontSize": "11px"}), width=5),
        ], className="py-1 border-bottom"))
    return html.Div(rows)
