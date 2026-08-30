"""Hesabim sayfasi — profil, gorunum tercihi ve oturum guvenligi.

Her rol erisir, viewer dahil: burada yonetilenler yetki degil kisisel hesap
ayarlaridir. Kullanici yonetimi (/dash/users) admin sayfasidir ve ayri kalir.

Iki farkli calisma bicimi bir arada — kasitli:

1. **Tema secimi sunucu turu GEREKTIRMEZ.** Dugmelere `data-theme-set` konur,
   assets/theme-toggle.js tiklamayi yakalar, damgayi ve cerezi gunceller,
   sonra PATCH /auth/preferences ile hesaba yazar. Anlik tepki icin.
2. **Profil/oturum islemleri sunucu callback'i kullanir** (`/api/account/*`).
   Bunlar DB durumu degistirir ve sonucu geri okunmali.

Bu yuzden tema karti callback ciktilarinin DISINDA durur: yeniden cizilirse
theme-toggle.js'in koydugu `active` sinifi ve `aria-pressed` durumu silinir.
"""

from dash import html, dcc, no_update
from dash import Input, Output, State
from dash.dash_table import DataTable
import dash_bootstrap_components as dbc

import dashboard.api_client as api
from dashboard.auth_context import current_user, display_name
from dashboard.components.page_header import create_page_header
from dashboard.components.state_block import create_state_block
from dashboard.components.table import TABLE_STYLES
from dashboard.theme import BORDER, CARD2, TEXT, TEXT_MUTED

ROLE_LABELS = {"admin": "Yonetici", "user": "Kullanici", "viewer": "Izleyici"}

ROLE_HELP = {
    "admin": "Tum yetkiler ve kullanici yonetimi.",
    "user": "Kendi calisma alaninda egitim, tahmin ve karar.",
    "viewer": "Yalnizca okuma.",
}

THEME_OPTIONS = [
    ("light", "Aydinlik", "bi-sun", "Her zaman acik zemin."),
    ("dark", "Koyu", "bi-moon-stars", "Her zaman koyu zemin."),
    ("system", "Sistem", "bi-circle-half", "Isletim sisteminizi izler."),
]

# Denetim kaydindaki ham eylem adlari kullaniciya gosterilmez; /dash/users
# (admin) ham kodu gosteriyor, burada okunur karsiligi veriliyor.
ACTION_LABELS = {
    "login": "Giris",
    "logout": "Cikis",
    "password.change": "Parola degisikligi",
    "account.update": "Profil guncellemesi",
    "account.revoke_sessions": "Diger oturumlar kapatildi",
    "session.reuse_detected": "Oturum jetonu tekrar kullanildi",
    # Admin rolundeki kullanici kendi etkinliginde bunlari da gorur
    "user.create": "Kullanici olusturuldu",
    "user.update": "Kullanici guncellendi",
    "user.delete": "Kullanici silindi",
    "user.password_reset": "Kullanici parolasi sifirlandi",
    "user.revoke_sessions": "Kullanicinin oturumlari kapatildi",
}

LOGIN_FAIL_REASONS = {
    "bad_password": "parola hatali",
    "locked": "hesap kilitli",
    "inactive": "hesap devre disi",
    "lockout_triggered": "cok fazla hatali deneme — hesap kilitlendi",
}


# ── Bicimlendirme yardimcilari ────────────────────────────────────────────

def _fmt_dt(value: str | None) -> str:
    """ISO zaman damgasini "30.08.2026 14:32 UTC" olarak yaz.

    UTC etiketi bilincli: kayitlar naive-UTC tutuluyor (bkz. models.utcnow).
    Etiketsiz gostermek "yerel saat" izlenimi verirdi ve son giris saatine
    bakan kullaniciyi yaniltirdi.
    """
    if not value:
        return "—"
    try:
        date, _, rest = value.partition("T")
        year, month, day = date.split("-")
        return f"{day}.{month}.{year} {rest[:5]} UTC"
    except Exception:
        return str(value)[:19].replace("T", " ")


def _fmt_size(num_bytes: int) -> str:
    size = float(num_bytes or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _initials(name: str) -> str:
    parts = [p for p in (name or "").replace(".", " ").split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _info_row(label: str, value, last: bool = False):
    return html.Div(
        [
            html.Span(label, style={"color": TEXT_MUTED, "fontSize": "13px"}),
            html.Span(value, style={"color": TEXT, "fontSize": "13px", "fontWeight": "500"}),
        ],
        style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "gap": "12px",
            "padding": "10px 0",
            "borderBottom": "none" if last else f"1px solid {BORDER}",
        },
    )


def _role_badge(role: str):
    return html.Span(
        [
            html.Span(
                ROLE_LABELS.get(role, role),
                style={"backgroundColor": CARD2, "color": TEXT, "fontSize": "11px",
                       "padding": "3px 9px", "borderRadius": "10px", "fontWeight": "600"},
            ),
            html.Small(ROLE_HELP.get(role, ""),
                       style={"color": TEXT_MUTED, "marginLeft": "8px"}),
        ]
    )


# ── Gorunum karti (clientside; callback ciktisi DEGIL) ────────────────────

def _theme_preview(kind: str):
    """Secenegin altindaki kucuk onizleme seridi.

    Renkler kasitli olarak TOKENLARDAN BAGIMSIZ: iki tema da ayni anda
    gorunmeli ki secim yapmadan once karsilastirilabilsin. Her seridin
    metin/zemin cifti kendi paletinden gelir. (test_theme_contrast bu dosyayi
    "kacak hex" denetiminden bu yuzden muaf tutuyor.)
    """
    palettes = {
        "light": {"bg": "#f6f8fb", "surface": "#ffffff", "border": "#e2e8f0",
                  "text": "#0f172a", "muted": "#556275"},
        "dark": {"bg": "#0f172a", "surface": "#1e293b", "border": "#334155",
                 "text": "#e2e8f0", "muted": "#a8b6c9"},
    }

    def strip(p):
        return html.Div(
            [
                html.Div(style={"height": "6px", "width": "60%", "borderRadius": "3px",
                                "backgroundColor": p["text"], "marginBottom": "4px"}),
                html.Div(style={"height": "5px", "width": "85%", "borderRadius": "3px",
                                "backgroundColor": p["muted"]}),
            ],
            style={
                "backgroundColor": p["surface"],
                "border": f"1px solid {p['border']}",
                "borderRadius": "4px",
                "padding": "8px",
            },
        )

    if kind == "system":
        # Sistem: iki tarafi yan yana goster — secenegin ne yaptigi boyle anlasilir
        return html.Div(
            [
                html.Div(strip(palettes["light"]),
                         style={"backgroundColor": palettes["light"]["bg"],
                                "padding": "6px", "flex": "1"}),
                html.Div(strip(palettes["dark"]),
                         style={"backgroundColor": palettes["dark"]["bg"],
                                "padding": "6px", "flex": "1"}),
            ],
            style={"display": "flex", "borderRadius": "6px", "overflow": "hidden",
                   "border": f"1px solid {BORDER}"},
        )

    p = palettes[kind]
    return html.Div(
        strip(p),
        style={"backgroundColor": p["bg"], "padding": "6px", "borderRadius": "6px",
               "border": f"1px solid {BORDER}"},
    )


def _theme_card():
    options = []
    for value, label, icon, hint in THEME_OPTIONS:
        options.append(
            dbc.Col(
                html.Div(
                    [
                        html.Button(
                            [
                                html.I(className=f"bi {icon} me-2"),
                                html.Span(label),
                            ],
                            id=f"theme-option-{value}",
                            className="btn theme-option",
                            **{"data-theme-set": value, "aria-pressed": "false"},
                            type="button",
                        ),
                        _theme_preview(value),
                        html.Small(hint, style={"color": TEXT_MUTED, "fontSize": "12px",
                                                "display": "block", "marginTop": "6px"}),
                    ],
                    style={"display": "flex", "flexDirection": "column", "gap": "8px"},
                ),
                md=4, sm=12, className="mb-3",
            )
        )

    return dbc.Card(
        [
            dbc.CardHeader(html.Span("Gorunum", className="card-title-sm")),
            dbc.CardBody(
                [
                    html.P(
                        "Tercih hesabiniza kaydedilir — baska bir makineden "
                        "girdiginizde de gecerli olur.",
                        style={"color": TEXT_MUTED, "marginBottom": "16px"},
                    ),
                    dbc.Row(options, className="g-3"),
                ]
            ),
        ]
    )


# ── Profil karti (govdesi statik, degerleri callback doldurur) ────────────

def _profile_card():
    """Ad soyad duzenlemesi + degistirilemeyen kimlik alanlari.

    Girdi kutusu ve dugme LAYOUT'TA statik durur; callback yalnizca
    `value`/`children` doldurur. Boylece State(...) her zaman var olan bir
    bileseni gosterir ve kaydetme sirasinda odak kaybolmaz.
    """
    return dbc.Card(
        [
            dbc.CardHeader(html.Span("Profil", className="card-title-sm")),
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.Span(id="account-avatar", className="account-avatar-lg"),
                            html.Div(
                                [
                                    html.Div(id="account-display-name",
                                             className="account-display-name"),
                                    html.Div(id="account-role-line"),
                                ],
                                style={"minWidth": "0"},
                            ),
                        ],
                        className="account-identity",
                    ),
                    html.Hr(),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Ad Soyad", className="section-title"),
                                    # debounce YOK: `value` yalnizca State olarak
                                    # okunuyor, sunucuya trafik binmiyor. Debounce
                                    # acikken deger blur'da guncellendigi icin
                                    # "yaz ve hemen Kaydet'e tikla" akisinda eski
                                    # deger gonderilme riski var.
                                    dbc.Input(id="account-name-input", type="text",
                                              placeholder="Ad Soyad", maxLength=120),
                                ],
                                md=8,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    [html.I(className="bi bi-check2 me-1"), "Kaydet"],
                                    id="account-name-save", color="primary",
                                    className="w-100",
                                ),
                                md=4, className="d-flex align-items-end",
                            ),
                        ],
                        className="g-3",
                    ),
                    html.Div(
                        [
                            html.I(className="bi bi-lock me-2"),
                            html.Span(id="account-email"),
                            html.Small(
                                " · giris adresiniz, yalnizca yonetici degistirebilir",
                                style={"color": TEXT_MUTED},
                            ),
                        ],
                        style={"color": TEXT_MUTED, "fontSize": "12px",
                               "marginTop": "12px"},
                    ),
                    html.Div(id="account-alert", className="mt-3"),
                ]
            ),
        ]
    )


def _info_card():
    return dbc.Card(
        [
            dbc.CardHeader(html.Span("Hesap", className="card-title-sm")),
            dbc.CardBody(html.Div(id="account-info-body")),
        ]
    )


def _security_card():
    return dbc.Card(
        [
            dbc.CardHeader(html.Span("Guvenlik", className="card-title-sm")),
            dbc.CardBody(
                [
                    html.P(
                        "Parolanizi degistirdiginizde diger tum oturumlariniz kapanir.",
                        style={"color": TEXT_MUTED, "marginBottom": "14px"},
                    ),
                    html.A(
                        [html.I(className="bi bi-key me-2"), "Parolayi degistir"],
                        href="/change-password",
                        className="btn btn-outline-secondary",
                    ),
                    html.Hr(),
                    html.Label("Aktif oturumlar", className="section-title"),
                    html.Div(id="account-sessions-body"),
                    html.Div(id="account-session-alert", className="mt-2"),
                ]
            ),
        ]
    )


def _activity_card():
    """Kendi hesap etkinligi.

    Amaci tek bir soruya cevap vermek: "benden baska biri bu hesaba girmeye
    calisti mi?" Basarisiz girisler sebebiyle birlikte gorunur; yoneticinin bu
    hesap uzerindeki islemleri KAPSAM DISI (gerekce:
    `service.list_audit_for_user`).
    """
    return dbc.Card(
        [
            dbc.CardHeader(html.Span("Son etkinlik", className="card-title-sm")),
            dbc.CardBody(
                [
                    html.P(
                        "Hesabinizla ilgili son 20 olay. Tanimadiginiz bir giris "
                        "gorurseniz parolanizi degistirin ve diger oturumlari kapatin.",
                        style={"color": TEXT_MUTED, "marginBottom": "14px"},
                    ),
                    html.Div(id="account-activity-body"),
                ]
            ),
        ]
    )


# ── Layout ────────────────────────────────────────────────────────────────

def layout():
    return html.Div(
        [
            # Her yazma isleminden sonra artar; okuma callback'ini tetikler.
            dcc.Store(id="account-tick", data=0),

            create_page_header(
                "Hesabim",
                "Profil bilgileri, gorunum tercihi ve oturum guvenligi",
            ),
            dbc.Row(
                [
                    dbc.Col(_profile_card(), lg=7, className="mb-4"),
                    dbc.Col(_info_card(), lg=5, className="mb-4"),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(_theme_card(), lg=7, className="mb-4"),
                    dbc.Col(_security_card(), lg=5, className="mb-4"),
                ]
            ),
            dbc.Row(dbc.Col(_activity_card(), lg=12, className="mb-4")),
        ]
    )


# ── Callback'ler ──────────────────────────────────────────────────────────

def _session_row(item: dict, last: bool):
    """Tek bir aktif oturum satiri.

    `tokens` alani 1'den buyukse ayni tarayici icin birden fazla gecerli
    kayit var demektir (es zamanli sessiz yenileme). Bunu ayri oturum gibi
    saymak yaniltici olur; gruplama sunucuda yapiliyor, burada yalnizca
    gosteriliyor.
    """
    badges = []
    if item.get("current"):
        badges.append(
            html.Span("bu tarayici", className="badge bg-primary",
                      style={"fontSize": "10px", "marginLeft": "8px"})
        )

    return html.Div(
        [
            html.Div(
                [
                    html.I(className="bi bi-display me-2", style={"color": TEXT_MUTED}),
                    html.Span(item.get("device") or "Bilinmeyen istemci",
                              style={"color": TEXT, "fontSize": "13px",
                                     "fontWeight": "500"}),
                    *badges,
                ],
                style={"display": "flex", "alignItems": "center"},
            ),
            html.Div(
                f"{item.get('ip') or 'IP yok'} · son yenileme "
                f"{_fmt_dt(item.get('last_seen'))}",
                style={"color": TEXT_MUTED, "fontSize": "11px", "marginTop": "2px"},
            ),
        ],
        style={
            "padding": "9px 0",
            "borderBottom": "none" if last else f"1px solid {BORDER}",
        },
    )


def _activity_detail(entry: dict) -> str:
    """Ham `detail` sozlugunu tek satirlik okunur ozete cevir.

    Bilinmeyen bir eylem gelirse (yeni audit action eklenirse) `target`
    gosterilir; bos donmektense elde olani vermek dogru — ekran sessizce
    bilgi kaybetmesin.
    """
    action = entry.get("action", "")
    detail = entry.get("detail") or {}

    if action == "login" and not entry.get("success"):
        reason = detail.get("reason", "")
        return LOGIN_FAIL_REASONS.get(reason, reason)

    if action == "account.update":
        change = detail.get("full_name") or {}
        if change:
            return f"{change.get('from') or '(bos)'} -> {change.get('to') or '(bos)'}"

    if action in ("account.revoke_sessions", "user.revoke_sessions"):
        return f"{detail.get('revoked', 0)} oturum"

    if action == "user.create":
        role = detail.get("role", "")
        return f"{entry.get('target', '')} ({role})" if role else entry.get("target", "")

    return entry.get("target", "")


def register_callbacks(app):
    @app.callback(
        Output("account-avatar", "children"),
        Output("account-display-name", "children"),
        Output("account-role-line", "children"),
        Output("account-name-input", "value"),
        Output("account-name-input", "disabled"),
        Output("account-name-save", "disabled"),
        Output("account-email", "children"),
        Output("account-info-body", "children"),
        Input("account-tick", "data"),
    )
    def load_account(_tick):
        data = api.get_account() or {}
        user = data.get("user") or current_user() or {}
        persistent = bool(data.get("persistent"))
        workspace = data.get("workspace") or {}

        name = user.get("full_name") or display_name()
        role = user.get("role", "user")

        rows = [
            _info_row("Son giris", _fmt_dt(user.get("last_login_at"))),
            _info_row("Hesap acilisi", _fmt_dt(user.get("created_at"))),
            _info_row("Hesabi acan", user.get("created_by") or "—"),
        ]
        if workspace.get("exists"):
            rows.append(_info_row(
                "Calisma alani",
                f"{workspace.get('files', 0)} dosya · {_fmt_size(workspace.get('bytes', 0))}",
                last=True,
            ))
        else:
            # Henuz egitim/tahmin yapilmamis hesapta dizin bos olabilir.
            rows.append(_info_row("Calisma alani", "Henuz dosya yok", last=True))

        if not persistent:
            rows.insert(0, dbc.Alert(
                "Kimlik dogrulama kapali (AUTH_ENABLED=False). Hesap kaydi yok; "
                "gorunum tercihi yalnizca bu tarayicida saklanir.",
                color="warning", className="py-2",
                style={"fontSize": "12px"},
            ))

        return (
            _initials(name),
            name,
            _role_badge(role),
            user.get("full_name", ""),
            not persistent,
            not persistent,
            user.get("email", "—"),
            html.Div(rows),
        )

    @app.callback(
        Output("account-alert", "children"),
        Output("account-tick", "data"),
        Input("account-name-save", "n_clicks"),
        State("account-name-input", "value"),
        State("account-tick", "data"),
        prevent_initial_call=True,
    )
    def save_name(n_clicks, value, tick):
        if not n_clicks:
            return no_update, no_update

        name = (value or "").strip()
        if not name:
            # Sunucu da reddeder (min_length=1); burada durmak bir tur ASGI
            # cagrisini ve "422" gorunumunu engelliyor.
            return dbc.Alert("Ad soyad bos olamaz.", color="danger",
                             className="py-2 mb-0"), no_update

        result = api.update_profile(name)
        if result.get("ok"):
            return (
                dbc.Alert("Profil guncellendi.", color="success",
                          className="py-2 mb-0", duration=4000),
                (tick or 0) + 1,
            )

        detail = (result.get("body") or {}).get("detail") or "Guncelleme basarisiz."
        return dbc.Alert(str(detail), color="danger", className="py-2 mb-0"), no_update

    @app.callback(
        Output("account-sessions-body", "children"),
        Input("account-tick", "data"),
    )
    def load_sessions(_tick):
        data = api.get_own_sessions() or {}
        sessions = data.get("sessions") or []

        if not sessions:
            return create_state_block(
                "empty", "Aktif oturum bilgisi yok",
                "Kimlik dogrulama kapaliyken oturum kaydi tutulmaz.",
            )

        rows = [_session_row(item, last=(i == len(sessions) - 1))
                for i, item in enumerate(sessions)]

        others = sum(1 for s in sessions if not s.get("current"))
        rows.append(
            dbc.Button(
                [html.I(className="bi bi-box-arrow-right me-1"),
                 "Diger oturumlari kapat"],
                id="account-revoke-btn",
                color="danger", outline=True, size="sm",
                disabled=others == 0,
                className="mt-3",
            )
        )
        if others == 0:
            rows.append(html.Small(
                "Baska aktif oturum yok.",
                style={"color": TEXT_MUTED, "display": "block", "marginTop": "6px"},
            ))
        return html.Div(rows)

    @app.callback(
        Output("account-activity-body", "children"),
        Input("account-tick", "data"),
    )
    def load_activity(_tick):
        entries = api.get_own_activity(20)
        if not entries:
            return create_state_block(
                "empty", "Kayitli etkinlik yok",
                "Girisler, parola degisiklikleri ve oturum islemleri burada listelenir.",
            )

        rows = [
            {
                "ts": _fmt_dt(entry.get("ts")),
                "action": ACTION_LABELS.get(entry.get("action", ""),
                                            entry.get("action", "")),
                "result": "Basarili" if entry.get("success") else "Basarisiz",
                "ip": entry.get("ip") or "—",
                "detail": _activity_detail(entry),
            }
            for entry in entries
        ]
        return DataTable(
            id="account-activity-table",
            columns=[
                {"name": "Zaman", "id": "ts"},
                {"name": "Olay", "id": "action"},
                {"name": "Sonuc", "id": "result"},
                {"name": "IP", "id": "ip"},
                {"name": "Ayrinti", "id": "detail"},
            ],
            data=rows,
            page_size=8,
            sort_action="native",
            # Basarisiz satir goze carpsin: renk burada ANLAM tasiyor
            # (Faz 8 C.3 kurali), dekoratif degil.
            style_data_conditional=[{
                "if": {"filter_query": '{result} = "Basarisiz"', "column_id": "result"},
                "color": "var(--rlt-loss)",
                "fontWeight": "600",
            }],
            **TABLE_STYLES,
        )

    @app.callback(
        Output("account-session-alert", "children"),
        Output("account-tick", "data", allow_duplicate=True),
        Input("account-revoke-btn", "n_clicks"),
        State("account-tick", "data"),
        prevent_initial_call=True,
    )
    def revoke_others(n_clicks, tick):
        if not n_clicks:
            return no_update, no_update

        result = api.revoke_other_sessions()
        if result.get("ok"):
            count = (result.get("body") or {}).get("revoked", 0)
            message = (f"{count} oturum kapatildi." if count
                       else "Kapatilacak baska oturum yoktu.")
            return (
                dbc.Alert(message, color="success", className="py-2 mb-0",
                          duration=4000),
                (tick or 0) + 1,
            )

        detail = (result.get("body") or {}).get("detail") or "Islem basarisiz."
        return dbc.Alert(str(detail), color="danger", className="py-2 mb-0"), no_update
