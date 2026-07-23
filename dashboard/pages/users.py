"""Kullanici yonetimi sayfasi (yalnizca admin).

Kullanilan uc noktalar:
  GET    /api/admin/users
  POST   /api/admin/users
  PATCH  /api/admin/users/{id}
  POST   /api/admin/users/{id}/password
  POST   /api/admin/users/{id}/revoke-sessions
  DELETE /api/admin/users/{id}
  GET    /api/admin/audit
"""

from dash import html, dcc, no_update
from dash import Input, Output, State, ALL, ctx
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable

import dashboard.api_client as api
from dashboard.theme import (
    CARD, CARD2, TEXT, TEXT_MUTED, BORDER, BLUE, GREEN, RED, YELLOW,
)

ROLE_OPTIONS = [
    {"label": "Yonetici (admin)", "value": "admin"},
    {"label": "Kullanici (user)", "value": "user"},
    {"label": "Izleyici (viewer)", "value": "viewer"},
]

_TABLE_STYLE = {
    "style_table": {"overflowX": "auto"},
    "style_header": {
        "backgroundColor": CARD2, "color": TEXT_MUTED, "fontWeight": "600",
        "fontSize": "12px", "textTransform": "uppercase", "border": f"1px solid {BORDER}",
    },
    "style_cell": {
        "backgroundColor": CARD, "color": TEXT, "border": f"1px solid {BORDER}",
        "fontSize": "13px", "padding": "8px", "textAlign": "left",
    },
}


def _fmt_size(num_bytes: int) -> str:
    size = float(num_bytes or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


# ── Layout ────────────────────────────────────────────────────────────────

def layout():
    return html.Div([
        dcc.Store(id="users-refresh-tick", data=0),

        html.H4("Kullanici Yonetimi", style={"color": TEXT, "marginBottom": "4px"}),
        html.P(
            "Hesaplari yonetici acar; kullanici ilk giriste parolasini degistirir. "
            "Her kullanicinin modelleri, sonuclari ve gunluk kararlari kendi calisma alaninda tutulur.",
            style={"color": TEXT_MUTED, "marginBottom": "24px"},
        ),

        html.Div(id="users-alert"),

        # Yeni kullanici
        dbc.Card([
            dbc.CardHeader(html.Span("Yeni Kullanici", style={"color": TEXT, "fontWeight": "600"})),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("E-posta", className="section-title"),
                        dbc.Input(id="users-new-email", type="email", placeholder="ad.soyad@sirket.com"),
                    ], md=4),
                    dbc.Col([
                        html.Label("Ad Soyad", className="section-title"),
                        dbc.Input(id="users-new-name", type="text", placeholder="Ad Soyad"),
                    ], md=3),
                    dbc.Col([
                        html.Label("Rol", className="section-title"),
                        dbc.Select(id="users-new-role", options=ROLE_OPTIONS, value="user"),
                    ], md=3),
                    dbc.Col([
                        dbc.Button([html.I(className="bi bi-person-plus me-1"), "Olustur"],
                                   id="users-create-btn", color="primary", className="w-100"),
                    ], md=2, className="d-flex align-items-end"),
                ], className="g-3"),
                html.Small(
                    "Parola otomatik uretilir ve yalnizca bir kez gosterilir.",
                    style={"color": TEXT_MUTED},
                ),
            ]),
        ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}", "marginBottom": "24px"}),

        # Kullanici listesi
        dbc.Card([
            dbc.CardHeader(
                dbc.Row([
                    dbc.Col(html.Span("Kullanicilar", style={"color": TEXT, "fontWeight": "600"})),
                    dbc.Col(
                        dbc.Button([html.I(className="bi bi-arrow-clockwise me-1"), "Yenile"],
                                   id="users-refresh-btn", size="sm", color="secondary", outline=True),
                        width="auto",
                    ),
                ], align="center", justify="between"),
            ),
            dbc.CardBody(html.Div(id="users-list")),
        ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}", "marginBottom": "24px"}),

        # Denetim kaydi
        dbc.Card([
            dbc.CardHeader(html.Span("Denetim Kaydi (son 100)", style={"color": TEXT, "fontWeight": "600"})),
            dbc.CardBody(DataTable(
                id="users-audit-table",
                columns=[
                    {"name": "Zaman", "id": "ts"},
                    {"name": "Kullanici", "id": "email"},
                    {"name": "Islem", "id": "action"},
                    {"name": "Hedef", "id": "target"},
                    {"name": "Sonuc", "id": "success"},
                    {"name": "IP", "id": "ip"},
                ],
                data=[],
                page_size=10,
                sort_action="native",
                **_TABLE_STYLE,
            )),
        ], style={"backgroundColor": CARD, "border": f"1px solid {CARD2}"}),
    ])


def _user_row(user: dict):
    role = user.get("role", "user")
    active = user.get("is_active", True)
    workspace = user.get("workspace") or {}
    uid = user["id"]

    badges = [
        dbc.Badge(role, color={"admin": "success", "user": "primary"}.get(role, "secondary"),
                  className="me-2"),
    ]
    if not active:
        badges.append(dbc.Badge("pasif", color="danger", className="me-2"))
    if user.get("must_change_password"):
        badges.append(dbc.Badge("parola degistirmeli", color="warning", className="me-2"))

    return dbc.Row([
        dbc.Col([
            html.Div(user.get("email", ""), style={"color": TEXT, "fontWeight": "600"}),
            html.Small(user.get("full_name") or "-", style={"color": TEXT_MUTED}),
        ], md=3),
        dbc.Col(html.Div(badges), md=2),
        dbc.Col([
            html.Small(
                f"Son giris: {(user.get('last_login_at') or '-')[:19].replace('T', ' ')}",
                style={"color": TEXT_MUTED, "display": "block"},
            ),
            html.Small(
                f"Calisma alani: {_fmt_size(workspace.get('bytes', 0))} / {workspace.get('files', 0)} dosya",
                style={"color": TEXT_MUTED},
            ),
        ], md=3),
        dbc.Col([
            dbc.Select(
                id={"type": "users-role-select", "id": uid},
                options=ROLE_OPTIONS, value=role, size="sm",
            ),
        ], md=2),
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button(html.I(className="bi bi-check2-circle"),
                           id={"type": "users-save-btn", "id": uid},
                           size="sm", color="primary", outline=True, title="Rolu kaydet"),
                dbc.Button(html.I(className="bi bi-key"),
                           id={"type": "users-reset-btn", "id": uid},
                           size="sm", color="warning", outline=True, title="Parola sifirla"),
                dbc.Button(html.I(className="bi bi-slash-circle" if active else "bi bi-check-circle"),
                           id={"type": "users-toggle-btn", "id": uid},
                           size="sm", color="secondary", outline=True,
                           title="Pasiflestir" if active else "Aktiflestir"),
                dbc.Button(html.I(className="bi bi-trash"),
                           id={"type": "users-delete-btn", "id": uid},
                           size="sm", color="danger", outline=True, title="Sil"),
            ], size="sm"),
        ], md=2, className="d-flex align-items-center justify-content-end"),
    ], className="g-2 py-3", style={"borderBottom": f"1px solid {BORDER}"}, align="center")


def _alert(message: str, color: str = "danger"):
    return dbc.Alert(message, color=color, dismissable=True, duration=None if color == "success" else 8000)


# ── Callbacks ─────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("users-list", "children"),
        Output("users-audit-table", "data"),
        Input("users-refresh-btn", "n_clicks"),
        Input("users-refresh-tick", "data"),
    )
    def _load(_clicks, _tick):
        users = api.list_users()
        if not users:
            body = html.Div("Kullanici bulunamadi veya yetkiniz yok.",
                            style={"color": TEXT_MUTED, "padding": "12px"})
        else:
            body = html.Div([_user_row(u) for u in users])

        audit_rows = [
            {
                "ts": (row.get("ts") or "")[:19].replace("T", " "),
                "email": row.get("email", ""),
                "action": row.get("action", ""),
                "target": row.get("target", ""),
                "success": "OK" if row.get("success") else "HATA",
                "ip": row.get("ip", ""),
            }
            for row in api.get_audit_log(100)
        ]
        return body, audit_rows

    @app.callback(
        Output("users-alert", "children"),
        Output("users-refresh-tick", "data"),
        Output("users-new-email", "value"),
        Output("users-new-name", "value"),
        Input("users-create-btn", "n_clicks"),
        State("users-new-email", "value"),
        State("users-new-name", "value"),
        State("users-new-role", "value"),
        State("users-refresh-tick", "data"),
        prevent_initial_call=True,
    )
    def _create(_clicks, email, full_name, role, tick):
        if not email:
            return _alert("E-posta zorunlu"), no_update, no_update, no_update

        result = api.create_user({"email": email, "full_name": full_name or "", "role": role})
        if not result.get("ok"):
            detail = result.get("body", {}).get("detail", "Bilinmeyen hata")
            if isinstance(detail, list):  # pydantic dogrulama hatasi
                detail = "; ".join(str(item.get("msg", item)) for item in detail)
            return _alert(f"Olusturulamadi: {detail}"), no_update, no_update, no_update

        body = result.get("body", {})
        temp = body.get("temporary_password", "")
        message = [
            html.Div(f"{body.get('user', {}).get('email')} olusturuldu."),
            html.Div([
                "Gecici parola: ",
                html.Code(temp, style={"fontSize": "14px"}),
                html.Span(" — bu parola bir daha gosterilmez, kullaniciya guvenli sekilde iletin.",
                          style={"opacity": .85}),
            ]) if temp else None,
        ]
        return _alert(message, "success"), (tick or 0) + 1, "", ""

    @app.callback(
        Output("users-alert", "children", allow_duplicate=True),
        Output("users-refresh-tick", "data", allow_duplicate=True),
        Input({"type": "users-save-btn", "id": ALL}, "n_clicks"),
        Input({"type": "users-reset-btn", "id": ALL}, "n_clicks"),
        Input({"type": "users-toggle-btn", "id": ALL}, "n_clicks"),
        Input({"type": "users-delete-btn", "id": ALL}, "n_clicks"),
        State({"type": "users-role-select", "id": ALL}, "value"),
        State({"type": "users-role-select", "id": ALL}, "id"),
        State("users-refresh-tick", "data"),
        prevent_initial_call=True,
    )
    def _actions(_save, _reset, _toggle, _delete, role_values, role_ids, tick):
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or not ctx.triggered[0]["value"]:
            return no_update, no_update

        user_id = triggered["id"]
        action = triggered["type"]
        roles = {item["id"]: value for item, value in zip(role_ids, role_values)}
        next_tick = (tick or 0) + 1

        if action == "users-save-btn":
            result = api.update_user(user_id, {"role": roles.get(user_id)})
            if not result:
                return _alert("Rol guncellenemedi (son admin devre disi birakilamaz olabilir)"), next_tick
            return _alert(f"Rol guncellendi: {result.get('user', {}).get('email')}", "success"), next_tick

        if action == "users-reset-btn":
            result = api.reset_user_password(user_id)
            if not result.get("ok"):
                return _alert(f"Sifirlanamadi: {result.get('body', {}).get('detail')}"), next_tick
            temp = result.get("body", {}).get("temporary_password", "")
            return _alert([
                "Parola sifirlandi. Gecici parola: ", html.Code(temp),
                " — tum aktif oturumlari kapatildi.",
            ], "success"), next_tick

        if action == "users-toggle-btn":
            users = {u["id"]: u for u in api.list_users()}
            current = users.get(user_id, {}).get("is_active", True)
            result = api.update_user(user_id, {"is_active": not current})
            if not result:
                return _alert("Durum degistirilemedi (sistemdeki son admin olabilir)"), next_tick
            state = "aktif" if not current else "pasif"
            return _alert(f"Hesap {state} yapildi.", "success"), next_tick

        if action == "users-delete-btn":
            result = api.delete_user(user_id)
            if not result:
                return _alert("Silinemedi (kendi hesabiniz veya son admin olabilir)"), next_tick
            return _alert(
                f"{result.get('email')} silindi. Calisma alani dosyalari korundu: "
                f"{result.get('workspace_kept')}", "success",
            ), next_tick

        return no_update, no_update
