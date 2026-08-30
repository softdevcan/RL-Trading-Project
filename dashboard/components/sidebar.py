"""Sol kenar cubugu (Faz 8, C.8 + B.4).

Menu uc gruba ayrildi; duz bir 8 maddelik liste yerine ne aradigini bilen
kullanicinin dogrudan bulabilecegi bir yapi. Alt kisimda kullanici rozeti,
gorunum anahtari ve cikis.
"""

from dash import html
import dash_bootstrap_components as dbc

from dashboard.auth_context import current_user, display_name, is_admin
from dashboard.theme import BORDER, TEXT, TEXT_MUTED, SIDEBAR_STYLE

# Gruplar sirayla cizilir; admin grubu yalnizca admin icin eklenir.
NAV_GROUPS = [
    ("Analiz", [
        {"label": "Dashboard", "icon": "bi bi-speedometer2", "href": "/dash/"},
        {"label": "Modeller", "icon": "bi bi-diagram-3", "href": "/dash/models"},
        {"label": "Akademik", "icon": "bi bi-journal-bookmark", "href": "/dash/academic"},
    ]),
    ("Islem", [
        {"label": "Trading", "icon": "bi bi-graph-up-arrow", "href": "/dash/daily-trading"},
        {"label": "Tahmin", "icon": "bi bi-lightning-charge", "href": "/dash/prediction"},
    ]),
    ("Sistem", [
        {"label": "Egitim", "icon": "bi bi-cpu", "href": "/dash/training"},
        {"label": "Veri", "icon": "bi bi-database", "href": "/dash/data"},
        {"label": "HiperParam", "icon": "bi bi-sliders", "href": "/dash/hyperopt"},
    ]),
]

ADMIN_GROUP = ("Yonetim", [
    {"label": "Kullanicilar", "icon": "bi bi-people", "href": "/dash/users"},
])

ROLE_LABELS = {"admin": "Yonetici", "user": "Kullanici", "viewer": "Izleyici"}


def _nav_link(item: dict) -> dbc.NavLink:
    return dbc.NavLink(
        [html.I(className=f"{item['icon']} me-2"), item["label"]],
        href=item["href"],
        active="exact",
        className="sidebar-link",
        style={"padding": "9px 17px", "margin": "1px 8px"},
    )


def _theme_button() -> html.Button:
    """Uc durumu sirayla dolasan gorunum anahtari.

    Etiket ve ikon assets/theme-toggle.js tarafindan acilista ve her
    tiklamada guncellenir; buradaki degerler yalnizca ilk cizimde
    (script calismadan once) gorunen yer tutuculardir.
    """
    return html.Button(
        [
            html.I(className="bi bi-circle-half"),
            html.Span("Sistem", className="theme-label ms-2"),
        ],
        id="theme-toggle",
        type="button",
        title="Gorunum tercihini degistir",
        **{"data-theme-set": "__cycle__"},
    )


def _initials(name: str) -> str:
    """Avatar icin en fazla iki harf. E-posta gelirse ilk harfi kullanilir."""
    parts = [p for p in (name or "").replace(".", " ").split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _account_link(user: dict) -> html.Div:
    """Hesabim'a giden acikca tiklanabilir satir.

    Onceki hali dusuk gorunurluktendi: yalnizca ad, `textDecoration: none`
    ile duz metin gibi duruyordu — link oldugu ancak uzerine gelinince
    anlasiliyordu. Simdi avatar + ad + rol + chevron, hover zemini ve
    `active="exact"` ile aktif sayfa vurgusu var (dbc aktif durumu
    dcc.Location'dan cozer, sunucu turu gerekmez).

    Auth KAPALIYKEN de cizilir ("Misafir"): o modda hesap yok ama Hesabim
    sayfasindaki gorunum tercihi calisiyor ve oraya baska giris yolu
    kalmiyordu. Yan fayda: `sidebar-account-name`/`-avatar` kimlikleri her
    zaman DOM'da, boylece ad guncelleme callback'i var olmayan bir bilesene
    yazmaya calismiyor.

    `title` sarmalayan Div'de: dbc.NavLink yalnizca sayili prop kabul ediyor
    (active/href/target/...), `title` verilince tum Dash agaci render
    edilemiyor — /dash/ 500 donuyordu. Sarmalayici ayni alani kapladigi icin
    ipucu davranisi degismiyor.
    """
    name = display_name()
    role = user.get("role")
    subtitle = ROLE_LABELS.get(role, role) if role else "Kimlik dogrulama kapali"
    email = user.get("email", "")
    return html.Div(
        dbc.NavLink(
            [
                html.Span(_initials(name), id="sidebar-account-avatar",
                          className="account-avatar"),
                html.Span(
                    [
                        html.Span(name, id="sidebar-account-name",
                                  className="account-name"),
                        html.Span(subtitle, className="account-role"),
                    ],
                    className="account-text",
                ),
                html.I(className="bi bi-chevron-right account-chevron"),
            ],
            href="/dash/account",
            active="exact",
            className="sidebar-account",
        ),
        title=(f"{email} - Hesabim, gorunum ve guvenlik" if email
               else "Hesabim - gorunum ayarlari"),
    )


def _user_footer():
    """Hesap satiri + gorunum anahtari + cikis.

    Auth kapaliyken hesap satiri gosterilmez ama gorunum anahtari kalir —
    tema tercihi kimlik dogrulamadan bagimsiz calisir (cerezde saklanir).
    """
    user = current_user()

    # Hesap satiri her durumda cizilir; auth kapaliyken "Misafir" olur.
    rows = [_account_link(user or {})]

    controls = [_theme_button()]
    if user:
        controls.append(
            html.A(
                [html.I(className="bi bi-box-arrow-right me-1"), "Cikis"],
                href="/logout",
                className="sidebar-logout",
            )
        )
    rows.append(html.Div(controls, className="sidebar-controls"))

    return html.Div(rows, className="sidebar-footer")


def create_sidebar():
    """Kenar cubugunu dondur.

    Not: Kullanici rozeti ve admin grubu role gore degistiginden layout,
    `app.py` icinde her sayfa yuklemesinde yeniden uretilir (callable layout).
    """
    groups = list(NAV_GROUPS) + ([ADMIN_GROUP] if is_admin() else [])

    nav_children = []
    for title, items in groups:
        nav_children.append(html.Div(title, className="sidebar-group"))
        nav_children.extend(_nav_link(item) for item in items)

    return html.Div(
        [
            # Marka
            html.Div(
                [
                    html.Div(
                        [
                            html.I(className="bi bi-robot me-2",
                                   style={"color": "var(--primary)", "fontSize": "19px"}),
                            html.Span("RL Trading",
                                      style={"color": TEXT, "fontWeight": "700",
                                             "fontSize": "16px"}),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    html.Small("by softdevcan",
                               style={"color": TEXT_MUTED, "fontSize": "11px"}),
                ],
                style={
                    "padding": "20px 20px 16px",
                    "borderBottom": f"1px solid {BORDER}",
                    "marginBottom": "4px",
                },
            ),
            dbc.Nav(nav_children, vertical=True, pills=True),
            _user_footer(),
        ],
        style={**SIDEBAR_STYLE, "paddingBottom": "150px"},
        id="sidebar",
    )
