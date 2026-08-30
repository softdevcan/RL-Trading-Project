"""Ust cubuk — sayfa baglami + arama + gorunum anahtari (Faz 8, G).

Neden var
---------
Kenar cubugu "nereye gidebilirim"i anlatiyor; ust cubuk "neredeyim ve buradan
ne yapabilirim"i. Faz 8/F sonrasi tema anahtari kenar cubugunun dibinde,
menunun altinda kaliyordu — sik kullanilan bir kontrol icin yanlis yer.

Yerlesim: kenar cubugu tam boy kalir (marka ustte), ust cubuk YALNIZCA icerik
alanini kaplar (`left: SIDEBAR_WIDTH`). Boylece marka tek yerde durur ve
mevcut sabit kenar cubugu yerlesimi bozulmaz.

Icerik
------
- **Sol**: kirinti (grup > sayfa). Sayfa basligini TEKRARLAMAZ — `PageHeader`
  zaten baslik + alt metni veriyor; buradaki katki hangi grupta oldugun.
- **Sag**: baglamsal eylem, arama, gorunum anahtari.

Baglamsal eylem nasil seciliyor
-------------------------------
Rastgele kisayol degil: `docs/development/rl-stability-portfolio-analysis.md`
icinde belgelenen arayuz akisi (veri indir -> egit -> analiz) izleniyor.
Her sayfa icin "bu isten sonra sirada ne var" baglantisi konuyor; akista
karsiligi olmayan sayfada (Hesabim, Kullanicilar) slot bos kalir.
"""

from dash import html, dcc, no_update
from dash import Input, Output, State

from dashboard.auth_context import is_admin
from dashboard.theme import TOPBAR_STYLE
from dashboard.components.sidebar import ADMIN_GROUP, NAV_GROUPS

# Yol -> (grup, sayfa adi). Kenar cubugunun menu modelinden uretilir ki
# iki yerde ayri liste tutulmasin.
def _route_index() -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    for group, items in list(NAV_GROUPS) + [ADMIN_GROUP]:
        for item in items:
            index[item["href"]] = (group, item["label"])
    # Menude yer almayan ama gezilebilen sayfa
    index["/dash/account"] = ("Hesap", "Hesabim")
    return index


ROUTE_INDEX = _route_index()

# Belgelenen akis: veri indir -> egit -> analiz. Slot bos kalabilir.
NEXT_STEP = {
    "/dash/": ("Veri durumu", "bi bi-database", "/dash/data"),
    "/dash/data": ("Egitime gec", "bi bi-cpu", "/dash/training"),
    "/dash/training": ("Modelleri karsilastir", "bi bi-diagram-3", "/dash/models"),
    "/dash/models": ("Gunluk karar", "bi bi-graph-up-arrow", "/dash/daily-trading"),
    "/dash/hyperopt": ("Egitime gec", "bi bi-cpu", "/dash/training"),
    "/dash/prediction": ("Hiper parametre", "bi bi-sliders", "/dash/hyperopt"),
    "/dash/daily-trading": ("Akademik analiz", "bi bi-journal-bookmark", "/dash/academic"),
}


def _search_options() -> list[dict]:
    """Komut paleti secenekleri: gidilebilen her sayfa.

    Admin grubu yalnizca admin icin eklenir — arama kutusu yetkisi olmayan
    bir sayfayi onermemeli (rota zaten korumali, ama oneri de yaniltmamali).
    """
    groups = list(NAV_GROUPS) + ([ADMIN_GROUP] if is_admin() else [])
    options = [
        {"label": f"{group} · {item['label']}", "value": item["href"]}
        for group, items in groups
        for item in items
    ]
    options.append({"label": "Hesap · Hesabim", "value": "/dash/account"})
    return options


def create_topbar() -> html.Div:
    """Ust cubugu dondur. `app.py::serve_layout` her sayfa yuklemesinde cagirir."""
    return html.Div(
        [
            html.Div(id="topbar-crumb", className="topbar-crumb"),
            html.Div(className="topbar-spacer"),
            html.Div(id="topbar-actions", className="topbar-actions"),
            dcc.Dropdown(
                id="topbar-search",
                options=_search_options(),
                placeholder="Sayfa ara...",
                className="topbar-search",
                clearable=False,
                optionHeight=34,
            ),
            _theme_button(),
        ],
        className="topbar",
        id="topbar",
        style=TOPBAR_STYLE,
    )


def _theme_button() -> html.Button:
    """Uc durumu sirayla dolasan gorunum anahtari.

    Faz 8/B.4'te kenar cubugunun dibindeydi; ust cubuga tasindi (tek kopya,
    cogaltilmadi). Etiket ve ikonu `assets/theme-toggle.js` acilista ve her
    tiklamada gunceller; buradakiler yalnizca script calismadan onceki yer
    tutuculardir. JS elemani `getElementById` ile buldugu icin tasima onu
    etkilemiyor.
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


def register_callbacks(app):
    @app.callback(
        Output("topbar-crumb", "children"),
        Output("topbar-actions", "children"),
        Input("url", "pathname"),
    )
    def update_context(pathname: str):
        path = pathname or "/dash/"
        group, label = ROUTE_INDEX.get(path, ("", ""))

        crumb = []
        if group:
            crumb = [
                html.Span(group, className="topbar-group"),
                html.I(className="bi bi-chevron-right topbar-sep"),
                html.Span(label, className="topbar-page"),
            ]

        action = NEXT_STEP.get(path)
        actions = []
        if action:
            text, icon, href = action
            actions = [
                html.A(
                    [html.I(className=f"{icon} me-2"), text],
                    href=href,
                    className="btn btn-outline-secondary btn-sm topbar-next",
                )
            ]
        return crumb, actions

    @app.callback(
        Output("url", "pathname"),
        Output("topbar-search", "value"),
        Input("topbar-search", "value"),
        State("url", "pathname"),
        prevent_initial_call=True,
    )
    def go_to_page(target, current):
        """Aramadan secilen sayfaya git ve kutuyu bosalt.

        Kutu bosaltilmazsa ayni sayfa ikinci kez secilemez (deger degismedigi
        icin callback tetiklenmez). Bosaltma callback'i tekrar tetikler; None
        gelen tur hicbir sey yapmadan cikar.
        """
        if not target or target == current:
            # no_update: ayni degeri geri yazmak display_page'i bosuna
            # tetikleyip sayfayi yeniden cizerdi.
            return no_update, None
        return target, None
