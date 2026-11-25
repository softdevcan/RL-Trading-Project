"""Sayfa basligi bileseni (Faz 8, C.2).

Faz 8 oncesi 9 sayfanin her biri ayni blogu elle yaziyordu:

    html.H4("...", style={"color": TEXT, "marginBottom": "4px"}),
    html.P("...", style={"color": TEXT_MUTED, "margin": 0}),

Baslik boyutu, aralik ve renk sayfadan sayfaya kayiyordu. Tek yer burasi.
"""

from dash import html


# Faz 11 §3.15: tek ajanli (Faz 1-9) sayfalarin kapsam notu; M8/MARL sonuclari baska sayfalarda.
LEGACY_NOTE = ("Bu sayfa tek ajanlı RL modelleri içindir (Faz 1–9; günlük karar ve kâğıt portföy bunları "
               "kullanır). Çok ajanlı nihai test (M8) ve sanal para simülasyonu: Dashboard, Simülasyon, Rehber.")


def create_page_header(title: str, subtitle: str = "", actions=None, note=None) -> html.Div:
    """Sayfa basligi: solda baslik + alt metin, sagda eylemler.

    Parameters
    ----------
    title:    Sayfa adi (20px / 600 — bkz. custom.css `.page-title`)
    subtitle: Bir cumlelik aciklama. Bos birakilirsa satir hic uretilmez.
    actions:  Sagda gosterilecek bilesen(ler): rozet, dugme, `html.Div(id=...)`.
              Callback ile doldurulacaksa bos bir Div verip id'sini kullan.
    note:     Baslik altinda tek satirlik kapsam notu (Faz 11 §3.15): sayfanin sistemin hangi
              kismina ait oldugu (or. tek ajanli Faz 1-9 modelleri). Metin ya da bilesen listesi.
    """
    left = [html.H1(title, className="page-title")]
    if subtitle:
        left.append(html.P(subtitle, className="page-subtitle"))
    if note:
        left.append(html.P([html.I(className="bi bi-info-circle me-1"), *(note if isinstance(note, list) else [note])],
                           className="page-scope-note"))

    children = [html.Div(left, style={"minWidth": 0})]
    if actions is not None:
        children.append(
            html.Div(
                actions,
                style={"display": "flex", "alignItems": "center",
                       "gap": "8px", "flexShrink": 0},
            )
        )

    return html.Div(children, className="page-header")
