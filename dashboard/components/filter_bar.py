"""Filtre satiri bileseni (Faz 8, C.5).

Kalip: etiket USTTE (11px, `--muted`), kontrol altta, kontroller tek satirda
esit araliklarla, hepsi tek bir kart icinde. Faz 8 oncesi her sayfa kendi
`dbc.Row`/`dbc.Col` yerlesimini kuruyordu; etiketler kimi yerde solda, kimi
yerde ustte, bosluklar tutarsizdi.
"""

from dash import html
import dash_bootstrap_components as dbc


def create_filter_field(label: str, control, width: int | None = None, **col_kwargs):
    """Tek bir filtre alani: ustte etiket, altta kontrol.

    `width` verilirse `md` sutun genisligi olarak kullanilir; verilmezse
    alanlar esit paylasir.
    """
    kwargs = dict(col_kwargs)
    if width is not None:
        kwargs.setdefault("md", width)

    return dbc.Col(
        [
            html.Label(label, className="field-label"),
            control,
        ],
        **kwargs,
    )


def create_filter_bar(fields, actions=None, card: bool = True):
    """Filtre satiri.

    Parameters
    ----------
    fields:  `create_filter_field` ciktilarindan olusan liste.
    actions: Sagda yer alacak dugme(ler). Etiket hizasini korumasi icin
             gorunmez bir etiketle asagi hizalanir.
    card:    False verilirse kart sarmalayicisi olmadan yalnizca satir doner
             (baska bir kartin icine gomulecekse).
    """
    children = list(fields)

    if actions is not None:
        children.append(
            dbc.Col(
                [
                    # Etiket yuksekligi kadar bosluk — dugme kontrollerle hizalansin
                    html.Label("​", className="field-label", **{"aria-hidden": "true"}),
                    html.Div(actions, style={"display": "flex", "gap": "8px"}),
                ],
                md="auto",
                className="ms-auto",
            )
        )

    row = dbc.Row(children, className="g-3 align-items-end")

    if not card:
        return row
    return dbc.Card(dbc.CardBody(row), className="mb-4")
