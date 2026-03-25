"""Reusable KPI metric card component."""

from dash import html
import dash_bootstrap_components as dbc

from dashboard.theme import CARD, CARD2, TEXT, TEXT_MUTED, GREEN, RED


def create_metric_card(
    title: str,
    value: str = "—",
    subtitle: str = "",
    color: str = TEXT,
    icon: str = "bi bi-bar-chart",
    card_id: str = None,
) -> dbc.Card:
    """
    Return a styled KPI metric card.

    Parameters
    ----------
    title:    Card label (small text, top)
    value:    Large displayed value
    subtitle: Optional footnote/trend text below value
    color:    CSS color string for the value text
    icon:     Bootstrap icon class
    card_id:  Optional id dict for the value element, e.g. {"type": "metric", "index": "sharpe"}
    """
    value_props = {"style": {"color": color, "fontSize": "28px", "fontWeight": "700", "lineHeight": "1"}}
    if card_id:
        value_props["id"] = card_id

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Small(title, style={"color": TEXT_MUTED, "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
                        html.I(className=f"{icon}", style={"color": TEXT_MUTED, "fontSize": "14px"}),
                    ],
                    style={"display": "flex", "justifyContent": "space-between", "marginBottom": "8px"},
                ),
                html.Div(value, **value_props),
                html.Small(subtitle, style={"color": TEXT_MUTED, "fontSize": "12px"}) if subtitle else html.Span(),
            ]
        ),
        style={
            "backgroundColor": CARD,
            "border": f"1px solid {CARD2}",
            "borderRadius": "8px",
        },
        className="h-100",
    )
