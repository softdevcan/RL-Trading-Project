"""Color constants and Plotly dark template for RL Trading Dashboard."""

# --- Color palette ---------------------------------------------------------
BG = "#0f172a"          # Page background (slate-900)
CARD = "#1e293b"        # Card surface (slate-800)
CARD2 = "#334155"       # Secondary surface (slate-700)
TEXT = "#e2e8f0"        # Primary text (slate-200)
TEXT_MUTED = "#94a3b8"  # Muted text (slate-400)
BORDER = "#475569"      # Border (slate-600)

# Accent colors
GREEN = "#22c55e"       # Profit / success
RED = "#ef4444"         # Loss / danger
BLUE = "#3b82f6"        # Info / primary
YELLOW = "#eab308"      # Warning
PURPLE = "#a855f7"      # Secondary accent
CYAN = "#06b6d4"        # Tertiary accent
ORANGE = "#f97316"      # Warning accent
GOLD = "#f59e0b"        # Gold price

# Algorithm colors
ALGO_COLORS = {
    "PPO": BLUE,
    "A2C": GREEN,
    "SAC": PURPLE,
    "TD3": ORANGE,
}

# Sidebar styling
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "220px",
    "padding": "0",
    "backgroundColor": CARD,
    "borderRight": f"1px solid {BORDER}",
    "overflowY": "auto",
    "zIndex": 1000,
}

CONTENT_STYLE = {
    "marginLeft": "220px",
    "padding": "24px",
    "backgroundColor": BG,
    "minHeight": "100vh",
}

# --- Plotly dark template ---------------------------------------------------
DARK_TEMPLATE = {
    "layout": {
        "paper_bgcolor": CARD,
        "plot_bgcolor": CARD,
        "font": {"color": TEXT, "family": "Inter, system-ui, sans-serif"},
        "title": {"font": {"color": TEXT}},
        "xaxis": {
            "gridcolor": CARD2,
            "linecolor": BORDER,
            "zerolinecolor": BORDER,
            "tickcolor": TEXT_MUTED,
            "tickfont": {"color": TEXT_MUTED},
        },
        "yaxis": {
            "gridcolor": CARD2,
            "linecolor": BORDER,
            "zerolinecolor": BORDER,
            "tickcolor": TEXT_MUTED,
            "tickfont": {"color": TEXT_MUTED},
        },
        "legend": {
            "bgcolor": CARD2,
            "bordercolor": BORDER,
            "font": {"color": TEXT},
        },
        "hoverlabel": {
            "bgcolor": CARD2,
            "bordercolor": BORDER,
            "font": {"color": TEXT},
        },
        "margin": {"l": 50, "r": 20, "t": 40, "b": 50},
        "colorway": [BLUE, GREEN, PURPLE, ORANGE, CYAN, YELLOW, RED],
    }
}


def apply_dark_template(fig):
    """Apply dark template and return figure."""
    import plotly.graph_objects as go
    fig.update_layout(**DARK_TEMPLATE["layout"])
    return fig


def empty_figure(message="Veri yok"):
    """Return an empty figure with a centered message."""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font={"color": TEXT_MUTED, "size": 14},
    )
    apply_dark_template(fig)
    return fig
