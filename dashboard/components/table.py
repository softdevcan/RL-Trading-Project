"""DataTable stil sozlukleri (Faz 8, C.6).

Faz 8 oncesi her sayfa kendi `style_header` / `style_cell` sozlugunu tasiyordu;
satir yuksekligi, yazi boyutu ve kenarlik sayfadan sayfaya kayiyordu.

Kararlar:
  - Zebra serit YOK. Satirlari tek bir ince cizgi ayirir; zebra uzun
    tablolarda okumayi kolaylastirmaz, gorsel gurultu ekler.
  - Sayisal sutunlar SAGA DAYALI ve `tabular-nums` — basamaklar alt alta
    hizalanmazsa iki sayiyi karsilastirmak gozle mumkun olmuyor.
  - Renkler token; koyu/aydinlik ayrimi burada yok.

Kullanim:

    from dashboard.components.table import TABLE_STYLES, numeric_columns

    DataTable(
        columns=[...],
        data=rows,
        **TABLE_STYLES,
        style_cell_conditional=numeric_columns("getiri", "sharpe"),
    )
"""

# DataTable inline stil ister; CSS sinifi yeterli degil. Degerler yine de
# token — tarayici cozer, tema degisince kendiliginden guncellenir.
TABLE_STYLES = {
    "style_table": {"overflowX": "auto"},
    "style_as_list_view": True,          # dikey cizgileri kaldirir
    "style_header": {
        "backgroundColor": "var(--surface-2)",
        "color": "var(--muted)",
        "fontWeight": "600",
        "fontSize": "11px",
        "textTransform": "uppercase",
        "letterSpacing": "0.06em",
        "border": "none",
        "borderBottom": "1px solid var(--border-strong)",
        "padding": "10px 12px",
    },
    "style_cell": {
        "backgroundColor": "var(--surface)",
        "color": "var(--text)",
        "border": "none",
        "borderBottom": "1px solid var(--border)",
        "fontSize": "13px",
        "height": "38px",
        "padding": "0 12px",
        "textAlign": "left",
        "fontVariantNumeric": "tabular-nums",
        "fontFamily": "inherit",
    },
    "style_data": {
        "backgroundColor": "var(--surface)",
        "color": "var(--text)",
    },
}


def numeric_columns(*column_ids: str) -> list[dict]:
    """Verilen sutunlari saga dayali yapar (C.6).

    Sayisal sutunlar sola dayali kaldiginda basamak sayisi degisen degerler
    hizalanmiyor ve tablo okunmuyor.
    """
    return [
        {"if": {"column_id": cid}, "textAlign": "right"}
        for cid in column_ids
    ]


def tone_rules(column_id: str) -> list[dict]:
    """Bir sutundaki negatif degerleri kirmizi, pozitifleri yesil yapar.

    Renk yalnizca yon tasiyan sutunlarda kullanilmali (getiri, degisim);
    her sutunu renklendirmek anlami yok eder.
    """
    return [
        {
            "if": {"filter_query": f"{{{column_id}}} < 0", "column_id": column_id},
            "color": "var(--loss)",
        },
        {
            "if": {"filter_query": f"{{{column_id}}} > 0", "column_id": column_id},
            "color": "var(--profit)",
        },
    ]
