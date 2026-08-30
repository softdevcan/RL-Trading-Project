"""Yukleniyor / bos / hata durum blogu (Faz 8, C.7).

Faz 8 oncesi her sayfa kendi satirini yaziyordu:
`html.P("Yukleniyor...", style={"color": TEXT_MUTED})`. Ikon, hizalama ve
bosluk sayfadan sayfaya farkliydi; hata durumu cogu yerde hic yoktu.
"""

from dash import html

_KINDS = {
    "loading": ("bi bi-hourglass-split", "Yukleniyor..."),
    "empty": ("bi bi-inbox", "Kayit yok"),
    "error": ("bi bi-exclamation-triangle", "Bir hata olustu"),
}


def create_state_block(kind: str = "empty", message: str = "", hint: str = "") -> html.Div:
    """Bos/yukleniyor/hata durumu icin ortalanmis blok.

    Parameters
    ----------
    kind:    "loading" | "empty" | "error". Bilinmeyen deger "empty" sayilir.
    message: Ana metin. Verilmezse turun varsayilan metni kullanilir.
    hint:    Altta kucuk yardim metni — kullanicinin ne yapacagini soyler.
    """
    icon, default_message = _KINDS.get(kind, _KINDS["empty"])

    children = [
        html.I(className=icon),
        html.Div(message or default_message),
    ]
    if hint:
        children.append(html.Small(hint, style={"opacity": ".85"}))

    css = "state-block" + (" state-error" if kind == "error" else "")
    return html.Div(children, className=css)
