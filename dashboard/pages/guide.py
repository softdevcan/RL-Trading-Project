"""
Rehber sayfasi (K28 R2; plan docs/development/system-guide-plan.md §5): sistem ne yapar, veri nasil akar, ne degildir,
sozluk aramasi. YALNIZ OKUR: API ucu yok, yoklama (dcc.Interval) yok; giris yapmis her rol gorur.

Icerik tek kaynak `dashboard/content/guide.yaml` + `glossary.yaml` (imaja girer; calisma aninda `docs/` YOK, .dockerignore).
`docs/sozluk.md` ve `snapshot.json` ayni kaynaklardan `scripts/build_guide.py` ile uretilir. Bolum 3 (asama cizelgesi,
system-map.md'den) ve 4 (deney kartlari: etiket + en fazla 2 sayi, her biri kaynak JSON + alan) YALNIZ snapshot.json'dan
okunur; sayfa olcum dosyasi acmaz. SSS (bolum 7) R4'te. Terim capasi `#terim-<id>` (sayfa bilesen id'leriyle cakismasin diye onekli).
"""
from __future__ import annotations

import os
from functools import lru_cache

import dash_bootstrap_components as dbc
import json

import yaml
from dash import Input, Output, dcc, html

from dashboard.components.page_header import create_page_header
from dashboard.components.state_block import create_state_block

CONTENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "content")
# scripts/build_guide.py::CATEGORIES ile ayni (test_guide [7] esitligi denetler; scripts/ imaja girmez)
CATEGORIES = {
    "sistem": "Sistem ve aşamalar",
    "deney": "Deney kodları ve etiketler",
    "degerlendirme": "Değerlendirme ve istatistik",
    "marl": "Çok ajanlı öğrenme",
    "rl": "Tek ajan ve genel pekiştirmeli öğrenme",
    "tahmin": "Tahmin hattı",
    "veri": "Veri",
    "yazilim": "Yazılım ve ürün",
}
ALL = "hepsi"
_FOLD = str.maketrans("ıİIşŞğĞüÜöÖçÇâÂîÎûÛ", "iiissgguuoocc" + "aaiiuu")


@lru_cache(maxsize=1)
def load_content() -> tuple[dict, list]:
    """(guide, glossary). Icerik imajla gelir; degisince yeniden derleme gerekir, onbellek bu yuzden kalicidir."""
    with open(os.path.join(CONTENT_DIR, "guide.yaml"), encoding="utf-8") as fh:
        guide = yaml.safe_load(fh)
    with open(os.path.join(CONTENT_DIR, "glossary.yaml"), encoding="utf-8") as fh:
        glossary = yaml.safe_load(fh)
    return guide, glossary


@lru_cache(maxsize=1)
def load_snapshot() -> dict:
    with open(os.path.join(CONTENT_DIR, "snapshot.json"), encoding="utf-8") as fh:
        return json.load(fh)


STAGE_GROUPS = (("suruyor", "Sürüyor", "bi bi-play-circle"), ("bitti", "Bitti", "bi bi-check-circle"),
                ("planli", "Planlı ya da açık", "bi bi-circle"))
LABEL_CLASS = {"GEÇTİ": "gecti", "KISMİ": "kismi", "GÖZLENMEDİ": "gozlenmedi", "PİLOT_İŞARETİ": "pilot",
               "GEÇMEDİ": "gecmedi", "FARK VAR: MARL": "fark", "FARK VAR: M-none": "fark",
               "FARK GÖZLENMEDİ": "gozlenmedi"}


def fold(text: str) -> str:
    """Aramada Turkce harf ve buyuk/kucuk farkini yok say ('Kısa' ~ 'kisa', 'İŞ' ~ 'is')."""
    return (text or "").translate(_FOLD).lower()


def filter_terms(terms: list, query: str | None, category: str | None) -> list:
    """Sorgudaki her kelime terim, kisaltma, sade, teknik ya da id'de geciyorsa eslesir; kategori 'hepsi' ise suzulmez."""
    words = fold((query or "").strip()).split()
    out = []
    for t in terms:
        if category and category != ALL and t["kategori"] != category:
            continue
        hay = fold(" ".join(str(t.get(k) or "") for k in ("terim", "kisaltma", "sade", "teknik", "id")))
        if all(w in hay for w in words):
            out.append(t)
    return out


GLOSSARY_PATH_PREFIX = "/dash/guide#terim-"  # ust cubuk aramasi (topbar._glossary_options) bu capaya gider


def _anchor(term_id: str) -> str:
    return f"terim-{term_id}"


def _term_link(term_id: str, names: dict, cls: str = "guide-link") -> html.A:
    return html.A(names.get(term_id, term_id), href=f"#{_anchor(term_id)}", className=cls)


def term_card(t: dict, names: dict) -> html.Div:
    head = [html.Span(t["terim"], className="guide-term-name")]
    if t.get("kisaltma") and t["kisaltma"] not in t["terim"]:
        head.append(html.Span(t["kisaltma"], className="guide-abbr"))
    head.append(html.Span(CATEGORIES.get(t["kategori"], t["kategori"]), className="guide-cat"))
    body = [html.Div(head, className="guide-term-head"), html.P(t["sade"].strip(), className="guide-sade")]
    if t.get("benzetme"):
        body.append(html.P([html.I(className="bi bi-lightbulb me-1"), t["benzetme"].strip()], className="guide-analogy"))
    details = [html.Summary("Teknik ayrıntı"), html.P(t["teknik"].strip(), className="guide-teknik")]
    if t.get("kodda"):
        details.append(html.P(["Kodda: "] + _join([html.Code(r) for r in t["kodda"]]), className="guide-meta"))
    if t.get("ilgili"):
        details.append(html.P(["İlgili: "] + _join([_term_link(r, names) for r in t["ilgili"]]), className="guide-meta"))
    if t.get("kaynak"):
        details.append(html.P("Kaynak: " + "; ".join(t["kaynak"]), className="guide-meta"))
    body.append(html.Details(details, className="guide-details"))
    if t.get("uyari"):
        body.append(html.P([html.I(className="bi bi-exclamation-triangle me-1"), t["uyari"].strip()],
                           className="guide-warn"))
    return html.Div(body, id=_anchor(t["id"]), className="guide-term")


def _join(items: list) -> list:
    out = []
    for i, x in enumerate(items):
        out += ([", "] if i else []) + [x]
    return out


def term_list(terms: list, names: dict):
    if not terms:
        return create_state_block("empty", "Eşleşen terim yok.",
                                  "Başka bir kelime deneyin ya da kategoriyi 'Hepsi' yapın.")
    return html.Div([term_card(t, names) for t in terms], className="guide-terms")


def _count_text(n: int, total: int) -> str:
    return f"{total} terimin {n} tanesi gösteriliyor." if n != total else f"{total} terim."


def _section(title: str, children: list, anchor: str) -> dbc.Card:
    return dbc.Card(dbc.CardBody([html.H2(title, className="guide-section-title"), *children]),
                    id=f"rehber-{anchor}", className="mb-4")


def section_intro(g: dict, names: dict) -> dbc.Card:
    s = g["ne_yapar"]
    return _section(s["baslik"], [
        *[html.P(p.strip(), className="guide-lead") for p in s["paragraflar"]],
        html.Div([html.Span("İlgili terimler:", className="guide-muted")]
                 + [_term_link(t, names, "guide-chip") for t in s.get("terimler", [])], className="guide-chips"),
    ], "ne-yapar")


def section_flow(g: dict, names: dict) -> dbc.Card:
    s = g["akis"]
    steps = []
    for i, a in enumerate(s["adimlar"]):
        if i:
            steps.append(html.Div(html.I(className="bi bi-arrow-right"), className="guide-arrow"))
        box = [html.Div(f"{i + 1}. {a['ad']}", className="guide-step-name"), html.P(a["metin"].strip(), className="guide-step-text")]
        if a.get("terim"):
            box.append(_term_link(a["terim"], names, "guide-link guide-step-link"))
        steps.append(html.Div(box, className="guide-step"))
    return _section(s["baslik"], [html.P(s["aciklama"].strip(), className="guide-muted mb-3"),
                                  html.Div(steps, className="guide-flow")], "akis")


def section_limits(g: dict, names: dict) -> dbc.Card:
    s = g["ne_degildir"]
    items = []
    for m in s["maddeler"]:
        text = [html.Div(m["baslik"], className="guide-limit-title"), html.P(m["metin"].strip(), className="guide-limit-text")]
        if m.get("terim"):
            text.append(_term_link(m["terim"], names))
        items.append(html.Li([html.I(className="bi bi-exclamation-circle"), html.Div(text)], className="guide-limit"))
    return _section(s["baslik"], [html.P(s["aciklama"].strip(), className="guide-muted mb-2"),
                                  html.Ul(items, className="guide-limits")], "ne-degildir")


def section_stages(g: dict, snap: dict) -> dbc.Card:
    s = g["neredeyiz"]
    cols = []
    for key, title, icon in STAGE_GROUPS:
        rows = [a for a in snap["asamalar"] if a["sinif"] == key]
        if not rows:
            continue
        items = []
        for a in rows:
            body = [html.Div(a["ad"], className="guide-stage-name"), html.Div(a["durum"], className="guide-stage-status")]
            if a.get("siradaki"):
                body.append(html.Div("Sıradaki: " + a["siradaki"], className="guide-stage-next"))
            items.append(html.Li(body, className=f"guide-stage guide-stage-{key}"))
        cols.append(html.Div([html.Div([html.I(className=f"{icon} me-1"), f"{title} ({len(rows)})"],
                                       className=f"guide-stage-head guide-stage-head-{key}"),
                              html.Ul(items, className="guide-stage-list")], className="guide-stage-col"))
    return _section(s["baslik"], [html.P(s["aciklama"].strip(), className="guide-muted mb-3"),
                                  html.Div(cols, className="guide-stages"),
                                  html.P(f"Kaynak: {s['kaynak']}", className="guide-meta mt-2")], "neredeyiz")


def experiment_card(c: dict, names: dict) -> html.Div:
    head = html.Div([html.Span(c["etiket"], className=f"guide-label guide-label-{LABEL_CLASS.get(c['etiket'], 'kismi')}"),
                     html.Span(c["rozet"], className=f"guide-badge guide-badge-{c.get('kademe', 'gelistirme')}")],
                    className="guide-exp-tags")
    nums = [html.Div([html.Div(n["deger"], className="guide-num-value"), html.Div(n["ad"], className="guide-num-name"),
                      html.Div(f"{n['kaynak'].rsplit('/', 1)[-1]} · {n['alan']}", className="guide-num-src",
                               title=n["kaynak"])], className="guide-num") for n in c["sayilar"]]
    links = [_term_link(t, names, "guide-chip") for t in c.get("terimler", [])]
    return html.Div([
        html.Div(c["baslik"], className="guide-exp-title"), head,
        html.P(c["soru"], className="guide-exp-q"),
        html.Div(nums, className="guide-nums"),
        html.P([html.Strong("Okuma: "), c["okuma"]], className="guide-exp-read"),
        html.Div(links, className="guide-chips"),
        html.P(f"Belge: {c['belge']}", className="guide-meta mt-2"),
    ], id=f"deney-{c['id']}", className=f"guide-exp guide-exp-{c.get('kademe', 'gelistirme')}")


def section_experiments(g: dict, snap: dict, names: dict) -> dbc.Card:
    s = g["deneyler"]
    return _section(s["baslik"], [html.P(s["aciklama"].strip(), className="guide-muted mb-3"),
                                  html.Div([experiment_card(c, names) for c in snap["kartlar"]], className="guide-exps")],
                    "deneyler")


def section_faq(g: dict, names: dict) -> dbc.Card:
    s = g["sss"]
    items = []
    for q in s["sorular"]:
        body = [html.P(q["cevap"].strip(), className="guide-faq-a")]
        if q.get("terimler"):
            body.append(html.Div([_term_link(t, names, "guide-chip") for t in q["terimler"]], className="guide-chips"))
        items.append(html.Details([html.Summary(q["soru"])] + body, className="guide-faq"))
    return _section(s["baslik"], items, "sss")


def section_glossary(terms: list, names: dict) -> dbc.Card:
    options = [{"label": "Hepsi", "value": ALL}] + [
        {"label": label, "value": key} for key, label in CATEGORIES.items() if any(t["kategori"] == key for t in terms)]
    controls = html.Div([
        html.Div([html.Label("Ara", htmlFor="guide-search"),
                  dcc.Input(id="guide-search", type="search", value="", debounce=False,
                            placeholder="Terim, kısaltma ya da kelime (örneğin sharpe, tohum, kol)")],
                 className="guide-control"),
        html.Div([html.Label("Kategori", htmlFor="guide-category"),
                  dcc.Dropdown(id="guide-category", options=options, value=ALL, clearable=False)],
                 className="guide-control"),
    ], className="guide-controls")
    return _section("Sözlük", [
        html.P("Her terimde önce sade tanım, altında açılır teknik ayrıntı (koddaki karşılığı ve kaynağı) var.",
               className="guide-muted mb-3"),
        controls,
        html.Div(_count_text(len(terms), len(terms)), id="guide-count", className="guide-count"),
        html.Div(term_list(terms, names), id="guide-terms"),
    ], "sozluk")


def layout():
    guide, terms = load_content()
    snap = load_snapshot()
    names = {t["id"]: t["terim"] for t in terms}
    return html.Div([
        create_page_header("Rehber", "Sistemin ne yaptığı, verinin nasıl aktığı, neredeyiz, deneyler ne buldu, "
                                     "sınırlar ve terimler sözlüğü"),
        section_intro(guide, names),
        section_flow(guide, names),
        section_stages(guide, snap),
        section_experiments(guide, snap, names),
        section_limits(guide, names),
        section_faq(guide, names),
        section_glossary(terms, names),
        # sayfa acilisinda URL'deki capaya (#terim-<id>, #deney-<id>) bir kez kaydir; icerik Dash ile sonradan ciziliyor
        dcc.Store(id="guide-scroll-trigger", data=1),
        dcc.Store(id="guide-scroll-done"),
    ])


SCROLL_TO_HASH_JS = """
function(_trigger, hash) {
    var h = hash || window.location.hash;
    if (h && h.length > 1) {
        var el = document.getElementById(decodeURIComponent(h.slice(1)));
        if (el) { el.scrollIntoView({block: "start"}); }
    }
    return window.dash_clientside.no_update;
}
"""


def register_callbacks(app):
    # Acilista (tetik deposu bir kez cizilir) ve ust cubuk aramasi capayi degistirince (url.hash);
    # sozluk icindeki aramayla yeniden cizimde kaydirmaz.
    app.clientside_callback(SCROLL_TO_HASH_JS, Output("guide-scroll-done", "data"),
                            Input("guide-scroll-trigger", "data"), Input("url", "hash"))

    @app.callback(Output("guide-terms", "children"), Output("guide-count", "children"),
                  Input("guide-search", "value"), Input("guide-category", "value"))
    def _filter(query, category):
        _, terms = load_content()
        names = {t["id"]: t["terim"] for t in terms}
        shown = filter_terms(terms, query, category)
        return term_list(shown, names), _count_text(len(shown), len(terms))
