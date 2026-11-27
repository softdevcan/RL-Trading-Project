"""
Ana sayfa (Faz 11 §3.14): sistemin bugünkü hâli tek bakışta.

Sıra:
  1. Durum şeridi: ileri veri testi seansı n/126, son günlük çalışma (TRT), fiyat marjı denetiminin geçerlilik ufku,
     sürüm. Kaynak `/api/marl/forward/status`.
  2. "Öne çıkan model": M8 nihai testinin (2021–2026, 20 bps, ön kayıtlı) özeti. Öğrenen kolların ortalama fazla Sharpe'ı
     kural portföylerinin yanında. Bir SIRALAMA DEĞİL: M8 en iyi kolu seçmez (M8 §6.5); "MARL tek ajandan iyi mi"
     sorusunu yanıtlar, "basit kurallardan iyi mi" sorusunu yanıtlamaz (§7.3(b)). Sayılar `dashboard/content/snapshot.json`
     `m8_ozet` alanından (scripts/build_guide.py üretir; kaynak JSON + alan yolu). Pano ölçüm dosyası açmaz.
  3. Simülasyon: dondurulmuş defterlerin ileri veri (ARA GÖRÜNÜM) ya da 2026 test penceresi görünümü; kol 12 tohum
     ortalaması ± std. Kaynak `/api/marl/forward/view`. Ayrıntı `/dash/simulation`.
  4. Önceki fazların tek ajanlı modelleri (Faz 1–9) ve tahmin modelleri: katlanır bölüm; M8 ile aynı protokolde değil.

API: GET /api/trading/health, /api/marl/forward/status, /api/marl/forward/view, /api/trading/models,
     /api/prediction/models
"""

from __future__ import annotations

import json
import logging
import os

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

import dashboard.api_client as api
from dashboard.components import plain
from dashboard.components.metric_card import create_metric_card
from dashboard.components.page_header import create_page_header
from dashboard.components.state_block import create_state_block
from dashboard.pages.simulation import ALL_KEYS as SIM_ALL_KEYS
from dashboard.pages.simulation import LABELS as SIM_LABELS
from dashboard.pages.simulation import (NOTE_FORWARD, NOTE_TEST, READ_SESSIONS, RUN_STATUS, _num, d_full, d_short,
                                        select_view, spct)
from dashboard.theme import (CARD2, GOLD, GREEN, RED, TEXT, TEXT_MUTED, YELLOW, algo_badge_class, apply_theme_template,
                             empty_figure, plot_palette, plot_rgba)

_log = logging.getLogger(__name__)

CONTENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "content")
SIM_KEYS = ["M-none", "rule/ew", "rule/minvar"]          # simülasyon sayfasının varsayılanı (K-F11-1)
RULE_LABELS = {"minvar": "Min-varyans (aylık)", "hrp": "HRP (aylık)", "bh": "Al-tut", "ew": "1/N eşit ağırlık"}
ARM_SHORT = {"M-none": "MARL iletişimsiz (MAPPO)", "null": "Boş kol (negatif kontrol)", "M-self": "MARL kendine dikkat",
             "M-gat": "MARL graf dikkati", "S-agr": "SARL ağırlık", "S-esm": "SARL eşmimari"}
LEARNED_NOTE = ("Boş kol bir negatif kontroldür (aynı yapılandırmanın ikinci tohum grubu), aday model değildir.")


# ── veri ─────────────────────────────────────────────────────────────────────
def load_m8_summary() -> dict | None:
    """snapshot.json -> m8_ozet (yoksa None; sayfa yine kurulur)."""
    try:
        with open(os.path.join(CONTENT_DIR, "snapshot.json"), encoding="utf-8") as fh:
            return json.load(fh).get("m8_ozet")
    except (OSError, ValueError) as exc:
        _log.warning("snapshot.json okunamadi: %s", exc)
        return None


def featured(m8: dict) -> dict:
    """Öğrenen kollar arasında (boş kol hariç: negatif kontrol) ortalaması en yüksek kol ve en yüksek kural portföyü."""
    cands = [a for a in m8["kollar"] if a["kol"] != "null"]
    arm = max(cands, key=lambda a: a["ort"])
    rule = max(m8["kurallar"], key=lambda r: r["ort"])
    return {"arm": arm, "rule": rule, "rules_above": sum(r["ort"] > arm["ort"] for r in m8["kurallar"]),
            "n_rules": len(m8["kurallar"])}


def sr(x: float) -> str:
    """Sharpe: Türkçe ondalık virgül, gerçek eksi."""
    return ("−" if x < 0 else "") + _num(x, 2)


def _version() -> str:
    try:
        from app import __version__
        return f"v{__version__}"
    except Exception:  # pragma: no cover - pano tek basina
        return "—"


# ── 0. sade özet (§3.18): alan dışından bir okur yalnız bunu okuyup çıkabilmeli ──
INTRO = ("Borsa İstanbul'un en büyük 30 hissesi (BIST-30) arasında parayı nasıl paylaştıracağını deneme-yanılmayla kendi "
         "kendine öğrenen yapay zekâ modelleri (pekiştirmeli öğrenme) eğitiyor ve bunları, öğrenme içermeyen basit "
         "kurallarla aynı koşullarda karşılaştırıyoruz: aynı hisseler, aynı günler, aynı işlem maliyeti.")
HOME_TERMS = ["sharpe", "tohum", "baseline-1n", "marl", "sarl", "ileri-veri-testi"]


def _label_is(m8: dict, h: str, prefix: str) -> bool:
    return str((m8.get("etiketler") or {}).get(h, "")).startswith(prefix)


def plain_answers(m8: dict | None, st: dict | None) -> list:
    """Saf: üç soru için {soru, yanit, metin, sayi, not}. Sayılar ve etiketler snapshot.json m8_ozet'ten (elle sayı yok);
    cümleler M8 tasarımı §7.3 yazım kuralına bağlı (H1/H2 'daha iyi portföy seçiyor' diye yazılmaz)."""
    out = []
    if m8:
        f = featured(m8)
        years = f"{m8['katlar'][0]}–{m8['katlar'][-1]}"
        weak = min(m8["kurallar"], key=lambda r: r["ort"])
        allb = f["rules_above"] == f["n_rules"]
        out.append({
            "soru": "Yapay zekâ, basit kurallardan daha iyi yatırım yaptı mı?",
            "yanit": "Hayır." if allb else ("Kısmen." if f["rules_above"] else "Evet."),
            "metin": (f"{years} arasındaki {len(m8['katlar'])} yıllık sınavda hiçbir yapay zekâ modeli, riske göre "
                      "getiride basit kuralların hiçbirine yetişemedi." if allb else
                      f"{f['n_rules']} basit kuralın {f['rules_above']}'ü en iyi yapay zekâ modelinin üstünde."),
            "sayi": [("En iyi yapay zekâ modeli", sr(f["arm"]["ort"])),
                     (f"En zayıf basit kural ({RULE_LABELS.get(weak['kural'], weak['kural'])})", sr(weak["ort"]))],
            "not": "Riske göre getiri (Sharpe, yüksek = iyi), 6 yıl ortalaması, işlem maliyeti dahil. Maliyetsiz "
                   "durumda en iyi model en basit kurala ancak yetişiyor. Betimsel karşılaştırma: bu soru için önceden "
                   "planlanmış bir istatistik testi yok (M8 §7.3b).",
        })
        arms = {a["kol"]: a["ort"] for a in m8["kollar"]}
        if _label_is(m8, "H1", "FARK_VAR") and _label_is(m8, "H2", "FARK_VAR"):
            yearly = (" Fark her yıl aynı yönde değil." if "yila_bagli" in str(m8["etiketler"].get("H1", "")) else "")
            comm = (" Ajanların birbirine bilgi göndermesi ölçülebilir bir fark yaratmadı."
                    if _label_is(m8, "H3", "FARK_GOZLENMEDI") else "")
            out.append({
                "soru": "Her hisseye ayrı ajan, tek ajandan iyi mi?",
                "yanit": "Evet, 6 yılın ortalamasında.",
                "metin": ("Her hisseye ayrı ajan atayan model, bütün portföyü tek ajanla yöneten iki modelden daha "
                          "yüksek riske göre getiri verdi. Farkın önemli bir kısmı, daha az alım-satım yapıp daha az "
                          "işlem maliyeti ödemesinden geliyor." + yearly + comm),
                "sayi": [(plain.name(k), sr(arms[k])) for k in ("M-none", "S-agr", "S-esm") if k in arms],
                "not": "Ön kayıtlı eşli test, 6 yıl × 12 tohum (M8 H1–H3). Ayrıntı Rehber'deki M8 kartlarında.",
            })
    st = st or {}
    if st.get("available") is not False and st:
        n, need = int(st.get("n_sessions") or 0), int(st.get("read_sessions") or READ_SESSIONS)
        rem = max(0, need - n)
        out.append({
            "soru": "Şimdi ne yapılıyor?",
            "yanit": f"Gerçek piyasada canlı deneme: {n} / {need} gün.",
            "metin": ("Eğitimi bitmiş modeller dondurularak değiştirilmeden deneniyor: 23 Kasım 2026'dan beri her "
                      f"işlem günü sanal parayla. Sonuç {need}. günde, önceden yazılmış kurala göre bir kez okunacak; "
                      "o güne kadar günlük rakamlar yorumlanmaz."),
            "sayi": [("Kalan işlem günü", f"{rem} (~{max(1, round(rem / 21)) if rem else 0} ay)")],
            "not": "Bu bir 'ara görünüm'dür; hiçbir karar ona göre değişmez (ileri veri eki §9).",
            "progress": min(100.0, n / need * 100) if need else 0.0,
        })
    return out


def plain_summary(m8: dict | None, st: dict | None):
    def card(a):
        body = [html.Div(a["soru"], className="home-q"), html.Div(a["yanit"], className="home-a"),
                html.P(a["metin"], className="home-q-text")]
        if a.get("progress") is not None:
            body.append(dbc.Progress(value=a["progress"], className="sim-progress mb-2"))
        body.append(html.Div([html.Div([html.Span(k, className="home-q-key"), html.Strong(v, className="home-num")],
                                       className="home-q-row") for k, v in a["sayi"]], className="home-q-nums"))
        body.append(html.Div(a["not"], className="home-q-note"))
        return dbc.Col(dbc.Card(dbc.CardBody(body), className="h-100 home-q-card"), lg=4, md=12, className="mb-3")
    answers = plain_answers(m8, st)
    return html.Div([
        dbc.Card(dbc.CardBody([
            html.Div("Bu proje ne yapıyor?", className="home-intro-title"),
            html.P(INTRO, className="home-intro"),
        ]), className="mb-3"),
        dbc.Row([card(a) for a in answers], id="home-plain"),
        dbc.Card(dbc.CardBody([
            html.Div("Terimler nasıl okunur?", className="home-intro-title"),
            plain.how_to_read(HOME_TERMS, cls="home"),
        ]), className="mb-4"),
        html.Div([html.Div("Ayrıntılar", className="home-section-title"),
                  html.Div("Aşağısı akademik okur için: günlük işin durumu, 6 yıllık sınavın kol kol sonucu ve "
                           "simülasyon.", className="home-muted")], className="home-section"),
    ])


# ── 1. durum şeridi ──────────────────────────────────────────────────────────
def status_values(st: dict) -> dict:
    """Saf: forward status sözlüğü -> şerit metinleri (test edilebilir)."""
    st = st or {}
    if st.get("available") is False or not st:
        return {"forward": "—", "forward_sub": "Dondurulmuş M8 arşivi bu sunucuda yok",
                "run": "—", "run_sub": "", "margin": "—", "margin_sub": ""}
    n, need = int(st.get("n_sessions") or 0), int(st.get("read_sessions") or READ_SESSIONS)
    last = st.get("last_run") or {}
    fin = (last.get("finished_trt") or "")[:16].replace("T", " ")
    run = "çalışıyor" if st.get("running") else RUN_STATUS.get(last.get("status"), last.get("status") or "—")
    mc = last.get("margin_coverage") or {}
    return {"forward": f"{n} / {need}",
            "forward_sub": f"Son defter günü {d_full(st['last_date']) if st.get('last_date') else '—'}",
            "run": run, "run_sub": f"{fin} TRT" if fin else "",
            "margin": d_full(mc["horizon_end"]) if mc.get("horizon_end") else "—",
            "margin_sub": f"Son denetim {d_full(mc['last_check'])}" if mc.get("last_check") else ""}


def _strip(vals: dict) -> dbc.Row:
    def col(card):
        return dbc.Col(card, lg=3, sm=6, xs=12, className="mb-3")
    return dbc.Row([
        col(create_metric_card("İleri veri testi", vals["forward"], vals["forward_sub"], icon="bi bi-hourglass-split")),
        col(create_metric_card("Son günlük çalışma", vals["run"], vals["run_sub"], icon="bi bi-arrow-repeat")),
        col(create_metric_card("Fiyat marjı geçerli", vals["margin"], vals["margin_sub"], icon="bi bi-shield-check")),
        col(create_metric_card("Sürüm", _version(), "Faz 11: entegrasyon + sanal para", icon="bi bi-tag")),
    ], className="mb-2", id="home-strip")


# ── 2. öne çıkan model (M8) ──────────────────────────────────────────────────
def m8_figure(m8: dict, theme: str | None = None):
    """Yatay çubuk: öğrenen kollar ve kural portföyleri, 6 yıl ortalaması fazla Sharpe (20 bps). Öne çıkan kol vurgulu."""
    p = plot_palette(theme)
    top = featured(m8)["arm"]["kol"]
    rows = [(RULE_LABELS.get(r["kural"], r["kural"]), r["ort"], None, "rule") for r in m8["kurallar"]]
    rows += [(ARM_SHORT.get(a["kol"], a["kol"]), a["ort"], a["std"], "top" if a["kol"] == top else
              ("null" if a["kol"] == "null" else "arm")) for a in m8["kollar"]]
    rows.sort(key=lambda r: r[1])
    color = {"rule": p["gold"], "top": p["blue"], "arm": plot_rgba("blue", 0.38, theme), "null": plot_rgba("muted", 0.45, theme)}
    hover = [(f"{n}<br>ort. {sr(v)} · ± std {sr(s)} (72 hücre: 12 tohum × 6 yıl)" if s is not None
              else f"{n}<br>ort. {sr(v)} (6 yıl; kural tek seri)") for n, v, s, _ in rows]
    fig = go.Figure(go.Bar(
        y=[r[0] for r in rows], x=[r[1] for r in rows], orientation="h",
        marker={"color": [color[r[3]] for r in rows]}, text=[sr(r[1]) for r in rows], textposition="outside",
        hovertext=hover, hoverinfo="text", cliponaxis=False))
    fig.add_vline(x=0, line={"color": p["line"], "width": 1})
    lo, hi = min(0.0, min(r[1] for r in rows)), max(r[1] for r in rows)
    pad = 0.12 * (hi - lo)                       # dış etiketler (eksi çubukta solda) eksen yazısına binmesin
    fig.update_layout(height=330, margin={"l": 10, "r": 40, "t": 8, "b": 36}, showlegend=False, separators=",.",
                      xaxis={"title": "Fazla Sharpe, 6 yıl ortalaması (20 bps)", "zeroline": False,
                             "range": [lo - (pad if lo < 0 else 0), hi + pad]},
                      yaxis={"automargin": True}, bargap=0.28)
    apply_theme_template(fig, theme)
    return fig


def featured_block(m8: dict | None):
    if not m8:
        return create_state_block("empty", "M8 özeti bulunamadı (snapshot.json). `python scripts/build_guide.py` kos.")
    f = featured(m8)
    arm, rule = f["arm"], f["rule"]
    hero = html.Div([
        html.Div([
            html.Div("Öğrenen modeller arasında en yüksek ortalama", className="home-kicker"),
            html.Div(ARM_SHORT.get(arm["kol"], arm["kol"]), className="home-hero-name"),
            html.Div([html.Span(sr(arm["ort"]), className="home-hero-value"),
                      html.Span(f"fazla Sharpe · ± std {sr(arm['std'])} · {m8['hucre_kol_basi']} hücre "
                                f"(12 tohum × {len(m8['katlar'])} yıl) · {m8['maliyet_bps']} bps", className="home-muted")],
                     className="home-hero-line"),
        ], className="home-hero-main"),
        html.Div([
            html.Div("Aynı maliyetle en yüksek kural portföyü", className="home-kicker"),
            html.Div(RULE_LABELS.get(rule["kural"], rule["kural"]), className="home-hero-name"),
            html.Div([html.Span(sr(rule["ort"]), className="home-hero-value home-hero-rule"),
                      html.Span("6 yıl ortalaması, öğrenme yok", className="home-muted")], className="home-hero-line"),
        ], className="home-hero-side"),
    ], className="home-hero")
    note = html.Ul([
        html.Li(f"{f['n_rules']} kural portföyünün {f['rules_above']}'ü öğrenen kolların hepsinin üstünde. M8 \"çok ajanlı "
                "öğrenme tek ajandan iyi mi\" sorusunu yanıtlar; \"basit kurallardan iyi mi\" sorusunu yanıtlamaz."),
        html.Li("Bu bir sıralama değildir: M8 en iyi kolu seçmez. Kol karşılaştırmaları ön kayıtlı eşli testlerle yapıldı; "
                "etiketleri Rehber'deki M8 kartlarında."),
        html.Li(["Tohumlar arası yayılım büyük (± std); tek bir tohum ya da yıl sonuç değildir. ", LEARNED_NOTE]),
    ], className="home-notes")
    return html.Div([
        hero,
        dcc.Graph(id="home-m8-chart", figure=empty_figure(), config={"displayModeBar": False}),
        note,
        html.Div([dbc.Button([html.I(className="bi bi-book me-1"), "M8 kartları (Rehber)"], href="/dash/guide#rehber-deneyler",
                             color="secondary", outline=True, size="sm", className="me-2"),
                  html.Span(f"Kaynak: {m8['kaynak'].rsplit('/', 1)[-1]} · {m8['on_kayit']}", className="home-src")],
                 className="home-actions"),
    ])


# ── 3. simülasyon ────────────────────────────────────────────────────────────
def sim_figure(view: dict, theme: str | None = None):
    """Toplam getiri (%): kol ortalaması ± std bandı + kural portföyleri (kesirli defter, aynı maliyet)."""
    dates = (view or {}).get("dates") or []
    lines = (view or {}).get("lines") or {}
    if not dates or not lines:
        return empty_figure("Henüz ileri veri seansı yok" if (view or {}).get("window") == "forward" else "Veri yok")
    p = plot_palette(theme)
    cap = float(view.get("capital") or 100_000)
    colors = {"M-none": "blue", "rule/ew": "orange", "rule/minvar": "gold"}
    fig = go.Figure()
    if len(dates) <= 40:                         # kısa pencere: tarih ekseni saat çizer; seans etiketiyle göster
        dates = [d_short(d) for d in dates]
    for key, ln in lines.items():
        mean = [v / cap - 1 for v in ln["mean"]]
        c = colors.get(key, "purple")
        if ln.get("n", 1) > 1:
            up = [(m + s) / cap - 1 for m, s in zip(ln["mean"], ln["std"])]
            lo = [(m - s) / cap - 1 for m, s in zip(ln["mean"], ln["std"])]
            fig.add_trace(go.Scatter(x=dates + dates[::-1], y=up + lo[::-1], fill="toself", line={"width": 0},
                                     fillcolor=plot_rgba(c, 0.14, theme), hoverinfo="skip", showlegend=False))
        label = SIM_LABELS.get(key, ln["label"]) + (f" (ort., N={ln['n']})" if ln.get("n", 1) > 1 else "")
        fig.add_trace(go.Scatter(x=dates, y=mean, mode="lines+markers" if len(dates) <= 10 else "lines", name=label,
                                 line={"color": p[c], "width": 2.2 if key in ("M-none",) else 1.6},
                                 hovertemplate="%{x}<br>%{y:+.2%}<extra>" + label + "</extra>"))
    fig.add_hline(y=0, line={"color": p["line"], "width": 1})
    fig.update_layout(height=280, margin={"l": 10, "r": 10, "t": 8, "b": 10}, separators=",.",
                      yaxis={"tickformat": "+.1%", "automargin": True},
                      legend={"orientation": "h", "y": -0.18, "x": 0, "font": {"size": 11}})
    apply_theme_template(fig, theme)
    return fig


def sim_table(view: dict):
    rows = (view or {}).get("table") or []
    if not rows:
        return html.Div()

    def pm(r, f):
        if r.get(f) is None:
            return "—"
        s = spct(r[f], 2)
        return s + (f" ± {_num(r[f + '_std'] * 100, 2)}" if r.get(f + "_std") is not None else "")
    body = [html.Tr([html.Td(SIM_LABELS.get(r["key"], r["label"]) + (f" (N={r['n']})" if r["n"] > 1 else "")),
                     html.Td(pm(r, "cum_return"), className="home-num"),
                     html.Td(pm(r, "max_drawdown"), className="home-num")]) for r in rows]
    return html.Table([html.Thead(html.Tr([html.Th("Strateji"), html.Th("Toplam getiri", className="home-num"),
                                           html.Th("En büyük düşüş", className="home-num")])),
                       html.Tbody(body)], className="home-table")


def sim_note(view: dict) -> str:
    v = view or {}
    n = v.get("n_sessions") or 0
    base = NOTE_FORWARD if v.get("window") == "forward" else NOTE_TEST
    return f"{n} seans · {v.get('cost_bps_roundtrip', 20):g} bps gidiş-dönüş · " + base


# ── 4. önceki fazlar ─────────────────────────────────────────────────────────
def _build_models_list(models):
    if not models:
        return create_state_block("empty", "Eğitilmiş tek ajanlı model bulunamadı.")
    rows = []
    for m in models:
        algo = m.get("algorithm", "—").upper()
        rows.append(dbc.Row([
            dbc.Col(dbc.Badge(algo, pill=True, className=algo_badge_class(algo)), width="auto"),
            dbc.Col(html.Span(m.get("name", "—"), style={"color": TEXT, "fontSize": "13px"}), width=5),
            dbc.Col(html.Small(f"Faz {m.get('phase', '—')}", style={"color": TEXT_MUTED}), width="auto"),
            dbc.Col(html.Small(str(m.get("created_at", m.get("timestamp", "—")))[:10], style={"color": TEXT_MUTED}),
                    width="auto", className="ms-auto"),
        ], className="align-items-center py-2", style={"borderBottom": f"1px solid {CARD2}"}))
    return html.Div(rows)


def _build_pred_models_list(models):
    if not models:
        return html.P("Eğitilmiş tahmin modeli yok. Tahmin sayfasından model eğitebilirsiniz.",
                      style={"color": TEXT_MUTED, "fontSize": "13px"})
    rows = []
    for m in sorted(models, key=lambda x: (x.get("symbol", ""), x.get("horizon", ""))):
        horizon = m.get("horizon", "?")
        mape, dir_acc = m.get("test_mape"), m.get("test_direction_accuracy")
        perf = []
        if mape is not None:
            perf.append(html.Small(f"MAPE {mape:.1f}%", style={"color": GREEN if mape < 3 else (YELLOW if mape < 7 else RED),
                                                                "marginRight": "8px"}))
        if dir_acc is not None:
            perf.append(html.Small(f"Yön {dir_acc:.0f}%", style={"color": GREEN if dir_acc >= 55 else RED}))
        if not perf:
            perf = [html.Small(f"{len(m.get('feature_cols', []))} özellik", style={"color": TEXT_MUTED})]
        rows.append(dbc.Row([
            dbc.Col(html.Span(m.get("symbol", "?"), style={"color": GOLD, "fontWeight": "600", "fontSize": "13px"}), width=3),
            dbc.Col(dbc.Badge("Günlük" if horizon == "daily" else "Haftalık",
                              color="primary" if horizon == "daily" else "info", pill=True, className="me-2"), width="auto"),
            dbc.Col(html.Span(perf), width=4),
            dbc.Col(html.Small((m.get("saved_at", "") or "")[:10] or "—", style={"color": TEXT_MUTED}), width="auto",
                    className="ms-auto"),
        ], className="align-items-center py-2", style={"borderBottom": f"1px solid {CARD2}"}))
    return html.Div([html.P(f"{len(models)} model", style={"color": TEXT_MUTED, "fontSize": "12px", "marginBottom": "8px"}),
                     *rows])


# ── sayfa ────────────────────────────────────────────────────────────────────
def layout():
    m8 = load_m8_summary()
    status = api.get_marl_forward_status()
    return html.Div([
        dcc.Interval(id="home-interval", interval=300_000, n_intervals=0),     # defter günde bir değişir
        create_page_header(
            "Dashboard",
            "Yapay zekâ ile portföy yönetimi: ne öğrendik ve şu an gerçek piyasada ne deneniyor",
            actions=html.Div(id="home-status-badge"),
        ),
        plain_summary(m8, status),
        html.Div(_strip(status_values(status)), id="home-strip-wrap"),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.Div([html.Span("Öne çıkan model · M8 nihai testi (2021–2026)", className="card-title-sm"),
                                         html.Span("Ön kayıtlı nihai deney", className="home-badge")],
                                        className="home-card-head")),
                dbc.CardBody(featured_block(m8)),
            ], className="h-100"), lg=7, className="mb-4"),
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.Div([
                    html.Span("Simülasyon · dondurulmuş defterler", className="card-title-sm"),
                    dbc.RadioItems(id="home-sim-window", inline=True, value="forward", className="home-window",
                                   options=[{"label": "İleri veri", "value": "forward"},
                                            {"label": "2026 test", "value": "test2026"}]),
                ], className="home-card-head")),
                dbc.CardBody([
                    dcc.Loading(dcc.Graph(id="home-sim-chart", figure=empty_figure(), config={"displayModeBar": False})),
                    html.Div(id="home-sim-table"),
                    html.Div(id="home-sim-note", className="home-note mt-2"),
                    html.Div(dbc.Button([html.I(className="bi bi-cash-coin me-1"), "Simülasyon ayrıntısı"],
                                        href="/dash/simulation", color="primary", size="sm"), className="home-actions"),
                ]),
            ], className="h-100"), lg=5, className="mb-4"),
        ]),
        html.Details([
            html.Summary("Önceki fazların modelleri (tek ajan, Faz 1–9) ve tahmin modelleri", className="home-legacy-sum"),
            html.P("Bu modeller M8 ile aynı protokolde (kat, tohum, maliyet, ön kayıt) değerlendirilmedi; yukarıdaki "
                   "sonuçlarla karşılaştırılmaz. Günlük karar ve kâğıt portföy bunları kullanır.", className="home-muted mt-2"),
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader(html.Span("Tek ajanlı RL modelleri", className="card-title-sm")),
                                  dbc.CardBody(html.Div(id="home-models-list", children=create_state_block("loading")))]),
                        md=6, className="mb-3"),
                dbc.Col(dbc.Card([dbc.CardHeader(html.Span("Tahmin modelleri", className="card-title-sm")),
                                  dbc.CardBody(html.Div(id="home-pred-models-list",
                                                        children=create_state_block("loading")))]),
                        md=6, className="mb-3"),
            ]),
        ], className="home-legacy"),
    ])


def register_callbacks(app):

    @app.callback(Output("home-status-badge", "children"), Output("home-strip-wrap", "children"),
                  Input("home-interval", "n_intervals"))
    def update_status(_n):
        status = api.get_health().get("status", "unknown")
        badge = dbc.Badge([html.I(className="bi bi-circle-fill me-1", style={"fontSize": "8px"}),
                           "SAĞLIKLI" if status == "healthy" else status.upper()],
                          color="success" if status == "healthy" else "danger", pill=True)
        return badge, _strip(status_values(api.get_marl_forward_status()))

    @app.callback(Output("home-m8-chart", "figure"), Input("home-interval", "n_intervals"))
    def update_m8(_n):
        m8 = load_m8_summary()
        return m8_figure(m8) if m8 else empty_figure("M8 özeti yok")

    @app.callback(Output("home-sim-chart", "figure"), Output("home-sim-table", "children"),
                  Output("home-sim-note", "children"),
                  Input("home-sim-window", "value"), Input("home-interval", "n_intervals"))
    def update_sim(window, _n):
        view = api.get_marl_forward_view(window or "forward", SIM_ALL_KEYS)   # tablo hepsi, grafik SIM_KEYS
        if not view:
            return empty_figure("Simülasyon verisi alınamadı (arşiv yok ya da sunucuya ulaşılamıyor)"), html.Div(), ""
        return sim_figure(select_view(view, SIM_KEYS)), sim_table(view), sim_note(view)

    @app.callback(Output("home-models-list", "children"), Output("home-pred-models-list", "children"),
                  Input("home-interval", "n_intervals"))
    def update_legacy(_n):
        models = _build_models_list(api.get_models())
        try:
            preds = _build_pred_models_list(api.get_prediction_models() or [])
        except Exception as exc:
            _log.warning("get_prediction_models failed: %s", exc)
            preds = dbc.Alert("Tahmin modelleri yüklenemedi.", color="warning", className="mb-0")
        return models, preds
