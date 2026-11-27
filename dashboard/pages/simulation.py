"""
Sanal para simülasyonu (Faz 11 S5 + §3.13 yeniden tasarım; plan phase-11-integration-plan.md, K-F11-1/2/6).

Sayfa üç soruyu cevaplar: portföyde hangi hisseler var, gün gün ne kazandı/kaybetti, toplam K/Z ne.
- İncelenen portföy: seçilen strateji (kol + tohum ya da kural), kullanıcının sermayesiyle TAM SAYI lot
  (`/api/marl/forward/product`, `book` = `forward_views.daily_book`). Kartlar, gün özeti, eldeki hisseler ve dağılım
  bundan. Bir güne tıklanınca (ya da önceki/sonraki) bütün bu bölümler o günün kapanışına geçer.
- Karşılaştırma: aynı pencere ve aynı maliyetteki kesirli defterler (`/view`; kol = 12 tohum ort. ± std), sermayeye
  ölçeklenir (kesirli defterde sermaye getiriyi değiştirmez). Varsayılan görünüm K-F11-1.
- İleri veri görünümü ARA GÖRÜNÜMDÜR: tez sonucu değil; ileri test 126. seansta ön kayıtlı kurala göre bir kez okunur
  (phase-11-m8-forward-annex.md). Ürün portföyü (tam lot) ileri teste girmez, kayıt tutmaz.

API: GET /api/marl/forward/status | /view | /product ; POST /api/marl/forward/run (yönetici).
Sayfa açılışta durumu sorar ve iş çalışıyorsa yoklamayı açar (dashboard/CLAUDE.md uzun işler kuralı).
Render fonksiyonları saf (store sözlüğü -> bileşen); `tests/test_marl_sim_book.py` sentetik defterle çağırır.
"""

from __future__ import annotations

from datetime import date as _date

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html, no_update
from dash.dash_table import DataTable
from plotly.subplots import make_subplots

import dashboard.api_client as api
from dashboard.auth_context import is_admin
from dashboard.components import plain
from dashboard.components.metric_card import create_metric_card
from dashboard.components.page_header import create_page_header
from dashboard.components.table import TABLE_STYLES, numeric_columns
from dashboard.theme import (BLUE, CYAN, GOLD, ORANGE, PURPLE, TEXT_MUTED, apply_theme_template, empty_figure,
                             plot_palette, plot_rgba)

ARMS = ("M-none", "S-esm", "S-agr", "M-gat", "M-self", "null")
# Sade ad + teknik kod tek yerden (components/plain.py); sıra sabit, sıralama değil
LABELS = {k: plain.name(k) for k in plain.ORDER}
ARM_OPTIONS = [{"label": LABELS[k], "value": k} for k in plain.ORDER]
MIN_SHARPE_SESSIONS = 20      # daha kısa pencerede yıllık Sharpe gürültüdür; tabloda "—" (4 günden yıllık sayı çıkmaz)
DEFAULT_KEYS = ["M-none", "rule/ew", "rule/minvar"]          # K-F11-1 (grafikteki kesikli çizgiler)
ALL_KEYS = [o["value"] for o in ARM_OPTIONS]                   # karşılaştırma tablosu hepsini sabit sırayla gösterir
SEEDS = list(range(201, 213))
NULL_SEEDS = list(range(213, 225))           # boş kol (negatif kontrol) ayrı tohumlarla eğitildi (M8 §6)


def seeds_for(focus: str) -> list:
    return NULL_SEEDS if focus == "null" else SEEDS
READ_SESSIONS = 126
FORWARD_START = "2026-11-23"
RUN_STATUS = {"yazildi": "yazıldı", "degisiklik_yok": "değişiklik yok"}     # forward_job durum kodu -> metin
RANGES = [{"label": "1A", "value": "1m"}, {"label": "3A", "value": "3m"}, {"label": "Tümü", "value": "all"}]
RANGE_LEN = {"1m": 21, "3m": 63, "all": None}
LINE_KEYS = ("orange", "purple", "cyan", "gold", "yellow")    # kıyas çizgileri; mavi incelenen portföyün
GROUP_KEYS = ("blue", "cyan", "purple", "orange", "gold")     # ilk 5 hisse; sonra diğer + nakit
GROUP_CSS = (BLUE, CYAN, PURPLE, ORANGE, GOLD)
OTHER_CSS, CASH_CSS = TEXT_MUTED, "var(--rlt-border-strong)"
MINUS = "−"
MON = ("Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara")
MONL = ("Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık")
DAYL = ("Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar")

NOTE_FORWARD = ("Ara görünüm — tez sonucu değildir. İleri veri testi 2026-11-23'ten itibaren TAM 126 seans dolunca ön "
                "kayıtlı kurala göre bir kez okunur (phase-11-m8-forward-annex.md). Buradaki sayılara göre hiçbir "
                "kural, kol ya da pencere değiştirilmez.")
NOTE_TEST = ("2026 test penceresi (2026-01-02..10-30): M8 değerlendirmesinde zaten okunmuş geçmiş pencere; aynı defterle "
             "yeniden oynatıldı (S3, 72/72 bit-eş).")


# ── biçim ─────────────────────────────────────────────────────────────────────
def _num(x: float, d: int = 0) -> str:
    return f"{abs(x):,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _sign(x: float, eps: float) -> str:
    return "+" if x > eps else (MINUS if x < -eps else "")


def tl(x: float) -> str:
    return ("" if x >= 0 else MINUS) + _num(x) + " TL"


def stl(x: float) -> str:
    return _sign(round(x), 0) + _num(round(x)) + " TL"


def pct(x: float, d: int = 1) -> str:
    return "%" + _num(x * 100, d)


def spct(x: float, d: int = 2) -> str:
    return _sign(x, 0.5 * 10 ** -(d + 2)) + "%" + _num(x * 100, d)


def _d(s: str) -> _date:
    return _date.fromisoformat(str(s)[:10])


def d_short(s: str) -> str:
    d = _d(s)
    return f"{d.day} {MON[d.month - 1]}"


def d_full(s: str) -> str:
    d = _d(s)
    return f"{d.day} {MON[d.month - 1]} {d.year}"


def d_long(s: str) -> str:
    d = _d(s)
    return f"{d.day} {MONL[d.month - 1]} {d.year}, {DAYL[d.weekday()]}"


def _tone(x: float) -> str:
    return "profit" if x > 1e-9 else ("loss" if x < -1e-9 else "neutral")


def _sign_rules(*cols: str) -> list:
    """Biçimlenmiş metinde yön rengi: '+' kâr, '−' (U+2212) zarar (tone_rules_formatted ASCII '-' arar)."""
    out = []
    for c in cols:
        out += [{"if": {"filter_query": f'{{{c}}} contains "{MINUS}"', "column_id": c}, "color": "var(--rlt-loss)"},
                {"if": {"filter_query": f'{{{c}}} contains "+"', "column_id": c}, "color": "var(--rlt-profit)"}]
    return out


def seed_of(focus: str, seed) -> int:
    """Kolun tohumu; seçili tohum o kolda yoksa (ör. boş kola geçilince 201) kolun ilk tohumu."""
    s = seeds_for(focus)
    return int(seed) if seed is not None and int(seed) in s else s[0]


def product_sid(focus: str, seed) -> str:
    return f"policy/{focus}/s{seed_of(focus, seed)}" if focus in ARMS else focus


def sel_index(book: dict, sel) -> int:
    dates = book.get("dates") or []
    return dates.index(sel) if sel in dates else len(dates) - 1


# ── defterden türetilenler ───────────────────────────────────────────────────
def _range_start(n: int, rng: str) -> int:
    L = RANGE_LEN.get(rng)
    return 0 if not L else max(0, n - L)


def _top_groups(book: dict, i: int) -> list:
    lots, mark = book["lots"][i], book["mark"][i]
    held = [j for j in range(len(lots)) if lots[j] > 0]
    return sorted(held, key=lambda j: -lots[j] * mark[j])[:5]


def _shares(book: dict, i: int, top: list) -> list:
    """[ilk 5 hisse..., diğer, nakit] piyasa değeri payları (gün sonu)."""
    v, lots, mark = book["value"][i], book["lots"][i], book["mark"][i]
    out = [lots[j] * mark[j] / v for j in top]
    out.append(sum(lots[j] * mark[j] for j in range(len(lots)) if j not in top) / v)
    out.append(book["cash"][i] / v)
    return out


def _drawdown(values: list, capital: float) -> list:
    pk, out = capital, []
    for v in values:
        pk = max(pk, v)
        out.append(v / pk - 1)
    return out


def _line_at(view: dict, key: str, date: str):
    if not view or key not in (view.get("lines") or {}):
        return None
    dates = view.get("dates") or []
    if date not in dates:
        return None
    return view["lines"][key]["mean"][dates.index(date)] / float(view.get("capital") or 1)


def date_axis(dates: list) -> dict:
    """İşlem günü ekseni: hafta sonu boşluğu yok; kısa pencerede etiket yalnız defter günlerinde (tekrar eden tarih yok)."""
    ax = {"tickformat": "%d.%m.%y", "hoverformat": "%d.%m.%Y", "rangebreaks": [{"bounds": ["sat", "mon"]}]}
    if len(dates) <= 25:
        ax["tickvals"] = list(dates)
    return ax


# ── bileşenler ───────────────────────────────────────────────────────────────
def _t(title: str, uid: str, key: str = "", term: str = ""):
    """Kart başlığı + ⓘ (not sözlüğü ya da sözlük terimi)."""
    return [title, " ", plain.info(f"sim-i-{uid}", term=None if key else term or None,
                                   text=plain.NOTES.get(key) if key else None, title=title if key else None)]


def kpi_cards(book: dict, view: dict, i: int, focus: str | None = None, seed=None) -> list:
    """Kol seçiliyse incelenen portföy TEK TOHUMDUR; 'Kol ortalaması' kartı 12 tohumun ortalamasını yanına koyar
    (tek tohumla kollar karşılaştırılınca ortalamanın tersi bir sıra çıkabiliyordu: ileri veri, S-esm s201)."""
    cap, V, dates = book["capital"], book["value"], book["dates"]
    v, prev = V[i], (V[i - 1] if i else cap)
    day, tot = v - prev, v - cap
    cost = sum(book["commission"][: i + 1])
    pk, pk_i, mdd, mdd_pk, mdd_tr = cap, -1, 0.0, -1, -1
    for t in range(i + 1):
        if V[t] > pk:
            pk, pk_i = V[t], t
        if V[t] / pk - 1 < mdd:
            mdd, mdd_pk, mdd_tr = V[t] / pk - 1, pk_i, t
    dd_sub = ("henüz düşüş yok" if mdd >= 0 else
              f"tepe {d_short(dates[mdd_pk]) if mdd_pk >= 0 else 'başlangıç'} → dip {d_short(dates[mdd_tr])}")
    bench = _line_at(view, "rule/ew", dates[i])
    if bench is None:
        bench_card = create_metric_card(_t("1/N kıyası", "kpi-ew", term="baseline-1n"), "—", "1/N'i karşılaştırmaya ekleyin", icon="bi bi-sliders")
    else:
        gap = tot / cap - (bench - 1)
        bench_card = create_metric_card(_t("1/N kıyası", "kpi-ew", term="baseline-1n"), f"{_sign(gap, 5e-5)}{_num(gap * 100, 1)} puan",
                                        f"aynı dönem, aynı maliyet · 1/N {spct(bench - 1, 1)}", icon="bi bi-sliders",
                                        tone=_tone(gap))
    is_arm = focus in ARMS
    one = f" · tek tohum ({seed_of(focus, seed)}), kol ortalaması değil" if is_arm else ""
    cards = [
        create_metric_card("Portföy değeri", tl(v), f"başlangıç {tl(cap)}{one}", icon="bi bi-wallet2"),
        create_metric_card(f"Gün K/Z · {d_short(dates[i])}", stl(day), f"{spct(day / prev)} önceki kapanışa göre",
                           icon="bi bi-calendar-day", tone=_tone(day)),
        create_metric_card(_t("Toplam K/Z", "kpi-tot", key="total"), stl(tot), f"{spct(tot / cap)} · işlem maliyeti {tl(cost)} dahil",
                           icon="bi bi-graph-up-arrow", tone=_tone(tot)),
        create_metric_card(_t("En büyük düşüş", "kpi-dd", key="drawdown"), spct(mdd, 1), dd_sub, icon="bi bi-arrow-down-right",
                           tone="loss" if mdd < 0 else "neutral"),
        bench_card,
    ]
    arm = _line_at(view, focus, dates[i]) if is_arm else None
    if arm is not None:
        line = view["lines"][focus]
        sd = line["std"][(view.get("dates") or []).index(dates[i])] / float(view.get("capital") or 1)
        gap = tot / cap - (arm - 1)
        cards.insert(3, create_metric_card(
            _t(f"Kol ortalaması · {line['n']} tohum", "kpi-arm", key="arm_mean", term="tohum"), spct(arm - 1),
            f"± {_num(sd * 100, 2)} puan std · bu tohum {_sign(gap, 5e-5)}{_num(gap * 100, 2)} puan",
            icon="bi bi-people", tone=_tone(arm - 1)))
    return [dbc.Col(c, xs=12, sm=6, lg=True, className="mb-3") for c in cards]


def main_figure(book: dict, view: dict, i: int, rng: str, focus_label: str, theme: str | None = None):
    p = plot_palette(theme)
    dates, V, cap = book["dates"], book["value"], book["capital"]
    k0 = _range_start(len(dates), rng)
    xs, ys = dates[k0:], V[k0:]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.62, 0.2, 0.18], vertical_spacing=0.04)
    scale = cap / float((view or {}).get("capital") or cap)
    vdates = (view or {}).get("dates") or []
    keep = [t for t, d in enumerate(vdates) if d >= xs[0]] if xs else []
    for n_line, (key, line) in enumerate(((view or {}).get("lines") or {}).items()):
        c = p[LINE_KEYS[n_line % len(LINE_KEYS)]]
        x = [vdates[t] for t in keep]
        m = [line["mean"][t] * scale for t in keep]
        if line["n"] > 1:
            up = [line["mean"][t] * scale + line["std"][t] * scale for t in keep]
            lo = [line["mean"][t] * scale - line["std"][t] * scale for t in keep]
            fig.add_trace(go.Scatter(x=x, y=up, line={"width": 0}, showlegend=False, hoverinfo="skip"), 1, 1)
            fig.add_trace(go.Scatter(x=x, y=lo, line={"width": 0}, fill="tonexty", fillcolor=plot_rgba(
                LINE_KEYS[n_line % len(LINE_KEYS)], 0.13, theme), showlegend=False, hoverinfo="skip"), 1, 1)
        # Türkçe etiket sayfanın kendi sözlüğünden (API etiketi ASCII); §3.14
        name = LABELS.get(key, line["label"]) + (f" (ort. ± std, N={line['n']})" if line["n"] > 1 else "")
        fig.add_trace(go.Scatter(x=x, y=m, name=name, line={"color": c, "width": 1.6, "dash": "dash"},
                                 hovertemplate="%{y:,.0f} TL<extra>" + LABELS.get(key, key) + "</extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=xs, y=ys, name=f"{focus_label} · incelenen (tam lot)",
                             line={"color": p["blue"], "width": 3},
                             hovertemplate="%{y:,.0f} TL<extra>incelenen</extra>"), 1, 1)
    if k0 <= i:
        fig.add_trace(go.Scatter(x=[dates[i]], y=[V[i]], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker={"size": 11, "color": p["blue"], "line": {"width": 2, "color": p["bg"]}}), 1, 1)
    fig.add_hline(y=cap, line={"color": p["line"], "width": 1, "dash": "dot"}, row=1, col=1)

    pnl = [V[t] - (V[t - 1] if t else cap) for t in range(k0, len(V))]
    fig.add_trace(go.Bar(x=xs, y=pnl, name="Günlük K/Z", showlegend=False,
                         marker={"color": [p["green"] if v >= 0 else p["red"] for v in pnl]},
                         hovertemplate="%{y:+,.0f} TL<extra>günlük K/Z</extra>"), 2, 1)
    dd = _drawdown(V, cap)[k0:]
    fig.add_trace(go.Scatter(x=xs, y=dd, name="Tepeden düşüş", showlegend=False, fill="tozeroy",
                             line={"color": p["red"], "width": 1.2}, fillcolor=plot_rgba("red", 0.18, theme),
                             hovertemplate="%{y:.1%}<extra>tepeden düşüş</extra>"), 3, 1)
    if k0 <= i:
        for r in (1, 2, 3):
            fig.add_vline(x=dates[i], line={"color": p["muted"], "width": 1, "dash": "dot"}, row=r, col=1)
    # Lejant çizim alanının ÜSTÜNDE: y=1.06 + t=10 iken 4 satıra sarılan lejant çizgilerin üstüne biniyordu (§3.14)
    n_leg = len(((view or {}).get("lines") or {})) + 1
    fig.update_layout(height=600 + 20 * ((n_leg + 1) // 2), margin={"l": 64, "r": 16, "t": 14 + 22 * ((n_leg + 1) // 2),
                      "b": 30}, hovermode="x unified", separators=",.", bargap=0.35,
                      legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0, "font": {"size": 11.5}})
    fig.update_xaxes(**date_axis(xs))                    # plotly yerel ay adları İngilizce: sayısal tarih biçimi
    fig.update_yaxes(title_text="Değer (TL)", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(title_text="Gün K/Z", tickformat=",.0f", row=2, col=1)
    fig.update_yaxes(title_text="Düşüş", tickformat=".0%", row=3, col=1)
    apply_theme_template(fig, theme)
    return fig


def day_panel(book: dict, i: int, is_rule: bool) -> list:
    dates, V, cap, syms = book["dates"], book["value"], book["capital"], book["symbols"]
    prev = V[i - 1] if i else cap
    day = V[i] - prev
    contrib = book["contrib"][i]
    held_prev = book["lots"][i - 1] if i else [0] * len(syms)
    items = sorted([(j, contrib[j]) for j in range(len(syms)) if held_prev[j] > 0 or book["lots"][i][j] > 0],
                   key=lambda x: -abs(x[1]))[:6]
    cmax = max([abs(v) for _, v in items] + [1.0])
    best = max(items, key=lambda x: x[1], default=None)
    worst = min(items, key=lambda x: x[1], default=None)
    if i == 0:
        sentence = f"İlk gün: {tl(cap)} sermaye {len(book['trades'][0])} emirle hisselere dağıtıldı."
    else:
        sentence = f"Gün kapanışı {tl(V[i])}."
        if best and best[1] > 0:
            sentence += f" En çok kazandıran {syms[best[0]]} ({stl(best[1])})"
        if worst and worst[1] < 0:
            sentence += f"{',' if best and best[1] > 0 else ''} en çok kaybettiren {syms[worst[0]]} ({stl(worst[1])})"
        sentence += "."
    bars = []
    for j, v in items:
        w = abs(v) / cmax * 50
        bar_style = ({"left": "50%", "background": "var(--rlt-profit)"} if v >= 0
                     else {"right": "50%", "background": "var(--rlt-loss)"})
        bars.append(html.Div([
            html.Span(syms[j], className="sim-contrib-sym"),
            html.Span(html.Span(className="sim-contrib-bar", style={**bar_style, "width": f"{w:.1f}%"}),
                      className="sim-contrib-track"),
            html.Span(stl(v), className=f"sim-contrib-val {'sim-pos' if v > 0 else 'sim-neg' if v < 0 else ''}"),
        ], className="sim-contrib-row"))
    trades = book["trades"][i]
    nb = sum(t["action"] == "BUY" for t in trades)
    trade_rows = [html.Div([
        html.Span("ALIŞ" if t["action"] == "BUY" else "SATIŞ",
                  className="sim-side sim-side-buy" if t["action"] == "BUY" else "sim-side sim-side-sell"),
        html.Span(t["symbol"], className="sim-trade-sym"),
        html.Span(f"{_num(t['shares'])} lot × {_num(t['price'], 2)}", className="sim-trade-qty"),
        html.Span(tl(t["amount"]), className="sim-trade-amt"),
    ], className="sim-trade-row") for t in trades]
    if len(trade_rows) > 8:
        trade_rows = trade_rows[:8] + [html.Details([html.Summary(f"Tüm emirler ({len(trade_rows)})",
                                                                  className="sim-explain-sum"), *trade_rows[8:]])]
    if not trade_rows:
        trade_rows = [html.P("Yeniden dengeleme günü değil; emir yok." if is_rule else
                             "Politika bu gün tam lot eşiğini aşan bir değişiklik yapmadı; emir yok.",
                             className="sim-muted mb-0")]
    tone = "sim-pos" if day > 0 else ("sim-neg" if day < 0 else "")
    return [
        html.Div(d_long(dates[i]), className="sim-muted mb-2"),
        html.Div([html.Span(stl(day), className=f"sim-day-pnl {tone}"),
                  html.Span(spct(day / prev), className=f"sim-day-pct {tone}")], className="sim-day-big"),
        html.P(sentence, className="sim-muted"),
        html.Div("Hisse bazında günün katkısı", className="section-title"),
        html.Div(bars or [html.P("Önceki gün pozisyon yok.", className="sim-muted mb-0")], className="mb-3"),
        html.Div(f"Emirler · {len(trades)}" + (f" ({nb} alış, {len(trades) - nb} satış)" if trades else ""),
                 className="section-title"),
        html.Div(trade_rows, className="mb-2"),
        html.P([f"Nakit faizi {stl(book['cash_interest'][i])} · işlem maliyeti {tl(book['commission'][i])} · "
                "emirler o günün açılışında yürütülür"], className="sim-muted small mb-0"),
    ]


def holdings_block(book: dict, i: int, theme: str | None = None):
    """(halka grafik, renk açıklaması, tablo) — seçili günün kapanışı."""
    p = plot_palette(theme)
    syms, lots, mark, avg, real = (book[k] for k in ("symbols", "lots", "mark", "avg_cost", "realized"))
    L, M, A, R, V = lots[i], mark[i], avg[i], real[i], book["value"][i]
    Lp, Mp = (lots[i - 1], mark[i - 1]) if i else ([0] * len(syms), M)
    top = _top_groups(book, i)
    sh = _shares(book, i, top)
    names = [syms[j] for j in top] + ["Diğer hisseler", "Nakit"]
    colors = [p[GROUP_KEYS[g]] for g in range(len(top))] + [p["muted"], p["line"]]
    css = list(GROUP_CSS[:len(top)]) + [OTHER_CSS, CASH_CSS]
    keep = [g for g in range(len(names)) if sh[g] > 1e-9 or names[g] == "Nakit"]
    donut = go.Figure(go.Pie(labels=[names[g] for g in keep], values=[sh[g] for g in keep], hole=0.64, sort=False,
                             direction="clockwise", textinfo="none",
                             marker={"colors": [colors[g] for g in keep], "line": {"color": p["bg"], "width": 2}},
                             hovertemplate="%{label}: %{percent}<extra></extra>"))
    donut.update_layout(height=230, margin={"l": 0, "r": 0, "t": 0, "b": 0}, showlegend=False, separators=",.",
                        annotations=[{"text": f"Toplam<br><b>{tl(V)}</b>", "showarrow": False,
                                      "font": {"size": 14}}])
    apply_theme_template(donut, theme)
    legend = [html.Div([html.Span(className="sim-dot", style={"background": css[g]}),
                        html.Span(names[g], className="sim-legend-name"),
                        html.Span(pct(sh[g]), className="sim-legend-val")], className="sim-legend-row")
              for g in keep]
    rows, day_tot, tot_tot = [], 0.0, 0.0
    for j in sorted([j for j in range(len(syms)) if L[j] > 0], key=lambda j: -L[j] * M[j]):
        mv, chg = L[j] * M[j], (M[j] / Mp[j] - 1 if i and Mp[j] > 0 else 0.0)
        dpnl, tpnl, traded = book["contrib"][i][j], L[j] * (M[j] - A[j]) + R[j], L[j] - Lp[j]
        day_tot, tot_tot = day_tot + dpnl, tot_tot + tpnl
        rows.append({"symbol": syms[j], "lots": _num(L[j]), "avg": _num(A[j], 2), "close": _num(M[j], 2),
                     "chg": spct(chg), "value": tl(mv), "w": pct(mv / V), "day": stl(dpnl), "tot": stl(tpnl),
                     "trade": "—" if traded == 0 else _sign(traded, 0) + _num(traded)})
    rows.append({"symbol": "Nakit", "lots": "", "avg": "", "close": "", "chg": "", "value": tl(book["cash"][i]),
                 "w": pct(book["cash"][i] / V), "day": stl(book["cash_interest"][i]), "tot": "", "trade": ""})
    rows.append({"symbol": "Toplam", "lots": "", "avg": "", "close": "", "chg": "", "value": tl(V), "w": "%100,0",
                 "day": stl(V - (book["value"][i - 1] if i else book["capital"])), "tot": stl(V - book["capital"]),
                 "trade": ""})
    table = DataTable(
        columns=[{"name": "Hisse", "id": "symbol"}, {"name": "Lot", "id": "lots"}, {"name": "Ort. maliyet", "id": "avg"},
                 {"name": "Kapanış", "id": "close"}, {"name": "Gün", "id": "chg"},
                 {"name": "Piyasa değeri", "id": "value"}, {"name": "Ağırlık", "id": "w"},
                 {"name": "Gün K/Z", "id": "day"}, {"name": "Toplam K/Z", "id": "tot"},
                 {"name": "Bugün işlem", "id": "trade"}],
        data=rows, sort_action="none", **TABLE_STYLES,
        style_cell_conditional=numeric_columns("lots", "avg", "close", "chg", "value", "w", "day", "tot", "trade"),
        style_data_conditional=_sign_rules("chg", "day", "tot", "trade") + [
            {"if": {"filter_query": '{symbol} = "Toplam" || {symbol} = "Nakit"'}, "fontWeight": "600"},
            {"if": {"filter_query": '{symbol} = "Toplam"'}, "backgroundColor": "var(--rlt-surface-2)"}],
    )
    foot = html.P("1 lot = 1 pay · fiyatlar düzeltilmiş kapanış (yfinance) · ortalama maliyet ve hisse K/Z'si işlem "
                  "maliyeti hariç; Toplam satırı maliyet dahil portföy sonucu. Toplam K/Z = lot × (kapanış − ort. "
                  "maliyet) + gerçekleşen satış kârı. Bugün işlem: o günün açılışında alınan (+) / satılan (−) adet.",
                  className="sim-muted small mt-2 mb-0")
    return donut, legend, html.Div([table, foot])


def weights_figure(book: dict, i: int, rng: str, theme: str | None = None):
    p = plot_palette(theme)
    dates = book["dates"]
    k0 = _range_start(len(dates), rng)
    top = _top_groups(book, i)
    sh = [_shares(book, t, top) for t in range(k0, len(dates))]
    names = [book["symbols"][j] for j in top] + ["Diğer hisseler", "Nakit"]
    colors = [GROUP_KEYS[g] for g in range(len(top))] + ["muted", "line"]
    fig = go.Figure()
    for g, name in enumerate(names):
        fig.add_trace(go.Scatter(x=dates[k0:], y=[s[g] for s in sh], name=name, stackgroup="one", mode="lines",
                                 line={"width": 0.6, "color": p[colors[g]]}, fillcolor=plot_rgba(colors[g], 0.85, theme),
                                 hovertemplate="%{y:.1%}<extra>" + name + "</extra>"))
    if k0 <= i:
        fig.add_vline(x=dates[i], line={"color": p["text"], "width": 1, "dash": "dot"})
    fig.update_layout(height=260, margin={"l": 50, "r": 16, "t": 10, "b": 30}, hovermode="x unified",
                      separators=",.", yaxis={"tickformat": ".0%", "range": [0, 1]}, xaxis=date_axis(dates[k0:]),
                      legend={"orientation": "h", "y": -0.2})
    apply_theme_template(fig, theme)
    return fig


def seed_figure(view: dict, window: str, theme: str | None = None, focus: str | None = None, seed=None):
    """İncelenen kolun (yoksa ilk seçili kolun) 12 tohumunun pencere sonu toplam getirisi; incelenen tohum halkalı;
    kural portföyleri referans çizgisi. Nokta tıklanınca o tohum incelenir (customdata = tohum numarası)."""
    rows = (view or {}).get("table") or []
    arm = (next((r for r in rows if r.get("members") and r.get("key") == focus), None)
           or next((r for r in rows if r.get("members")), None))
    if not arm:
        return empty_figure("Tohum dağılımı için karşılaştırmaya bir kol ekleyin"), ""
    p = plot_palette(theme)
    vals = [m["cum_return"] for m in arm["members"]]
    order = sorted(range(len(vals)), key=lambda k: vals[k])
    ys = [((r % 3) - 1) * 0.28 for r in range(len(order))]
    cur = f"s{seed_of(arm['key'], seed)}" if arm.get("key") == focus else None
    sel = [arm["members"][k]["seed"] == cur for k in order]
    fig = go.Figure(go.Scatter(
        x=[vals[k] for k in order], y=ys, mode="markers", name=LABELS.get(arm.get("key"), arm["label"]),
        text=[arm["members"][k]["seed"].lstrip("s") for k in order],
        customdata=[int(arm["members"][k]["seed"].lstrip("s")) for k in order],
        marker={"size": [17 if x else 12 for x in sel], "color": p["blue"],
                "line": {"width": [3 if x else 1.5 for x in sel], "color": [p["text"] if x else p["bg"] for x in sel]}},
        hovertemplate="tohum %{text}: %{x:+.2%}<br><i>tıklayın: bu tohumu inceleyin</i><extra></extra>"))
    mean = sum(vals) / len(vals)
    fig.add_vline(x=mean, line={"color": p["text"], "width": 1.5},
                  annotation={"text": f"ort. {spct(mean, 1)}", "font": {"size": 11}}, annotation_position="top left")
    for n_ref, r in enumerate([r for r in rows if str(r.get("key", "")).startswith("rule/")]):
        c = p[LINE_KEYS[(n_ref + 1) % len(LINE_KEYS)]]
        fig.add_vline(x=r["cum_return"], line={"color": c, "width": 2, "dash": "dash"},
                      annotation={"text": f"{LABELS.get(r['key'], r['label'])} {spct(r['cum_return'], 1)}",
                                  "font": {"size": 11, "color": c}},
                      annotation_position="top" if n_ref % 2 == 0 else "bottom right")
    fig.update_layout(height=220, margin={"l": 20, "r": 20, "t": 30, "b": 36}, showlegend=False, separators=",.",
                      xaxis={"tickformat": "+.1%", "nticks": 6, "title": "Pencere sonu toplam getiri"},
                      yaxis={"visible": False, "range": [-1, 1]})
    apply_theme_template(fig, theme)
    head = f"{LABELS.get(arm.get('key'), arm['label'])} — " if arm.get("key") else ""
    if window == "forward":
        note = (head + "Ara görünüm: tohumların kural portföylerine göre konumu ileri testte yorumlanmaz; "
                "karşılaştırma 126. seansta ön kayıtlı kuralla bir kez okunur.")
    else:
        rule = next((r for r in rows if r.get("key") == "rule/ew"), None)
        above = f"12 tohumun {sum(v > rule['cum_return'] for v in vals)}'i 1/N'in üzerinde. " if rule else ""
        sd = (sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
        note = (f"{head}{above}Tohumlar arası yayılım (std {_num(sd * 100, 2)} puan) tek bir tohuma bakarak sonuç "
                "çıkarmamanın nedeni.")
    return fig, note


def select_view(view_all: dict, keys) -> dict:
    """Tüm stratejili görünümden yalnız seçilenler (grafik çizgileri + tohum grafiği); API'ye yeniden gitmez."""
    if not view_all:
        return view_all
    keys = [k for k in (keys or []) if k in (view_all.get("lines") or {})]
    rows = {r["key"]: r for r in view_all.get("table") or []}
    return {**view_all, "lines": {k: view_all["lines"][k] for k in keys},
            "table": [rows[k] for k in keys if k in rows]}


HEADER_TIPS = {"plot": plain.note_text("col_plot"), "n": plain.note_text("col_n"), "ret": plain.note_text("col_ret"),
               "sharpe": plain.note_text(term="sharpe", extra=f"Yıllık; en az {MIN_SHARPE_SESSIONS} işlem günü gerekir."),
               "dd": plain.note_text("drawdown"), "turn": plain.note_text(term="devir"),
               "cash": plain.note_text("col_cash")}


def compare_table(view: dict, shown=None):
    """Tüm stratejiler SABİT sırada (kollar, sonra kurallar). Sıralama yok: ileri veri ara görünümünde 'en iyi'
    seçilmez (ek §9, dashboard/CLAUDE.md ana sayfa kuralı). Grafikte çizilenler vurgulanır."""
    def pm(m, s, fmt):
        if m is None:
            return "—"
        return fmt(m) + ("" if s is None else f" ± {_num(abs(s) * (100 if fmt is spct else 1), 2)}")

    def num(x):
        return "—" if x is None else _sign(x, 5e-3) + _num(x, 2)
    shown = set(shown or [])
    src = view["table"]
    short = int(view.get("n_sessions") or 0) < MIN_SHARPE_SESSIONS
    rows = [{"label": LABELS.get(r["key"], r["label"]), "n": r["n"],
             "plot": "●" if r["key"] in shown else "",
             "ret": pm(r["cum_return"], r.get("cum_return_std"), spct),
             "sharpe": "—" if short else pm(r["excess_sharpe"], r.get("excess_sharpe_std"), num),
             "dd": spct(r["max_drawdown"], 1),
             "turn": pct(r["turnover"] or 0), "cash": pct(r["cash"] or 0)} for r in src]
    hl = [{"if": {"row_index": k}, "backgroundColor": "var(--rlt-surface-2)"}
          for k, r in enumerate(src) if r["key"] in shown]
    table = DataTable(
        columns=[{"name": "Strateji", "id": "label"}, {"name": "Grafikte", "id": "plot"}, {"name": "N", "id": "n"},
                 {"name": "Toplam getiri", "id": "ret"}, {"name": "Fazla getiri Sharpe", "id": "sharpe"},
                 {"name": "En büyük düşüş", "id": "dd"}, {"name": "Günlük devir", "id": "turn"},
                 {"name": "Ort. nakit", "id": "cash"}],
        data=rows, sort_action="none", **TABLE_STYLES,
        style_cell_conditional=numeric_columns("n", "ret", "sharpe", "dd", "turn", "cash"),
        style_data_conditional=hl + _sign_rules("ret"),
        tooltip_header=HEADER_TIPS, tooltip_delay=0, tooltip_duration=None,
    )
    foot = html.P(f"{view['n_sessions']} seans · maliyet {view['cost_bps_roundtrip']:.0f} bps gidiş-dönüş · kesirli "
                  "defter (sermaye getiriyi değiştirmez) · kollar 12 tohumun ortalaması ± std · fazla getiri Sharpe: "
                  "TL risksiz faiz üstü, yıllık" + (f" ({MIN_SHARPE_SESSIONS} seanstan önce gösterilmez: bu kadar kısa "
                                                    "seriden yıllık sayı gürültüdür)" if short else "") +
                  " · sıra sabittir, sıralama değildir", className="sim-muted small mt-2 mb-0")
    return html.Div([table, foot])


# ── sade özet (sayfanın üstü; §3.18) ────────────────────────────────────────
GROUP_COLOR = {"ai_multi": "blue", "ai_single": "blue", "control": "muted", "rule": "gold"}


def summary_figure(view: dict, theme: str | None = None):
    """'100.000 TL ile başlasaydınız': 10 strateji sabit sırada, yapay zekâ ± 12 tohum std. Gün kaydırıcısı ve Oynat
    düğmesi her günün durumunu gösterir (kareler istemcide; sunucuya gidilmez). Nokta tıklanınca strateji incelenir
    (customdata[2] = anahtar)."""
    lines = (view or {}).get("lines") or {}
    keys = [k for k in plain.ORDER if k in lines]
    dates = (view or {}).get("dates") or []
    if not keys or not dates:
        return empty_figure("Veri yok")
    p = plot_palette(theme)
    cap = float(view.get("capital") or 100_000)
    T = len(dates)
    groups = [(g, [k for k in keys if plain.group(k) == g]) for g in ("ai_multi", "ai_single", "control", "rule")]
    groups = [(g, ks) for g, ks in groups if ks]

    def val(k, t):
        return lines[k]["mean"][t]

    def sd(k, t):
        return lines[k]["std"][t] if lines[k].get("n", 1) > 1 else 0.0

    lo = min([cap] + [val(k, t) - sd(k, t) for k in keys for t in range(T)])
    hi = max([cap] + [val(k, t) + sd(k, t) for k in keys for t in range(T)])
    pad = 0.30 * (hi - lo or cap * 0.01)

    def day(t):
        out = []
        for g, ks in groups:
            x, err = [val(k, t) for k in ks], [sd(k, t) for k in ks]
            out.append(go.Scatter(
                x=x, y=[plain.name(k) for k in ks], mode="markers", name=plain.GROUPS[g],
                marker={"size": 12, "color": p[GROUP_COLOR[g]], "line": {"width": 1.5, "color": p["bg"]}},
                error_x={"type": "data", "array": err, "visible": g != "rule", "thickness": 2, "width": 4,
                         "color": plot_rgba(GROUP_COLOR[g], 0.6, theme)},
                customdata=[[spct(v / cap - 1), f"± {_num(e, 0)} TL (12 eğitim arası yayılım)" if e else "tek seri", k]
                            for k, v, e in zip(ks, x, err)],
                hovertemplate="%{y}<br>%{x:,.0f} TL · %{customdata[0]}<br>%{customdata[1]}<br>"
                              "<i>tıklayın: aşağıda inceleyin</i><extra></extra>"))
        # değerler sağda ayrı sütun: nokta ve yayılım çubuğunun üstüne binmesin
        out.append(go.Scatter(x=[hi + pad * 0.95] * len(keys), y=[plain.name(k) for k in keys], mode="text",
                              text=[f"{tl(val(k, t))}  ({spct(val(k, t) / cap - 1, 1)})" for k in keys],
                              textposition="middle left", textfont={"size": 11.5, "color": p["text"]},
                              showlegend=False, hoverinfo="skip"))
        return out

    fig = go.Figure(data=day(T - 1))
    fig.add_vline(x=cap, line={"color": p["line"], "width": 1.2, "dash": "dot"},
                  annotation={"text": f"başlangıç {tl(cap)}", "font": {"size": 11}},
                  annotation_position="bottom right")
    layout = {"height": 70 + 30 * len(keys), "margin": {"l": 10, "r": 10, "t": 34, "b": 30}, "separators": ",.",
              "showlegend": True, "legend": {"orientation": "h", "y": 1.0, "yanchor": "bottom", "x": 0,
                                             "font": {"size": 11}},
              "xaxis": {"tickformat": ",.0f", "range": [lo - pad * 0.4, hi + pad], "title": None},
              "yaxis": {"autorange": "reversed", "automargin": True}, "hovermode": "closest"}
    if T > 1:                                  # tek günde oynatılacak bir şey yok
        dur = int(max(60, min(500, 12_000 / T)))
        fig.frames = [go.Frame(data=day(t), name=str(t)) for t in range(T)]
        step = {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}
        layout["height"] += 80
        layout["margin"]["b"] = 96
        layout["sliders"] = [{
            "active": T - 1, "x": 0.14, "len": 0.86, "y": 0, "yanchor": "top", "pad": {"t": 40},
            "currentvalue": {"prefix": "Gün: ", "font": {"size": 12}}, "font": {"size": 10},
            "steps": [{"label": d_short(dates[t]), "method": "animate", "args": [[str(t)], step]} for t in range(T)]}]
        layout["updatemenus"] = [{
            "type": "buttons", "direction": "left", "showactive": False, "x": 0, "y": 0, "yanchor": "top",
            "xanchor": "left", "pad": {"t": 62, "r": 6}, "font": {"size": 11},
            "buttons": [
                {"label": "▶ Oynat", "method": "animate",
                 "args": [None, {"frame": {"duration": dur, "redraw": True}, "transition": {"duration": 0},
                                 "fromcurrent": False, "mode": "immediate"}]},
                {"label": "❚❚", "method": "animate", "args": [[None], {**step, "frame": {"duration": 0, "redraw": False}}]},
            ]}]
    fig.update_layout(**layout)
    apply_theme_template(fig, theme)
    return fig


def _range_tl(vals: list) -> str:
    lo, hi = min(vals), max(vals)
    return tl(lo) if abs(hi - lo) < 0.5 else f"{_num(lo)} – {tl(hi)}"


def summary_text(view: dict, window: str) -> list:
    """Üç satır sonuç + bir satır 'nasıl yorumlanmalı'. Sayılar görünümden; yorum dili ön kayda bağlı (ek §9)."""
    rows = {r["key"]: r for r in (view or {}).get("table") or []}
    if not rows:
        return [html.P("Henüz veri yok.", className="sim-muted")]
    cap = float(view.get("capital") or 100_000)
    val = {k: cap * (1 + r["cum_return"]) for k, r in rows.items()}
    ai = [val[k] for k in plain.AI_KEYS if k in val]
    rules = [val[k] for k in plain.RULE_KEYS if k in val]
    n, last = int(view.get("n_sessions") or 0), (view.get("dates") or [""])[-1]
    lines = [html.Div([html.Span(className="sim-dot sim-dot-ai"),
                       html.Span(f"Yapay zekâ modelleri ({len(ai)})", className="sim-sum-label"),
                       html.Strong(_range_tl(ai), className="sim-num")], className="sim-sum-row")] if ai else []
    if rules:
        lines.append(html.Div([html.Span(className="sim-dot sim-dot-rule"),
                               html.Span(f"Basit kurallar ({len(rules)})", className="sim-sum-label"),
                               html.Strong(_range_tl(rules), className="sim-num")], className="sim-sum-row"))
    if "null" in val:
        lines.append(html.Div([html.Span(className="sim-dot sim-dot-control"),
                               html.Span("Kontrol grubu", className="sim-sum-label"),
                               html.Strong(tl(val["null"]), className="sim-num")], className="sim-sum-row"))
    ew = rows.get("rule/ew")
    market = (f" Aynı dönemde parayı hisselere eşit bölmek {spct(ew['cum_return'], 1)} getirdi; bu, piyasanın genel "
              "yönü için kaba bir ölçü." if ew else "")
    if window == "forward":
        read = (f"Bu, {n} işlem günlük bir ara görünüm; değerlendirme {READ_SESSIONS}. günde, önceden yazılmış kurala "
                f"göre bir kez yapılacak. Bu kadar kısa sürede aradaki farklar yorumlanmaz.{market}")
        when = f"{d_full(last)} kapanışı itibarıyla · {n} / {READ_SESSIONS} gün" if last else ""
    else:
        read = ("Bu, 6 yıllık sınavın (M8) yalnız 2026 bölümü; sonucu zaten okundu. Tek yılın sonucu genellenmez, 6 "
                f"yılın sonucu ana sayfada.{market}")
        when = f"{d_full(view['dates'][0])} – {d_full(last)} · {n} işlem günü" if last else ""
    return [html.Div(when, className="sim-muted small mb-2"), html.Div(lines, className="sim-sum-rows"),
            html.P(read, className="sim-sum-read")]


def diff_figure(view: dict, theme: str | None = None):
    """Eşli fark: her gün strateji − 1/N (aynı günler), puan. Kol: 12 tohumun farklarının ortalaması ± std."""
    lines = {k: v for k, v in ((view or {}).get("lines") or {}).items() if k != "rule/ew" and v.get("diff_ew")}
    if not lines:
        return empty_figure("Kıyas için 1/N dışında bir strateji seçin")
    p = plot_palette(theme)
    cap = float(view.get("capital") or 100_000)
    dates = view["dates"]
    order = list(view["lines"])              # renk ana grafikteki sırayla aynı (aynı strateji aynı renk)
    fig = go.Figure()
    for key, line in lines.items():
        ck = LINE_KEYS[order.index(key) % len(LINE_KEYS)]
        m = [x / cap * 100 for x in line["diff_ew"]["mean"]]
        s = [x / cap * 100 for x in line["diff_ew"]["std"]]
        if line["n"] > 1:
            fig.add_trace(go.Scatter(x=dates + dates[::-1], y=[a + b for a, b in zip(m, s)] + [a - b for a, b in
                                     zip(m[::-1], s[::-1])], fill="toself", line={"width": 0}, hoverinfo="skip",
                                     fillcolor=plot_rgba(ck, 0.13, theme), showlegend=False))
        fig.add_trace(go.Scatter(x=dates, y=m, name=LABELS.get(key, key), mode="lines+markers" if len(dates) <= 10
                                 else "lines", line={"color": p[ck], "width": 2},
                                 hovertemplate="%{y:+.2f} puan<extra>" + LABELS.get(key, key) + "</extra>"))
    fig.add_hline(y=0, line={"color": p["line"], "width": 1.2})
    fig.update_layout(height=300, margin={"l": 56, "r": 16, "t": 10, "b": 30}, hovermode="x unified", separators=",.",
                      yaxis={"title": "1/N'e göre fark (puan)", "tickformat": "+.1f"}, xaxis=date_axis(dates),
                      legend={"orientation": "h", "y": -0.18, "x": 0, "font": {"size": 11}})
    apply_theme_template(fig, theme)
    return fig


def banner(window: str, status: dict) -> list:
    s = status or {}
    if s.get("available") is False:
        return [dbc.Alert("Dondurulmuş M8 arşivi bu sunucuda yok (data/marl/frozen).", color="secondary",
                          className="mb-0")]
    n, need = int(s.get("n_sessions") or 0), int(s.get("read_sessions") or READ_SESSIONS)
    last = s.get("last_run") or {}
    run_txt = "çalışıyor" if s.get("running") else RUN_STATUS.get(last.get("status"), last.get("status") or "—")
    margin = (last.get("margin_coverage") or {}).get("horizon_end") or "—"
    fin = (last.get("finished_trt") or "")[:16].replace("T", " ")     # TRT (§3.14); ham `finished` yazanin yerel saati
    meta = (f"Son defter günü {s.get('last_date') or '—'} · son çalışma {run_txt} {fin + ' TRT' if fin else ''} · "
            f"marj denetimi geçerli {margin}")
    if window == "forward":
        title, text = "Ara görünüm — tez sonucu değildir", NOTE_FORWARD
        bar = html.Div([
            html.Div([html.Span("İleri veri seansı", className="sim-muted"),
                      html.Strong(f"{n} / {need}", className="sim-num")], className="sim-progress-head"),
            dbc.Progress(value=min(100.0, n / need * 100), className="sim-progress"),
            html.Div(f"İlk gün {d_full(FORWARD_START)} · {meta}", className="sim-muted small"),
        ], className="sim-banner-side")
    else:
        title, text = "Geçmiş pencere — 2026 test", NOTE_TEST
        bar = html.Div(html.Div(meta, className="sim-muted small"), className="sim-banner-side")
    return [html.Div([
        html.Div([html.I(className="bi bi-info-circle sim-banner-icon"),
                  html.Div([html.Div(title, className="sim-banner-title"), html.Div(text, className="sim-muted small")])],
                 className="sim-banner-main"),
        bar,
    ], className="sim-banner")]


# ── sayfa ────────────────────────────────────────────────────────────────────
def layout():
    status = api.get_marl_forward_status()
    running = bool(status.get("running"))
    n = int(status.get("n_sessions") or 0)
    default_window = "forward" if n > 0 else "test2026"
    admin_btn = dbc.Button([html.I(className="bi bi-play-circle me-2"), "Şimdi çalıştır"],
                           id="sim-run-btn", color="primary", size="sm") if is_admin() else html.Span(id="sim-run-btn")
    return html.Div([
        dcc.Interval(id="sim-poll", interval=5_000, disabled=not running, n_intervals=0),
        dcc.Store(id="sim-view-all"), dcc.Store(id="sim-view"), dcc.Store(id="sim-book"), dcc.Store(id="sim-sel"),
        create_page_header("Sanal Para Simülasyonu",
                           "Eğitimi bitmiş ve artık değiştirilmeyen yapay zekâ modelleri ile basit kurallar, aynı sanal "
                           "parayla, aynı günlerde ve aynı işlem maliyetiyle yatırım yapıyor",
                           actions=admin_btn),
        html.Div(id="sim-run-msg", className="mb-2"),
        # 1. sade özet: alan dışından bir okur yalnız bu kartı okuyup çıkabilmeli (§3.18)
        dbc.Card([
            dbc.CardHeader(html.Div([
                html.Span(["100.000 TL ile başlasaydınız, bugün ", plain.info("sim-i-summary", text=plain.NOTES["summary"],
                                                                         title="Nasıl okunur?")],
                          className="card-title-sm"),
                dbc.RadioItems(id="sim-window", inline=True, value=default_window, className="sim-range", options=[
                    {"label": "Gerçek piyasada canlı deneme (23 Kas →)", "value": "forward"},
                    {"label": "2026 sınav dönemi", "value": "test2026"}]),
            ], className="sim-card-head")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(dcc.Loading(dcc.Graph(id="sim-summary-chart", figure=empty_figure(),
                                                  config={"displayModeBar": False})), lg=7, className="mb-2"),
                    dbc.Col(html.Div(id="sim-summary-text"), lg=5, className="mb-2"),
                ]),
                html.Details([html.Summary("Bu stratejiler ne? (sade açıklama)", className="sim-explain-sum"),
                              html.Div(plain.strategy_list("sim"), className="mt-2")], className="sim-explain"),
                html.Details([html.Summary("Terimler nasıl okunur?", className="sim-explain-sum"),
                              html.Div(plain.how_to_read(["tohum", "baseline-1n", "kural-portfoyleri",
                                                          "ileri-veri-testi", "sharpe"], cls="sim"),
                                       className="mt-2")], className="sim-explain"),
            ]),
        ], className="mb-3"),
        dbc.Card(dbc.CardBody(html.Div(id="sim-banner", children=banner(default_window, status))), className="mb-4"),
        # 2. ayrıntı: tek bir portföyün içi
        dcc.Store(id="sim-scroll"),
        html.Div([html.Div("Tek bir portföyün içine bakın", className="sim-section-title"),
                  html.Div("Bir strateji ve tohum seçin: gerçek lot sayılarıyla gün gün ne aldığını, ne sattığını ve "
                           "ne kazandığını görün. İpucu: yukarıdaki grafikte bir noktaya tıklamak da strateji seçer.",
                           className="sim-muted small")], className="sim-section", id="sim-detail"),
        dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col([html.Label("İncelenen portföy", className="section-title"),
                     dcc.Dropdown(id="sim-focus", options=ARM_OPTIONS, value="M-none", clearable=False)],
                    lg=5, md=6, className="mb-2"),
            dbc.Col([html.Label("Tohum (12 eğitimden hangisi)", className="section-title"),
                     dcc.Dropdown(id="sim-seed", options=[{"label": str(s), "value": s} for s in SEEDS], value=201,
                                  clearable=False)], lg=3, md=3, className="mb-2"),
            dbc.Col([html.Label("Sermaye (TL)", className="section-title"),
                     dbc.Input(id="sim-capital", type="number", value=100_000, min=1_000, step=1_000, debounce=True)],
                    lg=4, md=3, className="mb-2"),
        ], className="g-3")), className="mb-1"),
        html.P(id="sim-seed-hint", className="sim-muted small mb-3",
               children="İncelenen portföy seçili TEK tohumdur; kolun 12 tohum ortalaması grafikte kesikli çizgi ve "
                        "'Kol ortalaması' kartında. Ortalamanın lot karşılığı yoktur. Kural portföylerinde tohum yoktur."),
        # Yalnız portföy (API) yüklenirken döner ve içerik görünür kalır; aralık / gün seçimi bütün bloğu örtmez
        dcc.Loading(target_components={"sim-book-msg": "children"}, delay_show=250,
                    overlay_style={"visibility": "visible", "opacity": 0.6}, children=html.Div([
            html.Div(id="sim-book-msg"),
            dbc.Row(id="sim-kpis", className="g-3"),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader(html.Div([
                        html.Div([html.Span("Portföy değeri", className="card-title-sm"),
                                  html.Div("Kalın mavi çizgi incelenen portföy (tam lot); kesikli çizgiler aynı "
                                           "sermayeyle kıyas. Bir güne tıklayın.", className="sim-muted small")]),
                        dbc.RadioItems(id="sim-range", options=RANGES, value="all", inline=True,
                                       className="sim-range"),
                    ], className="sim-card-head")),
                    dbc.CardBody(dcc.Graph(id="sim-chart", figure=empty_figure(),
                                           config={"displayModeBar": False}, style={"height": "600px"})),
                ], className="h-100"), lg=8, className="mb-3"),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.Div([
                        html.Span("Gün özeti", className="card-title-sm"),
                        html.Div([
                            dbc.Button(html.I(className="bi bi-chevron-left"), id="sim-prev", color="secondary",
                                       outline=True, size="sm", title="Önceki gün"),
                            dbc.Button(html.I(className="bi bi-chevron-right"), id="sim-next", color="secondary",
                                       outline=True, size="sm", title="Sonraki gün", className="ms-1"),
                        ]),
                    ], className="sim-day-head"),
                    html.Div(id="sim-day"),
                ]), className="h-100"), lg=4, className="mb-3"),
            ]),
            dbc.Card([
                dbc.CardHeader(html.Div([
                    html.Div([html.Span("Portföyümde ne var", className="card-title-sm"),
                              html.Div(id="sim-hold-sub", className="sim-muted small")]),
                ], className="sim-card-head")),
                dbc.CardBody(dbc.Row([
                    dbc.Col([dcc.Graph(id="sim-donut", figure=empty_figure(), config={"displayModeBar": False},
                                       style={"height": "230px"}),
                             html.Div(id="sim-donut-legend", className="sim-legend")], lg=3, className="mb-3"),
                    dbc.Col(html.Div(id="sim-hold-table"), lg=9),
                ])),
            ], className="mb-3"),
            dbc.Card([
                dbc.CardHeader(html.Div([
                    html.Div([html.Span("Dağılım zamanla", className="card-title-sm"),
                              html.Div("Portföyün her gün hisselere ve nakde bölünüşü (piyasa değeri payı). Renkler "
                                       "yukarıdaki dağılımla aynı.", className="sim-muted small")]),
                ], className="sim-card-head")),
                dbc.CardBody(dcc.Graph(id="sim-weights", figure=empty_figure(), config={"displayModeBar": False},
                                       style={"height": "260px"})),
            ], className="mb-3"),
        ])),
        # 3. ayrıntı: teknik karşılaştırma
        html.Div([html.Div("Stratejileri karşılaştırın (ayrıntılı)", className="sim-section-title"),
                  html.Div("Akademik okur için: aynı pencere ve maliyette ortalama ± std, riske göre getiri, eşli fark "
                           "ve 12 eğitimin tek tek sonuçları.", className="sim-muted small")], className="sim-section"),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.Span("Strateji karşılaştırması", className="card-title-sm")),
                dbc.CardBody([
                    html.Label("Grafikte kesikli çizgi olarak gösterilenler (tablo hepsini gösterir)",
                               className="section-title"),
                    dcc.Dropdown(id="sim-keys", options=ARM_OPTIONS, value=DEFAULT_KEYS, multi=True, clearable=False,
                                 className="mb-2"),
                    html.Div(id="sim-note", className="sim-muted small mb-2"),
                    dcc.Loading(html.Div(id="sim-table")),
                ]),
            ], className="h-100"), lg=8, className="mb-3"),
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.Span(["İncelenen kolun 12 tohumu ", plain.info(
                    "sim-i-seeds", text=plain.NOTES["seeds"], title="12 tohum")], className="card-title-sm")),
                dbc.CardBody([
                    html.P("Her tohum ayrı bir politika; ortalama ± std bandının arkasındaki tek tek sonuçlar.",
                           className="sim-muted small"),
                    dcc.Graph(id="sim-seeds", figure=empty_figure(), config={"displayModeBar": False},
                              style={"height": "220px"}),
                    html.Div(id="sim-seed-note", className="sim-note"),
                ]),
            ], className="h-100"), lg=4, className="mb-3"),
        ]),
        dbc.Card([
            dbc.CardHeader(html.Div([
                html.Div([html.Span(["Eşit ağırlığa (1/N) göre fark ", plain.info(
                    "sim-i-diff", text=plain.NOTES["diff"], title="Eşli fark")], className="card-title-sm"),
                          html.Div("Her gün, aynı günün 1/N sonucuyla eşleştirilmiş birikimli fark (puan = yüzde puanı). "
                                   "Sıfırın üstü: 1/N'den fazla kazandırdı. Bant: 12 eğitim arasındaki yayılım. "
                                   "Betimseldir, istatistik testi değildir.", className="sim-muted small")]),
            ], className="sim-card-head")),
            dbc.CardBody(dcc.Graph(id="sim-diff", figure=empty_figure(), config={"displayModeBar": False})),
        ], className="mb-3"),
        dbc.Card([
            dbc.CardHeader(html.Span("Eski tek ajan modelleri", className="card-title-sm")),
            dbc.CardBody([
                html.P("Faz 1-9'un tek ajanlı modelleri farklı bir ortam, gözlem ve muhasebeyle eğitildi; bu "
                       "defterlerle doğrudan kıyaslanmaz.", className="sim-muted small"),
                dbc.Button("Modeller", href="/dash/models", color="secondary", outline=True, size="sm",
                           className="me-2"),
                dbc.Button("Günlük karar ve kağıt portföy", href="/dash/daily-trading", color="secondary",
                           outline=True, size="sm"),
            ]),
        ]),
    ])


def _focus_label(focus: str, seed) -> str:
    return LABELS.get(focus, focus) + (f" · tohum {seed_of(focus, seed)}" if focus in ARMS else "")


def plotted_keys(keys, focus) -> list:
    """Grafikte çizilenler: seçilen kıyaslar + incelenen kolun 12 tohum ortalaması (tek tohum yalnız kalmasın)."""
    keys = list(keys or [])
    return keys + [focus] if focus in ARMS and focus not in keys else keys


def render_chart(book: dict, view: dict, sel, rng: str, focus: str, seed):
    """Yalnız ana grafik: aralık (1A/3A/Tümü) değişince başka hiçbir bölüm yeniden üretilmez."""
    return main_figure(book, view, sel_index(book, sel), rng, _focus_label(focus, seed))


def render_panels(book: dict, view: dict, sel, focus: str, seed):
    """Grafik dışındaki defter bölümleri (kartlar, gün özeti, eldeki hisseler, dağılım zamanla); aralıktan bağımsız."""
    i = sel_index(book, sel)
    label = _focus_label(focus, seed)
    donut, legend, table = holdings_block(book, i)
    sub = f"{d_full(book['dates'][i])} kapanışı · {label} · 1 lot = 1 pay"
    diff = book.get("diff") or {}
    if diff.get("final_diff_pct") is not None:
        sub += (f" · kesirli deftere göre son değer farkı {spct(diff['final_diff_pct'])}, yıllık izleme hatası "
                f"{pct(diff.get('tracking_error_ann') or 0, 2)}, {diff.get('n_orders') or 0} emir")
    return (kpi_cards(book, view, i, focus, seed), day_panel(book, i, focus.startswith("rule/")),
            donut, legend, table, weights_figure(book, i, "all"), sub, i == 0, i == len(book["dates"]) - 1)


def render(book: dict, view: dict, sel, rng: str, focus: str, seed):
    """İki callback'in birleşik çıktısı (saf; test sentetik defterle çağırır): grafik + 9 bölüm."""
    return (render_chart(book, view, sel, rng, focus, seed),) + render_panels(book, view, sel, focus, seed)


def register_callbacks(app):
    @app.callback(
        Output("sim-view-all", "data"), Output("sim-note", "children"),
        Input("sim-window", "value"),
    )
    def load_view(window):
        """Pencere başına TEK istek: 10 stratejinin hepsi (ham seriler sunucuda bir kez okunur)."""
        note = NOTE_FORWARD if window == "forward" else NOTE_TEST
        v = api.get_marl_forward_view(window, ALL_KEYS)
        if not v or not v.get("dates"):
            msg = ("Henüz ileri veri seansı yok (ilk gün 2026-11-23, 19:00 TRT sonrası işlenir)"
                   if window == "forward" else "Veri alınamadı")
            return {"empty": msg}, note
        return v, note

    @app.callback(
        Output("sim-view", "data"), Output("sim-table", "children"), Output("sim-diff", "figure"),
        Input("sim-view-all", "data"), Input("sim-keys", "value"), Input("sim-focus", "value"),
        State("sim-window", "value"),
    )
    def update_view(v, keys, focus, window):
        if not v:
            return no_update, no_update, no_update
        if v.get("empty"):
            return None, html.P(v["empty"], className="sim-muted small"), empty_figure(v["empty"])
        shown = plotted_keys(keys, focus)
        sub = select_view(v, shown)
        return sub, compare_table(v, shown), diff_figure(sub)

    @app.callback(
        Output("sim-seeds", "figure"), Output("sim-seed-note", "children"),
        Input("sim-view", "data"), Input("sim-focus", "value"), Input("sim-seed", "value"),
        State("sim-window", "value"),
    )
    def update_seeds(view, focus, seed, window):
        if not view or not view.get("table"):
            return empty_figure("Tohum dağılımı için bir kol seçin"), ""
        return seed_figure(view, window, focus=focus, seed=seed)

    @app.callback(Output("sim-focus", "value"), Input("sim-summary-chart", "clickData"), prevent_initial_call=True)
    def pick_strategy(click):
        """Özet grafikte bir noktaya tıklama: o strateji incelenir (yazı sütununda customdata yok)."""
        try:
            key = click["points"][0]["customdata"][2]
        except (TypeError, KeyError, IndexError):
            return no_update
        return key if key in LABELS else no_update

    @app.callback(Output("sim-seed", "value", allow_duplicate=True), Input("sim-seeds", "clickData"),
                  State("sim-focus", "value"), prevent_initial_call=True)
    def pick_seed(click, focus):
        try:
            seed = int(click["points"][0]["customdata"])
        except (TypeError, KeyError, IndexError, ValueError):
            return no_update
        return seed if focus in ARMS and seed in seeds_for(focus) else no_update

    # tıklamadan sonra ayrıntı bölümüne kaydır (istemcide; sunucuya gitmez)
    app.clientside_callback(
        """function(c) { if (c && c.points && c.points[0] && c.points[0].customdata) {
               var e = document.getElementById('sim-detail'); if (e) { e.scrollIntoView({behavior: 'smooth'}); } }
           return window.dash_clientside.no_update; }""",
        Output("sim-scroll", "data"), Input("sim-summary-chart", "clickData"), prevent_initial_call=True)

    @app.callback(
        Output("sim-summary-chart", "figure"), Output("sim-summary-text", "children"),
        Input("sim-view-all", "data"), State("sim-window", "value"),
    )
    def update_summary(v, window):
        if not v or v.get("empty"):
            msg = (v or {}).get("empty") or "Veri alınamadı"
            return empty_figure(msg), html.P(msg, className="sim-muted")
        return summary_figure(v), summary_text(v, window)

    @app.callback(
        Output("sim-book", "data"), Output("sim-sel", "data"), Output("sim-book-msg", "children"),
        Input("sim-window", "value"), Input("sim-focus", "value"), Input("sim-seed", "value"),
        Input("sim-capital", "value"),
    )
    def update_book(window, focus, seed, capital):
        if not focus or not capital or float(capital) <= 0:
            return None, None, dbc.Alert("Strateji ve pozitif sermaye girin.", color="warning", className="py-2")
        r = api.get_marl_product(product_sid(focus, seed), float(capital), window)
        if not r:
            return None, None, dbc.Alert("Portföy hesaplanamadı.", color="danger", className="py-2")
        if not r.get("dates") or not r.get("book"):
            return None, None, html.P(r.get("note") or "Bu pencerede veri yok.", className="sim-muted")
        book = r["book"]
        book["diff"] = {"final_diff_pct": r.get("final_diff_pct"), "tracking_error_ann": r.get("tracking_error_ann"),
                        "n_orders": r.get("n_orders")}
        return book, None, None

    @app.callback(
        Output("sim-sel", "data", allow_duplicate=True),
        Input("sim-chart", "clickData"), Input("sim-prev", "n_clicks"), Input("sim-next", "n_clicks"),
        State("sim-sel", "data"), State("sim-book", "data"), prevent_initial_call=True,
    )
    def select_day(click, _p, _n, sel, book):
        if not book:
            return no_update
        dates, trig = book["dates"], ctx.triggered_id
        if trig == "sim-chart":
            try:
                x = str(click["points"][0]["x"])[:10]
            except (TypeError, KeyError, IndexError):
                return no_update
            return x if x in dates else no_update
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            return no_update        # gün paneli yeniden çizilince düğme n_clicks=None ile tetiklenir
        i = sel_index(book, sel)
        i = max(0, i - 1) if trig == "sim-prev" else min(len(dates) - 1, i + 1)
        return dates[i]

    @app.callback(
        Output("sim-chart", "figure"),
        Input("sim-book", "data"), Input("sim-view", "data"), Input("sim-sel", "data"), Input("sim-range", "value"),
        State("sim-focus", "value"), State("sim-seed", "value"),
    )
    def update_chart(book, view, sel, rng, focus, seed):
        if not book or not book.get("dates"):
            return empty_figure("Veri yok")
        return render_chart(book, view, sel, rng or "all", focus or "M-none", seed)

    @app.callback(
        Output("sim-kpis", "children"), Output("sim-day", "children"),
        Output("sim-donut", "figure"), Output("sim-donut-legend", "children"), Output("sim-hold-table", "children"),
        Output("sim-weights", "figure"), Output("sim-hold-sub", "children"),
        Output("sim-prev", "disabled"), Output("sim-next", "disabled"),
        Input("sim-book", "data"), Input("sim-view", "data"), Input("sim-sel", "data"),
        State("sim-focus", "value"), State("sim-seed", "value"),
    )
    def update_panels(book, view, sel, focus, seed):
        if not book or not book.get("dates"):
            e = empty_figure("Veri yok")
            return ([], html.P("Gün seçmek için önce portföy yüklenmeli.", className="sim-muted"), e, [], None, e,
                    "", True, True)
        return render_panels(book, view, sel, focus or "M-none", seed)

    @app.callback(Output("sim-seed", "disabled"), Output("sim-seed", "options"), Output("sim-seed", "value"),
                  Input("sim-focus", "value"), State("sim-seed", "value"))
    def seed_enabled(focus, seed):
        s = seeds_for(focus)
        return focus not in ARMS, [{"label": str(x), "value": x} for x in s], seed_of(focus, seed)

    @app.callback(
        Output("sim-run-msg", "children"), Output("sim-poll", "disabled", allow_duplicate=True),
        Input("sim-run-btn", "n_clicks"), prevent_initial_call=True,
    )
    def run_now(n):
        if not n:
            return no_update, no_update
        r = api.run_marl_forward()
        if not r:
            return dbc.Alert("İş başlatılamadı.", color="danger", className="py-2 mb-0"), no_update
        msg = "İş başlatıldı." if r.get("started") else "İş zaten çalışıyor."
        return dbc.Alert(msg, color="info", className="py-2 mb-0"), False

    @app.callback(
        Output("sim-banner", "children"), Output("sim-poll", "disabled"),
        Input("sim-window", "value"), Input("sim-poll", "n_intervals"), prevent_initial_call=True,
    )
    def poll(window, _):
        s = api.get_marl_forward_status()
        return banner(window, s), not bool(s.get("running"))
