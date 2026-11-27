"""
Sade dil katmanı (Faz 11 §3.18): ana sayfa ve simülasyon sayfasının ÜSTÜ alan dışından birinin (hoca, yatırımcı,
meraklı) okuyabileceği dilde; ayrıntı aşağı doğru artar.

- Strateji adları ve tek cümlelik açıklamalar TEK YER burası (simülasyon + ana sayfa ithal eder). Teknik kod
  (M-none, S-esm, ...) adın sonunda parantezde kalır ki akademik okur M8 belgeleriyle eşleştirebilsin.
- Terim tanımları ELLE YAZILMAZ: `dashboard/content/glossary.yaml` `sade` alanından okunur (K28 tek kaynak) ve
  Rehber'deki terim çapasına (`/dash/guide#terim-<id>`) bağlanır.
- Burada deney sonucu sayısı YOK; sayılar çağıranın verdiği kaynaktan (snapshot.json m8_ozet, /forward/view).
"""

from __future__ import annotations

import logging
import os

import yaml
from dash import html

_log = logging.getLogger(__name__)
CONTENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "content")

# anahtar -> (sade ad, tek cümle, grup). Grup: ai_multi | ai_single | control | rule
STRATEGIES = {
    "M-none": ("Çok ajanlı · iletişimsiz (M-none)",
               "Her hisseye ayrı bir ajan karar verir. Ajanlar aynı öğrenilmiş modeli paylaşır ama birbirine bilgi "
               "göndermez.", "ai_multi"),
    "M-gat": ("Çok ajanlı · komşulara bakan (M-gat)",
              "Her ajan, aynı sektördeki ve fiyatı birlikte hareket eden hisselerin ajanlarından bilgi alır.",
              "ai_multi"),
    "M-self": ("Çok ajanlı · iletişim kontrolü (M-self)",
               "M-gat ile aynı büyüklükte ama her ajan yalnız kendine bakar; iletişimin bir şey katıp katmadığını "
               "ayırmak için.", "ai_multi"),
    "S-esm": ("Tek ajanlı · eş yapı (S-esm)",
              "Bütün portföy kararını tek bir ajan verir; ağ yapısı çok ajanlı modelinkiyle aynı.", "ai_single"),
    "S-agr": ("Tek ajanlı · tek büyük ağ (S-agr)",
              "Bütün portföy kararını tek bir ajan verir; 30 hissenin hepsini tek büyük ağla birlikte görür.",
              "ai_single"),
    "null": ("Kontrol grubu (boş kol)",
             "Çok ajanlı iletişimsiz modelin ayrı tohumlarla eğitilmiş ikinci kopyası. Aday değildir: iki kopya "
             "arasındaki fark, sonuçların şansa ne kadar bağlı olduğunu gösterir.", "control"),
    "rule/ew": ("Eşit ağırlık (1/N)", "Parayı 30 hisseye eşit böler ve her gün yeniden eşitler.", "rule"),
    "rule/bh": ("Al ve tut", "İlk gün parayı eşit böler, sonra hiç işlem yapmaz.", "rule"),
    "rule/minvar": ("En düşük oynaklık (min-varyans)",
                    "Son bir yılın verisine göre en az dalgalanan hisse karışımını seçer; ayda bir günceller.", "rule"),
    "rule/hrp": ("Risk dengeli (HRP)",
                 "Benzer hisseleri gruplar, riski gruplar arasında dengeli dağıtır; ayda bir günceller.", "rule"),
}
ORDER = list(STRATEGIES)          # sabit sıra: yapay zekâ, kontrol, kurallar (SIRALAMA DEĞİL)
GROUPS = {"ai_multi": "Yapay zekâ · çok ajanlı", "ai_single": "Yapay zekâ · tek ajanlı", "control": "Kontrol grubu",
          "rule": "Basit kurallar (öğrenme yok)"}
AI_KEYS = [k for k, v in STRATEGIES.items() if v[2] in ("ai_multi", "ai_single")]
RULE_KEYS = [k for k, v in STRATEGIES.items() if v[2] == "rule"]
SEED_NOTE = ("Her yapay zekâ modeli 12 kez, 12 farklı rastgele başlangıçla (tohum) eğitildi; aksi yazmadıkça "
             "12'sinin ortalaması gösterilir.")


def name(key: str) -> str:
    return STRATEGIES.get(key, (key,))[0]


def group(key: str) -> str:
    return STRATEGIES.get(key, ("", "", "rule"))[2]


_GLOSSARY: dict | None = None


def glossary() -> dict:
    """{id: {'terim', 'sade'}} — glossary.yaml'dan, bir kez okunur. Dosya yoksa boş (sayfa yine kurulur)."""
    global _GLOSSARY
    if _GLOSSARY is None:
        try:
            with open(os.path.join(CONTENT_DIR, "glossary.yaml"), encoding="utf-8") as fh:
                _GLOSSARY = {t["id"]: t for t in yaml.safe_load(fh)}
        except (OSError, ValueError, TypeError, KeyError) as exc:
            _log.warning("glossary.yaml okunamadi: %s", exc)
            _GLOSSARY = {}
    return _GLOSSARY


def term_href(term_id: str) -> str:
    return f"/dash/guide#terim-{term_id}"


def how_to_read(term_ids: list, cls: str = "home") -> html.Div:
    """'Nasıl okunur' şeridi: her terim için sözlükteki sade tanım + Rehber bağlantısı."""
    g = glossary()
    items = []
    for tid in term_ids:
        t = g.get(tid)
        if not t:
            continue
        items.append(html.Div([
            html.A(t["terim"], href=term_href(tid), className=f"{cls}-term-name"),
            html.Div(str(t.get("sade", "")).strip(), className=f"{cls}-term-text"),
        ], className=f"{cls}-term"))
    return html.Div(items, className=f"{cls}-terms")


def strategy_list(cls: str = "sim") -> html.Div:
    """Stratejilerin sade açıklaması, gruplu ve sabit sırada."""
    out = []
    for gkey, glabel in GROUPS.items():
        rows = [html.Li([html.Strong(v[0]), html.Span(" — " + v[1])]) for k, v in STRATEGIES.items() if v[2] == gkey]
        out += [html.Div(glabel, className=f"{cls}-explain-group"), html.Ul(rows, className=f"{cls}-explain-list")]
    out.append(html.P(SEED_NOTE, className=f"{cls}-muted small mb-0"))
    return html.Div(out)
