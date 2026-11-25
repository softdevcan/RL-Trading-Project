"""Veri kataloğu (Faz 11 §3.17): diskteki veri setleri, hangi işe yaradıkları ve nasıl güncellendikleri.

Veri sayfası yalnız dört "ham piyasa" kaynağını gösteriyordu (BIST-30, altın, makro, fundamental). Değerlendirmenin
referans serileri (TÜFE, USD/TRY, XU030/XU100), tarihsel evren, M8'in dondurulmuş kaynağı ve ileri veri paneli sayfada
yoktu; "12 iş günü eksik" görünce M8 verisinin de eksik olduğu sanılabiliyordu. Katalog her veri setini GRUBU, son tarihi,
kaynağı, KULLANANI ve GÜNCELLEME YOLUYLA tek satırda verir.

Salt okuma, ağ yok. Ham piyasa kaynaklarının son tarih / eksik gün bilgisi `/trading/data/status`'tan gelir (aynı hesap
iki kez yapılmaz); bu modül diğerlerini okur.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Callable

GROUPS = [
    ("ham", "Ham piyasa verisi",
     "Bu sayfadaki Güncelle düğmesiyle yenilenir. Tek ajanlı RL, tahmin hattı ve günlük karar kullanır."),
    ("referans", "Değerlendirme referans serileri",
     "Risksiz faiz makro güncellemesiyle (bu sayfa), diğerleri komutla yenilenir. Fazla getiri, reel / USD getiri ve "
     "rejim etiketi kullanır."),
    ("m8", "M8 ve ileri veri testi",
     "Bu sayfadan GÜNCELLENMEZ. M8 kaynağı dondurulmuştur; ileri veri panelini sunucunun günlük işi her akşam yazar."),
    ("evren", "Evren",
     "BIST-30 tarihsel üyeliği; betikle kurulur."),
]

# (anahtar, ad, dosya, kullanan, güncelleme) — referans serileri; kaynak kaydı <ad>_source.json
REFERENCE = [
    ("risk_free", "Risksiz faiz (TCMB)", "data/macro/risk_free.csv",
     "Fazla getiri Sharpe (tek ajanlı değerlendirme). M8 panelinde dondurulmuş kopyası var; ileri veri işi kendi çeker",
     "Makro güncellemesiyle birlikte (bu sayfa)"),
    ("cpi_index", "TÜFE endeksi (EVDS)", "data/macro/cpi_index.csv",
     "Reel getiri (baseline raporu)", "Komut yok: MacroDataFetcher().fetch_cpi_index() (data/macro_fetcher.py)"),
    ("usdtry", "USD/TRY (yfinance)", "data/macro/usdtry.csv",
     "USD bazlı getiri, rejim duyarlılık kolu", "python -m data.index_fetcher (XU100 ile birlikte)"),
    ("xu030", "BIST-30 endeksi XU030 (yfinance)", "data/macro/xu030.csv",
     "Endeks referansı (kıyas değil; M8 D1)",
     "Komut yok: fetch_index_close('XU030.IS', path='data/macro/xu030.csv') (data/index_fetcher.py)"),
    ("xu100", "BIST-100 endeksi XU100 (yfinance)", "data/macro/xu100.csv",
     "Rejim etiketi (Faz 9 B6, Faz 10 M6)", "python -m data.index_fetcher"),
]


def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _source_end(src: dict) -> str | None:
    """Kaynak kaydından serinin son tarihi (biçimler farklı: 'end' ya da 'range'[1])."""
    if src.get("end"):
        return str(src["end"])[:10]
    rng = src.get("range")
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        return str(rng[1])[:10]
    return None


def _source_text(src: dict) -> str:
    code = src.get("code") or src.get("symbol") or src.get("funding_rate")
    where = src.get("source") or ("EVDS" if src.get("code") or src.get("funding_rate") else None)
    fetched = str(src.get("fetched_at") or "")[:10]
    parts = [p for p in (where, code, f"çekim {fetched}" if fetched else None) if p]
    return " · ".join(parts)


def _npz_last_date(path: str) -> str | None:
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as z:
            return str(z["dates"][-1])[:10] if "dates" in z.files else None
    except Exception:
        return None


def _row(group, key, name, path, last_date=None, state="", tone="secondary", used_by="", update="", source="",
         missing=None) -> dict:
    return {"group": group, "key": key, "name": name, "path": path, "last_date": last_date, "missing_days": missing,
            "state": state, "tone": tone, "used_by": used_by, "update": update, "source": source,
            "exists": bool(path) and any(os.path.exists(p) for p in path.split(" + "))}


def _freshness(last_date: str | None, missing_fn: Callable[[str], int] | None) -> tuple[str, str, int | None]:
    if not last_date:
        return "tarih yok", "secondary", None
    n = missing_fn(last_date) if missing_fn else None
    if n is None:
        return f"son {last_date}", "secondary", None
    if n <= 1:
        return "güncel", "success", n
    return f"{n} iş günü eksik", ("warning" if n <= 20 else "danger"), n


def build_catalog(raw_status: dict, missing_fn: Callable[[str], int] | None = None,
                  forward: dict | None = None) -> dict:
    """raw_status: /trading/data/status 'status' sözlüğü. forward: marl_forward.info() (yoksa okunmaz)."""
    rows = []

    # 1. Ham piyasa verisi (bu sayfadan güncellenir) — durum /data/status'tan
    ham = [
        ("bist_stocks", "BIST-30 günlük fiyat + göstergeler", "data/bist/raw_stock_data.csv",
         "Tek ajanlı RL, tahmin hattı, günlük karar", "Bu sayfa: Güncelle", "yfinance"),
        ("gold", "Altın ve döviz", "data/gold/gold_borsapy.csv + data/gold/gold_yfinance.csv",
         "Tahmin hattı (altın)", "Bu sayfa: Güncelle", "borsapy ya da yfinance"),
        ("macro", "Makro (EVDS + EUR/TRY, BIST-100)", "data/macro/macro_data.csv",
         "Tahmin hattı ve Faz 2 RL gözlemi", "Bu sayfa: Güncelle (risksiz faizi de yeniler)", "EVDS + yfinance"),
        ("fundamental", "Fundamental oranlar (anlık görüntü)", "data/fundamental/fundamental_data.csv",
         "Faz 2 RL gözlemi", "Bu sayfa: Güncelle", "yfinance"),
    ]
    for key, name, path, used, upd, src_name in ham:
        st = raw_status.get(key) or {}
        last = st.get("last_date")
        if key == "gold":
            dates = [f.get("last_date") for f in st.get("files", []) if f.get("last_date")]
            last = max(dates) if dates else None
        if key == "fundamental" and st.get("exists"):
            state, tone, n = "anlık görüntü (tarihsiz)", "info", None
        elif st.get("error"):
            state, tone, n = "okunamadı", "danger", None
        elif not st.get("exists", False):
            state, tone, n = "dosya yok", "danger", None
        else:
            state, tone, n = _freshness(last, missing_fn)
        rows.append(_row("ham", key, name, path, last, state, tone, used, upd, src_name, n))

    # 2. Referans serileri (kaynak kaydından)
    for key, name, path, used, upd in REFERENCE:
        src = _read_json(os.path.splitext(path)[0] + "_source.json")
        last = _source_end(src)
        state, tone, n = _freshness(last, missing_fn) if os.path.exists(path) else ("dosya yok", "danger", None)
        if key == "cpi_index" and last:          # aylık seri: iş günü eksiği anlamsız
            state, tone, n = f"aylık, son ay {last[:7]}", "info", None
        rows.append(_row("referans", key, name, path, last, state, tone, used, upd, _source_text(src), n))

    # 3. M8 ve ileri veri testi (dondurulmuş / otomatik)
    man = _read_json("data/marl/frozen/m8-source/manifest.json")
    rows.append(_row("m8", "m8_source", "M8 dondurulmuş kaynağı (BIST-30 + İş Yatırım)",
                     "data/marl/frozen/m8-source/stock_data_with_indicators.csv",
                     None, f"dondurulmuş {str(man.get('frozen', ''))[:10]}" if man else "yok",
                     "info" if man else "secondary", "İleri veri panelinin başlangıcı",
                     "Değişmez (SHA'yla denetlenir)", f"csv SHA {str(man.get('csv_sha256', ''))[:12]}" if man else ""))
    p8 = "data/marl/panel_m8.npz"
    rows.append(_row("m8", "panel_m8", "M8 paneli (eğitim + 6 test yılı)", p8, _npz_last_date(p8),
                     "dondurulmuş (m8-v1)" if os.path.exists(p8) else "yok", "info", "M8 eğitim ve okuması",
                     "Değişmez", ""))
    fwd = forward or {}
    if fwd.get("available") is False or not fwd:
        f_state, f_tone = "bu sunucuda yok", "secondary"
    else:
        n_s, need = int(fwd.get("n_sessions") or 0), int(fwd.get("read_sessions") or 126)
        f_state, f_tone = f"{n_s} / {need} seans · otomatik", "info"
    last_run = (fwd.get("last_run") or {})
    rows.append(_row("m8", "forward", "İleri veri paneli ve defter (Faz 11)", "data/marl/forward/m8",
                     fwd.get("last_date"), f_state, f_tone, "Simülasyon sayfası, ileri veri testi",
                     "Sunucunun günlük işi (hafta içi 19:00 TRT sonrası; tek yazar)",
                     f"son çalışma {str(last_run.get('finished_trt') or '')[:16].replace('T', ' ')}".strip()
                     if last_run.get("finished_trt") else ""))

    # 4. Evren
    meta = _read_json("data/universe/bist30_membership_meta.json")
    rows.append(_row("evren", "membership", "BIST-30 tarihsel üyelik", "data/universe/bist30_membership.csv",
                     meta.get("anchor_as_of"),
                     {"provisional": "geçici (dönem içi olaylar eksik)", "final": "kesin"}.get(meta.get("status"),
                                                                                         meta.get("status", "—")),
                     "secondary",
                     "Hayatta kalma düzeltmesi (Faz 9 E)", "python scripts/build_membership.py chain",
                     f"{meta.get('n_segments', '—')} dönem"))

    # Yedekler: elle alinmis eski panel kopyalari. backup_2026-07-27'yi R1 olcum betikleri "eski panel" olarak okur
    # (scripts/analysis/measure_r1_exposure.py, measure_r1_folds.py): silinmez.
    backups = sorted(os.path.basename(p) for p in glob.glob("data/bist/backup_*") if os.path.isdir(p))
    return {"groups": [{"key": k, "title": t, "note": n} for k, t, n in GROUPS], "rows": rows,
            "backups": backups}


def group_of(path: str) -> str:
    """Dosya listesindeki bir yolun grubu (Mevcut dosyalar tablosu için)."""
    p = path.replace("\\", "/")
    if "/marl/" in p:
        return "M8 / ileri veri" if ("/forward/" in p or "/frozen/" in p or "panel_m" in p) else "MARL ara çıktı"
    if "/bist/backup_" in p:
        return "Yedek"
    if "/universe/" in p:
        return "Evren"
    if any(f"/macro/{n}" in p for n in ("risk_free", "cpi_index", "usdtry", "xu030", "xu100")):
        return "Referans seri"
    if any(f"/{d}/" in p for d in ("bist", "gold", "macro", "fundamental")):
        return "Ham piyasa"
    return "Diğer"
