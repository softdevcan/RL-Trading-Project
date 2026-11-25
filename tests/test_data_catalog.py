"""Faz 11 §3.17 veri kataloğu (app/services/data_catalog.py) — veri gerektirmez, ağ yok.

1. build_catalog: dört grup; ham kaynakların durumu /data/status'tan (eksik gün, dosya yok, okunamadı, anlık görüntü);
   "Bu sayfa: Güncelle" YALNIZ ham grupta (sayfanın vaadi: güncelleme M8 / ileri veriyi yazmaz); ileri veri satırı
   forward durumundan (n / 126), forward yoksa "bu sunucuda yok".
2. group_of: dosya listesindeki yolların grubu (ham, referans, M8 / ileri veri, yedek, evren).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import data_catalog as C  # noqa: E402

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    print(f"[{'OK' if ok else 'HATA'}] {name}" + (f"  {detail}" if not ok and detail else ""))
    passed, failed = passed + bool(ok), failed + (not ok)


raw = {
    "bist_stocks": {"exists": True, "last_date": "2026-11-09"},
    "gold": {"exists": True, "files": [{"last_date": "2026-07-28"}, {"last_date": "2026-07-01"}]},
    "macro": {"exists": True, "error": "bozuk"},
    "fundamental": {"exists": True},
}
miss = {"2026-11-09": 13, "2026-07-28": 87}
cat = C.build_catalog(raw, lambda d: miss.get(d, 0), {"available": True, "n_sessions": 3, "read_sessions": 126,
                                                     "last_date": "2026-11-25", "last_run": {}})
rows = {r["key"]: r for r in cat["rows"]}
ok = [g["key"] for g in cat["groups"]] == ["ham", "referans", "m8", "evren"]
ok &= rows["bist_stocks"]["state"] == "13 iş günü eksik" and rows["bist_stocks"]["tone"] == "warning"
ok &= rows["gold"]["last_date"] == "2026-07-28" and rows["gold"]["tone"] == "danger"       # en yeni dosya, > 20 gün
ok &= rows["macro"]["state"] == "okunamadı" and rows["fundamental"]["tone"] == "info"
page_rows = {r["key"] for r in cat["rows"] if r["update"].startswith("Bu sayfa")}
ok &= page_rows == {"bist_stocks", "gold", "macro", "fundamental"}
ok &= all(r["group"] != "m8" or not r["update"].startswith("Bu sayfa") for r in cat["rows"])
ok &= rows["forward"]["state"].startswith("3 / 126") and rows["forward"]["last_date"] == "2026-11-25"
none_fwd = {r["key"]: r for r in C.build_catalog(raw, None, {"available": False})["rows"]}
ok &= none_fwd["forward"]["state"] == "bu sunucuda yok"
check("1. build_catalog: 4 grup, ham durumu /data/status'tan, 'Bu sayfa' yalniz ham grupta, ileri veri n/126", ok,
      str({k: (r['state'], r['tone']) for k, r in rows.items()}))

paths = {"data/bist/raw_stock_data.csv": "Ham piyasa", "data/macro/macro_data.csv": "Ham piyasa",
         "data/macro/risk_free.csv": "Referans seri", "data/macro/xu030.csv": "Referans seri",
         "data/marl/forward/m8/rows.csv": "M8 / ileri veri", "data/marl/frozen/m8-source/x.csv": "M8 / ileri veri",
         "data/bist/backup_2026-07-27/raw_stock_data.csv": "Yedek", "data/universe/bist30_membership.csv": "Evren",
         "data\\marl\\forward\\m8\\ledger\\rows.csv": "M8 / ileri veri"}
got = {p: C.group_of(p) for p in paths}
check("2. group_of: ham / referans / M8 / yedek / evren (Windows ayiraci dahil)", got == paths,
      str({p: g for p, g in got.items() if g != paths[p]}))

print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
