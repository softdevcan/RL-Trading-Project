import json, sys, time, urllib.request, urllib.parse, re
lit = sys.argv[1]
R = json.load(open(lit + "/resolved.json", encoding="utf-8"))
core = ["Zhang2025_GAT","Fang2023","Li2024_MASA","Lee2020_MAPS","Huang2022_MSPM","Sun2025_CAIS","Kim2025_MAMA",
        "Khemlichi2025_MPLS","Das2019_TarMAC","Niu2021_MAGIC","Yu2024_ADMAC","Sun2021_IMMAC","Yoffe2024_DebUnc",
        "Carnelossi2017_EMDbias","Yang2024_EMDleak","Zhang2024_RollingCEEMDAN","Beyhan2026_EPFAD","Candar2022_Thesis",
        "Ansari2024","Yang2020_Ensemble","Gao2023_StockFormer","Choi2024_DeepClair","Ayari2025","Millea2022_HRP","Chen2026_MARS"]
seen = {}
for k in core:
    oid = (R.get(k) or {}).get("openalex")
    if not oid: continue
    wid = oid.rsplit("/", 1)[-1]
    url = f"https://api.openalex.org/works?filter=cites:{wid},from_publication_date:2025-01-01&per-page=200&mailto=akyildirimcan@gmail.com"
    try:
        res = json.load(urllib.request.urlopen(url, timeout=60))["results"]
    except Exception as e:
        print("ERR", k, e); continue
    for w in res:
        s = ((w.get("primary_location") or {}).get("source") or {})
        e = seen.setdefault(w["id"], dict(title=w.get("title"), year=w.get("publication_year"), venue=s.get("display_name"),
              vtype=s.get("type"), issn=s.get("issn_l"), doi=w.get("doi"), cites=w.get("cited_by_count"), via=[]))
        e["via"].append(k)
    time.sleep(0.3)
json.dump(seen, open(lit + "/snowball.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("unique citing works 2025+:", len(seen))
