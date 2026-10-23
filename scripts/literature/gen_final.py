import json, sys, importlib.util, collections

tools, lit, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location("meta", tools + "/meta.py")
meta = importlib.util.module_from_spec(spec); spec.loader.exec_module(meta)

R = json.load(open(lit + "/resolved.json", encoding="utf-8"))
S2 = json.load(open(lit + "/s2.json", encoding="utf-8"))
NOW = 2026

GROUPS = {
 "0-Ana": "0. Ana referans",
 "A-MARLportfoy": "A. MARL ile portföy yönetimi ve alım-satım",
 "B-Iletisim": "B. MARL iletişimi ve güven/belirsizlik",
 "C-MARLalgo": "C. MARL algoritmaları, derlemeler ve araçlar",
 "J-TemelDRL": "D. Tek-ajan DRL: temel algoritmalar, finans uygulamaları, derlemeler",
 "D-TahminRL": "E. Tahmin-destekli RL",
 "K-TahminModel": "F. Tahmin modelleri, topluluk, özellik seçimi, HPO, açıklanabilirlik",
 "E-TahminSizinti": "G. Tahmin katmanı, ayrıştırma sızıntısı, kalibrasyon",
 "F-OdulRisk": "H. Ödül, risk ve portföy baseline'ları",
 "G-Degerlendirme": "I. Değerlendirme metodolojisi ve istatistiksel testler",
 "H-XAI": "J. Açıklanabilirlik",
 "I-BIST": "K. BIST, Türkiye ve gelişmekte olan piyasalar",
}
ROLE_TXT = {"Z": "**Z** zorunlu", "Y": "**Y** yöntem", "B": "B bağlam", "I": "İ ikincil"}

def esc(s): return (s or "").replace("|", "/").replace("\n", " ").strip()

rows = []
for key, v in R.items():
    man = meta.MANUAL.get(key, {})
    title = man.get("title") or v.get("title")
    authors = man.get("authors") or v.get("authors") or []
    kw = man.get("keywords") or v.get("keywords") or []
    issn = man.get("issn") or v.get("issn_l")
    if key in meta.VENUE:
        venue, vtype, tier, year = meta.VENUE[key]
    else:
        venue, year = v.get("venue") or "?", v.get("year")
        vt = v.get("venue_type")
        vtype = "dergi" if vt == "journal" else ("ön baskı" if vt == "repository" else (vt or "?"))
        tier = None
    if key == "Bettini2024_BenchMARL": issn = "JMLR"
    idx, basis = meta.INDEX.get(issn, ("—", "")) if vtype == "dergi" else ("—", "")
    if tier is None:
        if vtype == "dergi":
            tier = "K1" if (("SCIE" in idx or "SSCI" in idx) and "değil" not in idx) else "K3"
        elif vtype == "ön baskı":
            tier = "K4"
        else:
            tier = "K3"
    if key == "Chaudhari2026_MAETS": idx, basis, tier = "—", "", "K4"
    s2 = S2.get(key) or {}
    c_s2 = s2.get("citationCount")
    c_oa = man.get("oa", v.get("cited_by")) if "oa" in man else v.get("cited_by")
    c = max([x for x in (c_s2, c_oa) if isinstance(x, int)] or [0])
    age = max(1, NOW - int(year)) if year else 1
    doi = meta.DOI_OVERRIDE[key] if key in meta.DOI_OVERRIDE else (v.get("doi") or "").replace("https://doi.org/", "")
    year = getattr(meta, "YEAR", {}).get(key, year)
    if "zenodo" in doi: doi = doi
    links = []
    if doi and not doi.lower().startswith("10.48550"):
        links.append(f"[DOI](https://doi.org/{doi})")
    ident = v.get("ident", "")
    if ident.startswith("arxiv:"):
        links.append(f"[arXiv](https://arxiv.org/abs/{ident[6:]})")
    if man.get("url") and not any("arxiv" in l for l in links):
        links.append(f"[bağlantı]({man['url']})")
    oa = v.get("oa_url")
    if oa and "arxiv" not in oa and "_reference.pdf" not in oa and not any(oa in l for l in links):
        links.append(f"[açık erişim]({oa})")
    if key in meta.PDF:
        links.append(f"[arşiv](../reference/papers/{meta.PDF[key]})")
    role, note = meta.ROLE[key]
    rows.append(dict(key=key, ident=v.get("ident", ""), group=v["group"], title=title, authors=authors, year=year, venue=venue,
        vtype=vtype, tier=tier, idx=idx, basis=basis, c_s2=c_s2, c_oa=c_oa, c=c, cpy=c / age,
        kw=kw, links=links, role=role, note=note, doi=doi))

TIER_ORDER = {"K1": 0, "K2": 1, "K3": 2, "K4": 3}
def fmt_auth(a):
    return ", ".join(a[:4]) + (" vd." if len(a) > 4 else "")
def fmt_idx(r):
    if r["vtype"] != "dergi":
        return f"{r['tier']} · {r['vtype']}"
    tag = {"WD": " [WD]", "MJL": " [MJL]", "B": " [doğrula]"}.get(r["basis"], "")
    return f"{r['tier']} · {r['idx']}{tag}"
def fmt_c(r):
    a = "—" if r["c_s2"] is None else str(r["c_s2"])
    b = "—" if r["c_oa"] is None else str(r["c_oa"])
    return f"{a} · {b} ({r['cpy']:.0f}/yıl)"


# ---- Nihai cikti (render_final) — gen_table.py'nin satir uretim kismindan sonra calisir ----
spec2 = importlib.util.spec_from_file_location("coverage", tools + "/coverage.py")
cov = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(cov)

STRONGER = {
 "Kazemdehbashi2026_TATS": "Yok — yöntemin tek kaynağı",
 "Torres2011_CEEMDAN": "Yok — yöntemin tek kaynağı (2.400+ atıf)",
 "Hao2025": "Yok — karma ödülün en yakın öncülü",
 "Shi2025": "Yok — G5 motivasyonu; yayına girerse künye güncellenir",
 "Hung2026_SelfAttn": "Yok — G1'in en yakın 2026 örneği",
 "deWitt2020_IPPO": "Yu vd. 2022 (NeurIPS) ile birlikte atıf",
 "Candar2022_Thesis": "LNNS bildirisi (DOI'li) tezin yerine birincil atıf",
 "Sun2021_IMMAC": "Yok — G2 alt hattının kaynağı",
 "Beyhan2026_EPFAD": "Yok — BIST DRL'nin en güncel hakemli örneği",
 "Khemlichi2025_MPLS": "Sun vd. 2025 (CAIS, SCIE) hiyerarşi için; sektör dağıtıcının kendisi için yok",
 "Ozturk2026_CWDE": "Yok — Türk jürisinin bilebileceği en yakın güven ağırlıklı çalışma",
 "Schulman2017_PPO": "Yok — algoritmanın özgün kaynağı (31.000+ atıf)",
 "Towers2024_Gymnasium": "Yok — yazılımın kaynağı",
 "Fey2019_PyG": "Yok — yazılımın kaynağı",
 "LopezdePrado2018_AFML": "Kısmen Bailey vd. 2017 (JCF, SSCI); purge/embargo için kitap tek kaynak",
 "Kelly1956": "MacLean vd. 2010 (QF, SCIE/SSCI) ile birlikte",
 "Sortino1994": "Yok — oranın özgün kaynağı",
 "Wu2009_EEMD": "Yok — EEMD'nin özgün kaynağı",
}
REPLACE = {
 "Ren2025_SAMP": "Rejim bazında raporlama örneği için Nystrup vd. 2018 (QF)",
 "Kang2025_RLBHRP": "G3 için MPLS + Sun vd. 2025 (CAIS) yeterli",
 "Wittner2026_Survey": "Zhu, Dastani & Wang 2024 (JAAMAS, SCIE)",
 "Farr2026_Herding": "Sürüleşme bekçisi tasarım varsayımı olarak yazılır; kaynak gerekmez",
 "Valdivieso2026_LeakNotDecomp": "Xiong vd. 2025 (Applied Intelligence, SCIE) + Carnelossi Furlaneto 2017 (ESWA)",
 "ManiShankar2026": "Tek vs çok ajan sorusu tezin kendi deneyi",
 "Akyapak2025_ASYU": "Türk makro bağlamı için Taşkıran 2026 (BIR, SSCI)",
 "Ergenc2025": "BIST tahmin bağlamı için Altuner & Kilimci 2022 (TJEECS, SCIE)",
 "CBAYAR2026_Rebalancing": "Uluslararası metinde DeMiguel vd. 2009; Makale 4'te (TR) kullanılabilir",
 "Ege2024_Thesis": "Candar & Üstündağ 2022 (LNNS)",
 "Chaudhari2026_MAETS": "Hakemli sürüm çıkarsa yeniden değerlendirilir",
 "Coriat2025_HARLF": "LLM kapsam dışı gerekçesi Pippas vd. 2025 (CSUR) ile",
 "Zou2025_StockMARL": "—",
 "Yang2018_MeanField": "Kapsam dışı paragrafında anılabilir",
 "Guo2023_Email": "—", "Jiang2024_Teammate": "—", "Fan2025_GraphModeling": "—", "Li2025_ExpoComm": "—",
 "Wojtala2025_MACTAS": "—", "SumerliSarigul2024": "Altuner & Kilimci 2022 (TJEECS, SCIE)",
 "Rout2026_Review": "—", "Erdag2026": "Makale 4 (TR) için uygun; uluslararası metinde Altuner & Kilimci 2022", "Rahimikia2025_TSFM": "GBM tercihinin gerekçesi Gu vd. 2020 (RFS) ve Fischer & Krauß 2018 (EJOR) ile", "Hu2024_CommFormer": "Hu vd. 2026 (TKDE, SCIE) dergi sürümü",
}

def verdict(r):
    if r["role"] in ("Z", "Y"): return None
    if r["role"] == "I": return "ikincil"
    if r["tier"] in ("K3", "K4"):
        if r["cpy"] >= 20: return None
        return f"{r['tier']} kademe"
    if r["group"] == "I-BIST": return None
    if r["year"] and NOW - int(r["year"]) >= 3 and r["cpy"] < 5: return f"atıf eşiği altı ({r['cpy']:.1f}/yıl)"
    return None

for r in rows: r["drop"] = verdict(r)
sel = [r for r in rows if not r["drop"]]
dropped = [r for r in rows if r["drop"]]
by_key = {r["key"]: r for r in rows}

SURNAME = {
 "LopezdePrado2016_HRP": "López de Prado", "LopezdePrado2018_AFML": "López de Prado", "Bailey2014_DSR": "Bailey & López de Prado",
 "deWitt2020_IPPO": "de Witt vd.", "Carnelossi2017_EMDbias": "Carnelossi Furlaneto vd.", "ManiShankar2026": "Mani Shankar vd.",
 "SumerliSarigul2024": "Sümerli Sarıgül vd.", "BIR2026_InflationSurprise": "Taşkıran", "Kritzman2010_Turbulence": "Kritzman & Li",
 "Bailey2017_PBO": "Bailey vd.", "Niu2021_MAGIC": "Niu vd.",
}
def short(r):
    if r["key"] in SURNAME: return f"{SURNAME[r['key']]} ({r['year']})"
    a = r["authors"]
    if not a: return f"? ({r['year']})"
    def sur(n): return n.split(",")[0].strip() if "," in n else n.split()[-1]
    s = sur(a[0]) + (" & " + sur(a[1]) if len(a) == 2 else (" vd." if len(a) > 2 else ""))
    return f"{s} ({r['year']})"

L = []
cnt = collections.Counter(r["tier"] for r in sel)
rc = collections.Counter(r["role"] for r in sel)
yr = collections.Counter("2025–2026" if r["year"] and int(r["year"]) >= 2025 else ("2020–2024" if r["year"] and int(r["year"]) >= 2020 else "≤2019") for r in sel)
L.append("## 2. Seçilmiş kaynakça — özet\n")
L.append(f"Değerlendirilen **{len(rows)}** kayıttan **{len(sel)}** seçildi, **{len(dropped)}** elendi (Ek A).\n")
L.append("| Kademe | Tanım | Seçilen |\n|---|---|---|")
L.append(f"| K1 | SCIE / SSCI dergi | **{cnt['K1']}** |")
L.append(f"| K2 | Üst düzey hakemli konferans (CORE A*/A) | {cnt['K2']} |")
L.append(f"| K3 | Diğer hakemli (TR Dizin/ESCI dergi, LNCS/LNNS, kitap, sıralamasız konferans) | {cnt['K3']} |")
L.append(f"| K4 | Hakemsiz (ön baskı, tez) | {cnt['K4']} |\n")
L.append("| Rol | Seçilen |\n|---|---|")
L.append(f"| Z zorunlu atıf | {rc['Z']} |\n| Y yöntem kaynağı | {rc['Y']} |\n| B bağlam | {rc['B']} |\n")
L.append("| Dönem | Seçilen |\n|---|---|")
for k in ("2025–2026", "2020–2024", "≤2019"): L.append(f"| {k} | {yr[k]} |")
L.append("")

# Kapsam matrisi
L.append("### 2.1 Kapsam matrisi — her bileşen kaynakla dayanaklı mı?\n")
L.append("Her satır tezin ya da kodun bir bileşeni. **K1/K2** sütunu o bileşeni SCIE/SSCI dergi ya da üst düzey konferans kaynağıyla dayandırıp dayandırmadığımızı gösterir.\n")
L.append("| Bileşen | Nerede | Kaynaklar | K1/K2 |\n|---|---|---|---|")
gaps = []; covered = set()
for name, where, keys in cov.COVERAGE:
    items = []
    for k in keys:
        r = by_key.get(k)
        if r is None: items.append(f"⚠{k}"); continue
        if r["drop"]: continue
        covered.add(k); items.append(f"{short(r)} [{r['tier']}]")
    strong = sum(1 for k in keys if k in by_key and not by_key[k]["drop"] and by_key[k]["tier"] in ("K1", "K2"))
    flag = f"✓ {strong}" if strong else "**YOK**"
    if not strong: gaps.append(name)
    L.append(f"| {name} | {where} | {'; '.join(items)} | {flag} |")
L.append("")
orphan = [r for r in sel if r["key"] not in covered]
L.append(f"Kapsam açığı (K1/K2 kaynağı olmayan bileşen): **{len(gaps)}**" + (": " + "; ".join(gaps) if gaps else "") + ".")
L.append(f"Matriste yer almayan seçilmiş kayıt: {len(orphan)}" + (" — " + "; ".join(short(r) for r in orphan) if orphan else "") + ".\n")

# Tutulan dusuk kademeler
keep = [r for r in sel if r["tier"] in ("K3", "K4")]
L.append(f"### 2.2 K3/K4 olup seçilen kayıtlar ({len(keep)}) — neden tutuluyor\n")
L.append("| Kayıt | Kademe | Rol / neden | Daha güçlü muadil |\n|---|---|---|---|")
for r in sorted(keep, key=lambda r: (r["role"], r["tier"], r["key"])):
    why = STRONGER.get(r["key"], "Yerleşik etki istisnası (yılda ≥20 atıf)" if r["role"] == "B" else "—")
    L.append(f"| {short(r)} — {esc(r['title'])[:60]} | {r['tier']} | {r['role']} — {esc(r['note'])} | {why} |")
L.append("")

# Ana tablolar
L.append("## 3. Seçilmiş kaynakça tabloları\n")
L.append("Sütunlar: **Atıf** = Semantic Scholar · OpenAlex (2026-10-07; parantezde yıllık ortalama, büyük olandan). **İndeks**: `[MJL]` Clarivate Master Journal List'te elle kontrol edildi · `[WD]` Wikidata'da SCIE/SSCI kaydı var · `[doğrula]` henüz kontrol edilmedi (§4.2). **Anahtar kelimeler** OpenAlex'in makale düzeyinde otomatik atadıklarıdır. **Erişim**: `arşiv` = `docs/reference/papers/` altındaki tam metin.\n")
n = 0
for g, gname in GROUPS.items():
    grp = sorted([r for r in sel if r["group"] == g], key=lambda r: (TIER_ORDER[r["tier"]], -r["c"]))
    if not grp: continue
    L.append(f"### {gname} ({len(grp)})\n")
    L.append("| # | Başlık | Yazarlar | Yıl | Yayın yeri | Kademe · indeks | Atıf | Anahtar kelimeler | Erişim | Rol |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in grp:
        n += 1
        L.append(f"| {n} | {esc(r['title'])} | {esc(fmt_auth(r['authors']))} | {r['year']} | {esc(r['venue'])} | {fmt_idx(r)} | {fmt_c(r)} | {esc('; '.join(r['kw']))} | {' · '.join(r['links'])} | {ROLE_TXT[r['role']]} — {esc(r['note'])} |")
    L.append("")

# Indeks
mjl = sorted({(r["venue"], r["idx"]) for r in rows if r["vtype"] == "dergi" and r["basis"] == "MJL"})
ver = sorted({(r["venue"], r["idx"]) for r in sel if r["vtype"] == "dergi" and r["basis"] == "B"})
L.append("## 4. İndeks doğrulaması\n")
L.append(f"### 4.1 Clarivate MJL'de elle kontrol edilenler ({len(mjl)}) — 2026-10-07\n")
L.append("| Dergi | Sonuç |\n|---|---|")
for v, i in mjl: L.append(f"| {esc(v)} | {i} |")
L.append("")
L.append("> **İki farklı dergi:** Zhang, Zohren & Roberts (2020) *The Journal of Financial Data Science*'ta (Portfolio Management Research, ISSN 2640-3943, DOI öneki 10.3905) yayımlandı. MJL'de bulunan *Journal of Finance and Data Science* (KeAi) ayrı bir dergi. JFDS bu yüzden SCIE/SSCI sayılmadı; makale yılda ~46 atıfla yerleşik etki istisnasından seçildi.\n")
L.append("> **Koleksiyon satırı:** MJL ESCI dergilerini de listeler. Discover Computing, Patterns ve International Journal of Financial Studies için kayıttaki *Web of Science Core Collection* satırının SCIE/SSCI olduğu teyit edilmeli.\n")
L.append(f"### 4.2 Seçilmiş kaynaklarda hâlâ MJL'de kontrol edilmemiş dergiler ({len(ver)})\n")
L.append("Wikidata'da SCIE/SSCI kaydı bulunmadığı için durum bilgiye dayalıdır.\n")
L.append("| Dergi | Tablodaki durum |\n|---|---|")
for v, i in ver: L.append(f"| {esc(v)} | {i} |")
L.append("")

# Ek A
L.append(f"## Ek A. Elenen kayıtlar ({len(dropped)})\n")
L.append("Kural (§1.3) uygulandı. Elenen bir kaydın konusu tezde geçiyorsa yerine kullanılacak kaynak son sütunda.\n")
L.append("| Kayıt | Yayın yeri | Rol | Kademe | Atıf/yıl | Eleme nedeni | Yerine |\n|---|---|---|---|---|---|---|")
for r in sorted(dropped, key=lambda r: (r["group"], r["key"])):
    L.append(f"| {short(r)} — {esc(r['title'])[:70]} | {esc(r['venue'])[:40]} | {r['role']} | {r['tier']} | {r['cpy']:.1f} | {r['drop']} | {REPLACE.get(r['key'], '—')} |")
L.append("")

open(out_path, "w", encoding="utf-8").write("\n".join(L))
print("rows", len(rows), "selected", len(sel), dict(cnt), dict(rc), "dropped", len(dropped), "gaps", gaps, "orphans", len(orphan), "verify", len(ver))

# ---- Kaynakca ciktisi (vision-and-scope §9 icin) ----
import re as _re
bib_path = sys.argv[4]

NAME_FIX = {"Marcos López de Prado": "López de Prado, M.", "Christian Schroeder de Witt": "Schroeder de Witt, C.",
            "Dhiego Carnelossi Furlaneto": "Carnelossi Furlaneto, D.", "Christian A. Schroeder de Witt": "Schroeder de Witt, C. A."}
def fmt_name(n):
    n = n.strip()
    if n in NAME_FIX: return NAME_FIX[n]
    if "Prado" in n and "López" in n: return "López de Prado, M."
    if "Furlaneto" in n: return "Carnelossi Furlaneto, D."
    if "de Witt" in n: return "Schroeder de Witt, C."
    if "," in n:
        sur, given = [x.strip() for x in n.split(",", 1)]
    else:
        parts = n.split()
        particles = {"de", "van", "der", "von", "da", "del", "la", "le"}
        i = len(parts) - 1
        while i > 0 and parts[i - 1].lower() in particles: i -= 1
        sur, given = " ".join(parts[i:]), " ".join(parts[:i])
    ini = " ".join(p[0] + "." for p in _re.split(r"[\s\-]+", given) if p and p[0].isalpha())
    return f"{sur}, {ini}".strip().rstrip(",")

def fmt_authors(a):
    if not a: return "?"
    a = [fmt_name(x) for x in a]
    if len(a) > 6: return ", ".join(a[:6]) + " vd."
    if len(a) == 1: return a[0]
    return ", ".join(a[:-1]) + " & " + a[-1]

def ref(r):
    title = esc(r["title"]).rstrip(".")
    venue = esc(r["venue"])
    end = "" if title.endswith(("?", "!")) else "."
    s = f"- {fmt_authors(r['authors'])} ({r['year']}). {title}{end} *{venue}*."
    if r["doi"]: s += f" DOI {r['doi'].lower() if r['doi'].lower().startswith('10.') else r['doi']}"
    elif r["key"] in meta.MANUAL and meta.MANUAL[r["key"]].get("url"): s += f" {meta.MANUAL[r['key']]['url']}"
    elif r["ident"].startswith("arxiv:"): s += f" arXiv {r['ident'][6:]}"
    tag = {"K1": "SCIE/SSCI", "K2": "konf. A*/A", "K3": "hakemli", "K4": "hakemsiz"}[r["tier"]]
    return s + f" `[{r['tier']} {tag}]`"

B = []
for g, gname in GROUPS.items():
    grp = sorted([r for r in sel if r["group"] == g], key=lambda r: (fmt_authors(r["authors"]).lower(), r["year"] or 0))
    if not grp: continue
    B.append(f"### {gname}\n")
    B.extend(ref(r) for r in grp)
    B.append("")
open(bib_path, "w", encoding="utf-8").write("\n".join(B))
print("bib entries", len(sel))
