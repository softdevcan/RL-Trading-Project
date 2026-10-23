import json, sys, time, urllib.request, urllib.parse, re
M = "mailto=akyildirimcan@gmail.com"
def get(url):
    for k in range(3):
        try:
            return json.load(urllib.request.urlopen(url, timeout=60))
        except Exception as e:
            err = e; time.sleep(2)
    raise err
def by_doi(d):
    return get("https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(d, safe="/()") + "?" + M)
def norm(t): return re.sub(r"[^a-z0-9]", "", (t or "").lower())
def search_title(t):
    q = urllib.parse.quote(re.sub(r"[:,]", " ", t))
    r = get(f"https://api.openalex.org/works?filter=title.search:{q}&per-page=25&{M}")["results"]
    same = [w for w in r if norm(w.get("title"))[:60] == norm(t)[:60]]
    return same or r[:1]
out = {}
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1], encoding="utf-8")][1:]
for key, group, ident, note in rows:
    try:
        kind, val = ident.split(":", 1)
        cands = []
        if kind == "doi":
            cands = [by_doi(val)]
            if "arxiv" not in val.lower():
                pass
        elif kind == "arxiv":
            w = by_doi("10.48550/arxiv." + val)
            cands = [w] + search_title(w["title"])
        else:
            cands = search_title(val)
        best = max(cands, key=lambda w: w.get("cited_by_count", 0))
        total_cites = best.get("cited_by_count", 0)
        loc = best.get("primary_location") or {}
        src = loc.get("source") or {}
        # prefer a non-repository location for venue
        venues = []
        for c in cands:
            for L in [c.get("primary_location") or {}] + (c.get("locations") or []):
                s = (L or {}).get("source") or {}
                if s.get("display_name"): venues.append((s.get("type"), s.get("display_name"), s.get("issn_l")))
        jv = [v for v in venues if v[0] in ("journal", "conference")]
        venue = jv[0] if jv else (src.get("type"), src.get("display_name"), src.get("issn_l"))
        oa = best.get("best_oa_location") or {}
        out[key] = dict(group=group, ident=ident, note=note, title=best.get("title"),
            year=best.get("publication_year"), authors=[a["author"]["display_name"] for a in best.get("authorships", [])],
            venue_type=venue[0], venue=venue[1], issn_l=venue[2], all_venues=sorted(set(v[1] for v in venues)),
            cited_by=total_cites, cited_split=[c.get("cited_by_count",0) for c in cands],
            doi=best.get("doi"), oa_url=oa.get("pdf_url") or oa.get("landing_page_url"),
            keywords=[k["display_name"] for k in (best.get("keywords") or [])][:5],
            topic=((best.get("primary_topic") or {}).get("display_name")), openalex=best.get("id"))
    except Exception as e:
        out[key] = dict(group=group, ident=ident, note=note, error=str(e))
    time.sleep(0.2)
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(out), "resolved;", sum(1 for v in out.values() if "error" in v), "errors")
