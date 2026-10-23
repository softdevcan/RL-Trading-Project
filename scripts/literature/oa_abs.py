import json, sys, urllib.request, urllib.parse
for doi in sys.argv[1:]:
    url = "https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(doi, safe="/") + "?mailto=akyildirimcan@gmail.com"
    try:
        w = json.load(urllib.request.urlopen(url, timeout=60))
    except Exception as e:
        print("##", doi, "ERR", e); continue
    inv = w.get("abstract_inverted_index") or {}
    pos = sorted((p, word) for word, ps in inv.items() for p in ps)
    ab = " ".join(word for _, word in pos)
    au = ", ".join(a["author"]["display_name"] for a in w.get("authorships", [])[:4])
    src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
    print(f"## {doi}\n{w.get('title')}\n{au} | {src} | {w.get('publication_date')} | cited_by={w.get('cited_by_count')}\nABS: {ab[:1300] or '(özet yok)'}\n")
