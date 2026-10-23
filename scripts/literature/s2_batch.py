import json, sys, urllib.request, time
d = json.load(open(sys.argv[1], encoding="utf-8"))
ids, keys = [], []
for k, v in d.items():
    kind, val = v["ident"].split(":", 1)
    if kind == "doi": ids.append("DOI:" + val); keys.append(k)
    elif kind == "arxiv": ids.append("ARXIV:" + val); keys.append(k)
extra = {"Velickovic2018_GAT": "ARXIV:1710.10903", "deWitt2020_IPPO": "ARXIV:2011.09533"}
for k, i in extra.items():
    if k not in keys: ids.append(i); keys.append(k)
body = json.dumps({"ids": ids}).encode()
url = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,year,citationCount,influentialCitationCount,venue,publicationVenue,externalIds,authors"
for attempt in range(5):
    try:
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        res = json.load(urllib.request.urlopen(req, timeout=120)); break
    except Exception as e:
        print("retry", e); time.sleep(5 * (attempt + 1))
s2 = {k: r for k, r in zip(keys, res)}
json.dump(s2, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k, r in s2.items():
    print(f"{k[:24]:24s} " + ("NONE" if not r else f"c={r['citationCount']} infl={r['influentialCitationCount']} | {(r.get('venue') or '')[:50]}"))
