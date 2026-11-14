"""Faz 9 Y29 — mevcut tahmin takibi kayitlarinin gecerlilik dokumu (Y28/Y29 kurallariyla).

    python scripts/analysis/measure_y29_tracker_records.py --out docs/development/measurements/phase9-y29/tracker_records.json

data/predictions ve workspaces/*/data/predictions altindaki *_predictions.json dosyalarini okur
(yazmaz). Her kayit icin tracker_rules.invalid_reason; degerlendirilmis olanlar ayrica sayilir.
Kullanici kimligi yazilmaz (calisma alani sayisi yeter).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prediction.tracker_rules import invalid_reason  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = glob.glob("data/predictions/*_predictions.json") + glob.glob("workspaces/*/data/predictions/*_predictions.json")
    c, ce = Counter(), Counter()
    dates = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        for pd_, hz in data.items():
            for h, e in hz.items():
                r = invalid_reason(e) or "valid"
                c[f"{h}/{r}"] += 1
                if e.get("actual_close") is not None:
                    ce[f"{h}/{r}"] += 1
                dates.append(pd_)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"command": "python scripts/analysis/measure_y29_tracker_records.py " + " ".join(sys.argv[1:]),
           "git": git, "n_files": len(files),
           "n_workspaces_with_records": len({f.split(os.sep)[1] for f in files if f.startswith("workspaces")}),
           "records_by_horizon_reason": dict(c), "evaluated_by_horizon_reason": dict(ce),
           "prediction_date_min": min(dates) if dates else None, "prediction_date_max": max(dates) if dates else None}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
