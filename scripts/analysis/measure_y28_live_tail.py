"""Faz 9 Y28 — gercek veride canli cikarim matrisinin son tarihi (makro + capraz varlik + temel dahil).

    python scripts/analysis/measure_y28_live_tail.py --out docs/development/measurements/phase9-y28/live_tail.json

Her sembol/ufuk icin: df'in son tarihi, build_inference_features (predict_next yolu) ve
build_features (egitim yolu) son tarihi, fazla satir sayisi (beklenen h: gunluk 1, haftalik 5),
ozellik kolonlarinin esitligi. Ag yok (yerel CSV; prediction_service yardimcilari).
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services import prediction_service as ps  # noqa: E402
from prediction.feature_engineer import PredictionFeatureEngineer as FE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="THYAO.IS,GARAN.IS,ASELS.IS")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    macro, cross = ps._fetch_macro_data(a.start), ps._fetch_cross_asset_data(a.start)
    rows = {}
    for sym in a.symbols.split(","):
        df = ps._fetch_symbol_data(sym, a.start, source=ps._resolve_source(sym, None))
        fund = ps._fetch_fundamental_data([sym])
        for hz in ("daily", "weekly"):
            fe = FE(hz)
            kw = dict(macro_df=macro, fundamental_df=fund, cross_asset_df=cross)
            inf, tr = fe.build_inference_features(df, sym, **kw), fe.build_features(df, sym, **kw)
            rows[f"{sym}/{hz}"] = {
                "df_last": str(df.index[-1])[:10], "inference_last": str(inf.index[-1])[:10],
                "train_last": str(tr.index[-1])[:10], "extra_rows": len(inf) - len(tr),
                "expected_extra_rows": 1 if hz == "daily" else 5,
                "same_feature_cols": fe.get_feature_columns(inf) == fe.get_feature_columns(tr),
                "inference_last_is_df_last": inf.index[-1] == df.index[-1],
            }
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "prediction"], capture_output=True, text=True).stdout.strip())
    out = {"command": " ".join(["python", "scripts/analysis/measure_y28_live_tail.py"] + sys.argv[1:]),
           "git": git, "prediction_dir_dirty": dirty,
           "all_ok": all(r["inference_last_is_df_last"] and r["extra_rows"] == r["expected_extra_rows"]
                         and r["same_feature_cols"] for r in rows.values()),
           "rows": rows}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"all_ok": out["all_ok"], "n": len(rows)}))


if __name__ == "__main__":
    main()
