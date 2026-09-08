"""
Var olan modeller icin tanim dosyasi (`<model>.meta.json`) uret.

Neden gerekli: model ADI egitildigi sembol evrenini ANLATMIYOR. Egitim rotasi
`get_symbols(phase)` listesini yalnizca veri cekerken kullanir, egitimi
yuklenen panelin tamamiyla yapar — bu yuzden "ppo_phase1_..." adli bir model
pekala 30 sembolle egitilmis olabilir (gozlem uzayi 331). Gunluk karar ucu
evreni artik modelin kendisinden cozuyor; tanim dosyasi olmayan eski modeller
icin egitim panelini yeniden uretiyor.

Bu betik o cozumlemeyi bir kez yapip DONDURUYOR. Panel sonradan degisirse
(yeni sembol indirilirse, tarih araligi kayarsa) yeniden uretim baska bir
sonuc verebilir; tanim dosyasi varsa model kendi evrenini tasir.

Kullanim:
    python scripts/backfill_model_meta.py                 # ne yapacagini yazar
    python scripts/backfill_model_meta.py --write         # dosyalari yazar
    python scripts/backfill_model_meta.py --write --force # var olanlari da tazeler
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.daily_trading import (  # noqa: E402
    model_meta_path,
    read_model_meta,
    resolve_trade_universe,
    write_model_meta,
)


def _load(model_path: str):
    from stable_baselines3 import PPO, A2C, TD3, SAC

    name = os.path.basename(model_path).lower()
    for key, cls in (("ppo", PPO), ("a2c", A2C), ("td3", TD3), ("sac", SAC)):
        if key in name:
            return cls.load(model_path, device="cpu")
    return None


def _model_dirs() -> list:
    roots = ["models"]
    roots += sorted(glob.glob(os.path.join("workspaces", "*", "models")))
    return [r for r in roots if os.path.isdir(r)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="dosyalari gercekten yaz")
    ap.add_argument("--force", action="store_true", help="var olan tanimi da tazele")
    args = ap.parse_args()

    zips = []
    for root in _model_dirs():
        zips += sorted(glob.glob(os.path.join(root, "*.zip")))

    if not zips:
        print("Model bulunamadi.")
        return 0

    written = skipped = failed = 0
    for zip_path in zips:
        model_path = zip_path[:-4]
        name = os.path.basename(model_path)
        meta_name = os.path.basename(model_meta_path(model_path))

        if read_model_meta(model_path) is not None and not args.force:
            print(f"  - {name}: tanim zaten var ({meta_name}), atlandi")
            skipped += 1
            continue

        model = _load(model_path)
        if model is None:
            print(f"  ! {name}: algoritma adindan cozulemedi (ppo/a2c/td3/sac yok)")
            failed += 1
            continue

        try:
            symbols, source = resolve_trade_universe(model, model_path, name)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {name}: evren cozumlenemedi — {exc}")
            failed += 1
            continue

        obs = getattr(model.observation_space, "shape", (0,))[0]
        act = getattr(model.action_space, "shape", (0,))[0]
        print(f"  + {name}: {len(symbols)} sembol (obs={obs}, act={act}) "
              f"[kaynak: {source}]")

        if args.write:
            write_model_meta(model_path, {
                "model_name": name,
                "symbols": symbols,
                "n_symbols": len(symbols),
                "obs_dim": int(obs),
                "action_dim": int(act),
                "backfilled_from": source,
            })
            written += 1

    print()
    if args.write:
        print(f"Yazildi: {written} | Atlandi: {skipped} | Basarisiz: {failed}")
    else:
        print(f"Deneme kosumu — hicbir dosya yazilmadi. Yazmak icin: --write")
        print(f"Aday: {len(zips) - skipped - failed} | Atlandi: {skipped} | "
              f"Basarisiz: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
