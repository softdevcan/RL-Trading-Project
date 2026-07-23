"""
Faz 6 (G.3+G.4) — Cok-Sembol Tahmin Batch Egitimi + Manifest + Resume

Butun BIST-30 ensemble'ini tek komutla egitir; her sembol icin yapilandirilmis
bir manifest kaydi tutar ve yarida kesilirse kaldigi yerden devam eder.

"Sorunsuz egitim"in dayaniklilik ayagi:
  - Bir sembol patlarsa batch DURMAZ (strict degilse); manifest'te 'failed'
    olarak isaretlenir, digerleri devam eder.
  - Beklenen 5 modelden azi cikarsa sembol 'degraded' isaretlenir (sessiz
    3-model ensemble artik gorunur — R1).
  - Her sembol sonrasi manifest diske yazilir (checkpoint). Kesinti = sadece
    o anki sembol kaybi, oncekiler resume icin kalir.

Kullanim:
    # Tum BIST-30 gunluk ensemble
    python scripts/training/train_prediction_batch.py

    # Belirli semboller
    python scripts/training/train_prediction_batch.py --symbols GARAN AKBNK THYAO

    # Yarida kalan kosumu surdur (biten sembolleri atlar)
    python scripts/training/train_prediction_batch.py --resume 20260722_014530
    # ... veya en son manifest'i otomatik bul:
    python scripts/training/train_prediction_batch.py --resume-latest

    # Ilk hatada dur (fail-fast)
    python scripts/training/train_prediction_batch.py --strict

Cikti: results/training_runs/<run_id>.json (manifest). Cikis kodu 0 = tum
semboller ok, 1 = en az bir degraded/failed (CI icin).
"""

import argparse
import glob
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('train_prediction_batch')


def _latest_manifest_run_id() -> str:
    """En son yazilmis manifest'in run_id'sini dondur (--resume-latest icin)."""
    from prediction.manifest import RUNS_DIR
    files = sorted(glob.glob(os.path.join(RUNS_DIR, '*.json')))
    if not files:
        return ''
    return os.path.splitext(os.path.basename(files[-1]))[0]


def main() -> int:
    ap = argparse.ArgumentParser(description='Faz 6 batch tahmin egitimi (manifest + resume)')
    ap.add_argument('--symbols', nargs='*', default=None,
                    help='Egitilecek semboller (bos = tum BIST-30)')
    ap.add_argument('--horizon', default='daily', choices=['daily', 'weekly'])
    ap.add_argument('--start-date', default='2018-01-01')
    ap.add_argument('--target-type', default='log_return', choices=['log_return', 'abs_price'])
    ap.add_argument('--optimize', action='store_true', help='Her sembolde HPO calistir (yavas)')
    ap.add_argument('--n-hpo-trials', type=int, default=30)
    ap.add_argument('--resume', default=None, metavar='RUN_ID_OR_PATH',
                    help='Onceki manifest run_id/yolu — biten semboller atlanir')
    ap.add_argument('--resume-latest', action='store_true',
                    help='En son manifest kosumunu otomatik surdur')
    ap.add_argument('--strict', action='store_true', help='Ilk hatada dur (fail-fast)')
    args = ap.parse_args()

    # Sembol listesi
    if args.symbols:
        symbols = args.symbols
    else:
        from data.bist30_symbols import BIST30_SYMBOLS
        symbols = list(BIST30_SYMBOLS)
    logger.info(f"Batch: {len(symbols)} sembol, horizon={args.horizon}")

    # Resume kaynagi
    resume_from = args.resume
    if args.resume_latest and not resume_from:
        resume_from = _latest_manifest_run_id()
        if resume_from:
            logger.info(f"--resume-latest: en son manifest = {resume_from}")
        else:
            logger.info("--resume-latest: onceki manifest yok, sifirdan.")

    from app.services.prediction_service import PredictionService
    svc = PredictionService()
    out = svc.train_batch(
        symbols, horizon=args.horizon, start_date=args.start_date,
        optimize=args.optimize, n_hpo_trials=args.n_hpo_trials,
        target_type=args.target_type, resume_from=resume_from, strict=args.strict,
    )

    s = out['summary']
    print('\n' + '=' * 60)
    print(f"BATCH BITTI — run_id={out['run_id']}")
    print(f"  ok={s['ok']}  degraded={s['degraded']}  failed={s['failed']}"
          f"  skipped(resume)={len(out['skipped'])}")
    if out['failed']:
        print(f"  FAILED semboller: {out['failed']}")
    print(f"  Manifest: {out['manifest_path']}")
    print('=' * 60)

    # CI icin: temiz degilse non-zero
    return 0 if (s['degraded'] == 0 and s['failed'] == 0) else 1


if __name__ == '__main__':
    sys.exit(main())
