"""
Faz 6 (3.1/B7) — BiLSTM & TFT Egitim Dongusu A/B Olcumu

Uctan uca ensemble profili (profile_training.py) TFT'nin egitimin ~%90'i
oldugunu gosterdi. Bu script sadece DL modellerini, ayni veri ve ayni seed'le,
farkli yapilandirmalarda egitip karsilastirir:

    legacy   : DL_GPU_PRELOAD=0, DL_AMP=0   (eski DataLoader yolu)
    preload  : DL_GPU_PRELOAD=1, DL_AMP=0   (varsayilan — GPU-yerlesik batch)
    amp      : DL_GPU_PRELOAD=1, DL_AMP=1   (karisik hassasiyet)

Kabul kriteri (plan): tek BiLSTM egitimi >=%25 hizlanir, val MAPE +-%2.
`preload` sayisal olarak `legacy` ile AYNI olmali (batch sinirlari degismedi);
`amp` fp16 nedeniyle tolerans icinde sapabilir.

Kullanim:
    python scripts/benchmarking/profile_dl_training.py --symbol AKBNK.IS
    python scripts/benchmarking/profile_dl_training.py --models tft --repeat 2
    python scripts/benchmarking/profile_dl_training.py --out results/benchmarks/dl_ab.md
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np

logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')
logger = logging.getLogger('profile_dl')

CONFIGS = {
    'legacy':  {'DL_GPU_PRELOAD': False, 'DL_AMP': False, 'TFT_FAST_VSN': False},
    'preload': {'DL_GPU_PRELOAD': True,  'DL_AMP': False, 'TFT_FAST_VSN': False},
    'amp':     {'DL_GPU_PRELOAD': True,  'DL_AMP': True,  'TFT_FAST_VSN': False},
    # Sadece TFT icin anlamli: degisken-secim agi gruplanmis (batched) surum.
    'fastvsn': {'DL_GPU_PRELOAD': True,  'DL_AMP': False, 'TFT_FAST_VSN': True},
}


def build_dataset(symbol: str, start_date: str):
    """Cached OHLCV -> indikator -> feature matrisi -> (X_train, y_train, X_val, y_val).

    Ensemble'in base model egitiminde kullandigi ayni bolme mantigi (%60/%20).
    """
    from data.data_fetcher import DataFetcher
    from data.technical_indicators import add_indicators_to_multi_symbol_df
    from prediction.feature_engineer import PredictionFeatureEngineer

    fetcher = DataFetcher(start_date=start_date)
    df = fetcher.load_data('raw_stock_data.csv')
    have = df.index.get_level_values('symbol').unique().tolist()
    if symbol not in have:
        raise SystemExit(f"Cache'te {symbol} yok. Mevcut: {have}")
    df = df[df.index.get_level_values('symbol') == symbol]
    df = fetcher.clean_data(df)
    df = add_indicators_to_multi_symbol_df(df)
    single = df.xs(symbol, level='symbol')

    fe = PredictionFeatureEngineer('daily')
    feat = fe.build_features(single, symbol)
    cols = fe.get_feature_columns(feat)
    feat = feat.dropna(subset=['target'])
    X = feat[cols].values.astype(np.float64)
    y = feat['target'].values.astype(np.float64)
    # Ensemble/trainer ile ayni temizlik (ensemble.py:217) — inf birakilirsa
    # DL egitiminde NaN uretir ve BCELoss device-side assert atar.
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0)

    n = len(X)
    i_tr, i_va = int(n * 0.6), int(n * 0.8)
    return X[:i_tr], y[:i_tr], X[i_tr:i_va], y[i_tr:i_va]


def run_one(model_type: str, config_name: str, data, seed: int):
    """Tek model egitimi — sure + val metrikleri dondurur."""
    from app.core.config import get_settings
    from prediction.seeding import seed_everything

    s = get_settings()
    for key, val in CONFIGS[config_name].items():
        setattr(s, key, val)

    # torch_perf ayarlari cagri aninda okunur; modul yeniden yuklenmesine gerek yok.
    seed_everything(seed)

    if model_type == 'bilstm':
        from prediction.models.lstm_model import BiLSTMModel as Model
    else:
        from prediction.models.tft_model import TFTModel as Model

    X_tr, y_tr, X_va, y_va = data
    model = Model(horizon='daily')
    t0 = time.perf_counter()
    metrics = model._fit(X_tr, y_tr, X_va, y_va)
    seconds = time.perf_counter() - t0
    return {
        'seconds': seconds,
        'mape': metrics.get('mape'),
        'rmse': metrics.get('rmse'),
        'direction_accuracy': metrics.get('direction_accuracy'),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='Faz 6 (3.1) DL egitim A/B olcumu')
    ap.add_argument('--symbol', default='AKBNK.IS')
    ap.add_argument('--start-date', default='2020-01-01')
    ap.add_argument('--models', nargs='*', default=['bilstm', 'tft'],
                    choices=['bilstm', 'tft'])
    ap.add_argument('--configs', nargs='*', default=list(CONFIGS),
                    choices=list(CONFIGS))
    ap.add_argument('--repeat', type=int, default=1, help='Her olcumu N kez tekrarla (medyan)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--seeds', type=int, nargs='*', default=None,
                    help='Coklu seed: kalite karsilastirmasi icin (ortalama metrik). '
                         'Tek kosumdaki MAPE farki gurultu olabilir.')
    ap.add_argument('--out', default=None, help='Markdown cikti yolu')
    args = ap.parse_args()

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
    print(f"Cihaz: {device} ({gpu})")
    print(f"Sembol: {args.symbol} | veri hazirlaniyor...")

    data = build_dataset(args.symbol, args.start_date)
    print(f"  X_train={data[0].shape} X_val={data[2].shape}")

    seeds = args.seeds if args.seeds else [args.seed] * args.repeat
    results = {}
    for model_type in args.models:
        for cfg in args.configs:
            runs = []
            for seed in seeds:
                out = run_one(model_type, cfg, data, seed)
                runs.append(out)
                print(f"  {model_type:7s} {cfg:8s} seed={seed}: "
                      f"{out['seconds']:7.2f}s  MAPE={out['mape']}  "
                      f"RMSE={out['rmse']}  Dir={out['direction_accuracy']}")
            # Sure: medyan (tek kosumda gurultu az). Kalite: ortalama —
            # seed'ler arasi sapma tek kosumda yanlis karar verdirir.
            med = sorted(runs, key=lambda d: d['seconds'])[len(runs) // 2]
            agg = dict(med)
            if len(runs) > 1:
                for key in ('mape', 'rmse', 'direction_accuracy'):
                    vals = [r[key] for r in runs if r[key] is not None]
                    agg[key] = round(float(np.mean(vals)), 4) if vals else None
                    agg[key + '_std'] = round(float(np.std(vals)), 4) if vals else None
            agg['n_runs'] = len(runs)
            results[(model_type, cfg)] = agg

    # --- Ozet ---
    lines = []
    lines.append(f"# Faz 6 (3.1) — DL Egitim A/B\n")
    lines.append(f"**Tarih:** {datetime.now().isoformat(timespec='seconds')}  ")
    lines.append(f"**Cihaz:** {gpu}  ")
    lines.append(f"**Sembol:** {args.symbol} | X_train={data[0].shape} X_val={data[2].shape}  ")
    lines.append(f"**Tekrar:** {args.repeat} (medyan)\n")
    lines.append("| Model | Config | Sure (s) | Hizlanma | Val MAPE | Dir Acc |")
    lines.append("|-------|--------|----------|----------|----------|---------|")

    print("\n" + "=" * 72)
    for model_type in args.models:
        base = results.get((model_type, 'legacy'))
        for cfg in args.configs:
            r = results.get((model_type, cfg))
            if not r:
                continue
            if base and base['seconds'] > 0:
                speed = f"{100.0 * (1 - r['seconds'] / base['seconds']):+.1f}%"
            else:
                speed = '-'
            mape = r['mape']
            dacc = r['direction_accuracy']
            lines.append(f"| {model_type} | {cfg} | {r['seconds']:.2f} | {speed} | "
                         f"{mape} | {dacc} |")
            print(f"{model_type:7s} {cfg:8s} {r['seconds']:8.2f}s  {speed:>8s}  "
                  f"MAPE={mape}  Dir={dacc}")

        # Sayisal esdegerlik kontrolu: preload, legacy ile AYNI olmali
        pre = results.get((model_type, 'preload'))
        if base and pre and base['mape'] is not None and pre['mape'] is not None:
            same = abs(base['mape'] - pre['mape']) < 1e-9
            print(f"  -> [{model_type}] preload == legacy (sayisal): "
                  f"{'EVET' if same else 'HAYIR'} "
                  f"(legacy={base['mape']} preload={pre['mape']})")
            lines.append(f"\n> `{model_type}`: preload vs legacy sayisal esdegerlik: "
                         f"**{'AYNI' if same else 'FARKLI'}** "
                         f"(legacy MAPE={base['mape']}, preload MAPE={pre['mape']})\n")
    print("=" * 72)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"Yazildi: {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
