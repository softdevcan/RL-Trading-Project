"""
Faz 6 — Eğitim & Veri Pipeline Profilleme

"Önce ölç" ilkesinin aracı: hangi aşamaya ne kadar süre gittiğini ölçer.
Optimizasyon PR'ları bu script'in ürettiği baseline'a karşı "X s -> Y s"
kanıtı taşır.

Kullanım:
    # Uçtan uca eğitim profili (tek + çok sembol)
    python scripts/benchmarking/profile_training.py --stage train --symbols 1
    python scripts/benchmarking/profile_training.py --stage train --symbols 5

    # Veri pipeline profili (çekme/temizleme/indikatör/feature)
    python scripts/benchmarking/profile_training.py --stage data --symbols 5

    # İkisi + markdown baseline yaz
    python scripts/benchmarking/profile_training.py --stage all --symbols 5 \
        --out results/benchmarks/phase6_baseline.md

Notlar:
- Eğitim aşamasında ensemble'ın kendi iç model-breakdown log'u da görünür.
- --cached ile daha önce çekilmiş raw_stock_data.csv kullanılır (ağ olmadan
  eğitim ölçümü için). Yoksa yfinance'den canlı çeker.
"""

import argparse
import contextlib
import logging
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime

# Proje kökü
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

logger = logging.getLogger('profile_training')


class StageTimer:
    """Aşama-bazlı süre biriktirici. `with timer('ad'):` şeklinde kullanılır."""

    def __init__(self):
        self.stages: "OrderedDict[str, float]" = OrderedDict()

    @contextlib.contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.stages[name] = self.stages.get(name, 0.0) + dt
            logger.info(f"  [stage] {name:32s} {dt:8.2f}s")

    def as_rows(self):
        total = sum(self.stages.values()) or 1e-9
        return [
            (name, dt, 100.0 * dt / total)
            for name, dt in self.stages.items()
        ]

    def total(self) -> float:
        return sum(self.stages.values())


def _gpu_info() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            return f"{name} ({vram} GB)"
        return "CPU-only (CUDA yok)"
    except Exception as exc:
        return f"bilinmiyor ({exc})"


def _peak_rss_mb() -> float:
    """Basit peak RSS ölçümü (psutil varsa)."""
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1e6, 1)
    except Exception:
        return -1.0


def _load_prices(n_symbols: int, start_date: str, use_cached: bool, timer: StageTimer):
    """Fiyat verisini yükle (cache'ten veya yfinance'ten). Multi-index döner."""
    from data.bist30_symbols import BIST30_SYMBOLS
    from data.data_fetcher import DataFetcher

    symbols = BIST30_SYMBOLS[:n_symbols]
    fetcher = DataFetcher(start_date=start_date)

    if use_cached and os.path.exists(os.path.join('data', 'bist', 'raw_stock_data.csv')):
        with timer('data.load_cached'):
            df = fetcher.load_data('raw_stock_data.csv')
        have = df.index.get_level_values('symbol').unique().tolist()
        keep = [s for s in symbols if s in have]
        if len(keep) < n_symbols:
            logger.warning(
                f"  Cache'te {len(keep)}/{n_symbols} sembol var; "
                f"eksikler canlı çekilecek: {[s for s in symbols if s not in have]}"
            )
        df = df[df.index.get_level_values('symbol').isin(keep or symbols)]
        if keep:
            return df, keep, fetcher

    with timer('data.fetch_stock_data'):
        df = fetcher.fetch_stock_data(symbols, save=True)
    return df, symbols, fetcher


def run_data_stage(n_symbols, start_date, use_cached, timer):
    """Veri pipeline aşamalarını ölç: çek -> temizle -> indikatör -> feature."""
    from data.technical_indicators import add_indicators_to_multi_symbol_df
    from prediction.feature_engineer import PredictionFeatureEngineer

    df, symbols, fetcher = _load_prices(n_symbols, start_date, use_cached, timer)

    with timer('data.clean'):
        df = fetcher.clean_data(df)

    with timer('data.technical_indicators'):
        df = add_indicators_to_multi_symbol_df(df)

    # Feature engineering — ilk sembol üzerinde tek ölçüm (temsili)
    fe = PredictionFeatureEngineer('daily')
    first = symbols[0]
    single = df.xs(first, level='symbol')
    with timer('data.build_features(1 sym)'):
        fe.build_features(single, first)

    return df, symbols


def run_train_stage(n_symbols, start_date, use_cached, timer):
    """Uçtan uca eğitim: veri hazırlığı + sembol başına ensemble eğitimi."""
    from data.technical_indicators import add_indicators_to_multi_symbol_df
    from prediction.trainer import WalkForwardTrainer

    df, symbols, fetcher = _load_prices(n_symbols, start_date, use_cached, timer)

    with timer('data.clean'):
        df = fetcher.clean_data(df)
    with timer('data.technical_indicators'):
        df = add_indicators_to_multi_symbol_df(df)

    trainer = WalkForwardTrainer(horizon='daily', select_features=True)

    per_symbol = OrderedDict()
    for sym in symbols:
        single = df.xs(sym, level='symbol')
        label = f'train.ensemble[{sym}]'
        t0 = time.perf_counter()
        with timer(label):
            res = trainer.train_final_models(single, sym)
        per_symbol[sym] = {
            'seconds': round(time.perf_counter() - t0, 1),
            'n_models': res.get('n_models'),
            'models': res.get('models_trained'),
            'test_mape': res.get('ensemble_test_metrics', {}).get('mape'),
        }
        logger.info(
            f"  -> {sym}: {per_symbol[sym]['n_models']} model, "
            f"MAPE={per_symbol[sym]['test_mape']}, {per_symbol[sym]['seconds']}s"
        )

    return per_symbol


def write_markdown(path, meta, timer, per_symbol):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    lines.append("# Faz 6 — Eğitim/Veri Pipeline Baseline\n")
    lines.append(f"**Oluşturuldu:** {meta['when']}  ")
    lines.append(f"**Git commit:** `{meta['commit']}`  ")
    lines.append(f"**GPU:** {meta['gpu']}  ")
    lines.append(f"**Sembol sayısı:** {meta['n_symbols']}  ")
    lines.append(f"**Başlangıç tarihi:** {meta['start_date']}  ")
    lines.append(f"**Peak RSS:** {meta['peak_rss_mb']} MB  ")
    lines.append(f"**Toplam wall-clock:** {timer.total():.1f}s\n")

    lines.append("## Aşama Kırılımı\n")
    lines.append("| Aşama | Süre (s) | % |")
    lines.append("|-------|----------|---|")
    for name, dt, pct in timer.as_rows():
        lines.append(f"| {name} | {dt:.2f} | {pct:.1f}% |")

    if per_symbol:
        lines.append("\n## Sembol Bazlı Eğitim\n")
        lines.append("| Sembol | Model sayısı | Modeller | Test MAPE | Süre (s) |")
        lines.append("|--------|-------------|----------|-----------|----------|")
        for sym, d in per_symbol.items():
            models = ', '.join(d.get('models') or [])
            lines.append(
                f"| {sym} | {d.get('n_models')} | {models} | "
                f"{d.get('test_mape')} | {d.get('seconds')} |"
            )

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    logger.info(f"Baseline yazıldı: {path}")


def _git_commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], text=True
        ).strip()
    except Exception:
        return 'unknown'


def main():
    ap = argparse.ArgumentParser(description='Faz 6 eğitim/veri profilleme')
    ap.add_argument('--stage', choices=['train', 'data', 'all'], default='train')
    ap.add_argument('--symbols', type=int, default=1, help='Kaç sembol')
    ap.add_argument('--start-date', default='2020-01-01')
    ap.add_argument('--cached', action='store_true',
                    help='raw_stock_data.csv varsa ağ olmadan kullan')
    ap.add_argument('--out', default=None, help='Markdown baseline çıktı yolu')
    ap.add_argument('--quiet-training', action='store_true',
                    help='Ensemble iç loglarını bastır (sadece aşama süreleri)')
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )
    if args.quiet_training:
        for noisy in ('prediction', 'data'):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    timer = StageTimer()
    per_symbol = OrderedDict()

    logger.info("=" * 64)
    logger.info(f"FAZ 6 PROFILLEME — stage={args.stage} symbols={args.symbols}")
    logger.info(f"GPU: {_gpu_info()}")
    logger.info("=" * 64)

    if args.stage in ('data', 'all'):
        run_data_stage(args.symbols, args.start_date, args.cached, timer)

    if args.stage in ('train', 'all'):
        per_symbol = run_train_stage(args.symbols, args.start_date, args.cached, timer)

    logger.info("-" * 64)
    logger.info("AŞAMA ÖZETİ:")
    for name, dt, pct in timer.as_rows():
        logger.info(f"  {name:32s} {dt:8.2f}s  ({pct:4.1f}%)")
    logger.info(f"  {'TOPLAM':32s} {timer.total():8.2f}s")
    logger.info(f"  Peak RSS: {_peak_rss_mb()} MB")

    if args.out:
        meta = {
            'when': datetime.now().isoformat(timespec='seconds'),
            'commit': _git_commit(),
            'gpu': _gpu_info(),
            'n_symbols': args.symbols,
            'start_date': args.start_date,
            'peak_rss_mb': _peak_rss_mb(),
        }
        write_markdown(args.out, meta, timer, per_symbol)


if __name__ == '__main__':
    main()
