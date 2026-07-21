"""
Faz 6 (G.3) — Eğitim Manifesti / Run Kaydı

Gözlemlenebilirliğin kalbi. Her batch eğitim koşumu için yapılandırılmış bir
JSON kayıt üretir: hangi semboller OK/degraded/failed oldu, hangi modeller
eğitildi, veri sürümü + kalite bayrakları, süre kırılımı, git commit.

Amaç: gece çalışan bir batch'in sabah TEK BAKIŞTA teşhis edilmesi —
"3 sembol degraded, GARAN'da TFT patladı, makro fallback'teydi" — dağınık log
satırlarını taramak yerine.

Kullanım:
    m = TrainingManifest(run_kind='batch_train', symbols=syms)
    m.set_data_quality(macro_df.attrs.get('data_quality'))
    ...
    for sym in syms:
        res = trainer.train_final_models(single, sym)
        m.record_symbol(sym, res)   # status'u otomatik cikartir
    m.finalize()   # results/training_runs/<run_id>.json yazar
"""

import json
import os
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

RUNS_DIR = os.path.join('results', 'training_runs')

# Beklenen tam model seti — bundan azı 'degraded' sayılır.
EXPECTED_MODELS = {'xgboost', 'lightgbm', 'catboost', 'bilstm', 'tft'}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return 'unknown'


def _gpu_info() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return 'cpu'
    except Exception:
        return 'unknown'


class TrainingManifest:
    """Bir batch eğitim koşumunun yapılandırılmış kaydı."""

    def __init__(
        self,
        run_kind: str = 'batch_train',
        symbols: Optional[List[str]] = None,
        horizon: str = 'daily',
        expected_models: Optional[set] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_kind = run_kind
        self.horizon = horizon
        self.requested_symbols = list(symbols or [])
        self.expected_models = expected_models or EXPECTED_MODELS
        self.extra = extra or {}
        self._t0 = time.perf_counter()

        self.symbols: Dict[str, Dict[str, Any]] = {}
        self.data_quality: Dict[str, str] = {}
        self._finalized = False
        os.makedirs(RUNS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    def set_data_quality(self, dq: Optional[Dict[str, str]]):
        """Makro/veri kalite bayraklarını kaydet (G.2 ile entegre)."""
        if dq:
            self.data_quality.update(dq)

    def record_symbol(
        self,
        symbol: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        seconds: Optional[float] = None,
    ):
        """Bir sembolün eğitim sonucunu kaydet, status'u çıkart.

        status:
          - failed   : hiç sonuç yok / exception
          - degraded : eğitildi ama beklenen 5 modelden azı çıktı
          - ok       : tam model seti
        """
        if error is not None or result is None:
            self.symbols[symbol] = {
                'status': 'failed',
                'error': error or 'sonuç yok',
                'seconds': round(seconds, 1) if seconds else None,
            }
            return

        trained = set(result.get('models_trained', []))
        missing = sorted(self.expected_models - trained)
        status = 'ok' if not missing else 'degraded'

        self.symbols[symbol] = {
            'status': status,
            'models_trained': sorted(trained),
            'missing_models': missing,
            'n_models': result.get('n_models'),
            'test_mape': result.get('ensemble_test_metrics', {}).get('mape'),
            'test_dir_acc': result.get('ensemble_test_metrics', {}).get('direction_accuracy'),
            'seconds': round(seconds, 1) if seconds else result.get('training_time_seconds'),
        }

    # ------------------------------------------------------------------
    def _summary(self) -> Dict[str, int]:
        counts = {'ok': 0, 'degraded': 0, 'failed': 0}
        for d in self.symbols.values():
            counts[d.get('status', 'failed')] = counts.get(d.get('status', 'failed'), 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            'run_id': self.run_id,
            'run_kind': self.run_kind,
            'horizon': self.horizon,
            'started_at': self.run_id,
            'finished_at': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'wall_clock_seconds': round(time.perf_counter() - self._t0, 1),
            'git_commit': _git_commit(),
            'gpu': _gpu_info(),
            'requested_symbols': self.requested_symbols,
            'summary': self._summary(),
            'data_quality': self.data_quality or {},
            'symbols': self.symbols,
            **({'extra': self.extra} if self.extra else {}),
        }

    def finalize(self, write: bool = True) -> str:
        """Manifesti diske yaz ve yolunu döndür."""
        self._finalized = True
        path = os.path.join(RUNS_DIR, f'{self.run_id}.json')
        if write:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    def completed_symbols(self) -> List[str]:
        """G.4 (resume) için: başarıyla (ok) tamamlanmış semboller."""
        return [s for s, d in self.symbols.items() if d.get('status') == 'ok']
