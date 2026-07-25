"""
Faz 6 (3.1/B7) — DL Egitim Dongusu Ince Ayari

BiLSTM ve TFT egitim donguleri ayni desende yazilmis: `TensorDataset` +
`DataLoader(num_workers=0)` ve her batch'te ayri ayri `.to(device)`. Bu
projedeki veri kucuk (tek sembol ~2-3 bin sekans, ~30-50 batch/epoch), yani
GPU hesabi degil **batch basina Python + transfer masrafi** baskin. Olcum
(2026-07-24, RTX 4060): TFT sembol basina ~135s ile egitimin ~%90'i, epoch
basina ~3s — bu boyuttaki bir ag icin cok yuksek.

Bu modul iki kaldirac sunar:

1. **GPU-yerlesik batch'leme** (`iter_batches`) — tum sekanslar bir kez
   cihaza tasinir, epoch'lar dilimleyerek ilerler. Batch icerigi/sirasi
   DataLoader ile ayni (shuffle=False) ve tensor degerleri bit-es; tek bir
   model ayni-seed ile calistirildiginda cikti da bit-es. ANCAK tam pipeline'da
   (tek global seed, modeller ardisik) RNG tuketim sirasi degistiginden uctan
   uca cikti deterministik olarak kayar (olcum: golden MAPE 139->177). Bu
   yuzden opt-in (`DL_GPU_PRELOAD`, default KAPALI) — davranis dondurma.

2. **Karisik hassasiyet** (`amp_components`) — forward autocast, loss FP32.
   Kucuk aglarda cast/scaler masrafi kazanci yiyor (olcum: daha yavas) ve
   sayisal davranisi degistiriyor -> varsayilan KAPALI (`DL_AMP`), onerilmez.

Ayrica `configure_threads()` ile `torch.set_num_threads` sabitlenebilir
(Epic 2.2 process paralelliginde CPU asiri-abonelugini onlemek icin).
"""

import contextlib
import logging
import threading
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Epic 2.2: semboller paralel egitildiginde GPU'da es zamanli DL egitimini
# sinirlayan yuva. Tek GPU + 8GB VRAM -> sinirsiz paralellik OOM demektir.
_gpu_semaphore: Optional[threading.Semaphore] = None
_gpu_sem_lock = threading.Lock()


def _settings():
    """Config'e ulasmaya calis; app katmani yoksa varsayilanlarla devam et."""
    try:
        from app.core.config import get_settings

        return get_settings()
    except Exception:
        return None


def gpu_preload_enabled() -> bool:
    """Sekanslar cihaza bir kez tasinip dilimlensin mi (DataLoader yerine).

    Config okunamazsa (app katmani yok / Settings patladi) KAPALI kabul edilir:
    acik olmasi RNG tuketim sirasini kaydirip uctan uca ciktiyi degistirir
    (golden MAPE 139->177) — sessiz davranis degisikligi olmasin.
    """
    s = _settings()
    return False if s is None else bool(getattr(s, 'DL_GPU_PRELOAD', False))


def fast_vsn_enabled() -> bool:
    """TFT'nin degisken-secim agi gruplanmis (batched) surumde mi calissin.

    Egitim yorungesini degistirir (dropout RNG tuketimi farkli) -> config'ten
    okunur, karar A/B olcumune birakilir.
    """
    s = _settings()
    return False if s is None else bool(getattr(s, 'TFT_FAST_VSN', False))


def amp_enabled(device: torch.device) -> bool:
    """AMP yalnizca CUDA'da ve acikca istendiginde."""
    if device.type != 'cuda':
        return False
    s = _settings()
    return False if s is None else bool(getattr(s, 'DL_AMP', False))


def configure_threads() -> None:
    """`DL_TORCH_THREADS` > 0 ise torch CPU is parcacigi sayisini sabitle."""
    s = _settings()
    n = int(getattr(s, 'DL_TORCH_THREADS', 0) or 0) if s is not None else 0
    if n > 0:
        try:
            torch.set_num_threads(n)
        except Exception as exc:  # pragma: no cover — platforma bagli
            logger.debug(f"set_num_threads({n}) basarisiz: {exc}")


@contextlib.contextmanager
def gpu_slot() -> Iterator[None]:
    """DL egitimi icin GPU yuvasi (Epic 2.2 · VRAM korumasi).

    Seri kosumda (tek is parcacigi) hicbir sey yapmaz. Semboller paralel
    egitildiginde ayni anda en fazla `DL_GPU_SLOTS` sembolun DL adimi GPU'ya
    girer; agac modelleri (CPU) bu kisittan etkilenmez.

    Yuva sayisi ilk kullanimda sabitlenir — kosum ortasinda degismez.
    """
    global _gpu_semaphore
    if _gpu_semaphore is None:
        with _gpu_sem_lock:
            if _gpu_semaphore is None:
                s = _settings()
                slots = int(getattr(s, 'DL_GPU_SLOTS', 1) or 1) if s is not None else 1
                _gpu_semaphore = threading.Semaphore(max(1, slots))
    with _gpu_semaphore:
        yield


def to_device_tensors(
    arrays: Sequence[np.ndarray], device: torch.device,
) -> List[torch.Tensor]:
    """Numpy dizilerini tek seferde float32 tensor olarak cihaza tasi."""
    return [torch.as_tensor(a, dtype=torch.float32, device=device) for a in arrays]


def iter_batches(
    tensors: Sequence[torch.Tensor], batch_size: int,
) -> Iterator[Tuple[torch.Tensor, ...]]:
    """Cihazda duran tensorlari dilimleyerek batch uret.

    `DataLoader(dataset, batch_size, shuffle=False)` ile ayni batch sinirlari:
    son batch kisa kalabilir. Kopya yok — dilim (view) dondurulur.
    """
    n = int(tensors[0].shape[0])
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        yield tuple(t[start:stop] for t in tensors)


def n_batches(n_samples: int, batch_size: int) -> int:
    """DataLoader'in `len()` degeriyle ayni (val loss ortalamasi icin)."""
    return (n_samples + batch_size - 1) // batch_size


def amp_components(device: torch.device) -> Tuple[Any, Optional[Any]]:
    """(autocast fabrikasi, GradScaler) dondurur.

    AMP kapaliysa autocast yerine `nullcontext`, scaler yerine None gelir —
    cagri yeri tek kod yoluyla calisir.
    """
    if not amp_enabled(device):
        return (lambda: contextlib.nullcontext()), None

    scaler = torch.amp.GradScaler('cuda')

    def _autocast():
        return torch.amp.autocast('cuda', dtype=torch.float16)

    logger.info("  [dl] AMP acik (forward fp16, loss fp32)")
    return _autocast, scaler


def backward_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[Any],
    parameters,
    clip_norm: float = 1.0,
) -> None:
    """Geri yayilim + gradyan kirpma + optimizer adimi (AMP'li ya da degil).

    AMP'de kirpma oncesi `unscale_` sart — aksi halde olceklenmis gradyan
    kirpilir ve efektif kirpma esigi degisir.
    """
    optimizer.zero_grad(set_to_none=True)
    if scaler is None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, clip_norm)
        optimizer.step()
        return

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(parameters, clip_norm)
    scaler.step(optimizer)
    scaler.update()
