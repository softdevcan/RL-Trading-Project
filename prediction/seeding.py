"""
Merkezi Determinizm / Seed Politikasi (Faz 6 · G.5)

Amac: "Ayni seed + seri mod -> tekrar uretilebilir cikti". Eskiden seed
degeri (42) 11 ayri dosyada hardcode'du ve SUREC-DUZEYINDE global RNG'ler
(numpy, python `random`, torch) yalnizca regresyon testinde tohumlaniyordu —
gercek egitim kosumlari degil. Bu modul ikisini de tek yerde toplar:

  - GLOBAL_SEED: tum model `random_state`/`random_seed`/`seed` degerlerinin
    okudugu tek kaynak (varsayilan 42 — mevcut egitilmis modeller ve golden
    dosya bit-bit korunur).
  - seed_everything(): bir egitim/batch kosumunun BASINDA cagrilinca surecin
    tum RNG'lerini sabitler.

NOT (paralellik uyarisi): Epic 2.2 (ProcessPoolExecutor ile sembol paraleli)
devreye girince float toplama sirasi degisip determinizmi kirabilir. O yuzden
regresyon testi paralel modda tolerans esigi kullanmali; seri mod bit-eş kalir.
Bu, planin G.5 kabul kriteridir.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Tek kaynak: tum tahmin modellerinin random_state'i buradan gelir.
# Env `PREDICTION_SEED` ile override edilebilir (config ile hizali).
try:
    GLOBAL_SEED = int(os.environ.get('PREDICTION_SEED', '42'))
except ValueError:
    GLOBAL_SEED = 42


def seed_everything(seed: int = GLOBAL_SEED, *, deterministic_torch: bool = False) -> int:
    """Surecin tum RNG'lerini tohumla. Egitim/batch kosumunun basinda cagir.

    Tohumlananlar: PYTHONHASHSEED, python `random`, numpy, torch (CPU + CUDA).
     Agac modelleri (xgboost/lightgbm/catboost) kendi `random_state`'lerini
    parametre uzerinden alir (bkz. GLOBAL_SEED) — burada ayrica gerek yok.

    Args:
        seed: Tohum degeri (varsayilan GLOBAL_SEED=42).
        deterministic_torch: True ise cuDNN deterministik moda alinir
            (biraz yavas, ama tam tekrar uretilebilir). Varsayilan False —
            mevcut hiz/davranis korunur; sadece istenirse aktive edilir.

    Returns:
        Kullanilan seed (loglama/manifest icin).
    """
    os.environ['PYTHONHASHSEED'] = str(seed)

    import random
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            # Tam determinizm (cuDNN). Performans maliyeti olabilir.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger.debug("seed_everything(%d) uygulandi (deterministic_torch=%s)",
                 seed, deterministic_torch)
    return seed
