"""Faz 11: M8 ileri veri simulasyonu uclari (plan phase-11-integration-plan.md §3 S4, S5).

Okuma herkese (CurrentUser; viewer dahil), yazma (isi tetikleme) yalniz yonetici: defter ortak arastirma alanidir
(K-F11-3). Pano bu uclari `api_client` uzerinden cagirir.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from app.auth.deps import CurrentUser, RequireAdmin
from app.services import marl_forward

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/marl", tags=["marl"])


@router.get("/forward/status")
def forward_status(user: CurrentUser):
    return marl_forward.info()


@router.post("/forward/run")
def forward_run(user: RequireAdmin):
    if not marl_forward.available():
        raise HTTPException(status_code=409, detail="Dondurulmus M8 arsivi/kaynagi bu sunucuda yok (data/marl/frozen)")
    started = marl_forward.run_now('manual')
    try:
        from app.auth import service
        from app.auth.db import session_scope

        with session_scope() as db:
            service.audit(db, "marl.forward_run", user=user, detail={"started": started})
    except Exception as exc:  # pragma: no cover - denetim kaydi kritik yol degil
        logger.warning(f"MARL ileri veri isi denetim kaydina yazilamadi: {exc}")
    return {"started": started, "already_running": not started}


@router.get("/forward/view")
def forward_view(user: CurrentUser, window: str = Query("forward"), keys: str = Query("")):
    """Pano grafigi + ozet: kol (12 tohum ortalamasi +- std) ve kural portfoyleri; ayni pencere, ayni maliyet."""
    from marl import forward_views as V
    if not marl_forward.available():
        raise HTTPException(status_code=409, detail="Dondurulmus M8 arsivi bu sunucuda yok")
    if window not in V.WINDOWS:
        raise HTTPException(status_code=422, detail=f"pencere {V.WINDOWS}")
    sel = tuple(k for k in keys.split(',') if k) or V.DEFAULT_VIEW
    allowed = set(V.ARMS) | {k for k in V.ARM_LABELS if k.startswith('rule/')}
    if not set(sel) <= allowed:
        raise HTTPException(status_code=422, detail=f"bilinmeyen strateji: {sorted(set(sel) - allowed)}")
    out = V.view(window, sel)
    out['labels'] = V.ARM_LABELS
    return out


@router.get("/forward/product")
def forward_product(user: CurrentUser, sid: str = Query(...), capital: float = Query(100_000.0, gt=0, le=1e12),
                    window: str = Query("forward")):
    """Urun portfoyu (K-F11-2): secilen strateji tam sayi lotla ve kesirli; fark. Ileri teste girmez, kayit tutmaz."""
    from marl import forward_views as V
    if not marl_forward.available():
        raise HTTPException(status_code=409, detail="Dondurulmus M8 arsivi bu sunucuda yok")
    if window not in V.WINDOWS:
        raise HTTPException(status_code=422, detail=f"pencere {V.WINDOWS}")
    try:
        return V.product(sid, capital, window)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
