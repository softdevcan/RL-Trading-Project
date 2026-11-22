"""Faz 11: M8 ileri veri simulasyonu uclari (plan phase-11-integration-plan.md §3 S4, S5).

Okuma herkese (CurrentUser; viewer dahil), yazma (isi tetikleme) yalniz yonetici: defter ortak arastirma alanidir
(K-F11-3). Pano bu uclari `api_client` uzerinden cagirir.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.auth.deps import CurrentUser, RequireAdmin
from app.services import marl_forward

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/marl", tags=["marl"])


@router.get("/forward/status")
async def forward_status(user: CurrentUser):
    return marl_forward.info()


@router.post("/forward/run")
async def forward_run(user: RequireAdmin):
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
