"""Kullanicinin KENDI hesabina dair uc noktalar (her rol, viewer dahil).

Neden /api/account/*, neden /auth/* degil?
------------------------------------------
`/auth/*` tarayicinin **dogrudan** cagirdigi yuzeydir: giris formu ve
clientside tema anahtari. Orada CSRF'i ucun kendisi kontrol etmek zorunda
(bkz. routes.py::update_preferences), cunku AuthGateMiddleware CSRF'i
yalnizca `/api/*` yazmalarinda dogruluyor.

Buradaki uclari ise pano callback'leri `dashboard/api_client.py` uzerinden
cagiriyor; o istemci oturum cerezini ve `X-CSRF-Token` basligini zaten
tasiyor. `/api/*` altinda durunca CSRF, RBAC ve calisma alani baglami
mevcut kapidan bedava geliyor — `/api/admin/*` ile ayni kalip.

Yetki: hepsi `CurrentUser`. Hedef **her zaman** oturumdaki kullanicidir;
govdeden kullanici kimligi ALINMAZ, dolayisiyla baskasinin hesabina yazmak
mumkun degil. Viewer da kendi adini/oturumlarini yonetebilir — bu yetki
degil, kisisel hesap ayaridir.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import security, service
from app.auth.db import get_db
from app.auth.deps import CurrentUser
from app.auth.models import SessionToken, User, as_utc
from app.auth.routes import client_ip
from app.auth.workspace import workspace_usage
from app.core.config import get_settings

router = APIRouter(prefix="/account", tags=["account"])


class ProfileUpdate(BaseModel):
    # E-posta bilincli olarak yok: giris anahtari ve audit kaydinin kimligi.
    # Degistirmek admin isi (/api/admin/users/{id}).
    full_name: str = Field(min_length=1, max_length=120)


def _persisted(db: Session, user: User) -> User:
    """Oturumdaki kullanicinin DB satiri.

    AUTH_ENABLED=False iken `get_current_user` DB'de olmayan sanal bir admin
    dondurur; o modda yazacak bir hesap kaydi yoktur. 404 yerine 409: istek
    hatali degil, kurulum bu islemi desteklemiyor.
    """
    row = service.get_user(db, user.id)
    if row is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Kimlik dogrulama kapali; kaydedilecek hesap yok",
        )
    return row


def _device_label(user_agent: str) -> str:
    """Ham UA yerine "Chrome · Windows" gibi okunur bir etiket.

    Amac tam tespit degil, kullanicinin "bu benim makinem mi" sorusuna
    cevap verebilmesi. Taninmayan UA oldugu gibi kisaltilarak gosterilir ki
    hicbir sey uydurulmus olmasin.
    """
    ua = user_agent or ""
    low = ua.lower()

    browser = ""
    for needle, label in (
        ("edg/", "Edge"), ("opr/", "Opera"), ("firefox", "Firefox"),
        ("chrome", "Chrome"), ("safari", "Safari"),
    ):
        if needle in low:
            browser = label
            break

    platform = ""
    for needle, label in (
        ("windows", "Windows"), ("android", "Android"), ("iphone", "iPhone"),
        ("ipad", "iPad"), ("mac os", "macOS"), ("linux", "Linux"),
    ):
        if needle in low:
            platform = label
            break

    if browser and platform:
        return f"{browser} · {platform}"
    if browser or platform:
        return browser or platform
    return (ua[:48] + "...") if len(ua) > 48 else (ua or "Bilinmeyen istemci")


# "Yakinda bitti" penceresi. Daha uzun tutmak son kosumun sonucunu gunlerce
# zilde asili birakirdi; daha kisa tutmak gece biten bir egitimi kacirtirdi.
NOTIFY_RECENT_SECONDS = 12 * 3600


def _is_recent(epoch: float | None, now: float) -> bool:
    """Epoch damgasi pencere icinde mi. Damga yoksa GOSTERME —
    "ne zaman bittigini bilmiyorum" ile "az once bitti" ayni sey degil."""
    if not epoch:
        return False
    return 0 <= (now - float(epoch)) <= NOTIFY_RECENT_SECONDS


def _is_recent_iso(stamp: str | None, now: float) -> bool:
    """ISO damgali surum (tahmin egitimi `finished_at` ISO yaziyor).

    Naive damga yerel saat kabul edilir: prediction_service `datetime.now()`
    ile yaziyor, epoch'a cevirirken ayni varsayim korunmali.
    """
    if not stamp:
        return False
    try:
        parsed = datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return False
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return _is_recent(parsed.timestamp(), now)


def _current_jti(request: Request) -> str:
    """Istegi yapan tarayicinin refresh token jti'si (yoksa bos)."""
    token = request.cookies.get(get_settings().REFRESH_COOKIE_NAME)
    if not token:
        return ""
    try:
        payload = security.decode_token(token, expected_type=security.REFRESH_TOKEN_TYPE)
    except security.TokenError:
        return ""
    return payload.get("jti", "")


def _group_sessions(rows: list[SessionToken], current_jti: str) -> list[dict]:
    """Oturumlari (ip, user_agent) ile grupla.

    Es zamanli sessiz yenileme (grace penceresi) ayni tarayici icin birden
    fazla gecerli kayit birakabiliyor. Kullaniciya "3 aktif oturum" demek
    yaniltici olurdu; grup basina tek satir gosterilir, `tokens` alani kac
    kayda karsilik geldigini saklamadan verir.
    """
    groups: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row.ip or "", row.user_agent or "")
        group = groups.get(key)
        if group is None:
            group = {
                "ip": row.ip or "",
                "device": _device_label(row.user_agent),
                "user_agent": row.user_agent or "",
                "last_seen": as_utc(row.issued_at).isoformat() if row.issued_at else None,
                "expires_at": as_utc(row.expires_at).isoformat() if row.expires_at else None,
                "tokens": 0,
                "current": False,
            }
            groups[key] = group
        group["tokens"] += 1
        if row.jti == current_jti and current_jti:
            group["current"] = True
        # list_sessions yeniden eskiye siralar; ilk satir zaten en yenisi.

    # Once son gorulmeye gore yeniden eskiye, sonra mevcut oturumu basa al
    # (sort kararli oldugu icin ikinci gecis ilkini bozmaz).
    ordered = sorted(groups.values(), key=lambda g: g["last_seen"] or "", reverse=True)
    ordered.sort(key=lambda g: not g["current"])
    return ordered


@router.get("/me")
async def get_account(user: CurrentUser, db: Annotated[Session, Depends(get_db)]) -> dict:
    """Profil karti icin tek cagri: hesap alanlari + calisma alani kullanimi."""
    row = service.get_user(db, user.id)
    data = (row or user).to_dict()
    return {
        "user": data,
        # AUTH_ENABLED=False iken hesap kalici degil; arayuz duzenleme
        # formunu buna bakarak gizler.
        "persistent": row is not None,
        "workspace": workspace_usage(user.id),
    }


@router.patch("/profile")
async def update_profile(
    payload: ProfileUpdate,
    user: CurrentUser,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Kullanicinin kendi ad soyadini guncellemesi.

    Rol, is_active ve e-posta bilincli olarak disarida: bunlar yetki/kimlik
    alanlari, admin ucundan yonetilir. Kullanici kendi rolunu yukseltemez.
    """
    row = _persisted(db, user)
    before = row.full_name

    service.update_user(db, row, full_name=payload.full_name)
    if row.full_name != before:
        service.audit(db, "account.update", user=row, target=row.email,
                      ip=client_ip(request),
                      detail={"full_name": {"from": before, "to": row.full_name}})
    db.commit()
    db.refresh(row)
    return {"status": "ok", "user": row.to_dict()}


@router.get("/sessions")
async def list_own_sessions(
    user: CurrentUser,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    rows = service.list_sessions(db, user.id)
    sessions = _group_sessions(rows, _current_jti(request))
    return {"sessions": sessions, "count": len(sessions), "tokens": len(rows)}


@router.get("/activity")
async def list_own_activity(
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    limit: int = Query(20, ge=1, le=100),
) -> dict:
    """Kendi hesap etkinligi — "birisi hesabima girmeye calisti mi" sorusu.

    Kapsam karari `service.list_audit_for_user` docstring'inde: yalnizca
    `user_id` eslesen satirlar, yoneticinin bu hesap uzerindeki islemleri
    degil.

    `detail` ayristirilip sozluk olarak donuyor (kullanicinin kendi verisi:
    basarisizlik sebebi, kapatilan oturum sayisi, ad degisikligi). Etiketlere
    cevirmek arayuzun isi — uc ham kalir.
    """
    rows = service.list_audit_for_user(db, user.id, limit=limit)

    entries = []
    for row in rows:
        try:
            detail = json.loads(row.detail) if row.detail else {}
        except (ValueError, TypeError):
            detail = {}
        entries.append({
            "ts": as_utc(row.ts).isoformat() if row.ts else None,
            "action": row.action,
            "target": row.target or "",
            "success": bool(row.success),
            "ip": row.ip or "",
            "detail": detail if isinstance(detail, dict) else {},
        })
    return {"entries": entries, "count": len(entries)}


@router.get("/notifications")
async def list_notifications(user: CurrentUser) -> dict:
    """Ust cubuktaki zil icin bildirimler.

    **Olay gunlugu degil, DURUM ozeti.** Kalici bir bildirim tablosu yok;
    burada okunanlar zaten bellekte tutulan calisma durumlaridir (RL egitimi
    ve tahmin egitimi, ikisi de kullanici bazli sozlukler). Bu yuzden uc
    ucuz: disk veya DB'ye gitmiyor, sik yoklanabilir.

    Kasitli olarak DISARIDA: veri tazeligi. `/trading/data/status` panelin
    tamamini CSV'den okuyor; zilin arkasina koymak her yoklamada o maliyeti
    odetirdi. Veri durumu kendi sayfasinda kaliyor.

    "Bitti" bildirimleri `NOTIFY_RECENT_SECONDS` penceresiyle sinirli — aksi
    halde son kosumun sonucu sonsuza kadar zilde asili kalirdi.
    """
    now = time.time()
    items: list[dict] = []

    # ── RL egitimi ────────────────────────────────────────────────────────
    try:
        from app.api.routes.trading import get_training_state

        rl = get_training_state(user.id) or {}
    except Exception:  # pragma: no cover - modul yuklenemezse zil bos kalir
        rl = {}

    if rl.get("is_training"):
        total = rl.get("total_steps") or 0
        current = rl.get("current_step") or 0
        percent = int(100 * current / total) if total else None
        body = f"%{percent}" if percent is not None else "Suruyor"
        phase = rl.get("phase_name")
        if phase and phase not in ("training", "completed", "error"):
            body = f"{body} · {phase}"
        items.append({
            "id": "rl-training",
            "kind": "info",
            "title": "Model egitimi suruyor",
            "body": body,
            "href": "/dash/training",
        })
    elif rl.get("state") == "error":
        items.append({
            "id": "rl-error",
            "kind": "error",
            "title": "Model egitimi hata verdi",
            "body": (rl.get("error") or "")[:160],
            "href": "/dash/training",
        })
    elif rl.get("state") == "completed" and _is_recent(rl.get("finished_ts"), now):
        items.append({
            "id": "rl-done",
            "kind": "success",
            "title": "Model egitimi tamamlandi",
            "body": "Sonuclar Modeller sayfasinda",
            "href": "/dash/models",
        })

    # ── Tahmin egitimi (sembol basina) ────────────────────────────────────
    try:
        from app.services.prediction_service import get_training_state as pred_states

        records = pred_states(user_id=user.id) or []
    except Exception:  # pragma: no cover
        records = []

    for record in records if isinstance(records, list) else []:
        symbol = record.get("symbol", "?")
        state = record.get("state")
        if state == "running":
            items.append({
                "id": f"pred-{symbol}",
                "kind": "info",
                "title": f"{symbol} tahmin egitimi suruyor",
                "body": record.get("source") or "",
                "href": "/dash/prediction",
            })
        elif state == "error":
            items.append({
                "id": f"pred-err-{symbol}",
                "kind": "error",
                "title": f"{symbol} tahmin egitimi hata verdi",
                "body": (record.get("error") or "")[:160],
                "href": "/dash/prediction",
            })
        elif state == "completed" and _is_recent_iso(record.get("finished_at"), now):
            items.append({
                "id": f"pred-done-{symbol}",
                "kind": "success",
                "title": f"{symbol} tahmin egitimi tamamlandi",
                "body": record.get("source") or "",
                "href": "/dash/prediction",
            })

    return {"items": items, "count": len(items)}


@router.post("/sessions/revoke-others")
async def revoke_other_sessions(
    user: CurrentUser,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Bu tarayici disindaki tum oturumlari kapat.

    Parola degistirmenin aksine parola sormaz — oturum zaten dogrulanmis ve
    islem yalnizca erisim KALDIRIYOR, vermiyor. Denetim kaydina yazilir:
    parola degisimi gibi bir guvenlik olayi.
    """
    row = _persisted(db, user)
    revoked = service.revoke_other_sessions(db, row.id, _current_jti(request))
    service.audit(db, "account.revoke_sessions", user=row, target=row.email,
                  ip=client_ip(request), detail={"revoked": revoked})
    db.commit()
    return {"status": "ok", "revoked": revoked}
