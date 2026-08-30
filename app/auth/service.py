"""Kullanici yonetimi ve kimlik dogrulama is mantigi."""

from __future__ import annotations

import json
import logging
from datetime import timedelta, timezone

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.auth import security
from app.auth.models import AuditLog, Role, SessionToken, User, utcnow
from app.core.config import get_settings

log = logging.getLogger(__name__)


class AuthError(Exception):
    """Kimlik dogrulama hatasi. `code` UI'da mesaja cevrilir."""

    def __init__(self, code: str, message: str, status_code: int = 401):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


# ── Audit ─────────────────────────────────────────────────────────────────

def audit(
    db: Session,
    action: str,
    *,
    user: User | None = None,
    email: str = "",
    target: str = "",
    success: bool = True,
    ip: str = "",
    detail: dict | None = None,
) -> None:
    db.add(AuditLog(
        user_id=user.id if user else None,
        email=email or (user.email if user else ""),
        action=action,
        target=target,
        success=success,
        ip=ip[:64],
        detail=json.dumps(detail, ensure_ascii=False)[:4000] if detail else "",
    ))


# ── Sorgular ──────────────────────────────────────────────────────────────

def get_user_by_email(db: Session, email: str) -> User | None:
    return db.execute(
        select(User).where(User.email == normalize_email(email))
    ).scalar_one_or_none()


def get_user(db: Session, user_id: str) -> User | None:
    return db.get(User, user_id)


def list_users(db: Session) -> list[User]:
    return list(db.execute(select(User).order_by(User.created_at)).scalars())


def user_count(db: Session) -> int:
    return len(list(db.execute(select(User.id)).scalars()))


# ── Kullanici CRUD ────────────────────────────────────────────────────────

def create_user(
    db: Session,
    *,
    email: str,
    password: str,
    full_name: str = "",
    role: str = Role.USER,
    created_by: str = "",
    must_change_password: bool = True,
    is_active: bool = True,
) -> User:
    email = normalize_email(email)
    if not email or "@" not in email:
        raise AuthError("invalid_email", "Gecerli bir e-posta adresi girin", 400)
    if role not in Role.ALL:
        raise AuthError("invalid_role", f"Gecersiz rol: {role}", 400)
    if get_user_by_email(db, email):
        raise AuthError("email_exists", "Bu e-posta zaten kayitli", 409)

    problems = security.validate_password_strength(password)
    if problems:
        raise AuthError("weak_password", "Parola politikasi: " + "; ".join(problems), 400)

    user = User(
        email=email,
        full_name=(full_name or "").strip()[:120],
        password_hash=security.hash_password(password),
        role=role,
        created_by=created_by,
        must_change_password=must_change_password,
        is_active=is_active,
    )
    db.add(user)
    db.flush()  # user.id gerekli

    # Hibrit izolasyon: kullanicinin calisma alani hemen hazirlanir.
    from app.auth.workspace import ensure_workspace
    ensure_workspace(user.id)
    return user


def update_user(
    db: Session,
    user: User,
    *,
    full_name: str | None = None,
    role: str | None = None,
    is_active: bool | None = None,
) -> User:
    if full_name is not None:
        user.full_name = full_name.strip()[:120]
    if role is not None:
        if role not in Role.ALL:
            raise AuthError("invalid_role", f"Gecersiz rol: {role}", 400)
        user.role = role
    if is_active is not None:
        user.is_active = is_active
        if not is_active:
            revoke_all_sessions(db, user.id)
    user.updated_at = utcnow()
    return user


def set_password(db: Session, user: User, new_password: str, *, must_change: bool = False) -> None:
    problems = security.validate_password_strength(new_password)
    if problems:
        raise AuthError("weak_password", "Parola politikasi: " + "; ".join(problems), 400)
    user.password_hash = security.hash_password(new_password)
    user.must_change_password = must_change
    user.failed_attempts = 0
    user.locked_until = None
    user.updated_at = utcnow()


def delete_user(db: Session, user: User) -> None:
    """Hesabi sil. Calisma alani dosyalari KASITLI olarak silinmez —
    egitilmis modeller/gecmis veri kaybi geri alinamaz. Temizlik operatorde."""
    revoke_all_sessions(db, user.id)
    db.delete(user)


# ── Kimlik dogrulama ──────────────────────────────────────────────────────

def authenticate(db: Session, email: str, password: str, *, ip: str = "") -> User:
    settings = get_settings()
    email = normalize_email(email)
    user = get_user_by_email(db, email)

    # Kullanici yoksa da bcrypt maliyeti odenir: varlik sizintisini onler.
    if user is None:
        security.verify_password(password, security.hash_password("dummy-timing-guard"))
        audit(db, "login", email=email, success=False, ip=ip, detail={"reason": "no_such_user"})
        raise AuthError("invalid_credentials", "E-posta veya parola hatali")

    locked_until = user.locked_until
    if locked_until and locked_until > utcnow():
        remaining = int((locked_until - utcnow()).total_seconds() // 60) + 1
        audit(db, "login", user=user, success=False, ip=ip, detail={"reason": "locked"})
        raise AuthError("account_locked", f"Hesap kilitli. {remaining} dakika sonra deneyin", 429)

    if not user.is_active:
        audit(db, "login", user=user, success=False, ip=ip, detail={"reason": "inactive"})
        raise AuthError("account_disabled", "Hesap devre disi. Yonetici ile gorusun", 403)

    if not security.verify_password(password, user.password_hash):
        user.failed_attempts += 1
        if user.failed_attempts >= settings.LOGIN_MAX_ATTEMPTS:
            user.locked_until = utcnow() + timedelta(minutes=settings.LOGIN_LOCKOUT_MINUTES)
            user.failed_attempts = 0
            audit(db, "login", user=user, success=False, ip=ip, detail={"reason": "lockout_triggered"})
            raise AuthError(
                "account_locked",
                f"Cok fazla hatali deneme. Hesap {settings.LOGIN_LOCKOUT_MINUTES} dakika kilitlendi",
                429,
            )
        audit(db, "login", user=user, success=False, ip=ip, detail={"reason": "bad_password"})
        raise AuthError("invalid_credentials", "E-posta veya parola hatali")

    user.failed_attempts = 0
    user.locked_until = None
    user.last_login_at = utcnow()
    audit(db, "login", user=user, success=True, ip=ip)
    return user


# ── Oturumlar (refresh token kayitlari) ───────────────────────────────────

def create_session(
    db: Session, user: User, *, ip: str = "", user_agent: str = ""
) -> tuple[str, str]:
    """(access_token, refresh_token) uretir ve refresh kaydini yazar."""
    access = security.create_access_token(user.id, user.email, user.role)
    refresh, jti, expires = security.create_refresh_token(user.id)
    db.add(SessionToken(
        user_id=user.id,
        jti=jti,
        expires_at=expires.astimezone(timezone.utc).replace(tzinfo=None),
        ip=ip[:64],
        user_agent=(user_agent or "")[:255],
    ))
    return access, refresh


# Dash sayfasi ayni anda onlarca callback atar; hepsi ayni anda sessiz
# yenilemeye girerse ilk istek token'i iptal eder ve digerleri "calinmis
# token" gibi gorunur. Bu pencere icinde tekrar kullanim normal kabul edilir.
REFRESH_REUSE_GRACE_SEC = 30


def rotate_session(db: Session, refresh_token: str, *, ip: str = "", user_agent: str = "") -> tuple[str, str, User]:
    """Refresh token'i dogrula, eskisini iptal et, yenisini ver (rotation).

    Iptal edilmis bir jti (grace penceresi disinda) tekrar kullanilirsa token
    calinmis sayilir ve kullanicinin TUM oturumlari kapatilir.
    """
    payload = security.decode_token(refresh_token, expected_type=security.REFRESH_TOKEN_TYPE)
    jti = payload.get("jti", "")
    record = db.execute(select(SessionToken).where(SessionToken.jti == jti)).scalar_one_or_none()
    if record is None:
        raise AuthError("session_unknown", "Oturum bulunamadi")

    if record.revoked_at is not None:
        age = (utcnow() - record.revoked_at).total_seconds()
        if age > REFRESH_REUSE_GRACE_SEC:
            revoke_all_sessions(db, record.user_id)
            audit(db, "session.reuse_detected", user=record.user, success=False, ip=ip)
            raise AuthError("session_reused", "Oturum gecersiz. Tekrar giris yapin")
        # Es zamanli yenileme yarisi — yeni bir oturum verip devam et.
        user = db.get(User, record.user_id)
        if user is None or not user.is_active:
            raise AuthError("account_disabled", "Hesap devre disi", 403)
        access, refresh = create_session(db, user, ip=ip, user_agent=user_agent)
        return access, refresh, user

    if not record.is_valid:
        raise AuthError("session_expired", "Oturum suresi doldu. Tekrar giris yapin")

    user = db.get(User, record.user_id)
    if user is None or not user.is_active:
        raise AuthError("account_disabled", "Hesap devre disi", 403)

    record.revoked_at = utcnow()
    access, refresh = create_session(db, user, ip=ip, user_agent=user_agent)
    return access, refresh, user


def revoke_session(db: Session, refresh_token: str) -> None:
    try:
        payload = security.decode_token(refresh_token, expected_type=security.REFRESH_TOKEN_TYPE)
    except security.TokenError:
        return
    record = db.execute(
        select(SessionToken).where(SessionToken.jti == payload.get("jti", ""))
    ).scalar_one_or_none()
    if record and record.revoked_at is None:
        record.revoked_at = utcnow()


def revoke_all_sessions(db: Session, user_id: str) -> int:
    rows = db.execute(
        select(SessionToken).where(
            SessionToken.user_id == user_id, SessionToken.revoked_at.is_(None)
        )
    ).scalars()
    count = 0
    now = utcnow()
    for row in rows:
        row.revoked_at = now
        count += 1
    return count


def list_sessions(db: Session, user_id: str) -> list[SessionToken]:
    """Kullanicinin gecerli oturum kayitlari, yeniden eskiye.

    "Gecerli" = iptal edilmemis VE suresi dolmamis. Rotasyon her yenilemede
    eskisini iptal edip yenisini yazdigi icin normalde tarayici basina tek
    satir kalir; es zamanli yenileme yarisinda (grace penceresi, bkz.
    REFRESH_REUSE_GRACE_SEC) ayni tarayici icin birden fazla gecerli satir
    olusabilir. Cagiran taraf bunlari (ip, user_agent) ile gruplar.
    """
    rows = db.execute(
        select(SessionToken)
        .where(
            SessionToken.user_id == user_id,
            SessionToken.revoked_at.is_(None),
            SessionToken.expires_at > utcnow(),
        )
        .order_by(desc(SessionToken.issued_at))
    ).scalars()
    return list(rows)


def revoke_other_sessions(db: Session, user_id: str, keep_jti: str) -> int:
    """Verilen jti disindaki tum oturumlari kapat (kasitli iptal).

    `revoke_all_sessions`tan iki farki var:

    1. Cagiran kullanici kendi oturumundan dusmez. `keep_jti` bos ise
       (refresh cerezi yoksa) hepsi kapanir — kimligi bilinmeyen bir oturumu
       "benimki" sayip ayakta birakmak yanlis olurdu.
    2. Kayitlar **silinir**, `revoked_at` ile isaretlenmez. Sebep: iptal
       edilmis bir jti REFRESH_REUSE_GRACE_SEC (30 sn) icinde tekrar
       kullanilirsa `rotate_session` bunu "es zamanli yenileme yarisi" sayip
       YENI bir oturum veriyor. O grace penceresi Dash'in paralel
       callback'leri icin var, ama kasitli iptali de gecersiz kilardi:
       /auth/refresh CSRF istemedigi icin, calinan bir refresh cerezini elinde
       tutan taraf saniyede bir yenileyerek "diger oturumlari kapat"i
       atlatabilirdi. Silinen kayitta `record is None` -> session_unknown ->
       401; grace yolu hic calismaz.

    Kaydin izi denetim kaydinda kalir (account.revoke_sessions).
    """
    count = 0
    rows = db.execute(
        select(SessionToken).where(SessionToken.user_id == user_id)
    ).scalars()
    for row in rows:
        if keep_jti and row.jti == keep_jti:
            continue
        db.delete(row)
        if row.revoked_at is None:
            count += 1  # kullaniciya "kac AKTIF oturum kapandi" denir
    return count


def list_audit_for_user(db: Session, user_id: str, limit: int = 20) -> list[AuditLog]:
    """Kullanicinin KENDI denetim satirlari, yeniden eskiye.

    Filtre `user_id` uzerinden — yani "benim yaptigim / benim oturumumda olan"
    olaylar: girisler (basarili ve basarisiz), cikis, parola degisikligi,
    profil guncellemesi, oturum iptali, jeton tekrar kullanimi tespiti.

    `target` eslesmesi KASITLI olarak disarida: bir yoneticinin bu hesap
    uzerinde yaptigi islemler (rol degisimi, parola sifirlama) o satirlarda
    `user_id` olarak YONETICIYI tasir; onlari buraya almak yoneticinin
    kimligini ve IP'sini yonetici olmayan bir yuzeye sizdirirdi. Yonetici
    tarafi zaten /dash/users -> Denetim Kaydi'nda gorunuyor.

    Basarisiz girislerde `no_such_user` satiri hic eslesmez (user_id NULL) —
    oturum acmis bir kullanici icin zaten anlamsiz.
    """
    rows = db.execute(
        select(AuditLog)
        .where(AuditLog.user_id == user_id)
        .order_by(desc(AuditLog.ts))
        .limit(limit)
    ).scalars()
    return list(rows)


def purge_expired_sessions(db: Session) -> int:
    """Suresi dolmus kayitlari temizle (acilista cagrilir)."""
    rows = db.execute(select(SessionToken).where(SessionToken.expires_at < utcnow())).scalars()
    count = 0
    for row in rows:
        db.delete(row)
        count += 1
    return count
