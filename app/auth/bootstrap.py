"""Ilk admin hesabinin olusturulmasi.

Sirasiyla:
1. BOOTSTRAP_ADMIN_EMAIL + BOOTSTRAP_ADMIN_PASSWORD tanimliysa onunla ac.
2. Degilse ve DEBUG ise rastgele parola uretip LOG'a bir kez yaz.
3. DEBUG degilse hicbir sey yapma — kullanici yoksa /login "kurulum gerekli"
   uyarisi gosterir. Sunucuda parolayi log'a dusurmek istemiyoruz.

Kullanici zaten varsa hicbir sey yapilmaz (idempotent).
"""

from __future__ import annotations

import logging
import secrets

from app.auth.db import session_scope
from app.auth.models import Role
from app.auth.service import audit, create_user, user_count
from app.core.config import get_settings

log = logging.getLogger(__name__)


def bootstrap_admin() -> None:
    settings = get_settings()
    with session_scope() as db:
        if user_count(db) > 0:
            return

        email = (settings.BOOTSTRAP_ADMIN_EMAIL or "").strip().lower()
        password = settings.BOOTSTRAP_ADMIN_PASSWORD
        generated = False

        if not email:
            if not settings.DEBUG:
                log.warning(
                    "Hic kullanici yok ve BOOTSTRAP_ADMIN_EMAIL tanimsiz. "
                    "Admin olusturmak icin: python scripts/create_admin.py"
                )
                return
            email = "admin@localhost"

        if not password:
            if not settings.DEBUG:
                log.warning("BOOTSTRAP_ADMIN_PASSWORD tanimsiz — admin olusturulmadi.")
                return
            password = f"Rlt{secrets.token_urlsafe(12)}1"
            generated = True

        user = create_user(
            db,
            email=email,
            password=password,
            full_name="System Admin",
            role=Role.ADMIN,
            created_by="bootstrap",
            # ENV'den gelen parola operatorde zaten var; uretilen parola ilk
            # giriste degistirilmeli.
            must_change_password=generated,
        )
        audit(db, "user.bootstrap", user=user, target=email, detail={"generated": generated})

        if generated:
            log.warning(
                "\n%s\n  ILK ADMIN OLUSTURULDU (yalnizca bu kez gosterilir)\n"
                "  E-posta : %s\n  Parola  : %s\n"
                "  Ilk giriste parola degistirmeniz istenecek.\n%s",
                "=" * 62, email, password, "=" * 62,
            )
        else:
            log.info("Ilk admin olusturuldu: %s", email)
