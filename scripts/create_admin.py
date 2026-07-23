"""Admin hesabi olustur / parola sifirla (sunucuda elle calistirilir).

Kullanim:
    python scripts/create_admin.py
    python scripts/create_admin.py --email admin@sirket.com --role admin
    python scripts/create_admin.py --email user@sirket.com --reset-password

Parola argumanla verilmez (shell gecmisine dusmesin) — istem ile sorulur.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.auth.db import init_db, session_scope  # noqa: E402
from app.auth.models import Role  # noqa: E402
from app.auth.service import (  # noqa: E402
    AuthError, audit, create_user, get_user_by_email, set_password, update_user,
)


def prompt_password() -> str:
    while True:
        first = getpass.getpass("Parola (min 10 karakter, buyuk+kucuk harf+rakam): ")
        second = getpass.getpass("Parola (tekrar): ")
        if first != second:
            print("  Parolalar eslesmiyor, tekrar deneyin.\n")
            continue
        return first


def main() -> int:
    parser = argparse.ArgumentParser(description="RL Trading admin hesabi yonetimi")
    parser.add_argument("--email", help="Hesap e-postasi")
    parser.add_argument("--full-name", default="", help="Ad soyad")
    parser.add_argument("--role", default=Role.ADMIN, choices=list(Role.ALL))
    parser.add_argument("--reset-password", action="store_true",
                        help="Var olan hesabin parolasini sifirla")
    args = parser.parse_args()

    email = args.email or input("E-posta: ").strip()
    if not email:
        print("E-posta zorunlu.")
        return 1

    init_db()
    with session_scope() as db:
        existing = get_user_by_email(db, email)

        if existing and not args.reset_password:
            print(f"'{email}' zaten kayitli (rol: {existing.role}).")
            answer = input("Parolayi sifirlamak ister misiniz? [e/H] ").strip().lower()
            if answer != "e":
                return 1
            args.reset_password = True

        password = prompt_password()

        try:
            if args.reset_password and existing:
                set_password(db, existing, password, must_change=False)
                if existing.role != args.role:
                    update_user(db, existing, role=args.role)
                audit(db, "user.password_reset", user=existing, target=email, detail={"via": "cli"})
                print(f"OK — '{email}' parolasi guncellendi (rol: {existing.role}).")
            else:
                user = create_user(
                    db,
                    email=email,
                    password=password,
                    full_name=args.full_name,
                    role=args.role,
                    created_by="cli",
                    must_change_password=False,
                )
                audit(db, "user.create", user=user, target=email, detail={"via": "cli"})
                print(f"OK — '{email}' olusturuldu (rol: {user.role}).")
        except AuthError as exc:
            print(f"HATA: {exc.message}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
