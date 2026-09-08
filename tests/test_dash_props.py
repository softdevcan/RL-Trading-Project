"""Dash/dbc bilesen prop uyumlulugu (pytest degil, `python` ile).

    python tests/test_dash_props.py

Neden var: Dash bilesenleri TANIMADIKLARI bir kwarg'i sessizce yok saymaz,
`TypeError` firlatir — ve bu hata **calisma aninda**, callback govdesinde
patlar. Layout render testi de yakalamaz. Faz 8'de bu sinif iki kez isirdi:

  1. `dbc.NavLink(title=...)`  -> tum Dash agaci render edilemedi, /dash/ 500
  2. `dbc.Spinner(className=...)` -> egitim/optimizasyon ilerleme callback'i
     her yoklamada 500 verdi (dbc 2.0.4'te dogru ad `spinner_class_name`)

Ikisi de yalnizca ilgili kod yolu CALISINCA goruluyordu. Bu denetim kaynagi
AST ile tarayip her `dbc.X` / `dcc.X` / `html.X` cagrisinin kwarg'larini
bilesenin GERCEK prop listesiyle karsilastirir; dbc/dash yukseltmesi bir prop
adini degistirirse burada kalir.
"""

from __future__ import annotations

import ast
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import dash_bootstrap_components as dbc  # noqa: E402
from dash import dcc, html  # noqa: E402

MODULES = {"dbc": dbc, "dcc": dcc, "html": html}

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [OK]   {name}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name} {detail}")


def allowed_props(component) -> set[str] | None:
    """Bilesenin kabul ettigi prop adlari.

    None = bu bir Dash BILESENI degil (ornegin `dcc.send_string` bir
    yardimci fonksiyon) ya da ornek uretilemedi. Bazi bilesenler `id`
    zorunlu istiyor (`dcc.Store`, `dcc.Location`), o yuzden ikinci deneme
    sahte bir id ile yapilir.
    """
    if not isinstance(component, type):
        return None  # fonksiyon/sabit — bilesen degil
    for kwargs in ({}, {"id": "_probe"}):
        try:
            return set(component(**kwargs)._prop_names)
        except Exception:
            continue
    return None


def main() -> int:
    files = sorted(glob.glob(os.path.join(ROOT, "dashboard", "**", "*.py"),
                             recursive=True))
    check("Taranacak dosya bulundu", len(files) > 5, f"({len(files)} dosya)")

    bad: list[str] = []
    skipped: set[str] = set()
    scanned = 0

    for path in files:
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
                continue
            if func.value.id not in MODULES:
                continue

            component = getattr(MODULES[func.value.id], func.attr, None)
            if component is None:
                continue
            allowed = allowed_props(component)
            if allowed is None:
                # Bilesen olmayan cagrilar (yardimci fonksiyonlar) sessizce
                # gecilir; ornek URETILEMEYEN bir BILESEN olsaydi denetim
                # sessizce kor kalirdi, o yuzden ayri sayilir.
                if isinstance(component, type):
                    skipped.add(f"{func.value.id}.{func.attr}")
                continue

            scanned += 1
            for keyword in node.keywords:
                # **kwargs yayilimi (kw.arg is None) statik olarak denetlenemez
                if keyword.arg and keyword.arg not in allowed:
                    rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
                    bad.append(
                        f"{rel}:{node.lineno} {func.value.id}.{func.attr}"
                        f"(... {keyword.arg}=...) — kabul edilenler: "
                        f"{', '.join(sorted(allowed)[:8])}..."
                    )

    check("Bilesen cagrilari tarandi", scanned > 100, f"({scanned} cagri)")
    check("Ornek uretilemeyen bilesen yok", not skipped, f"atlanan: {sorted(skipped)}")

    print("\n  -- bulunan uyumsuz kwarg'lar --")
    for line in sorted(set(bad)):
        print(f"     {line}")
    check("Her kwarg bilesenin prop listesinde var", not bad,
          f"({len(set(bad))} uyumsuz cagri)")

    print("\n" + "=" * 60)
    print(f"  Gecen: {len(PASSED)}   Kalan: {len(FAILED)}")
    if FAILED:
        for name in FAILED:
            print(f"   - {name}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
