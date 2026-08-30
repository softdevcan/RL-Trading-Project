"""Tema kontrast ve token butunlugu testi (pytest degil, dogrudan `python` ile).

    python tests/test_theme_contrast.py

Ne dogrular (Faz 8, E.1 + E.2):
  1. static/tokens.css'teki HER metin tokeni, kendi temasindaki UC yuzeyin
     (--rlt-bg / --rlt-surface / --rlt-surface-2) hepsinde WCAG AA (>= 4.5:1) geciyor mu.
  2. UI sinirlari ve dolgular >= 3:1, dolgu uzerindeki yazi >= 4.5:1.
  3. Koyu blok iki kez yaziliyor (damgasiz "system" hali + acik secim);
     ikisi BIREBIR ayni mi. Ayrisirlarsa bir durum sessizce bozulur.
  4. dashboard/theme.py'deki Plotly hex paleti tokens.css ile ayni mi.
  5. Sayfa kodunda kacak hex yok mu — renk yalnizca tokenlardan gelmeli.

Yeni renk eklerken bu testi calistir; "muhtemelen yeterlidir" kabul degil.
"""

from __future__ import annotations

import ast
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENS_CSS = os.path.join(ROOT, "static", "tokens.css")

AA_TEXT = 4.5      # normal metin
AA_NON_TEXT = 3.0  # UI sinirlari, dolgular

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [OK]   {name}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name} {detail}")


# ── WCAG 2.1 relative luminance ────────────────────────────────────────────

def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    channels = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255
        channels.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(fg: str, bg: str) -> float:
    a, b = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


# ── tokens.css ayristirma ──────────────────────────────────────────────────

_HEX = re.compile(r"^#[0-9a-fA-F]{3,8}$")


def _parse_block(css: str, start_index: int) -> dict[str, str]:
    """start_index'teki '{' ile eslesen '}' arasindaki --token: deger ciftleri."""
    depth = 0
    end = start_index
    for i in range(start_index, len(css)):
        if css[i] == "{":
            depth += 1
        elif css[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    body = css[start_index + 1:end]
    out: dict[str, str] = {}
    for name, value in re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", body):
        out[name.strip()] = value.strip()
    return out


def load_themes() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """(aydinlik, koyu-media, koyu-damga) token sozlukleri; yalnizca hex degerler."""
    css = open(TOKENS_CSS, encoding="utf-8").read()

    def block_after(pattern: str) -> dict[str, str]:
        m = re.search(pattern, css)
        if not m:
            return {}
        brace = css.index("{", m.end() - 1)
        return _parse_block(css, brace)

    # Aydinlik taban: dosyadaki ilk ":root {" blogu
    light = block_after(r":root\s*\{")
    dark_media = block_after(r":root:not\(\[data-theme=\"light\"\]\)\s*\{")
    dark_stamp = block_after(r":root\[data-theme=\"dark\"\]\s*\{")

    keep = lambda d: {k: v for k, v in d.items() if _HEX.match(v)}
    return keep(light), keep(dark_media), keep(dark_stamp)


# ── Kontrast beklentileri ──────────────────────────────────────────────────

SURFACES = ("--rlt-bg", "--rlt-surface", "--rlt-surface-2")

# Metin/ikon olarak kullanilan tokenlar: uc yuzeyin hepsinde AA gecmeli.
TEXT_TOKENS = (
    "--rlt-text", "--rlt-muted", "--rlt-primary", "--rlt-profit", "--rlt-loss",
    "--rlt-warn", "--rlt-info", "--rlt-accent", "--rlt-orange", "--rlt-gold",
)

# Dolgular: --rlt-on-fill yazisi okunmali (>=4.5) ve yuzeyden ayrilmali (>=3).
# --rlt-info dolgu tokeni degil ama .badge.bg-info zemininde kullaniliyor.
FILL_TOKENS = ("--rlt-primary-fill", "--rlt-profit-fill", "--rlt-loss-fill", "--rlt-warn-fill", "--rlt-info")

# Sinir: metin degil, 3:1 yeterli.
BORDER_TOKENS = ("--rlt-border-strong",)


def audit_theme(label: str, tokens: dict[str, str]) -> None:
    print(f"\n  -- {label} --")
    missing = [t for t in TEXT_TOKENS + FILL_TOKENS + BORDER_TOKENS + SURFACES
               if t not in tokens]
    check(f"{label}: tum tokenlar tanimli", not missing, f"eksik: {missing}")
    if missing:
        return

    for token in TEXT_TOKENS:
        ratios = {s: contrast(tokens[token], tokens[s]) for s in SURFACES}
        worst_surface = min(ratios, key=ratios.get)
        worst = ratios[worst_surface]
        check(
            f"{label}: {token} her yuzeyde AA ({worst:.2f})",
            worst >= AA_TEXT,
            f"en kotu {worst_surface} uzerinde {worst:.2f} < {AA_TEXT}",
        )

    for token in FILL_TOKENS:
        on_fill = contrast(tokens["--rlt-on-fill"], tokens[token])
        check(
            f"{label}: {token} uzerine --rlt-on-fill okunuyor ({on_fill:.2f})",
            on_fill >= AA_TEXT,
            f"{on_fill:.2f} < {AA_TEXT}",
        )
        sep = contrast(tokens[token], tokens["--rlt-surface"])
        check(
            f"{label}: {token} yuzeyden ayriliyor ({sep:.2f})",
            sep >= AA_NON_TEXT,
            f"{sep:.2f} < {AA_NON_TEXT}",
        )

    for token in BORDER_TOKENS:
        sep = contrast(tokens[token], tokens["--rlt-surface"])
        check(
            f"{label}: {token} gorunur sinir ({sep:.2f})",
            sep >= AA_NON_TEXT,
            f"{sep:.2f} < {AA_NON_TEXT}",
        )


def main() -> int:
    print("1) tokens.css ayristirma")
    light, dark_media, dark_stamp = load_themes()
    check("Aydinlik blok okundu", len(light) > 10, f"({len(light)} token)")
    check("Koyu media blogu okundu", len(dark_media) > 10, f"({len(dark_media)} token)")
    check("Koyu damga blogu okundu", len(dark_stamp) > 10, f"({len(dark_stamp)} token)")

    print("\n2) Koyu blogun iki kopyasi ayni mi")
    # "system" (damgasiz) ve acik secim ayni gorunmeli; ayrisirsa biri sessizce bozulur
    diff = {k for k in set(dark_media) | set(dark_stamp)
            if dark_media.get(k) != dark_stamp.get(k)}
    check("Koyu bloklar birebir ayni", not diff, f"ayrisan: {sorted(diff)}")

    print("\n3) Kontrast denetimi")
    audit_theme("aydinlik", light)
    audit_theme("koyu", dark_stamp)

    print("\n4) Plotly paleti tokens.css ile ayni mi")
    from dashboard.theme import PLOT

    mapping = {
        "text": "--rlt-text", "muted": "--rlt-muted", "blue": "--rlt-primary",
        "green": "--rlt-profit", "red": "--rlt-loss", "yellow": "--rlt-warn",
        "purple": "--rlt-accent", "cyan": "--rlt-info", "orange": "--rlt-orange",
        "gold": "--rlt-gold", "grid": "--rlt-border", "line": "--rlt-border-strong",
        "bg": "--rlt-surface", "hover_bg": "--rlt-surface-2",
    }
    for theme_name, tokens in (("light", light), ("dark", dark_stamp)):
        drift = {
            key: (PLOT[theme_name][key], tokens[token])
            for key, token in mapping.items()
            if PLOT[theme_name].get(key, "").lower() != tokens.get(token, "").lower()
        }
        check(f"PLOT['{theme_name}'] tokens.css ile ayni", not drift, f"ayrisan: {drift}")

    print("\n5) Sayfa kodunda kacak hex")
    import glob

    leaks: list[str] = []
    scan = (
        glob.glob(os.path.join(ROOT, "dashboard", "pages", "*.py"))
        + glob.glob(os.path.join(ROOT, "dashboard", "components", "*.py"))
        + [os.path.join(ROOT, "dashboard", "app.py")]
    )
    # account.py muaf: iki temanin onizleme seridini ayni anda gostermek zorunda,
    # o yuzden kasitli olarak tokenlardan bagimsiz hex tasir (bkz. _theme_preview).
    exempt = {"account.py"}
    for path in scan:
        if os.path.basename(path) in exempt:
            continue
        for i, line in enumerate(open(path, encoding="utf-8"), 1):
            if re.search(r"#[0-9a-fA-F]{6}\b", line):
                leaks.append(f"{os.path.relpath(path, ROOT)}:{i}")
    check("Sayfa/bilesen kodunda hex yok", not leaks, f"kacak: {leaks}")

    print("\n6) Token adi ucuncu parti ile cakisiyor mu")
    # Dash DataTable kendi bundle'inda --muted / --border / --accent tanimliyor
    # ve tablo icinde ayni adli tokenlari golgeliyor. Onek olmadan tablo
    # basligi #c8c8c8 cikiyordu (zemin uzerinde 1.35:1) — ne style_header ne
    # de custom.css kuraliyla duzelen bir sey; ad cakismasi.
    RESERVED = {
        "--accent", "--border", "--muted", "--hover", "--text-color",
        "--faded-text", "--faded-text-header", "--faded-dropdown",
        "--selected-background", "--background-color-ellipses",
    }
    all_tokens = set(light) | set(dark_stamp)
    collisions = sorted(all_tokens & RESERVED)
    check("Token adlari DataTable ile cakismiyor", not collisions,
          f"cakisan: {collisions}")
    unprefixed = sorted(t for t in all_tokens if not t.startswith("--rlt-"))
    check("Tum tokenlar --rlt- onekli", not unprefixed, f"oneksiz: {unprefixed}")

    print("\n7) theme sembolleri import edilmis mi")
    # Faz C'de olu importlar budanirken `f"1px solid {BORDER}"` gibi f-string
    # icindeki kullanimlar bir kez gozden kacti ve callback calisinca
    # NameError verdi. layout() render testi bunu YAKALAMAZ — hata callback
    # govdesinde. Bu denetim ucuz ve o sinifi kapatiyor.
    theme_tree = ast.parse(open(os.path.join(ROOT, "dashboard", "theme.py"),
                                encoding="utf-8").read())
    exported = {n.targets[0].id for n in theme_tree.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    exported |= {n.name for n in theme_tree.body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

    undefined: list[str] = []
    for path in scan + glob.glob(os.path.join(ROOT, "dashboard", "components", "*.py")):
        tree = ast.parse(open(path, encoding="utf-8").read())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported |= {a.asname or a.name for a in node.names}
            elif isinstance(node, ast.Import):
                imported |= {(a.asname or a.name).split(".")[0] for a in node.names}
        defined = {n.name for n in tree.body
                   if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                defined |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        used = {n.id for n in ast.walk(tree)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        for name in sorted((used & exported) - imported - defined):
            undefined.append(f"{os.path.relpath(path, ROOT)}:{name}")

    check("theme sembolleri her yerde import edilmis", not undefined,
          f"eksik: {undefined}")

    print("\n8) Kullanilan dcc bilesenleri temalanmis mi")
    custom_css = open(os.path.join(ROOT, "dashboard", "assets", "custom.css"),
                      encoding="utf-8").read()
    # Dash surumleri bilesenlerin DOM'unu degistiriyor: dcc.Dropdown artik
    # react-select degil `button.dash-dropdown`, dcc.Slider `dash-slider-*`.
    # Eski secicilerimiz bunlara uymuyordu — acilir kutu koyu temada beyaz
    # kaliyor, slider isaretleri gorunmez oluyordu. Sayfalarda kullanilan her
    # bilesen ailesinin custom.css'te karsiligi olmali.
    FAMILIES = {
        "Dropdown": "dash-dropdown",
        "Slider": "dash-slider",
        "RangeSlider": "dash-slider",
        "Checklist": "dash-options",
        "RadioItems": "dash-options",
        "Loading": "dash-spinner",
        "DatePickerSingle": "dash-datepicker",
        "DatePickerRange": "dash-datepicker",
    }
    used_components: set[str] = set()
    for path in scan:
        used_components.update(
            re.findall(r"dcc\.([A-Za-z]+)", open(path, encoding="utf-8").read())
        )
    for component in sorted(used_components & set(FAMILIES)):
        family = FAMILIES[component]
        check(f"dcc.{component} -> .{family}-* temalanmis", family in custom_css,
              "Dash varsayilani sizacak")

    print("\n9) Bootstrap renk varyantlari ezilmis mi")
    # Taban artik dbc.themes.BOOTSTRAP. Ezilmeyen her varyant Bootstrap'in
    # kendi rengini kullanir — .btn-outline-warning'in #ffc107'si beyaz zeminde
    # ~1.6:1 kaliyordu. Sayfalarda kullanilan her varyant burada tanimli olmali.
    used: set[str] = set()
    for path in scan:
        text = open(path, encoding="utf-8").read()
        used.update(re.findall(r'color="([a-z]+)"', text))
    variants = used & {"primary", "secondary", "success", "danger", "warning", "info"}

    for variant in sorted(variants):
        for selector in (f".btn-{variant}", f".btn-outline-{variant}",
                         f".badge.bg-{variant}", f".alert-{variant}"):
            check(f"{selector} tanimli", selector in custom_css,
                  "Bootstrap varsayilani sizacak")

    print("\n10) Kendi CSS siniflarimiz tanimli mi")
    # Baglanmayan bir sinif hata VERMEZ, bileseni sessizce bicimsiz birakir.
    # Yalnizca tasarim sistemimizin kendi onekleri taranir; Bootstrap yardimci
    # siniflari (me-2, d-flex, ...) kapsam disi.
    OWN_PREFIXES = ("sidebar-", "account-", "card-title", "state-", "metric-",
                    "theme-", "section-", "page-", "filter-", "topbar")
    # Stil TASIMAYAN, kasitli istisnalar:
    #   theme-label  — theme-toggle.js'in metnini gunceldigi kanca
    #   sidebar-link — isaret; bicimi `#sidebar .nav-link` kuralindan geliyor
    NON_STYLE = {"theme-label", "sidebar-link"}

    own_classes: set[str] = set()
    for path in scan:
        text = open(path, encoding="utf-8").read()
        for value in re.findall(r'className="([^"]+)"', text):
            own_classes.update(
                token for token in value.split() if token.startswith(OWN_PREFIXES)
            )
    undefined_css = sorted(
        name for name in own_classes - NON_STYLE if f".{name}" not in custom_css
    )
    check("Kendi siniflarimiz custom.css'te tanimli", not undefined_css,
          f"tanimsiz: {undefined_css}")

    print("\n" + "=" * 60)
    print(f"  Gecen: {len(PASSED)}   Kalan: {len(FAILED)}")
    if FAILED:
        for name in FAILED:
            print(f"   - {name}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
