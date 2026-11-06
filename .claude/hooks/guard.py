"""Claude Code proje hook'lari (bkz. .claude/settings.json).

Kullanim:  python .claude/hooks/guard.py pre    (PreToolUse,  Edit|Write)
           python .claude/hooks/guard.py post   (PostToolUse, Edit|Write)

pre  : 1) Uretilen literatur dosyalarini elle duzenlemeyi engeller
          (literature-table.md tamami, vision-and-scope.md §9 kaynakca listesi).
       2) docs/thesis/ altina YENI bir "leakage/sizinti yok" iddiasi eklenirken onay ister
          (dosyada zaten olan arsiv ifadeleri sayilmaz — yalnizca artis).
post : env/trading_env.py degisince RL env bekcilerini calistirir; kalirsa Claude'a geri bildirir.
Hook hatasi araci asla bozmaz: beklenmeyen her durumda sessizce cikar (exit 0).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "docs" / "thesis" / "literature-table.md"
VISION = ROOT / "docs" / "thesis" / "vision-and-scope.md"
THESIS = ROOT / "docs" / "thesis"
ENV_FILE = ROOT / "env" / "trading_env.py"
ENV_TESTS = ["tests/test_env_lookup_equivalence.py", "tests/test_env_price_guards.py"]

CLAIM = re.compile(
    r"(leakage|s[ıi]z[ıi]nt[ıi])[^.\n]{0,25}?\b(yok|yoktur|bulunmuyor|bulunmamaktad[ıi]r|i[çc]ermez|i[çc]ermiyor)\b"
    r"|\bno (data )?leakage\b|\bleak(age)?[- ]free\b",
    re.IGNORECASE,
)


def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False))
    sys.exit(0)


def pre_decision(decision, reason):
    emit({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                 "permissionDecision": decision,
                                 "permissionDecisionReason": reason}})


def resolve(p):
    try:
        q = Path(p)
        return (q if q.is_absolute() else ROOT / q).resolve()
    except Exception:
        return None


def read(p):
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def bib_start(text):
    """vision §9'da build.py'nin yeniden urettigi bolumun baslangici (ilk '### ')."""
    try:
        s9 = text.index("## 9. Kaynak")
        return text.index("\n### ", s9) + 1
    except ValueError:
        return None


def pre(data):
    tool, ti = data.get("tool_name"), data.get("tool_input") or {}
    path = resolve(ti.get("file_path", ""))
    if path is None:
        return
    gen_msg = ("Bu dosya URETILIR: scripts/literature/build.py yazar, elle duzenleme. "
               "Kaynak eklemek/duzeltmek icin /kaynak-ekle skill'ini kullan "
               "(seed.tsv -> resolve -> s2_batch -> meta.py -> coverage.py -> build.py).")

    # 1) Uretilen dosyalar
    if path == TABLE:
        pre_decision("deny", gen_msg)
    if path == VISION:
        cur = read(VISION)
        start = bib_start(cur)
        if start is not None:
            if tool == "Edit":
                old = ti.get("old_string", "")
                pos = cur.find(old) if old else -1
                if pos >= 0 and pos + len(old) > start:
                    pre_decision("deny", "vision-and-scope.md §9 kaynakca listesi " + gen_msg)
            elif tool == "Write":
                new = ti.get("content", "")
                ns = bib_start(new)
                if ns is None or new[ns:].strip() != cur[start:].strip():
                    pre_decision("deny", "vision-and-scope.md §9 kaynakca listesi " + gen_msg)

    # 2) Yeni sizinti-yok iddiasi
    try:
        path.relative_to(THESIS)
    except ValueError:
        return
    if path.suffix.lower() != ".md" or path.name == "CLAUDE.md":
        return
    if tool == "Edit":
        before, after = ti.get("old_string", ""), ti.get("new_string", "")
        if ti.get("replace_all"):
            n = read(path).count(before) if before else 0
            before, after = before * n, after * n
    else:
        before, after = read(path), ti.get("content", "")
    added = len(CLAIM.findall(after)) - len(CLAIM.findall(before))
    if added > 0:
        hit = CLAIM.search(after).group(0)
        pre_decision("ask", (
            f"Tez metnine yeni bir sizinti-yok iddiasi ekleniyor ('{hit}'). Kodda dogrulanmis sizinti "
            "kaynaklari var (makro bfill, TUFE/UFE gecikmesi, statik temel oranlar, seri butunune "
            "ICEEMDAN) — bkz. docs/development/roadmap.md 'Bilinen Metodolojik Aciklar'. "
            "Iddiayi kapsamla sinirla (hangi ozellik/konfigurasyon, hangi kanitla) ya da kaldir. "
            "Alinti/elestiri ise onaylayabilirsin."))


def post(data):
    ti = data.get("tool_input") or {}
    path = resolve(ti.get("file_path", ""))
    if path != ENV_FILE:
        return
    py = ROOT / "venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = ROOT / "venv" / "bin" / "python"
    exe = str(py) if py.exists() else sys.executable
    failed = []
    for t in ENV_TESTS:
        try:
            r = subprocess.run([exe, "-X", "utf8", t], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=240)
        except subprocess.TimeoutExpired:
            failed.append(f"{t}: zaman asimi (240 sn)")
            continue
        if r.returncode != 0:
            tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-25:])
            failed.append(f"{t} (exit {r.returncode}):\n{tail}")
    if failed:
        emit({"decision": "block",
              "reason": "env/trading_env.py degisikligi RL env bekcilerini kirdi "
                        "(state space / fiyat korumalari):\n\n" + "\n\n".join(failed)})
    emit({"hookSpecificOutput": {"hookEventName": "PostToolUse",
                                 "additionalContext": "RL env bekcileri gecti: " + ", ".join(ENV_TESTS)}})


def main():
    try:
        data = json.load(sys.stdin)
        {"pre": pre, "post": post}[sys.argv[1]](data)
    except SystemExit:
        raise
    except Exception:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
