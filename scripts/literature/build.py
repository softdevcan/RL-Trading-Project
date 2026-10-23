"""Seçilmiş kaynakçayı yeniden üretir.

Girdi  : scripts/literature/data/ (resolved.json, s2.json, head.md, foot.md) + meta.py + coverage.py
Çıktı  : docs/thesis/literature-table.md (tamamı) ve docs/thesis/vision-and-scope.md §9 (kaynakça listesi)

Kullanım:
    python scripts/literature/build.py

Yeni kaynak eklemek:
    1. data/seed.tsv'ye satır ekle (key, grup, doi:/arxiv:/title:, not)
    2. python scripts/literature/resolve.py data/seed.tsv data/resolved.json   (OpenAlex; ağ gerekir)
       python scripts/literature/s2_batch.py data/resolved.json data/s2.json    (Semantic Scholar atıfları)
    3. meta.py: ROLE (zorunlu), gerekirse VENUE / INDEX / YEAR / PDF
    4. coverage.py: kaynağı ilgili bileşene bağla
    5. python scripts/literature/build.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ROOT = HERE.parents[1]
TABLE = ROOT / "docs" / "thesis" / "literature-table.md"
VISION = ROOT / "docs" / "thesis" / "vision-and-scope.md"


def main() -> None:
    body, bib = DATA / "final_body.md", DATA / "bib.md"
    subprocess.run([sys.executable, "-I", "-X", "utf8", str(HERE / "gen_final.py"),
                    str(HERE), str(DATA), str(body), str(bib)], check=True)

    TABLE.write_text("".join(p.read_text(encoding="utf-8") for p in (DATA / "head.md", body, DATA / "foot.md")),
                     encoding="utf-8")

    # vision §9: giriş paragrafı korunur, ilk alt başlıktan dosya sonuna kadar liste yenilenir
    v = VISION.read_text(encoding="utf-8")
    s9 = v.index("## 9. Kaynakça")
    first_group = v.index("\n### ", s9)
    VISION.write_text(v[:first_group + 1] + bib.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")
    print(f"yazıldı: {TABLE.relative_to(ROOT)}, {VISION.relative_to(ROOT)} §9")


if __name__ == "__main__":
    main()
