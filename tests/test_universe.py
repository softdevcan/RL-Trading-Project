"""Faz 9 E — sembol evreni (E0, Y11/K7) ve sektor siniflamasi (E2), tarihsel uyelik (E1).

    python tests/test_universe.py

E0: evrenin tek kaynagi `data/universe/bist30_panel.json`; kod ve pano listeleri ondan okur ve RL
panelinin sembolleriyle (gorunme SIRASIYLA) ayni. Ag yok.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

FAILED: list[str] = []
PASSED: list[str] = []
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}]   {name} {'' if ok else detail}")
    (PASSED if ok else FAILED).append(name)


def test_e0():
    print("\nE0 — evren tek kaynak (Y11, K7)")
    with open(os.path.join(ROOT, 'data', 'universe', 'bist30_panel.json'), encoding='utf-8') as f:
        uni = json.load(f)
    syms = uni['symbols']
    check("evren dosyasi: 30 benzersiz .IS sembol, panel sha256 ve tarih kayitli",
          len(syms) == 30 == uni['n_symbols'] and len(set(syms)) == 30
          and all(s.endswith('.IS') for s in syms) and len(uni['panel']['sha256']) == 64
          and uni['frozen_at'] and uni['panel']['last_date'], str(uni.get('panel')))

    panel = os.path.join(ROOT, uni['panel']['file'])
    if os.path.exists(panel):
        order = list(dict.fromkeys(pd.read_csv(panel, usecols=['symbol'])['symbol']))
        check("RL paneli ile ayni semboller, ayni SIRA (env durum vektoru sirasi)", order == syms,
              f"(fark {sorted(set(order) ^ set(syms))})")
    else:
        print("  [ATLA]  panel CSV yok (temiz klon); sira kontrolu atlandi")

    from data.bist30_symbols import BIST30_SYMBOLS, STOCK_INFO, PHASE1_SYMBOLS, get_symbols
    check("BIST30_SYMBOLS = evren dosyasi; get_symbols(phase=2) ayni (egitim rotasi varsayilani)",
          BIST30_SYMBOLS == syms and get_symbols(phase=2) == syms)
    check("STOCK_INFO her sembolun adini tasiyor, fazlasi yok; PHASE1 evrenin alt kumesi",
          set(STOCK_INFO) == set(syms) and all(v.get('name') for v in STOCK_INFO.values())
          and set(PHASE1_SYMBOLS) <= set(syms))
    import dashboard.pages.data as pdata
    import dashboard.pages.daily_trading as pdaily
    check("pano listeleri (veri, gunluk karar) evren dosyasindan, ayni sira",
          [s for s, _ in pdata.BIST30_SYMBOLS] == syms and list(pdaily.BIST30_SYMBOLS) == syms
          and pdata.ALL_BIST_VALUES == syms)
    old = {'ALARK.IS', 'HEKTS.IS', 'KOZA.IS', 'KOZAA.IS', 'KOZAL.IS', 'TTKOM.IS'}
    check("eski listelerin panelde olmayan sembolleri (Y11) kalmadi", not (old & set(syms)))


def main() -> int:
    test_e0()
    print("\n" + "=" * 60)
    print(f"Sonuc: {len(PASSED)} gecti, {len(FAILED)} kaldi")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
