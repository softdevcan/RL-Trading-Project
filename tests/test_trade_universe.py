"""
Gunluk karar — sembol evreni cozumlemesi

Kusur: `/trading/daily-decision` evreni MODEL ADINDAN turetiyordu
("phase1" -> PHASE1_SYMBOLS, 5 sembol). Oysa egitim rotasi `get_symbols(phase)`
listesini yalnizca veri cekerken kullanir; egitimi yuklenen panelin tamamiyla
yapar. Boylece "ppo_phase1_..." adli model 30 sembolle egitiliyor (obs=331),
karar ucu 56 uzunlugunda durum uretiyor ve SB3 patliyordu:

    ValueError: Unexpected observation shape (56,) ... please use (331,)

Sunucu ve gercek model gerektirmez: sahte gozlem/eylem uzaylari kullanilir.

    python tests/test_trade_universe.py
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import daily_trading as dt

PASSED = 0
FAILED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [OK]   {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name}{(' — ' + detail) if detail else ''}")


class _Space:
    def __init__(self, n):
        self.shape = (n,)


class _FakeModel:
    """Sadece uzaylari tasiyan sahte SB3 modeli."""

    def __init__(self, n_symbols, phase=1):
        self.action_space = _Space(n_symbols)
        obs = 1 + n_symbols + 10 * n_symbols
        self.observation_space = _Space(obs)


PANEL_30 = [
    'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'EKGYO.IS', 'ENKAI.IS',
    'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'HALKB.IS', 'ISCTR.IS', 'KCHOL.IS',
    'KRDMD.IS', 'MGROS.IS', 'ODAS.IS', 'PETKM.IS', 'PGSUS.IS', 'SAHOL.IS',
    'SASA.IS', 'SISE.IS', 'SKBNK.IS', 'TAVHL.IS', 'TCELL.IS', 'THYAO.IS',
    'TKFEN.IS', 'TOASO.IS', 'TSKB.IS', 'TUPRS.IS', 'VAKBN.IS', 'YKBNK.IS',
]


def run():
    print("=" * 62)
    print("Gunluk karar — sembol evreni cozumlemesi")
    print("=" * 62)

    tmp = tempfile.mkdtemp(prefix="rlt_universe_")
    orig_panel = dt._panel_train_symbols
    try:
        model_path = os.path.join(tmp, "ppo_phase1_20260830_121012")

        print("\n[1] Kusur: 30 sembolle egitilmis 'phase1' modeli")
        dt._panel_train_symbols = lambda: list(PANEL_30)
        model = _FakeModel(30)
        syms, src = dt.resolve_trade_universe(model, model_path, os.path.basename(model_path))
        check("ad 'phase1' olsa da 30 sembol cozuldu", len(syms) == 30, str(len(syms)))
        check("panel sirasi korundu (sorted DEGIL)", syms == PANEL_30)
        check("kaynak panel olarak raporlandi", "panel" in src, src)
        check("durum uzunlugu modelin bekledigiyle esit",
              1 + len(syms) + 10 * len(syms) == model.observation_space.shape[0])

        print("\n[2] 5 sembollu gercek phase1 modeli hala calisiyor")
        from data.bist30_symbols import PHASE1_SYMBOLS
        dt._panel_train_symbols = lambda: list(PANEL_30)  # panel 30, model 5
        syms5, src5 = dt.resolve_trade_universe(_FakeModel(5), model_path, "ppo_phase1_x")
        check("5 sembole dusuldu", len(syms5) == 5, str(len(syms5)))
        check("PHASE1_SYMBOLS secildi", syms5 == list(PHASE1_SYMBOLS), str(syms5))
        check("kaynak ad olarak raporlandi", "model adi" in src5, src5)

        print("\n[3] Yan dosya panelden ONCE gelir")
        pinned = list(reversed(PANEL_30))
        dt.write_model_meta(model_path, {"symbols": pinned, "n_symbols": 30})
        check("meta dosyasi olustu", os.path.exists(dt.model_meta_path(model_path)))
        syms_meta, src_meta = dt.resolve_trade_universe(_FakeModel(30), model_path, "ppo_phase1_x")
        check("meta'daki sira kullanildi", syms_meta == pinned)
        check("kaynak tanim dosyasi", "tanim" in src_meta, src_meta)

        print("\n[4] Sayisi tutmayan yan dosya yok sayilir")
        dt.write_model_meta(model_path, {"symbols": PANEL_30[:7], "n_symbols": 7})
        syms_bad, src_bad = dt.resolve_trade_universe(_FakeModel(30), model_path, "ppo_phase1_x")
        check("bozuk meta yerine panel kullanildi", syms_bad == PANEL_30, src_bad)

        print("\n[5] Cozulemeyince ANLASILIR hata")
        os.remove(dt.model_meta_path(model_path))
        dt._panel_train_symbols = lambda: list(PANEL_30)
        try:
            dt.resolve_trade_universe(_FakeModel(17), model_path, "ppo_phase1_x")
            err = None
        except ValueError as exc:
            err = exc
        check("ValueError firlatildi", err is not None)
        check("mesaj sembol sayisini soyluyor", err is not None and "17" in str(err), str(err))
        check("mesaj ne yapilacagini soyluyor",
              err is not None and "yeniden egitin" in str(err), str(err))

        print("\n[6] Panel okunamazsa cokmez, ada duser")
        def _boom():
            raise FileNotFoundError("stock_data_with_indicators.csv")
        dt._panel_train_symbols = _boom
        syms_nf, src_nf = dt.resolve_trade_universe(_FakeModel(5), model_path, "ppo_phase1_x")
        check("panel yokken PHASE1 kullanildi", syms_nf == list(PHASE1_SYMBOLS))
        check("kaynak ad olarak raporlandi", "model adi" in src_nf, src_nf)

        print("\n[7] phase3 modeli altini kaybetmiyor")
        from data.bist30_symbols import PHASE3_SYMBOLS
        syms3, _ = dt.resolve_trade_universe(_FakeModel(6), model_path, "ppo_phase3_x")
        check("6 sembol + GOLD_GRAM_TRY", syms3 == list(PHASE3_SYMBOLS), str(syms3))

        print("\n[8] Bozuk/eksik meta dosyasi cokertmiyor")
        with open(dt.model_meta_path(model_path), "w", encoding="utf-8") as fh:
            fh.write("{ bu json degil")
        check("bozuk meta None doner", dt.read_model_meta(model_path) is None)
        dt._panel_train_symbols = lambda: list(PANEL_30)
        syms_c, _ = dt.resolve_trade_universe(_FakeModel(30), model_path, "ppo_phase1_x")
        check("bozuk metaya ragmen cozuldu", syms_c == PANEL_30)
        check("meta yolu .zip uzantisini yutuyor",
              dt.model_meta_path(model_path + ".zip") == dt.model_meta_path(model_path))

        print("\n[9] Yazilan meta gecerli JSON")
        dt.write_model_meta(model_path, {"symbols": PANEL_30, "n_symbols": 30, "phase": 1})
        with open(dt.model_meta_path(model_path), encoding="utf-8") as fh:
            data = json.load(fh)
        check("symbols geri okunuyor", data["symbols"] == PANEL_30)
        check("n_symbols geri okunuyor", data["n_symbols"] == 30)
    finally:
        dt._panel_train_symbols = orig_panel
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 62)
    print(f"Toplam: {PASSED + FAILED} | Gecti: {PASSED} | Kaldi: {FAILED}")
    print("=" * 62)
    return FAILED == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
