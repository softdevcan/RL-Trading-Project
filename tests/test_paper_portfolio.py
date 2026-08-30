"""
Kagit portfoy — kar/zarar takibi

Kusur: gunluk karar aliniyor ama "gun sonunda elimizde ne kaldi, kar ettik mi"
sorusunun cevabi hicbir yerde olusmuyordu. Dongunun uc halkasi da kopuktu:

  1. Portfoyun hafizasi yoktu. Pano bakiyeyi 100.000 varsayilaniyla ve elle
     doldurulan 5 satirlik formdan kuruyordu; `GET /portfolio/latest` ucu
     duruyordu ama hic cagrilmiyordu. Dunku pozisyonlar bugunku karara
     girmiyordu.
  2. `portfolio_history.csv` yalnizca "Uygula" basilinca yaziliyordu ve hic
     olusmamisti — portfoy grafigi bos kaliyordu.
  3. Gosterilen "gunluk getiri" kar/zarar DEGILDI. `portfolio_before` ile
     `portfolio_after` ayni gunun ayni fiyatlariyla hesaplaniyor; alim-satim
     nakit<->hisse takasi oldugu icin portfoy degeri degismez, geriye yalnizca
     komisyon kalir. Kayitli gercek karar: 100.000,00 -> 99.998,99, "-0,0010%",
     komisyon 1,01 TL. Metrik tanim geregi hep ~0 veya negatif cikar.

Bu testin cekirdegi [7]: kar/zarar ancak pozisyon BASKA BIR GUNUN fiyatiyla
degerlenince olusur.

Sunucu gerektirmez: calisma alani gecici bir dizine yonlendirilir.

    python tests/test_paper_portfolio.py
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.auth import workspace as ws
from app.services import portfolio as pf

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


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) < tol


COMMISSION = 0.001


def _buy(symbol, shares, price):
    """`interpret_actions_with_risk` ile ayni tanim: cost komisyon DAHIL."""
    cost = shares * price * (1 + COMMISSION)
    return {
        "symbol": symbol, "action": "BUY", "shares": shares, "price": price,
        "cost": cost, "revenue": 0.0, "commission": shares * price * COMMISSION,
        "raw_signal": 0.5, "reason": "test", "executed": True,
    }


def _sell(symbol, shares, price):
    """SELL -> revenue komisyon DUSULMUS."""
    revenue = shares * price * (1 - COMMISSION)
    return {
        "symbol": symbol, "action": "SELL", "shares": shares, "price": price,
        "cost": 0.0, "revenue": revenue, "commission": shares * price * COMMISSION,
        "raw_signal": -0.5, "reason": "test", "executed": True,
    }


def _hold(symbol, price):
    return {
        "symbol": symbol, "action": "HOLD", "shares": 0, "price": price,
        "cost": 0.0, "revenue": 0.0, "commission": 0.0,
        "raw_signal": 0.01, "reason": "zayif sinyal", "executed": False,
    }


def run():
    print("=" * 62)
    print("Kagit portfoy — kar/zarar takibi")
    print("=" * 62)

    tmp = tempfile.mkdtemp(prefix="rlt_portfolio_")
    orig_dir = ws.live_trading_dir
    ws.live_trading_dir = lambda: tmp
    pf.ws.live_trading_dir = lambda: tmp

    try:
        print("\n[1] Baslangic durumu")
        p = pf.load_portfolio(100_000)
        check("dosya yokken baslangic sermayesi", close(p["cash"], 100_000))
        check("pozisyon yok", p["positions"] == {})
        check("salt-okuma diske yazmadi",
              not os.path.exists(os.path.join(tmp, "portfolio.json")))

        print("\n[2] BUY: nakit duser, ortalama maliyet komisyon DAHIL")
        p, s = pf.apply_decisions(p, [_buy("GARAN.IS", 10, 100.0),
                                      _hold("THYAO.IS", 300.0)], "2026-08-28")
        # cost = 10*100*1.001 = 1001.0
        check("nakit dogru", close(p["cash"], 100_000 - 1001.0), str(p["cash"]))
        check("pozisyon acildi", p["positions"]["GARAN.IS"]["shares"] == 10)
        check("avg_cost komisyon dahil",
              close(p["positions"]["GARAN.IS"]["avg_cost"], 100.1),
              str(p["positions"]["GARAN.IS"]["avg_cost"]))
        check("HOLD islem sayilmadi", s["executed_trades"] == 1, str(s))
        check("komisyon birikti", close(p["total_commission"], 1.0))

        print("\n[3] Ek BUY: agirlikli ortalama maliyet")
        p, _ = pf.apply_decisions(p, [_buy("GARAN.IS", 10, 120.0)], "2026-08-29")
        # (10*100.1 + 10*120*1.001) / 20 = (1001 + 1201.2)/20 = 110.11
        check("agirlikli ortalama", close(p["positions"]["GARAN.IS"]["avg_cost"], 110.11),
              str(p["positions"]["GARAN.IS"]["avg_cost"]))
        check("adet toplandi", p["positions"]["GARAN.IS"]["shares"] == 20)

        print("\n[4] Kismi SELL: gerceklesmis kar, kalan maliyet korunur")
        before_cash = p["cash"]
        p, s = pf.apply_decisions(p, [_sell("GARAN.IS", 5, 150.0)], "2026-08-30")
        revenue = 5 * 150.0 * (1 - COMMISSION)   # 749.25
        expected_realized = revenue - 110.11 * 5  # 749.25 - 550.55 = 198.70
        check("nakit arttı", close(p["cash"], before_cash + revenue), str(p["cash"]))
        check("gerceklesmis kar dogru", close(p["realized_pnl"], expected_realized),
              f"{p['realized_pnl']} != {expected_realized}")
        check("kalan adet", p["positions"]["GARAN.IS"]["shares"] == 15)
        check("kalan maliyet degismedi",
              close(p["positions"]["GARAN.IS"]["avg_cost"], 110.11))

        print("\n[5] Tam SELL: pozisyon kapanir")
        p2, _ = pf.apply_decisions(p, [_sell("GARAN.IS", 15, 150.0)], "2026-08-31")
        check("pozisyon silindi", "GARAN.IS" not in p2["positions"],
              str(p2["positions"]))

        print("\n[6] Ayni tarih ikinci kez uygulanmaz")
        p3, s3 = pf.apply_decisions(p, [_buy("THYAO.IS", 5, 300.0)], "2026-08-30")
        check("already_applied bayragi", s3["already_applied"] is True)
        check("portfoy degismedi", p3["cash"] == p["cash"])
        check("pozisyon eklenmedi", "THYAO.IS" not in p3["positions"])

        print("\n[7] KAR/ZARAR ancak BASKA GUNUN fiyatiyla olusur")
        # Kusurun tam kalbi: alis fiyatiyla degerlersen kar sadece -komisyondur.
        fresh = pf.default_portfolio(100_000)
        fresh, _ = pf.apply_decisions(fresh, [_buy("GARAN.IS", 10, 100.0)],
                                      "2026-08-28")
        same_day = pf.value_portfolio(fresh, {"GARAN.IS": 100.0})
        check("ayni gun fiyatiyla toplam deger ~ baslangic - komisyon",
              close(same_day["total_value"], 100_000 - 1.0, tol=1e-6),
              str(same_day["total_value"]))
        check("ayni gun kar/zarar = -komisyon",
              close(same_day["total_pnl"], -1.0), str(same_day["total_pnl"]))

        next_day = pf.value_portfolio(fresh, {"GARAN.IS": 110.0})
        # pozisyon degeri 1100, nakit 98999 -> toplam 100099, P&L +99
        check("ertesi gun fiyati kar uretiyor", next_day["total_pnl"] > 0,
              str(next_day["total_pnl"]))
        check("kar dogru hesaplandi", close(next_day["total_pnl"], 99.0),
              str(next_day["total_pnl"]))
        check("gerceklesmemis kar ayri raporlaniyor",
              close(next_day["unrealized_pnl"], 1100 - 1001.0),
              str(next_day["unrealized_pnl"]))
        check("yuzde dogru", close(next_day["total_pnl_pct"], 0.099),
              str(next_day["total_pnl_pct"]))

        down = pf.value_portfolio(fresh, {"GARAN.IS": 90.0})
        check("dusen fiyat zarar uretiyor", down["total_pnl"] < 0,
              str(down["total_pnl"]))

        print("\n[8] Fiyati bilinmeyen sembol maliyetle degerlenir ve raporlanir")
        blind = pf.value_portfolio(fresh, {})
        check("missing_prices bildiriyor", blind["missing_prices"] == ["GARAN.IS"],
              str(blind["missing_prices"]))
        check("maliyetle degerlendi (sahte kar yok)",
              close(blind["unrealized_pnl"], 0.0), str(blind["unrealized_pnl"]))
        check("pozisyon price_available=False",
              blind["positions"][0]["price_available"] is False)

        print("\n[9] Nakit yetmezse islem atlanir, portfoy eksiye dusmez")
        poor = pf.default_portfolio(500)
        poor, s9 = pf.apply_decisions(poor, [_buy("TUPRS.IS", 10, 400.0)], "2026-09-01")
        check("nakit eksiye dusmedi", poor["cash"] >= 0, str(poor["cash"]))
        check("islem atlandi", s9["executed_trades"] == 0, str(s9))
        check("atlanan islem raporlandi", len(s9["skipped"]) == 1, str(s9["skipped"]))

        print("\n[10] Elde olmayan hisse satilamaz")
        empty = pf.default_portfolio(10_000)
        empty, s10 = pf.apply_decisions(empty, [_sell("SASA.IS", 5, 10.0)],
                                        "2026-09-02")
        check("satis atlandi", s10["executed_trades"] == 0)
        check("nakit degismedi", close(empty["cash"], 10_000))
        check("gerceklesmis kar uretilmedi", close(empty["realized_pnl"], 0.0))

        print("\n[11] Pozisyondan fazla satis oranlanir")
        part = pf.default_portfolio(100_000)
        part, _ = pf.apply_decisions(part, [_buy("SISE.IS", 4, 50.0)], "2026-09-03")
        cash_before = part["cash"]
        part, _ = pf.apply_decisions(part, [_sell("SISE.IS", 10, 60.0)], "2026-09-04")
        # yalnizca 4 adet satilabilir: revenue 4*60*0.999 = 239.76
        check("pozisyon kapandi", "SISE.IS" not in part["positions"])
        check("gelir fiilen satilan adede oranlandi",
              close(part["cash"], cash_before + 4 * 60 * (1 - COMMISSION)),
              str(part["cash"]))

        print("\n[12] Disk turu: kaydet/oku ve shares_map")
        pf.save_portfolio(p)
        again = pf.load_portfolio()
        check("nakit korundu", close(again["cash"], p["cash"]))
        check("pozisyon korundu",
              again["positions"]["GARAN.IS"]["shares"] == 15)
        check("shares_map karar ucunun bicimi",
              pf.shares_map(again) == {"GARAN.IS": 15}, str(pf.shares_map(again)))
        check("applied_dates korundu",
              again["last_applied_date"] == "2026-08-30",
              str(again.get("last_applied_date")))

        print("\n[13] Sifirlama")
        reset = pf.reset_portfolio(50_000)
        check("baslangic sermayesi uygulandi", close(reset["cash"], 50_000))
        check("pozisyonlar temizlendi", reset["positions"] == {})
        check("gerceklesmis kar sifirlandi", close(reset["realized_pnl"], 0.0))

    finally:
        ws.live_trading_dir = orig_dir
        pf.ws.live_trading_dir = orig_dir
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 62)
    print(f"Toplam: {PASSED + FAILED} | Gecti: {PASSED} | Kaldi: {FAILED}")
    print("=" * 62)
    return FAILED == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
