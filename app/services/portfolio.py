"""
Kagit portfoy — gunluk kararin sonucunu takip eden tek gercek kaynak.

Neden ayri bir modul: `daily_trading.py` kararin NASIL uretildigini biliyor,
burasi kararin SONUCUNU biliyor. Ikisi ayri omurlere sahip — karar her
istekte yeniden uretilir, portfoy gunler boyunca birikir.

Kapatilan dongu:

    karar al  ->  (Uygula)  ->  portfoy ilerler  ->  ertesi gunun karari
                                                      portfoyden beslenir

Onceki durumda halkalarin ucu de kopuktu:
  * pano bakiyeyi 100.000 varsayilaniyla ve elle doldurulan 5 satirlik
    formdan kuruyordu, `GET /portfolio/latest` hic cagrilmiyordu;
  * `portfolio_history.csv` yalnizca "Uygula" basilinca yaziliyordu ve hic
    olusmamisti;
  * "gunluk getiri" olarak gosterilen sayi kar/zarar DEGILDI. `portfolio_before`
    ile `portfolio_after` ayni gunun ayni fiyatlariyla hesaplaniyor; alim-satim
    nakit<->hisse takasi oldugu icin portfoy degeri degismez, geriye yalnizca
    komisyon kalir. Metrik tanim geregi hep ~0 veya negatif cikiyordu
    (olculdu: 100.000,00 -> 99.998,99, "-0,0010%", komisyon 1,01 TL).

Kar/zarar ancak POZISYON BASKA BIR GUNUN FIYATIYLA degerlenince olusur —
`value_portfolio()` bunu yapar.

Nakit akisi `daily_trading.interpret_actions_with_risk()` ile birebir ayni
tanimi kullanir:
    BUY  -> cost    = adet * fiyat * (1 + komisyon)   [komisyon DAHIL]
    SELL -> revenue = adet * fiyat * (1 - komisyon)   [komisyon DUSULMUS]
Ortalama maliyet de bu yuzden komisyon dahil tutulur: pozisyonun gercek
maliyeti odenen nakittir, aksi halde gerceklesmis kar komisyon kadar sisik
gorunur.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from filelock import FileLock

from app.auth import workspace as ws

logger = logging.getLogger(__name__)

PORTFOLIO_FILE = 'portfolio.json'
DEFAULT_INITIAL_CAPITAL = 100_000.0


def portfolio_path() -> str:
    return os.path.join(ws.live_trading_dir(), PORTFOLIO_FILE)


def default_portfolio(initial_capital: float = DEFAULT_INITIAL_CAPITAL) -> dict:
    now = datetime.now().isoformat()
    return {
        "initial_capital": float(initial_capital),
        "cash": float(initial_capital),
        # {sembol: {"shares": int, "avg_cost": float}} — avg_cost komisyon dahil
        "positions": {},
        "realized_pnl": 0.0,
        "total_commission": 0.0,
        "applied_dates": [],
        "last_applied_date": None,
        "created_at": now,
        "updated_at": now,
    }


def _normalize(portfolio: dict) -> dict:
    """Eksik alanlari tamamla — eski dosyalar ve elle duzenlemeler icin."""
    base = default_portfolio()
    out = {**base, **(portfolio or {})}
    positions = {}
    for sym, pos in (out.get("positions") or {}).items():
        if isinstance(pos, dict):
            shares = int(pos.get("shares", 0) or 0)
            avg = float(pos.get("avg_cost", 0.0) or 0.0)
        else:  # {sembol: adet} kisayolu
            shares = int(pos or 0)
            avg = 0.0
        if shares:
            positions[sym] = {"shares": shares, "avg_cost": avg}
    out["positions"] = positions
    return out


def load_portfolio(initial_capital: float = DEFAULT_INITIAL_CAPITAL) -> dict:
    """Portfoyu oku; yoksa baslangic sermayesiyle yeni bir tane uret.

    Dosya yoksa DISKE YAZMAZ: salt-okuma bir istegin (viewer'in portfoy
    gorunumu) kullaniciya sessizce dosya olusturmasi istenmez.
    """
    path = portfolio_path()
    if not os.path.exists(path):
        return default_portfolio(initial_capital)
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return _normalize(json.load(fh))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Portfoy okunamadi ({exc}); baslangic durumu kullanildi")
        return default_portfolio(initial_capital)


def save_portfolio(portfolio: dict) -> str:
    path = portfolio_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    portfolio = dict(portfolio)
    portfolio["updated_at"] = datetime.now().isoformat()
    with FileLock(path + '.lock'):
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(portfolio, fh, ensure_ascii=False, indent=2)
    return path


def reset_portfolio(initial_capital: float = DEFAULT_INITIAL_CAPITAL) -> dict:
    """Portfoyu sifirla. Gecmis dosyalarina DOKUNMAZ — bkz. rota."""
    fresh = default_portfolio(initial_capital)
    save_portfolio(fresh)
    logger.info(f"Portfoy sifirlandi: baslangic sermayesi {initial_capital:,.2f}")
    return fresh


def shares_map(portfolio: dict) -> Dict[str, int]:
    """Karar ucunun bekledigi {sembol: adet} bicimi."""
    return {sym: int(pos["shares"]) for sym, pos in portfolio["positions"].items()
            if pos.get("shares")}


def apply_decisions(
    portfolio: dict,
    decisions: List[dict],
    date: str,
) -> Tuple[dict, dict]:
    """Kararlari portfoye isle; yeni portfoy ve uygulama ozetini dondur.

    Ayni tarih ikinci kez uygulanmaz: "Uygula"ya iki kez basmak pozisyonu
    iki katina cikarirdi. `applied_dates` bekcisi bunu keser ve cagirana
    `already_applied` bayragiyla soyler.
    """
    if date in (portfolio.get("applied_dates") or []):
        return portfolio, {
            "already_applied": True,
            "date": date,
            "executed_trades": 0,
            "realized_pnl": 0.0,
            "commission": 0.0,
        }

    out = _normalize(portfolio)
    positions = out["positions"]
    cash = float(out["cash"])
    realized = 0.0
    commission = 0.0
    executed = 0
    skipped: List[str] = []

    for d in decisions:
        if not d.get("executed"):
            continue
        symbol = d["symbol"]
        action = d.get("action")
        qty = int(d.get("shares", 0) or 0)
        if qty <= 0:
            continue

        if action == "BUY":
            cost = float(d.get("cost", 0.0) or 0.0)
            if cost > cash + 1e-9:
                # Karar baska bir bakiyeyle uretilmis olabilir (pano elle
                # bakiye girmeye izin veriyor). Sessizce eksiye dusurmek
                # yerine islemi atla ve cagirana bildir.
                skipped.append(f"{symbol}: nakit yetmedi ({cost:,.2f} > {cash:,.2f})")
                continue
            prev = positions.get(symbol, {"shares": 0, "avg_cost": 0.0})
            total_shares = prev["shares"] + qty
            total_cost = prev["shares"] * prev["avg_cost"] + cost
            positions[symbol] = {
                "shares": total_shares,
                "avg_cost": total_cost / total_shares if total_shares else 0.0,
            }
            cash -= cost
            commission += float(d.get("commission", 0.0) or 0.0)
            executed += 1

        elif action == "SELL":
            prev = positions.get(symbol)
            if not prev or prev["shares"] <= 0:
                skipped.append(f"{symbol}: satilacak pozisyon yok")
                continue
            qty = min(qty, prev["shares"])
            revenue = float(d.get("revenue", 0.0) or 0.0)
            if d.get("shares") and int(d["shares"]) != qty:
                # Pozisyonun tamamini asan satis: geliri fiilen satilan adede
                # oranla. Aksi halde elde olmayan hisseden nakit yaratilirdi.
                revenue = revenue * qty / float(d["shares"])
            realized += revenue - prev["avg_cost"] * qty
            cash += revenue
            commission += float(d.get("commission", 0.0) or 0.0)
            remaining = prev["shares"] - qty
            if remaining > 0:
                positions[symbol] = {"shares": remaining, "avg_cost": prev["avg_cost"]}
            else:
                positions.pop(symbol, None)
            executed += 1

    out["cash"] = cash
    out["positions"] = positions
    out["realized_pnl"] = float(out["realized_pnl"]) + realized
    out["total_commission"] = float(out["total_commission"]) + commission
    out["applied_dates"] = sorted(set((out.get("applied_dates") or []) + [date]))
    out["last_applied_date"] = max(out["applied_dates"])

    summary = {
        "already_applied": False,
        "date": date,
        "executed_trades": executed,
        "realized_pnl": realized,
        "commission": commission,
        "skipped": skipped,
    }
    if skipped:
        logger.warning(f"{date}: {len(skipped)} islem uygulanamadi — {skipped}")
    return out, summary


def value_portfolio(portfolio: dict, prices: Dict[str, float]) -> dict:
    """Portfoyu VERILEN fiyatlarla degerle (mark-to-market).

    Kar/zarar ancak burada olusur: pozisyon alindigi gunun degil, bugunun
    fiyatiyla olculur. Fiyati bilinmeyen sembol ortalama maliyetiyle
    degerlenir (0 saymak portfoyu oldugundan kucuk gosterirdi) ve
    `missing_prices` ile raporlanir — sessiz kalmasi kar/zarari sahte
    gosterir.
    """
    p = _normalize(portfolio)
    positions_out = []
    position_value = 0.0
    unrealized = 0.0
    missing: List[str] = []

    for symbol, pos in sorted(p["positions"].items()):
        shares = int(pos["shares"])
        avg = float(pos["avg_cost"])
        price = float(prices.get(symbol) or 0.0)
        priced = price > 0
        if not priced:
            price = avg
            missing.append(symbol)
        value = shares * price
        cost_basis = shares * avg
        pnl = value - cost_basis
        position_value += value
        unrealized += pnl
        positions_out.append({
            "symbol": symbol,
            "shares": shares,
            "avg_cost": avg,
            "price": price,
            "value": value,
            "cost_basis": cost_basis,
            "unrealized_pnl": pnl,
            "unrealized_pnl_pct": (pnl / cost_basis * 100) if cost_basis else 0.0,
            "price_available": priced,
        })

    cash = float(p["cash"])
    total = cash + position_value
    initial = float(p["initial_capital"]) or 1.0
    total_pnl = total - float(p["initial_capital"])

    return {
        "initial_capital": float(p["initial_capital"]),
        "cash": cash,
        "position_value": position_value,
        "total_value": total,
        "realized_pnl": float(p["realized_pnl"]),
        "unrealized_pnl": unrealized,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl / initial * 100,
        "total_commission": float(p["total_commission"]),
        "positions": positions_out,
        "missing_prices": missing,
        "last_applied_date": p.get("last_applied_date"),
        "applied_days": len(p.get("applied_dates") or []),
    }
