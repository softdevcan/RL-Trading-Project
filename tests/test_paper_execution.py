"""Faz 11 S7: kagit portfoy yurutme zamani (K-F11-5, plan §8 #18) ve canli gozlem uyarilari (K13). Pytest degil.

    python tests/test_paper_execution.py

Ne dogrular (ag yok; piyasa verisi sahte):
  1. reprice_to_open: alim maliyeti / satis geliri / komisyon ertesi acilis fiyatiyla ve kararin kendi komisyon
     oraniyla; karar fiyati `decision_price`'ta kalir; acilisi olmayan sembolde islem yapilmaz; HOLD dokunulmaz.
  2. next_session_opens: karar gununden SONRAKI ilk seans (kapsam esigi altindaki gun atlanir); yoksa None.
  3. observation_warnings: meta yok / price_stats yok / causal fiyat tanimi + wilder ADX uyarilari.
  4. /apply-decision: ertesi seans yokken 409 ve portfoy degismez; varken islem ACILIS fiyatiyla (nakit ve maliyet
     acilisa gore), 'decision_close' eski davranis; gecersiz mod 422; viewer 403.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TMP = tempfile.mkdtemp(prefix="rlt_paper_exec_")
os.environ["AUTH_DB_URL"] = f"sqlite:///{os.path.join(_TMP, 'auth.db').replace(os.sep, '/')}"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-not-for-production"
os.environ["AUTH_ENABLED"] = "True"
os.environ["WORKSPACES_DIR"] = os.path.join(_TMP, "workspaces")
os.environ["BOOTSTRAP_ADMIN_EMAIL"] = ""
os.environ["BOOTSTRAP_ADMIN_PASSWORD"] = ""

import pandas as pd  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from fastapi.testclient import TestClient  # noqa: E402

from app.auth import security  # noqa: E402
from app.auth import workspace as ws  # noqa: E402
from app.auth.db import init_db, reset_engine, session_scope  # noqa: E402
from app.auth.models import Role, User  # noqa: E402
from app.auth.service import create_user  # noqa: E402
from app.services import daily_trading as DT  # noqa: E402
from app.services import portfolio as pf  # noqa: E402

passed = failed = 0


def check(name, ok):
    global passed, failed
    print(f"[{'OK' if ok else 'HATA'}] {name}")
    passed, failed = passed + bool(ok), failed + (not ok)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


# 1 ---------------------------------------------------------------------------------------------------------------
dec = [
    {"symbol": "GARAN.IS", "action": "BUY", "shares": 10, "price": 100.0, "cost": 1001.0, "commission": 1.0,
     "executed": True},
    {"symbol": "THYAO.IS", "action": "SELL", "shares": 4, "price": 300.0, "revenue": 1198.8, "commission": 1.2,
     "executed": True},
    {"symbol": "SASA.IS", "action": "BUY", "shares": 5, "price": 10.0, "cost": 50.05, "commission": 0.05,
     "executed": True},
    {"symbol": "AKBNK.IS", "action": "HOLD", "shares": 0, "price": 50.0, "executed": False},
]
rp, miss = pf.reprice_to_open(dec, {"GARAN.IS": 102.0, "THYAO.IS": 295.0}, "2026-11-24")
g, t, s, h = rp
ok1 = close(g["price"], 102.0) and close(g["cost"], 10 * 102 * 1.001) and close(g["commission"], 1.02)
ok1 &= g["decision_price"] == 100.0 and g["execution"] == "next_open" and g["execution_date"] == "2026-11-24"
ok1 &= close(t["revenue"], 4 * 295 * 0.999) and close(t["commission"], 4 * 295 * 0.001)
ok1 &= s["executed"] is False and miss == ["SASA.IS"] and h == dec[3] and dec[0]["price"] == 100.0
check("1. reprice_to_open: acilis fiyatiyla maliyet/gelir/komisyon (kararin orani), karar fiyati korunur, acilissiz "
      "sembolde islem yok, HOLD ve girdi degismez", ok1)


# 2 ---------------------------------------------------------------------------------------------------------------
def market(rows):
    df = pd.DataFrame(rows, columns=["symbol", "date", "open", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["symbol", "date"]).sort_index()


syms = [f"S{i}.IS" for i in range(10)]
rows = [(s, "2026-11-20", 10.0, 10.0) for s in syms]
rows += [(s, "2026-11-23", (11.0 if i < 5 else float("nan")), 11.0) for i, s in enumerate(syms)]   # %50 kapsam
rows += [(s, "2026-11-24", 12.0 + i, 12.0) for i, s in enumerate(syms)]
m = market(rows)
d1, o1 = DT.next_session_opens(m, "2026-11-20")
d2, o2 = DT.next_session_opens(m, "2026-11-24")
check("2. next_session_opens: kapsam esigi altindaki gun atlanir (11-23 -> 11-24), son gunden sonrasi None",
      d1 == "2026-11-24" and close(o1["S3.IS"], 15.0) and len(o1) == 10 and d2 is None and o2 == {})

# 3 ---------------------------------------------------------------------------------------------------------------
w0 = DT.observation_warnings(None)
w1 = DT.observation_warnings({"symbols": ["A"]})
w2 = DT.observation_warnings({"price_stats": {"A": {}}, "price_norm": {"mode": "causal"},
                              "indicator_spec": {"adx": "wilder"}})
w3 = DT.observation_warnings({"price_stats": {"A": {}}, "price_norm": {"mode": "stats"},
                              "indicator_spec": {"adx": "legacy"}})
check("3. observation_warnings: meta yok 1, price_stats yok 1, causal + wilder 3, stats + legacy 1 (olcek K13)",
      len(w0) == 1 and len(w1) == 1 and len(w2) == 3 and len(w3) == 1 and "K13" in w3[0])

# 4 ---------------------------------------------------------------------------------------------------------------
reset_engine()
security.reset_secret_cache()
init_db()
with session_scope() as db:
    create_user(db, email="w@test.local", password="WriterPass1", full_name="W", role=Role.USER, created_by="t",
                must_change_password=False)
    create_user(db, email="v@test.local", password="ViewerPass1", full_name="V", role=Role.VIEWER, created_by="t",
                must_change_password=False)
from app.main import app  # noqa: E402


def login(c, e, p):
    return c.post("/auth/login", json={"email": e, "password": p}).status_code == 200


def csrf(c):
    return {"X-CSRF-Token": c.cookies.get("rlt_csrf") or "", "X-Requested-With": "XMLHttpRequest"}


D = "2026-11-20"
decision = {"decisions": [{"symbol": "GARAN.IS", "action": "BUY", "shares": 10, "price": 100.0, "cost": 1001.0,
                           "commission": 1.0, "executed": True, "reason": "x"}],
            "summary": {"total_trades": 1}, "portfolio_before": {}, "portfolio_after": {}}
state = {"market": market([("GARAN.IS", D, 99.0, 100.0)])}


async def fake_fetch(symbols, target_date, lookback_days=30):
    return state["market"]

DT.fetch_latest_market_data = fake_fetch

with TestClient(app) as c:
    login(c, "w@test.local", "WriterPass1")
    with session_scope() as db:
        uid = db.query(User).filter(User.email == "w@test.local").first().id
    with ws.use_workspace(uid):
        os.makedirs(ws.live_trading_dir(), exist_ok=True)
        with open(os.path.join(ws.live_trading_dir(), "trade_decisions.json"), "w") as fh:
            json.dump({D: decision}, fh)
    r409 = c.post(f"/api/trading/apply-decision?date={D}", headers=csrf(c))
    with ws.use_workspace(uid):
        cash0 = pf.load_portfolio()["cash"]
    r422 = c.post(f"/api/trading/apply-decision?date={D}&execution=yok", headers=csrf(c))
    state["market"] = market([("GARAN.IS", D, 99.0, 100.0), ("GARAN.IS", "2026-11-23", 103.0, 104.0)])
    r200 = c.post(f"/api/trading/apply-decision?date={D}", headers=csrf(c))
    body = r200.json() if r200.status_code == 200 else {}
    with ws.use_workspace(uid):
        p = pf.load_portfolio()
    ok4 = r409.status_code == 409 and "ACILIS" in r409.json().get("detail", "") and close(cash0, 100_000)
    ok4 &= r422.status_code == 422
    ok4 &= r200.status_code == 200 and body.get("execution", {}).get("date") == "2026-11-23"
    ok4 &= close(p["cash"], 100_000 - 10 * 103 * 1.001) and close(p["positions"]["GARAN.IS"]["avg_cost"], 103 * 1.001)
    # eski davranis: ayni karar ayni gun kapanisiyla (yeni portfoyle)
    with ws.use_workspace(uid):
        pf.reset_portfolio(100_000)
    rc = c.post(f"/api/trading/apply-decision?date={D}&execution=decision_close", headers=csrf(c))
    with ws.use_workspace(uid):
        pc = pf.load_portfolio()
    ok4 &= rc.status_code == 200 and close(pc["cash"], 100_000 - 1001.0) and "note" in rc.json().get("execution", {})
    c.post("/auth/logout", headers=csrf(c))
    login(c, "v@test.local", "ViewerPass1")
    rv = c.post(f"/api/trading/apply-decision?date={D}", headers=csrf(c))
    ok4 &= rv.status_code == 403
check(f"4. /apply-decision: ertesi seans yok 409 ({r409.status_code}) + portfoy degismedi; ertesi acilisla islem "
      f"({r200.status_code}, nakit/maliyet acilisa gore); decision_close eski davranis; gecersiz mod 422; viewer 403",
      ok4)

print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
