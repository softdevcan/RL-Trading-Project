"""Faz 11 §3.19: rota isleyicisi `async def` ise `await` icermeli (pytest degil).

    python tests/test_api_async_guard.py

Neden: FastAPI `async def` isleyiciyi olay dongusunde kosar. Icinde CSV okuma, ag istegi, model tahmini gibi engelleyici
is varsa o surede SUNUCUNUN TAMAMI bekler (olculdu: altin fiyati cekilirken hafif bir istek 2,8 sn bekledi; `def`e
cevrilince medyan 11 ms). `await`'siz isleyici `def` olmali: FastAPI onu is parcacigi havuzunda kosar.
Istisna: websocket ve `asyncio` kullanan isleyiciler.
"""
import ast
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS = {"get", "post", "put", "delete", "patch"}


def offenders(paths) -> list:
    out = []
    for f in paths:
        src = open(f, encoding="utf-8").read()
        for n in ast.walk(ast.parse(src)):
            if not isinstance(n, ast.AsyncFunctionDef):
                continue
            if not any(isinstance(d, ast.Call) and getattr(d.func, "attr", "") in METHODS for d in n.decorator_list):
                continue
            inner = [x for x in ast.walk(n) if x is not n]
            if any(isinstance(x, (ast.Await, ast.AsyncWith, ast.AsyncFor, ast.Yield, ast.YieldFrom)) for x in inner):
                continue
            seg = ast.get_source_segment(src, n)
            if "asyncio." in seg or "WebSocket" in seg:
                continue
            out.append(f"{os.path.relpath(f, ROOT)}:{n.lineno} {n.name}")
    return out


routes = sorted(glob.glob(os.path.join(ROOT, "app", "api", "routes", "*.py")))
bad = offenders(routes)
# mutasyon: await'siz async isleyici yakalanmali
tmp = os.path.join(ROOT, "tests", "_async_guard_sample.py")
with open(tmp, "w", encoding="utf-8") as fh:
    fh.write("@router.get('/x')\nasync def x():\n    return open('a').read()\n")
try:
    caught = offenders([tmp])
finally:
    os.remove(tmp)
ok = not bad and len(caught) == 1
print(f"[{'OK' if ok else 'HATA'}] 1. {len(routes)} rota dosyasinda await'siz async isleyici yok "
      f"(mutasyon yakalandi: {len(caught) == 1})" + (f"; bulunan: {bad}" if bad else ""))
print(f"\n{int(ok)} gecti, {int(not ok)} kaldi")
sys.exit(0 if ok else 1)
