"""Faz 11 S4: M8 ileri veri gunluk isini elle ya da isletim sistemi zamanlayicisindan calistir (ANA venv).

Uygulama acikken ayni isi `app/services/marl_forward.py` zamanlar (K-F11-4); bu betik sunucu kapaliyken ya da elle
doldurma icin. Ayni kilidi kullanir (ikisi ayni anda calismaz). Is idempotent; kacirilan gunleri sirayla doldurur.

    python scripts/marl/forward_daily_m8.py                 # panel guncelle (ag) + defter
    python scripts/marl/forward_daily_m8.py --no-update     # yalniz panelde olan gunleri defterle
    python scripts/marl/forward_daily_m8.py --status
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from marl import forward_job as FJ  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-update', action='store_true')
    ap.add_argument('--as-of', default=None, help='ISO zaman (TRT); varsayilan simdi')
    ap.add_argument('--status', action='store_true')
    a = ap.parse_args()
    if a.status:
        print(json.dumps(FJ.status(), indent=1, ensure_ascii=False, default=str))
        return
    tz = ZoneInfo('Europe/Istanbul')
    as_of = datetime.fromisoformat(a.as_of).replace(tzinfo=tz) if a.as_of else datetime.now(tz)
    rep = FJ.run_daily(as_of, update_panel=not a.no_update)
    print(json.dumps({k: v for k, v in rep.items() if k != 'archive'}, indent=1, ensure_ascii=False, default=str))
    sys.exit(1 if rep.get('status') == 'hata' else 0)


if __name__ == '__main__':
    main()
