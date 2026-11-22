"""Faz 11 S4: fiyat marji takviminin ileri gunlere kapsami (plan phase-11-integration-plan.md §3 S4; S1 acik maddesi).

`accounting.PRICE_MARGIN_VERIFIED_UNTIL` M8 verisinin son gunu (2026-11-09). Ileri gunler icin takvim (2020-03-13'ten
%10) iki kanitla gecerli sayilir:
  1. Tarihli KAYNAK DENETIMI (`marl/margin_source_checks.json`, git'te): Borsa Istanbul kaynaklari elle okunur, takvimde
     degisiklik yoksa kayit eklenir. Bir denetim en fazla CHECK_HORIZON_DAYS takvim gunu ilerisini kapsar (marj
     degisikligi Borsa Istanbul duyurusuyla, onceden ilan edilir; aylik yeniden denetim yeter). Takvim degistiyse
     (`schedule_unchanged: false`) kapsam durur: once `accounting.PRICE_MARGIN_SCHEDULE` kaynakla guncellenir.
  2. VERI DENETIMI: ileri panelin her gunu S1 gun kapisindan gecti (kapanis/acilis +-%10,5 bandi; `forward_panel.day_gate`
     'marj_bandi_disi' gunu yazmaz). Panelde olan gun bu denetimi gecmistir.
Kapsam = min(panelin son gunu, son denetim + CHECK_HORIZON_DAYS). Denetim bayatsa defter o gunden sonrasini islemez
(sessiz gecmez; sebep gunluge yazilir). Kaynak denetimi elle yapilir; kod yalniz tarihi okur.
"""
from __future__ import annotations

import json
import os
from datetime import date, timedelta

from marl import accounting as A

CHECKS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'margin_source_checks.json')
CHECK_HORIZON_DAYS = 31


def load_checks(path: str = CHECKS_PATH) -> list:
    with open(path, encoding='utf-8') as fh:
        checks = json.load(fh)
    for c in checks:
        date.fromisoformat(c['checked'])
        if not c.get('sources'):
            raise ValueError(f"kaynak denetimi kaynaksiz: {c['checked']}")
    return sorted(checks, key=lambda c: c['checked'])


def coverage_until(panel_last_date: str, checks: list | None = None) -> dict:
    """Ileri kapsam: {'until': 'YYYY-MM-DD', 'last_check': ..., 'horizon_end': ..., 'limited_by': 'panel'|'check'}."""
    checks = load_checks() if checks is None else sorted(checks, key=lambda c: c['checked'])
    if not checks:
        raise ValueError("kaynak denetimi kaydi yok")
    last = checks[-1]
    if not last.get('schedule_unchanged', False):
        raise ValueError(f"son kaynak denetimi ({last['checked']}) takvim degisikligi bildiriyor: once "
                         "accounting.PRICE_MARGIN_SCHEDULE kaynakla guncellenmeli")
    horizon = (date.fromisoformat(last['checked']) + timedelta(days=CHECK_HORIZON_DAYS)).isoformat()
    panel_last = str(panel_last_date)[:10]
    until = max(A.PRICE_MARGIN_VERIFIED_UNTIL, min(panel_last, horizon))
    return {'until': until, 'last_check': last['checked'], 'horizon_end': horizon,
            'limited_by': 'panel' if panel_last <= horizon else 'check', 'horizon_days': CHECK_HORIZON_DAYS}
