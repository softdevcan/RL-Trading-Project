"""
Faz 11 §3.13: simulasyon sayfasinin gunluk defteri (`marl/forward_views.daily_book`) ve sayfa render'i. Ag, arsiv ve
panel GEREKMEZ: satirlar gercek tam sayi adimiyla (`orders.step_integer`) sentetik fiyatlarda uretilir ve
`sim_ledger.integer_fills` ile defterin yazdigi bicime cevrilir.

Kontroller:
  1. Muhasebe ozdesligi her gun: sum(hisse katkisi) + nakit faizi - komisyon = deger_t - deger_{t-1} (goreli 1e-9);
     sum(lot x degerleme) + nakit = deger.
  2. Ortalama maliyet bagimsiz denetim: hisse basina birikimli katki = lot x (degerleme - ort. maliyet) + gerceklesen
     (hareketli ortalama tanimi); islem goremeyen hissede o gun emir yok; adetler tam sayi, negatif yok.
  3. Mutasyon: bir emrin fiyati 1 kurus kayinca (1) bozulur; gerceklesen K/Z yok sayilinca (2) bozulur.
  4. Sayfa render'i (saf fonksiyonlar): 10 cikti; grafik izleri (kol bandi 2 + ortalama + kural + incelenen + secili
     gun + bar + dusus); 1A araligi 21 gun; tablo satiri = elde tutulan hisse + nakit + toplam; Toplam satiri deger
     ile ayni; secili ilk gunde 'onceki' kapali; gun secimi (tarih) kartlari o gune tasir.
  5. Ileri pencere on kayit dili: bant 'Ara gorunum' + seans ilerlemesi; tohum notu kiyas yorumu yapmaz; test
     penceresinde 1/N ustundeki tohum sayisi yazilir.

    python tests/test_marl_sim_book.py      (~3 sn; ana venv)
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from marl import forward_views as FV  # noqa: E402
from marl import orders as O  # noqa: E402
from marl import sim_ledger as SL  # noqa: E402

passed = failed = 0


def check(name, ok):
    global passed, failed
    print(f"[{'OK' if ok else 'HATA'}] {name}")
    passed, failed = passed + bool(ok), failed + (not ok)


def synthetic_rows(seed=0, T=60, N=6, capital=100_000.0, rf=0.0015, cost=0.001):
    rng = np.random.default_rng(seed)
    close = 40 * np.cumprod(1 + 0.02 * rng.standard_normal((T, N)), axis=0) * rng.uniform(0.5, 3, N)
    opn = close * (1 + 0.006 * rng.standard_normal((T, N)))
    px = {'close': close, 'open': opn}
    syms = [f'S{i}.IS' for i in range(N)]
    h, cash, value, w = np.zeros(N, dtype=np.int64), capital, capital, np.r_[1.0, np.zeros(N)]
    rows, blocked = [], []
    start = np.datetime64('2026-01-02')
    for t in range(T - 1):
        u = rng.dirichlet(np.ones(N + 1) * 0.8)
        tradable = np.ones(N, dtype=bool)
        if t % 7 == 3:
            tradable[t % N] = False
            blocked.append((t, t % N))
        g_on, g_day = opn[t + 1] / close[t], close[t + 1] / opn[t + 1]
        r = O.step_integer(h, cash, value, w, u, px, t, g_on, g_day, tradable, rf, None, cost, syms)
        h, cash, value, w = r['holdings'], r['cash'], r['value'], r['w']
        rows.append({'date': str(start + t + 1), 'holdings': h.copy(), 'cash': float(cash), 'value': float(value),
                     'rf': rf, 'fills': SL.integer_fills(r['decisions']), 'p_close': r['p_close'].copy()})
    return rows, [s.replace('.IS', '') for s in syms], capital, blocked


def identity_err(book):
    cap, err = book['capital'], 0.0
    for i, v in enumerate(book['value']):
        prev = book['value'][i - 1] if i else cap
        lhs = sum(book['contrib'][i]) + book['cash_interest'][i] - book['commission'][i]
        err = max(err, abs(lhs - (v - prev)) / cap)
    return err


def basis_err(book):
    n, err = len(book['symbols']), 0.0
    cum = np.zeros(n)
    for i in range(len(book['dates'])):
        cum += np.asarray(book['contrib'][i])
        L, M, A, R = (np.asarray(book[k][i]) for k in ('lots', 'mark', 'avg_cost', 'realized'))
        err = max(err, float(np.max(np.abs(cum - (L * (M - A) + R)))) / book['capital'])
    return err


rows, syms, cap, blocked = synthetic_rows()
book = FV.daily_book(rows, syms, cap)

# 1
e1 = identity_err(book)
val_err = max(abs(sum(l * m for l, m in zip(book['lots'][i], book['mark'][i])) + book['cash'][i] - book['value'][i])
              / cap for i in range(len(rows)))
n_tr = sum(len(t) for t in book['trades'])
check(f"1. gun ozdesligi sum(katki)+faiz-komisyon = dV (en buyuk goreli {e1:.1e}); lot x fiyat + nakit = deger "
      f"({val_err:.1e}); {len(rows)} gun, {n_tr} emir", e1 < 1e-9 and val_err < 1e-9 and n_tr > 50)

# 2
e2 = basis_err(book)
no_fill_blocked = all(all(f['i'] != j for f in rows[t]['fills']) for t, j in blocked)
lots_ok = all(isinstance(x, int) and x >= 0 for day in book['lots'] for x in day)
sold = sum(1 for d in book['trades'] for t in d if t['action'] == 'SELL')
check(f"2. birikimli katki = lot x (fiyat - ort. maliyet) + gerceklesen ({e2:.1e}; {sold} satis); islem goremeyen "
      f"hissede emir yok ({len(blocked)} gun); adet tam sayi >= 0", e2 < 1e-9 and no_fill_blocked and lots_ok and sold > 5)

# 3 mutasyon
mut = [dict(r) for r in rows]
k = next(t for t, r in enumerate(mut) if r['fills'])
mut[k] = dict(mut[k], fills=[dict(mut[k]['fills'][0], price=mut[k]['fills'][0]['price'] + 0.01)] + mut[k]['fills'][1:])
e3a = identity_err(FV.daily_book(mut, syms, cap))
no_real = dict(book, realized=[[0.0] * len(syms) for _ in book['realized']])
e3b = basis_err(no_real)
check(f"3. mutasyon: emir fiyati +0,01 -> ozdeslik {e3a:.1e}; gerceklesen yok sayilinca maliyet denetimi {e3b:.1e}",
      e3a > 1e-9 and e3b > 1e-6)

# 4 sayfa render'i
from dashboard.pages import simulation as S  # noqa: E402

dates = book['dates']
rr = np.random.default_rng(5)
members = [{'sid': f'policy/M-none/s{s}', 'seed': f's{s}', 'cum_return': float(rr.normal(0.05, 0.03)),
            'excess_sharpe': None} for s in S.SEEDS]
mean = list(np.linspace(1e5, 1.05e5, len(dates)))
view = {'window': 'test2026', 'dates': dates, 'n_sessions': len(dates), 'capital': 100_000.0, 'cost_bps_roundtrip': 20.0,
        'lines': {'M-none': {'label': 'MARL', 'n': 12, 'mean': mean, 'std': [500.0] * len(dates)},
                  'rule/ew': {'label': '1/N', 'n': 1, 'mean': mean[::-1], 'std': [0.0] * len(dates)}},
        'table': [{'key': 'M-none', 'label': 'MARL', 'n': 12, 'cum_return': 0.05, 'cum_return_std': 0.03,
                   'excess_sharpe': 0.4, 'excess_sharpe_std': 0.2, 'max_drawdown': -0.08, 'turnover': 0.1, 'cash': 0.05,
                   'members': members},
                  {'key': 'rule/ew', 'label': '1/N', 'n': 1, 'cum_return': 0.03, 'cum_return_std': None,
                   'excess_sharpe': None, 'excess_sharpe_std': None, 'max_drawdown': -0.1, 'turnover': 0.01,
                   'cash': 0.0}]}
out = S.render(book, view, None, 'all', 'M-none', 201)
fig = out[0]
held = sum(1 for x in book['lots'][-1] if x > 0)
tbl = out[5].children[0]
ok4 = len(out) == 10 and len(fig.data) == 8 and len(out[1]) == 5
ok4 &= len(tbl.data) == held + 2 and tbl.data[-1]['value'] == S.tl(book['value'][-1])
ok4 &= out[8] is False and out[9] is True                         # son gun: onceki acik, sonraki kapali
r1m = S.render(book, view, None, '1m', 'M-none', 201)
ok4 &= len([t for t in r1m[0].data if t.name and 'incelenen' in t.name][0].x) == 21
first = S.render(book, view, dates[0], 'all', 'rule/ew', None)
ok4 &= first[8] is True and S.d_short(dates[0]) in str(first[1]) and 'İlk gün' in str(first[2])
ok4 &= S.tl(book['value'][10]) in str(S.render(book, view, dates[10], 'all', 'M-none', 201)[1])
check(f"4. render: 10 cikti, grafik {len(fig.data)} iz, 5 kart, tablo {len(tbl.data)} satir (= {held} hisse + nakit + "
      f"toplam), toplam = deger; 1A 21 gun; ilk gunde onceki kapali; secili gun kartlari tasir", ok4)

# 5 on kayit dili
fwd = str(S.banner('forward', {'n_sessions': 3, 'read_sessions': 126}))
_, note_f = S.seed_figure(view, 'forward')
_, note_t = S.seed_figure(view, 'test2026')
n_above = sum(m['cum_return'] > 0.03 for m in members)
check("5. ileri bant 'Ara gorunum' + 3 / 126; ileri tohum notu kiyas yorumu yapmaz; testte 1/N ustu sayisi yazilir",
      'Ara görünüm' in fwd and '3 / 126' in fwd and 'yorumlanmaz' in note_f and "1/N'in üzerinde" not in note_f
      and f"12 tohumun {n_above}'i 1/N'in üzerinde" in note_t)

# 6 karsilastirma: tablo tum stratejileri sabit sirayla gosterir, grafik yalniz secilenleri cizer (API'ye gitmeden)
sub = S.select_view(view, ['rule/ew'])
ct = S.compare_table(view, ['rule/ew']).children[0]
check("6. select_view yalniz secilen cizgi + satir; tablo tum satirlar ayni sirada, secilen 'Grafikte' isaretli",
      list(sub['lines']) == ['rule/ew'] and [r['key'] for r in sub['table']] == ['rule/ew']
      and len(view['lines']) == 2 and [r['plot'] for r in ct.data] == ['', '●'] and len(ct.data) == len(view['table']))

print(f"\n{passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
