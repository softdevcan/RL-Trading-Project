"""Faz 11 S5: simulasyon panosunun veri katmani (plan phase-11-integration-plan.md §3 S5; K-F11-1, K-F11-2, K-F11-6).

Iki pencere (karsilastirma YALNIZ ayni pencere ve ayni maliyetteki defterler arasinda; plan S5):
  'forward'  : ileri veri defterleri (`forward_job` deposu, 2026-11-23'ten). Ara gorunum; tez sonucu DEGIL (ek §9).
  'test2026' : ayni defterin 2026 test penceresi yeniden oynatimi (S3; M8'de okunmus, gecmis). Onbellek
               `data/marl/forward/m8/cache/` (arsiv manifest SHA'siyla anahtarli).
Kol ozeti 12 tohumun ortalamasi +- std (N gosterilir). Urun portfoyu (`product`): secilen strateji, kullanicinin
sermayesiyle TAM SAYI lot (orders.step_integer; ortamla ayni fonksiyon) ve kesirli es; fark gosterilir. Urun portfoyu
ileri teste girmez.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from marl import forward_job as FJ
from marl import panel as PN
from marl import sim_ledger as SL
from marl.selection import excess_sharpe

ARMS = ('M-none', 'S-esm', 'S-agr', 'M-gat', 'M-self', 'null')
ARM_LABELS = {'M-none': 'MARL iletişimsiz (MAPPO)', 'S-esm': 'SARL eşmimari', 'S-agr': 'SARL ağırlık',
              'M-gat': 'MARL graf dikkati', 'M-self': 'MARL kendine dikkat (kontrol)', 'null': 'Boş kol (negatif kontrol)',
              'rule/ew': '1/N eşit ağırlık', 'rule/bh': 'Al-tut', 'rule/minvar': 'Min-varyans (aylık)',
              'rule/hrp': 'HRP (aylık)'}             # yalnız görüntü etiketi (pano); hesaba girmez
DEFAULT_VIEW = ('M-none', 'rule/ew', 'rule/minvar')        # K-F11-1
WINDOWS = ('forward', 'test2026')
CACHE_DIR = os.path.join('data', 'marl', 'forward', 'm8', 'cache')
PANEL_M8 = os.path.join('data', 'marl', 'panel_m8.npz')


def _manifest_sha() -> str:
    with open(os.path.join(FJ.ARCHIVE, 'manifest.json'), 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _test_rows() -> tuple[int, int]:
    p = FJ.manifest()['policies'][0]
    with open(os.path.join(FJ.ARCHIVE, p['arm'], p['seed'], 'fold_eval.json'), encoding='utf-8') as fh:
        return tuple(json.load(fh)['rows']['test'])


def _forward_market():
    from marl import forward_panel as FP
    from marl import margin_coverage as MC
    path = os.path.join(FP.FWD_ROOT, FP.PANEL_FWD)
    arrays = PN.load_npz(path if os.path.exists(path) else FP.PANEL_M8)
    cov = MC.coverage_until(str(arrays['dates'][-1]))
    arrays = FJ.truncate(arrays, cov['until'])
    return arrays, cov['until']


def raw_series(window: str) -> dict:
    """{'dates': [...], 'series': {sid: {'R', 'X', 'value', 'turnover', 'cash_weight'}}} (deger: 100.000 TL'den)."""
    if window == 'forward':
        rows = FJ.read_rows()
        if rows.empty:
            return {'dates': [], 'series': {}}
        out = {}
        for sid, g in rows.groupby('sid'):
            g = g.sort_values('row')
            out[sid] = {k: g[k].astype(float).tolist() for k in ('R', 'X', 'value', 'turnover', 'cash_weight')}
            dates = list(g['date'])
        return {'dates': dates, 'series': out}
    if window != 'test2026':
        raise ValueError(f"pencere {WINDOWS}: {window}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = os.path.join(CACHE_DIR, f'test2026_{_manifest_sha()[:12]}.json')
    if os.path.exists(cp):
        with open(cp, encoding='utf-8') as fh:
            return json.load(fh)
    ts, te = _test_rows()
    mk = SL.Market(PN.load_npz(PANEL_M8))
    out, dates = {}, None
    for sid in FJ.strategy_ids():
        _, rows = SL.run(mk, FJ.make_strategy(sid), SL.LedgerState.start(ts, mk.N), te)
        ser = SL.series(rows)
        out[sid] = {k: np.asarray(ser[k], float).tolist() for k in ('R', 'X', 'value', 'turnover', 'cash_weight')}
        dates = ser['date']
    res = {'dates': dates, 'series': out}
    with open(cp + '.tmp', 'w', encoding='utf-8') as fh:
        json.dump(res, fh)
    os.replace(cp + '.tmp', cp)
    return res


def _metrics(s: dict) -> dict:
    R, X, v = (np.asarray(s[k], float) for k in ('R', 'X', 'value'))
    sr = excess_sharpe(R, rf=R - X) if R.size >= 3 else None
    peak = np.maximum.accumulate(np.r_[SL.INITIAL_CAPITAL, v])
    dd = float(np.min(np.r_[SL.INITIAL_CAPITAL, v] / peak - 1)) if v.size else 0.0
    return {'cum_return': float(v[-1] / SL.INITIAL_CAPITAL - 1) if v.size else 0.0,
            'excess_sharpe': sr, 'max_drawdown': dd, 'turnover': float(np.mean(s['turnover'])) if v.size else 0.0,
            'cash': float(np.mean(s['cash_weight'])) if v.size else 1.0}


def view(window: str, keys=DEFAULT_VIEW) -> dict:
    """Secilen kollar (12 tohum ortalamasi +- std) ve kurallar; ozet tablo."""
    raw = raw_series(window)
    dates, ser = raw['dates'], raw['series']
    out = {'window': window, 'dates': dates, 'n_sessions': len(dates), 'lines': {}, 'table': [],
           'cost_bps_roundtrip': SL.COST_BPS_ROUNDTRIP, 'capital': SL.INITIAL_CAPITAL}
    if not dates:
        return out
    for key in keys:
        sids = [s for s in ser if s.startswith(f'policy/{key}/')] if key in ARMS else ([key] if key in ser else [])
        if not sids:
            continue
        V = np.array([ser[s]['value'] for s in sids])
        m = [_metrics(ser[s]) for s in sids]
        line = {'label': ARM_LABELS.get(key, key), 'n': len(sids), 'mean': V.mean(axis=0).tolist(),
                'std': (V.std(axis=0, ddof=1) if len(sids) > 1 else np.zeros(V.shape[1])).tolist()}
        out['lines'][key] = line
        row = {'key': key, 'label': line['label'], 'n': len(sids)}
        for f in ('cum_return', 'excess_sharpe', 'max_drawdown', 'turnover', 'cash'):
            vals = np.array([x[f] if x[f] is not None else np.nan for x in m], float)
            row[f] = float(np.nanmean(vals)) if np.isfinite(vals).any() else None
            row[f + '_std'] = float(np.nanstd(vals, ddof=1)) if len(sids) > 1 and np.isfinite(vals).sum() > 1 else None
        row['n_undefined_sharpe'] = int(sum(x['excess_sharpe'] is None for x in m))
        if key in ARMS:     # tohum dagilimi (plan §3.13): ortalama +- std'nin arkasindaki tek tek sonuclar
            row['members'] = [{'sid': s, 'seed': s.rsplit('/', 1)[-1], 'cum_return': x['cum_return'],
                               'excess_sharpe': x['excess_sharpe']} for s, x in zip(sids, m)]
        out['table'].append(row)
    return out


def daily_book(rows: list, symbols, capital: float) -> dict:
    """Tam sayi defterinin gun gun dokumu (plan §3.13; pano 'Gun ozeti' ve 'Portfoyumde ne var').

    `rows`: `sim_ledger.run` tam sayi modu satirlari (`holdings`, `cash`, `value`, `rf`, `fills`, `p_close`). Baslangic
    durumu: tum sermaye nakit. Ortalama maliyet komisyon HARIC hareketli ortalama (alista guncellenir, satista degismez,
    adet 0'a inince sifirlanir); gerceklesen K/Z = (satis fiyati - ort. maliyet) x adet, birikimli. Hisse katkisi
    = gun sonu pozisyon degeri - onceki gun sonu pozisyon degeri - pozisyona giren net nakit (komisyon haric).
    Ozdeslik (test): sum(katki) + nakit faizi - komisyon = deger_t - deger_{t-1}. Hesap defteri degistirmez.
    """
    n = len(symbols)
    h_prev, mark_prev = np.zeros(n, dtype=np.int64), np.zeros(n)
    cash_prev = float(capital)
    avg, real = np.zeros(n), np.zeros(n)
    out = {k: [] for k in ('dates', 'value', 'cash', 'lots', 'mark', 'avg_cost', 'realized', 'contrib',
                           'cash_interest', 'commission', 'trades')}
    for r in rows:
        h = np.asarray(r['holdings'], dtype=np.int64)
        mark = np.asarray(r['p_close'], dtype=np.float64)
        flow, comm, trades = np.zeros(n), 0.0, []
        for f in r['fills']:
            i, q, p = f['i'], f['shares'], f['price']
            if f['action'] == 'BUY':
                before = int(h[i]) - q          # h gun sonu adedi; simgede gunde tek karar
                avg[i] = (avg[i] * before + q * p) / (before + q)
                flow[i] += q * p
            else:
                real[i] += (p - avg[i]) * q
                flow[i] -= q * p
                if h[i] == 0:
                    avg[i] = 0.0
            comm += f['commission']
            trades.append({'symbol': symbols[i], 'action': f['action'], 'shares': q, 'price': p,
                           'amount': q * p, 'commission': f['commission']})
        interest = cash_prev * float(r['rf'])
        contrib = h * mark - h_prev * mark_prev - flow
        out['dates'].append(str(r['date']))
        out['value'].append(float(r['value']))
        out['cash'].append(float(r['cash']))
        out['lots'].append([int(x) for x in h])
        out['mark'].append([float(x) for x in mark])
        out['avg_cost'].append([float(x) for x in avg])
        out['realized'].append([float(x) for x in real])
        out['contrib'].append([float(x) for x in contrib])
        out['cash_interest'].append(interest)
        out['commission'].append(comm)
        out['trades'].append(sorted(trades, key=lambda x: -x['amount']))
        h_prev, mark_prev, cash_prev = h, mark, float(r['cash'])
    out['symbols'] = list(symbols)
    out['capital'] = float(capital)
    return out


def product(sid: str, capital: float, window: str = 'forward') -> dict:
    """Urun portfoyu: secilen strateji tam sayi lotla (kullanici sermayesi) ve kesirli es; fark (K-F11-2)."""
    if capital <= 0:
        raise ValueError('sermaye pozitif olmali')
    if sid not in FJ.strategy_ids():
        raise ValueError(f'bilinmeyen strateji: {sid}')
    if window == 'forward':
        arrays, until = _forward_market()
        r0 = FJ.start_row(arrays['dates'])
        if r0 is None:
            return {'sid': sid, 'window': window, 'dates': [], 'note': f'ileri veri {FJ.FORWARD_START} itibariyle'}
        mk = SL.Market(arrays, verified_until=until)
        te = len(arrays['dates']) - 1
    else:
        r0, te = _test_rows()
        mk = SL.Market(PN.load_npz(PANEL_M8))
    if te <= r0:
        return {'sid': sid, 'window': window, 'dates': []}
    _, fr = SL.run(mk, FJ.make_strategy(sid), SL.LedgerState.start(r0, mk.N, capital), te)
    _, it = SL.run(mk, FJ.make_strategy(sid), SL.LedgerState.start(r0, mk.N, capital, integer=True), te)
    vf, vi = np.array([r['value'] for r in fr]), np.array([r['value'] for r in it])
    dR = np.array([a['R'] - b['R'] for a, b in zip(it, fr)])
    # Lot tablosu: son gun kapanisinda eldeki adet (1 lot = 1 pay), kapanis fiyati, piyasa degeri, agirlik; kesirli
    # defterin ayni gunku agirligi ve son gunku alim/satim (adet farki). Fiyatlar yfinance duzeltilmis kapanis.
    h_last = np.asarray(it[-1]['holdings'], dtype=np.int64)
    h_prev = np.asarray(it[-2]['holdings'], dtype=np.int64) if len(it) > 1 else np.zeros_like(h_last)
    close = np.asarray(mk.px['close'][te], dtype=np.float64)
    w_int, w_frac = np.asarray(it[-1]['w'][1:]), np.asarray(fr[-1]['w'][1:])
    holdings = [{'symbol': s.replace('.IS', ''), 'lots': int(h_last[i]), 'close': float(close[i]),
                 'market_value': float(h_last[i] * close[i]), 'weight': float(w_int[i]), 'weight_frac': float(w_frac[i]),
                 'traded_lots': int(h_last[i] - h_prev[i])}
                for i, s in enumerate(mk.symbols) if h_last[i] != 0 or h_prev[i] != 0 or w_frac[i] > 1e-4]
    holdings.sort(key=lambda x: -x['market_value'])
    return {'sid': sid, 'label': ARM_LABELS.get(sid, sid), 'window': window, 'capital': capital,
            'dates': [r['date'] for r in fr], 'value_fractional': vf.tolist(), 'value_integer': vi.tolist(),
            'final_diff_pct': float(vi[-1] / vf[-1] - 1), 'tracking_error_ann': float(dR.std(ddof=1) * np.sqrt(252))
            if dR.size > 1 else 0.0, 'mean_weight_gap': float(np.mean([r['weight_gap'] for r in it])),
            'n_orders': int(sum(r['n_orders'] for r in it)), 'last_cash_integer': float(it[-1]['cash_weight']),
            'last_weights_integer': {s: float(w) for s, w in zip(mk.symbols, it[-1]['w'][1:]) if w > 1e-9},
            'holdings_date': str(it[-1]['date']), 'holdings': holdings, 'cash_integer_tl': float(it[-1]['cash']),
            'cash_weight_frac': float(fr[-1]['w'][0]),
            'book': daily_book(it, [s.replace('.IS', '') for s in mk.symbols], capital)}
