"""
Piyasa endeksi kapanis serisi (Faz 9 B6: rejim etiketi icin XU100).

`macro_data.csv` icindeki `bist100_index` B1'den beri 2018'de basliyor (zorunlu makro
sutunlari dolmadan once satirlar atiliyor). RL penceresi 2013-06'da basliyor ve 200 gunluk
trend isinma istiyor; rejim etiketi icin ayri, uzun bir seri tutulur.

Cikti: `data/macro/xu100.csv` (date, close) + `xu100_source.json` (sembol, kaynak, cekim
zamani, kutuphane surumu, SHA-256). CSV git disi (data/**/*.csv); olcum JSON'lari SHA'yi tasir.

Olculen (2026-11-15): yfinance 'XU100.IS' 2005-01-03..; |gunluk getiri| > %15 olan gun yok —
2020-07-27 endeks "sifir atma"si seride geriye donuk duzeltilmis gorunuyor (2020-07-24: 1192).
`fetch_index_close` her cekimde bu denetimi yapar; sicrama varsa yazmaz.

Calistirma:
    python -m data.index_fetcher            # XU100, 2005-01-01..bugun
"""

import hashlib
import json
import os
from datetime import datetime, timezone

import pandas as pd

DEFAULT_INDEX_CSV = os.path.join('data', 'macro', 'xu100.csv')
JUMP_LIMIT = 0.15  # gunluk |getiri| esigi: ustu seviye sicramasi sayilir (yazilmaz)


class IndexDataError(ValueError):
    """Endeks serisi bos ya da seviye sicramasi tasiyor."""


def check_index_series(close: pd.Series, jump_limit: float = JUMP_LIMIT) -> None:
    """Bos seri, pozitif olmayan deger ya da seviye sicramasi -> IndexDataError."""
    c = close.dropna()
    if c.empty:
        raise IndexDataError("endeks serisi bos")
    if (c <= 0).any():
        raise IndexDataError(f"pozitif olmayan kapanis: {int((c <= 0).sum())} gun")
    r = c.pct_change().dropna()
    jumps = r[r.abs() > jump_limit]
    if len(jumps):
        raise IndexDataError(
            "seviye sicramasi (ornek: " + ", ".join(f"{d.date()} {v:+.1%}" for d, v in jumps.head(3).items())
            + ") — sifir atma / bolunme duzeltilmeden rejim etiketi kurulamaz")


def fetch_index_close(symbol: str = 'XU100.IS', start: str = '2005-01-01',
                      path: str = DEFAULT_INDEX_CSV, save: bool = True) -> pd.Series:
    """yfinance'ten gunluk kapanis (auto_adjust=False; endekste temettu duzeltmesi yok)."""
    import yfinance as yf
    df = yf.download(symbol, start=start, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise IndexDataError(f"{symbol}: yfinance bos dondu")
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().astype(float)
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep='last')].sort_index()
    close.name = 'close'
    close.index.name = 'date'
    check_index_series(close)
    if save:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        close.to_csv(path, lineterminator='\n')
        with open(path, 'rb') as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        meta = {'symbol': symbol, 'source': 'yfinance', 'auto_adjust': False,
                'yfinance_version': getattr(yf, '__version__', None),
                'fetched_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'range': [str(close.index[0].date()), str(close.index[-1].date())],
                'n': int(len(close)), 'sha256': sha, 'jump_limit': JUMP_LIMIT}
        with open(os.path.splitext(path)[0] + '_source.json', 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, indent=1)
    return close


def load_index_close(path: str = DEFAULT_INDEX_CSV) -> pd.Series:
    """Kaydedilmis endeks kapanisi. Dosya yoksa FileNotFoundError (sessiz yedek yok)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} yok: `python -m data.index_fetcher` ile cek")
    s = pd.read_csv(path, parse_dates=['date'], index_col='date')['close'].astype(float)
    check_index_series(s)
    return s


if __name__ == '__main__':
    s = fetch_index_close()
    print(f"XU100: {s.index[0].date()}..{s.index[-1].date()} ({len(s)} gun) -> {DEFAULT_INDEX_CSV}")
