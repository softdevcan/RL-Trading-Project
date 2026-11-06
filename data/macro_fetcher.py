"""
Makroekonomik Veri Çekme Modülü (Faz 2)
TCMB EVDS API ve yfinance kullanarak makroekonomik göstergeleri çeker

Göstergeler:
1. TCMB Politika Faizi (%) - EVDS TP.BISPOLFAIZ.TUR (aylık; Fallback: 50.0 + kalite bayrağı)
2. TÜFE Enflasyon (YoY %) - EVDS (Endeksten hesaplanır)
3. ÜFE Enflasyon (YoY %) - EVDS (Endeksten hesaplanır)
4. USD/TRY Döviz Kuru - yfinance
5. EUR/TRY Döviz Kuru - yfinance
6. BIST-100 Endeks - yfinance

Faz 9 / B1–B3 (2026-11-06) — nokta-zamanlı (point-in-time) hizalama:
  * B2: eski politika faizi kodu `TP.YSSK.A01` EVDS'de yok (API hata veriyor);
    her çekim sabit 50'ye düşüyordu. TÜFE/ÜFE ise iki ayrıştırma hatasıyla
    düşüyordu: EVDS sütunları alt çizgili döner (`TP_FG_J0`) ve aylık seride
    `Tarih` 'YYYY-M' biçimindedir. `_evds_series` ikisini de çözer.
  * B3: aylık bir değer, açıklanmadan ÖNCEKİ günlere yazılmaz:
      TÜFE/ÜFE  ay m -> ay m+1'in CPI_RELEASE_DAY'inden (6) itibaren geçerli
      politika  ay m -> ay m+1'in 1'inden itibaren geçerli (BIS aylık seri)
    İkisi de TASARIM VARSAYIMI: TÜİK TÜFE'yi genelde ayın 3'ünde açıklar,
    hafta sonu kaymasına pay bırakılmıştır; resmi takvim kullanılmadı.
  * B1: `bfill` YOK. Serinin ilk değeri geçmişe yayılmaz; zorunlu sütunların
    hepsi dolmadan önceki baş satırlar atılır (`attrs['leading_rows_dropped']`).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import logging
import os
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# EVDS API için
try:
    from evds import evdsAPI
    EVDS_AVAILABLE = True
except ImportError:
    EVDS_AVAILABLE = False
    logging.warning("evds package not installed. Install with: pip install evds")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MacroDataFetcher:
    """TCMB EVDS API ve yfinance ile makroekonomik veri çekici"""

    # EVDS Seri Kodları (2026-11-06'da EVDS katalogundan dogrulandi —
    # get_series('bie_bispolfaiz' / 'bie_apifon'); eski 'TP.YSSK.A01' yok)
    EVDS_CODES = {
        'policy_rate': 'TP.BISPOLFAIZ.TUR',    # Türkiye Politika Faizi (BIS) - Aylık
        # TÜFE: eski 'TP.FG.J0' (2003=100) 2026-01'de BITIYOR (TÜİK 2025=100 bazina
        # gecti) — son cekimde Ocak 2026'dan beri ayni deger tasiniyordu.
        'cpi_index': 'TP.TUKFIY2025.GENEL',    # TÜFE Genel (2025=100), 2005'ten - Aylık
        # ÜFE: eski 'TP.FG.G0' EVDS'de yok (API hata veriyor).
        'ppi_index': 'TP.TUFE1YI.T1',          # Yurt İçi ÜFE Genel - Aylık
        'funding_rate': 'TP.APIFON4',          # TCMB Ağırlıklı Ort. Fonlama Maliyeti - İş günü
    }
    EVDS_MAX_ROWS = 1000      # EVDS tek istekte en fazla bu kadar satir dondurur (aralik SONUNDAN)
    EVDS_CHUNK_DAYS = 730
    CPI_RELEASE_DAY = 6       # B3: ay m TÜFE/ÜFE'si ay m+1'in bu gününden itibaren bilinir
    POLICY_AVAILABLE_DAY = 1  # B3: ay m politika faizi (ay sonu) ay m+1'in bu gününden itibaren
    RISK_FREE_FILE = 'risk_free.csv'

    # yfinance Sembolleri
    YF_SYMBOLS = {
        'usd_try': 'TRY=X',
        'eur_try': 'EURTRY=X',
        'bist100_index': 'XU100.IS',
        # Faz 3.4.3: Global gostergeler
        'vix': '^VIX',          # CBOE Volatility Index (korku endeksi)
        'us10y': '^TNX',        # ABD 10-Yil Hazine Getirisi
        'dxy': 'DX-Y.NYB',     # ABD Dolar Endeksi
        # Faz 4: Global makro genisleme
        'oil_wti': 'CL=F',     # Ham petrol (WTI vadeli)
        'gold_vix': '^GVZ',    # Altin volatilite endeksi
        'silver': 'SI=F',      # Gumus vadeli (altin/gumus orani icin)
    }

    def __init__(self, api_key: Optional[str] = None, start_date: str = "2018-01-01",
                 end_date: Optional[str] = None, strict_data: bool = False):
        """
        Args:
            api_key: TCMB EVDS API anahtarı. None ise EVDS_API_KEY env değişkeninden okunur.
            start_date: Başlangıç tarihi (DD-MM-YYYY veya YYYY-MM-DD)
            end_date: Bitiş tarihi (DD-MM-YYYY veya YYYY-MM-DD), None ise bugün
            strict_data: Faz 6 (G.2/R3) — True ise sabit-değer/zero-fill fallback
                devreye girerse HATA fırlatılır (production eğitimi yanlış makro
                veriyle sessizce çalışmasın). Varsayılan False: fallback kullanılır
                ama data_quality bayrağıyla işaretlenir (mevcut davranış).
        """
        self.strict_data = strict_data
        self.api_key = api_key or os.getenv("EVDS_API_KEY")
        if not self.api_key:
            raise ValueError("EVDS_API_KEY bulunamadı. .env dosyasına ekleyin veya api_key parametresi geçin.")
        self.start_date_evds = self._convert_date_format(start_date, to_evds=True)
        self.end_date_evds = self._convert_date_format(end_date, to_evds=True) if end_date else datetime.now().strftime("%d-%m-%Y")
        
        # yfinance için YYYY-MM-DD formatı
        self.start_date_yf = self._convert_date_format(start_date, to_evds=False)
        self.end_date_yf = self._convert_date_format(end_date, to_evds=False) if end_date else datetime.now().strftime("%Y-%m-%d")

        self.data_dir = "data/macro"

        # EVDS API client oluştur
        if EVDS_AVAILABLE:
            self.evds = evdsAPI(self.api_key)
        else:
            self.evds = None

        # Veri dizini oluştur
        os.makedirs(self.data_dir, exist_ok=True)

        # Faz 6 (G.2): Veri kalite bayraklari — fallback/sabit-deger kullanildiginda
        # isaretlenir. Egitim manifesti bunu kaydeder ki yanlis veriyle egitim
        # sessizce gerceklesmesin (bkz. R3). fetch_macro_data basinda sifirlanir.
        self.data_quality: Dict[str, str] = {}

        logger.info(f"MacroDataFetcher initialized")
        logger.info(f"Date range: {self.start_date_evds} to {self.end_date_evds}")

    def _convert_date_format(self, date_str: Optional[str], to_evds: bool = True) -> Optional[str]:
        """Tarih formatını çevir"""
        if not date_str:
            return None
        
        try:
            # Input formatı tahmin et
            if '-' in date_str:
                parts = date_str.split('-')
                if len(parts[0]) == 4: # YYYY-MM-DD
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                else: # DD-MM-YYYY
                    dt = datetime.strptime(date_str, "%d-%m-%Y")
            else:
                return date_str

            if to_evds:
                return dt.strftime("%d-%m-%Y")
            else:
                return dt.strftime("%Y-%m-%d")
        except Exception as e:
            logger.error(f"Date conversion error: {e}")
            return date_str

    # ------------------------------------------------------------------ #
    # Faz 9 / B2–B3: EVDS ayristirma ve nokta-zamanli hizalama
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_evds_dates(raw: pd.Series) -> pd.DatetimeIndex:
        """EVDS 'Tarih' sutunu: gunluk 'DD-MM-YYYY', aylik 'YYYY-M' / 'YYYY-MM'."""
        s = raw.astype(str).str.strip()
        daily = pd.to_datetime(s, format="%d-%m-%Y", errors='coerce')
        monthly = pd.to_datetime(s, format="%Y-%m", errors='coerce')
        out = daily.fillna(monthly)
        if out.isna().any():
            bad = s[out.isna()].head(3).tolist()
            raise ValueError(f"EVDS tarih bicimi taninmadi: {bad}")
        return pd.DatetimeIndex(out)

    def _evds_series(self, code: str, start_evds: Optional[str] = None,
                     end_evds: Optional[str] = None) -> pd.Series:
        """Tek EVDS serisini tarih indeksli float Series olarak dondur.

        EVDS sutun adini nokta yerine alt cizgiyle dondurur ('TP.FG.J0' ->
        'TP_FG_J0'); eski kod noktali adi aradigi icin seriyi hic bulamiyordu.
        Bos ya da tamamen NaN cevapta hata yukseltir (sessiz bos seri yok).
        """
        if not self.evds:
            raise RuntimeError("evds paketi yok")
        start = datetime.strptime(start_evds or self.start_date_evds, "%d-%m-%Y")
        end = datetime.strptime(end_evds or self.end_date_evds or datetime.now().strftime("%d-%m-%Y"),
                                "%d-%m-%Y")
        # EVDS tek istekte en fazla ~1000 satir veriyor ve fazlasini aralik
        # BASINDAN kesiyor (olculdu: 2013-01..2016-12 istegi 2013-03-04'ten basladi).
        # Aralik parcalara bolunur; tam 1000 satir donen parca ikiye yarilir.
        parts = []
        windows = []
        cur = start
        while cur <= end:
            nxt = min(cur + timedelta(days=self.EVDS_CHUNK_DAYS - 1), end)
            windows.append((cur, nxt))
            cur = nxt + timedelta(days=1)
        while windows:
            a, b = windows.pop(0)
            df = self.evds.get_data([code], startdate=a.strftime("%d-%m-%Y"),
                                    enddate=b.strftime("%d-%m-%Y"))
            if df is None or df.empty:
                continue
            if len(df) >= self.EVDS_MAX_ROWS and (b - a).days > 1:
                mid = a + (b - a) / 2
                windows[:0] = [(a, mid), (mid + timedelta(days=1), b)]
                continue
            col = next((c for c in (code, code.replace('.', '_')) if c in df.columns), None)
            if col is None:
                raise ValueError(f"EVDS cevabinda {code} sutunu yok: {list(df.columns)}")
            parts.append(pd.Series(pd.to_numeric(df[col], errors='coerce').to_numpy(),
                                   index=self._parse_evds_dates(df['Tarih']), name=code))
        if not parts:
            raise ValueError(f"EVDS bos cevap: {code}")
        s = pd.concat(parts)
        s = s[~s.index.duplicated(keep='last')].dropna().sort_index()
        if s.empty:
            raise ValueError(f"EVDS serisi tamamen bos: {code}")
        return s

    @staticmethod
    def _monthly_point_in_time(monthly: pd.Series, available_day: int) -> pd.Series:
        """Ay m degerini ay m+1'in `available_day`'inden itibaren GUNLUK seriye yay.

        Donen seri yalnizca degisim noktalarini tasir; cagiran gunluk
        indekse ffill ile hizalar. Ay m'nin kendi gunlerine ve m+1'in
        `available_day`'inden onceki gunlere ay m degeri YAZILMAZ (B3).
        """
        m = monthly.dropna().copy()
        periods = pd.DatetimeIndex(m.index).to_period('M')
        effective = (periods + 1).to_timestamp() + pd.Timedelta(days=available_day - 1)
        return pd.Series(m.to_numpy(), index=effective, name=monthly.name).sort_index()

    def _evds_start_minus(self, months: int) -> str:
        """YoY hesabi icin baslangictan `months` ay once (EVDS bicimi)."""
        start = datetime.strptime(self.start_date_evds, "%d-%m-%Y")
        return (pd.Timestamp(start) - pd.DateOffset(months=months)).strftime("%d-%m-%Y")

    def fetch_risk_free(self, start_date: str = "2011-01-01", save: bool = True) -> pd.DataFrame:
        """A8 — tek kaynak risksiz faiz dosyasi (data/macro/risk_free.csv).

        funding_rate: TCMB Agirlikli Ortalama Fonlama Maliyeti (TP.APIFON4), yillik %,
            is gunu — `evaluation.risk_free`'nin varsayilan serisi.
        policy_rate:  TP.BISPOLFAIZ.TUR, aylik, B3 kuraliyla gunluge hizali.
        Yedek deger YOK: cekilemezse hata (risksiz faiz uydurulmaz).
        """
        start_evds = self._convert_date_format(start_date, to_evds=True)
        funding = self._evds_series(self.EVDS_CODES['funding_rate'], start_evds=start_evds)
        policy = self._monthly_point_in_time(
            self._evds_series(self.EVDS_CODES['policy_rate'], start_evds=start_evds),
            self.POLICY_AVAILABLE_DAY)
        idx = funding.index.union(policy.index)
        out = pd.DataFrame({
            'funding_rate': funding.reindex(idx),          # yalnizca gozlenen gunler
            'policy_rate': policy.reindex(idx).ffill(),    # adim fonksiyonu
        }).sort_index()
        out = out[out['funding_rate'].notna()]
        out.index.name = 'date'
        out.attrs['source'] = {
            'funding_rate': self.EVDS_CODES['funding_rate'],
            'policy_rate': self.EVDS_CODES['policy_rate'],
            'policy_available_day': self.POLICY_AVAILABLE_DAY,
        }
        if save:
            path = os.path.join(self.data_dir, self.RISK_FREE_FILE)
            out.to_csv(path)
            with open(os.path.splitext(path)[0] + '_source.json', 'w', encoding='utf-8') as f:
                json.dump({'fetched_at': datetime.now().isoformat(), **out.attrs['source'],
                           'rows': int(len(out)), 'start': str(out.index.min().date()),
                           'end': str(out.index.max().date())}, f, indent=2)
            logger.info(f"Risksiz faiz kaydedildi: {path} ({len(out)} satir)")
        return out

    def fetch_macro_data(self, save=True) -> pd.DataFrame:
        """Tüm makro verileri çek ve birleştir"""
        data_dict = {}
        self.data_quality = {}  # Faz 6 (G.2): bu cekimde fallback kullanildi mi

        # 1. yfinance verilerini çek (Döviz + BIST100 + global gostergeler)
        # Faz 6 (1.3): sembolleri paralel çek (I/O-bound -> ThreadPool). Sonuçlar
        # YF_SYMBOLS sırasına göre toplanır — deterministik. EVDS seri kalır
        # (tek endpoint). Log/quality davranışı korunur.
        logger.info("Fetching market data from yfinance...")

        def _fetch_yf(item):
            name, symbol = item
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=self.start_date_yf, end=self.end_date_yf)
                if not hist.empty:
                    from pandas import DatetimeIndex
                    if isinstance(hist.index, DatetimeIndex) and hist.index.tz is not None:
                        hist.index = hist.index.tz_localize(None)
                    return name, hist['Close'], len(hist), None
                return name, None, 0, 'no_data'
            except Exception as e:
                return name, None, 0, str(e)

        items = list(self.YF_SYMBOLS.items())
        yf_results = {}
        if len(items) <= 1:
            for it in items:
                n, s, rows, err = _fetch_yf(it)
                yf_results[n] = (s, rows, err)
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(8, len(items))) as pool:
                for n, s, rows, err in pool.map(_fetch_yf, items):
                    yf_results[n] = (s, rows, err)

        for name, _symbol in items:  # deterministik sıra + log
            series, rows, err = yf_results.get(name, (None, 0, 'missing'))
            if series is not None:
                data_dict[name] = series
                logger.info(f"  ✓ {name}: {rows} rows")
            elif err == 'no_data':
                logger.warning(f"  ✗ {name}: No data found")
            else:
                logger.error(f"Error fetching {name} from yfinance: {err}")

        # 2. EVDS verilerini çek (Enflasyon + Faiz)
        if self.evds:
            logger.info("Fetching economic data from EVDS...")
            
            # 2a. Enflasyon (CPI/PPI) — yillik % degisim, ACIKLANMA tarihine hizali (B3).
            # YoY'nin ilk 12 ayi bos kalmasin diye endeks 13 ay erken cekilir;
            # bos kalan bas, eskiden bfill ile GELECEKTEKI degerle dolduruluyordu (B1).
            for key, out_col in (('cpi_index', 'cpi_inflation'), ('ppi_index', 'ppi_inflation')):
                try:
                    idx_series = self._evds_series(self.EVDS_CODES[key],
                                                   start_evds=self._evds_start_minus(13))
                    yoy = (idx_series.pct_change(12) * 100).dropna()
                    data_dict[out_col] = self._monthly_point_in_time(yoy, self.CPI_RELEASE_DAY)
                    logger.info(f"  ✓ {out_col}: {self.EVDS_CODES[key]} YoY, "
                                f"ay m -> ay m+1'in {self.CPI_RELEASE_DAY}. gunu")
                except Exception as e:
                    logger.error(f"Error fetching {out_col} from EVDS: {e}")

            # 2b. Politika Faizi (Fallback mekanizmalı)
            try:
                # Aylik BIS serisi; ay m (ay sonu) degeri ay m+1'in 1'inden itibaren (B3).
                # 1 ay erken cekilir ki baslangic ayi da dolu olsun.
                pr = self._evds_series(self.EVDS_CODES['policy_rate'],
                                       start_evds=self._evds_start_minus(1))
                data_dict['policy_rate'] = self._monthly_point_in_time(pr, self.POLICY_AVAILABLE_DAY)
                logger.info(f"  ✓ policy_rate: {self.EVDS_CODES['policy_rate']} "
                            f"(son deger {pr.iloc[-1]}, {pr.index[-1].strftime('%Y-%m')})")
            except Exception as e:
                # Faz 6 (G.2): strict_data ise yanlis veriyle egitmektense patla
                if self.strict_data:
                    raise ValueError(
                        f"strict_data: Politika faizi EVDS'ten cekilemedi ({e}) ve "
                        f"sabit-deger (50.0) fallback devre disi. Gercek veriyle egit "
                        f"ya da strict_data=False yap."
                    ) from e
                logger.warning(f"Could not fetch Policy Rate from EVDS ({e}). Using fallback value (50.0).")
                # Fallback: 50.0 sabit değer
                # Ensure dates are not None
                start = self.start_date_yf or "2018-01-01"
                end = self.end_date_yf or datetime.now().strftime("%Y-%m-%d")
                dates = pd.date_range(start=start, end=end, freq='D')
                data_dict['policy_rate'] = pd.Series(50.0, index=dates)
                # Faz 6 (G.2): sabit-deger fallback isaretle — egitim manifestine gecer
                self.data_quality['policy_rate'] = 'fallback_constant_50.0'

        # 3. Verileri Birleştir
        if not data_dict:
            raise ValueError("No macro data could be fetched!")

        # DataFrame oluştur
        macro_df = pd.DataFrame(data_dict)

        # Faz 4: Turetilmis global makro sütunlari
        # Reel faiz = US 10Y nominal getiri - ABD CPI enflasyon (tahmini)
        # CPI olarak TR enflasyon yerine sabit %3 ABD ortalamasi kullaniriz;
        # gerçek ABD CPI FRED API'sinden çekilebilir, simdilik yaklaşik.
        if 'us10y' in macro_df.columns:
            # ABD CPI yillık ~ long-run ortalama 3%; daha iyi bir kaynaktan beslenebilir
            us_cpi_approx = 3.0
            macro_df['us_real_rate'] = macro_df['us10y'] - us_cpi_approx
            logger.info("  ✓ us_real_rate: us10y - 3.0 (US CPI approx)")

        # Altin/Gumus orani
        if 'silver' in macro_df.columns:
            # Altin fiyati cross_asset verisinde degil; makro'da silver var,
            # gold_silver_ratio = GC/SI. GC makroda yok ama approximation olarak
            # dxy ile iliski kurabiliriz. Simdilik silver serisini birakalim,
            # feature_engineer gold/silver oranini hesaplar.
            pass

        # B1: yalnizca ILERI doldurma. Eskiden `.ffill().bfill()` idi; bfill her
        # serinin ilk gozlemini (gelecekteki degeri) baslangica kadar geriye
        # yayiyordu (Faz 9 S2 — varsayilan hatta etkin sizinti).
        macro_df = macro_df.sort_index().ffill()
        # YoY/politika icin erken cekilen aylar pencere disinda kalir
        start_ts = pd.Timestamp(self.start_date_yf) if self.start_date_yf else None
        if start_ts is not None:
            macro_df = macro_df[macro_df.index >= start_ts]
        macro_df = self._drop_leading_gaps(macro_df)

        # Son kontroller
        required_cols = ['policy_rate', 'cpi_inflation', 'ppi_inflation', 'usd_try', 'eur_try', 'bist100_index']
        missing_required = [c for c in required_cols if c not in macro_df.columns]
        # Faz 6 (G.2): strict_data ise zorunlu kolon eksikligini zero-fill'le
        # gizlemek yerine patla.
        if missing_required and self.strict_data:
            raise ValueError(
                f"strict_data: Zorunlu makro kolonlari eksik {missing_required} ve "
                f"zero-fill fallback devre disi."
            )
        for col in required_cols:
            if col not in macro_df.columns:
                logger.warning(f"Missing column {col}, filling with 0")
                macro_df[col] = 0.0
                # Faz 6 (G.2): eksik zorunlu kolon sifirla dolduruldu — isaretle
                self.data_quality[col] = 'fallback_zero'

        logger.info(f"\nMacro Data Summary:")
        logger.info(f"Rows: {len(macro_df)}")
        logger.info(f"Columns: {macro_df.columns.tolist()}")
        if self.data_quality:
            logger.warning(f"⚠ Veri kalite uyarisi (fallback kullanildi): {self.data_quality}")

        # Faz 6 (G.2): kalite bayraklarini DataFrame'e ilistir (downstream/manifest erisimi)
        macro_df.attrs['data_quality'] = dict(self.data_quality)

        if save:
            self.save_data(macro_df, 'macro_data.csv')

        return macro_df

    REQUIRED_COLS = ('policy_rate', 'cpi_inflation', 'ppi_inflation',
                     'usd_try', 'eur_try', 'bist100_index')

    @classmethod
    def _drop_leading_gaps(cls, macro_df: pd.DataFrame) -> pd.DataFrame:
        """B1: mevcut zorunlu sutunlarin HEPSI dolmadan onceki bas satirlari at.

        Doldurma (bfill) yerine atma: baslangictan once bilinmeyen bir deger
        uydurulmaz. Atilan satir sayisi `attrs['leading_rows_dropped']`'ta.
        Zorunlu olmayan sutunlarin (vix, oil...) bas NaN'lari DOKUNULMADAN kalir;
        tuketiciler (feature_engineer dropna, env) bunu zaten ele aliyor.
        """
        present = [c for c in cls.REQUIRED_COLS if c in macro_df.columns]
        if not present or macro_df.empty:
            macro_df.attrs['leading_rows_dropped'] = 0
            return macro_df
        ok = macro_df[present].notna().all(axis=1)
        if not ok.any():
            macro_df.attrs['leading_rows_dropped'] = len(macro_df)
            return macro_df.iloc[0:0]
        first = ok.idxmax()
        out = macro_df[macro_df.index >= first]
        out.attrs['leading_rows_dropped'] = int(len(macro_df) - len(out))
        return out

    @staticmethod
    def _quality_sidecar_path(filepath: str) -> str:
        """CSV'nin yanindaki kalite bayragi dosyasi (Faz 6 · G.2)."""
        return f"{os.path.splitext(filepath)[0]}_quality.json"

    def save_data(self, df: pd.DataFrame, filename: str):
        """Veriyi kaydet"""
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath)
        logger.info(f"Macro data saved to {filepath}")

        # Faz 6 (G.2): kalite bayraklari CSV'de tasinamiyor; yan dosyaya yaz ki
        # sonraki egitimler veriyi cache'ten okudugunda da fallback gorunur
        # kalsin. Temiz cekimde bos dict yazilir -> eski bayrak temizlenir.
        quality = dict(df.attrs.get('data_quality') or {})
        try:
            with open(self._quality_sidecar_path(filepath), 'w', encoding='utf-8') as f:
                json.dump({'fetched_at': datetime.now().isoformat(),
                           'data_quality': quality}, f, indent=2)
        except OSError as e:
            logger.warning(f"Kalite bayragi yazilamadi ({filepath}): {e}")

    def load_data(self, filename: str = 'macro_data.csv') -> pd.DataFrame:
        """Veriyi yükle"""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"Loaded macro data: {len(df)} rows")

        # Faz 6 (G.2): cache'ten okuma yolu da kalite bayragini tasimali —
        # aksi halde fallback'li bir CSV ile egitim yapilir ve manifest bunu
        # 'temiz' gosterir (R3'un tam olarak kacinilmak istenen hali).
        df.attrs['data_quality'] = self._load_quality(filepath)

        if self.strict_data and df.attrs['data_quality']:
            raise ValueError(
                f"strict_data: cache'lenmis makro veri kalite bayragi tasiyor "
                f"{df.attrs['data_quality']} — temiz veri icin yeniden cekin "
                f"(fetch_macro_data) ya da strict_data=False yapin."
            )
        return df

    def _load_quality(self, filepath: str) -> Dict[str, str]:
        """CSV'nin kalite bayraklarini yan dosyadan oku.

        Yan dosya yoksa CSV bu ozellikten once yazilmistir: 'temiz' demek yerine
        bilinmiyor olarak isaretlenir — sessiz yanlis guven olusmasin.
        """
        sidecar = self._quality_sidecar_path(filepath)
        if not os.path.exists(sidecar):
            return {'macro_csv': 'quality_unknown_legacy_file'}
        try:
            with open(sidecar, 'r', encoding='utf-8') as f:
                return dict(json.load(f).get('data_quality') or {})
        except (OSError, json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Kalite bayragi okunamadi ({sidecar}): {e}")
            return {'macro_csv': 'quality_unreadable'}

    # ------------------------------------------------------------------ #
    # Faz 4: Incremental veri akisi
    # ------------------------------------------------------------------ #

    def get_source_status(self, filename: str = 'macro_data.csv') -> dict:
        """Mevcut makro dosyanin tarih durumunu raporla."""
        today = datetime.now().date()
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            return {'exists': False, 'last_date': None, 'missing_days': None,
                    'columns': []}

        try:
            df = self.load_data(filename)
            idx = pd.to_datetime(df.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            last_date = idx.max().date()
            missing = sum(
                1 for i in range(1, (today - last_date).days + 1)
                if (last_date + timedelta(days=i)).weekday() < 5
            )
            return {
                'exists': True,
                'last_date': str(last_date),
                'missing_days': missing,
                'columns': df.columns.tolist(),
            }
        except Exception as exc:
            return {'exists': True, 'last_date': None, 'missing_days': None,
                    'error': str(exc), 'columns': []}

    def fetch_incremental(self, filename: str = 'macro_data.csv') -> dict:
        """Eksik gunleri cek, mevcut makro dosyasina ekle.

        - yfinance kismi: son tarihten bugune incremental
        - EVDS kismi: aylik veri (CPI/PPI) ucuz, faiz gunluk -> tum aralik
          tekrar cekilir; sonuc mevcutla birlestirilir.
        """
        today_str = datetime.now().strftime('%Y-%m-%d')
        filepath = os.path.join(self.data_dir, filename)

        existing: Optional[pd.DataFrame] = None
        if os.path.exists(filepath):
            try:
                existing = self.load_data(filename)
            except Exception as exc:
                logger.warning(f"Mevcut makro yuklenemedi, tam cekme yapilacak: {exc}")

        # Mevcut dosya yedek degerle (sabit 50 / sifir) ya da kalite bayragi
        # bilinmeyen eski bir surumle yazildiysa artimli ekleme sahte gecmisi
        # korurdu: tam cekim yapilir (Faz 9 B2).
        if existing is not None and (existing.attrs.get('data_quality') or {}):
            logger.warning(f"Mevcut makro dosya kalite bayragi tasiyor "
                           f"{existing.attrs.get('data_quality')} — tam cekim yapiliyor")
            existing = None

        if existing is None or existing.empty:
            logger.info("Mevcut macro_data.csv yok, tam cekme yapiliyor...")
            df = self.fetch_macro_data(save=True)
            return {'mode': 'full', 'new_rows': len(df), 'total_rows': len(df),
                    'last_date': str(df.index.max().date())}

        idx = pd.to_datetime(existing.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
            existing.index = idx
        last_date = idx.max().date()
        fetch_from = (datetime.combine(last_date, datetime.min.time())
                      + timedelta(days=1)).strftime('%Y-%m-%d')

        if fetch_from > today_str:
            logger.info("Macro veri guncel, cekme gerekmez")
            return {'mode': 'skip', 'new_rows': 0, 'total_rows': len(existing),
                    'last_date': str(last_date)}

        logger.info(f"Eksik macro verisi cekiliyor: {fetch_from} -> {today_str}")
        delta_fetcher = MacroDataFetcher(
            api_key=self.api_key,
            start_date=fetch_from,
            end_date=today_str,
        )
        try:
            new_data = delta_fetcher.fetch_macro_data(save=False)
        except Exception as exc:
            logger.error(f"Delta makro cekimi basarisiz: {exc}")
            return {'mode': 'error', 'new_rows': 0, 'total_rows': len(existing),
                    'last_date': str(last_date), 'error': str(exc)}

        # Birlestir: yeni satirlar onceligi alir
        combined = pd.concat([existing, new_data])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        # Yalnizca ileri doldurma (B1) — yfinance hafta sonu, EVDS aylik.
        # bfill kaldirildi: yeni cekimin ilk degeri eski satirlara yazilmasin.
        combined = combined.ffill()
        combined.attrs['data_quality'] = {**(existing.attrs.get('data_quality') or {}),
                                          **(new_data.attrs.get('data_quality') or {})}

        self.save_data(combined, filename)
        new_rows = len(combined) - len(existing)
        return {
            'mode': 'incremental',
            'new_rows': max(0, new_rows),
            'total_rows': len(combined),
            'fetch_from': fetch_from,
            'fetch_to': today_str,
            'last_date': str(combined.index.max().date()),
        }


def main():
    """Test script"""
    fetcher = MacroDataFetcher(start_date="2018-01-01")  # api_key .env'den okunur
    try:
        df = fetcher.fetch_macro_data(save=True)
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nLast 5 rows:")
        print(df.tail())
        print("\nStats:")
        print(df.describe())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
