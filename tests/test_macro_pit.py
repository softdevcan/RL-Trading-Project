"""
Faz 9 / B1–B3 + A8 — makro verisinin nokta-zamanli (point-in-time) uretimi.

Ag YOK: sahte bir EVDS istemcisi ve sahte yfinance kullanilir. Sahte EVDS,
gercek API'nin olculen iki davranisini taklit eder (2026-11-06):
  * sutun adi alt cizgili doner ('TP.X.Y' -> 'TP_X_Y'); aylik 'Tarih' 'YYYY-M'
  * tek istekte en fazla 1000 satir, fazlasi aralik BASINDAN kesilir

Kontroller:
  B2  tarih/sutun ayristirma; parcali cekim tam seriyi getiriyor
  B3  ay m TUFE'si ay m+1'in 6'sindan ONCE gorunmuyor; politika ay m+1'in 1'inde
  B1  bfill yok: (a) kesme degismezligi — bitis tarihi T1 ile T2 > T1 icin
      <= T1 satirlari BIT-ES; (b) bas bosluk atiliyor, ilk deger uydurulmuyor
  G2  EVDS hatasinda eski yedek + kalite bayragi; strict_data hata
  inc kalite bayrakli mevcut dosya artimli degil TAM cekimle yenileniyor
  A8  fetch_risk_free -> risk_free.csv -> evaluation.risk_free

Calistirma:
    python tests/test_macro_pit.py
"""

import json
import logging
import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import data.macro_fetcher as mf  # noqa: E402
from data.macro_fetcher import MacroDataFetcher  # noqa: E402

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [OK]   {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


# --------------------------------------------------------------------------
# Sahte kaynaklar
# --------------------------------------------------------------------------
MONTHS = pd.period_range("2010-01", "2026-10", freq="M")
_n = np.arange(len(MONTHS))
# Aylik degisim HER AY farkli (hakem-2 B2 #2: geometrik seride YoY sabitti ve
# B3 testi ileriye bakisi yakalayamiyordu)
TRUTH = {
    "TP.TUKFIY2025.GENEL": pd.Series(100 * np.cumprod(1.02 + 0.015 * np.sin(_n * 1.3)), index=MONTHS),
    "TP.TUFE1YI.T1": pd.Series(100 * np.cumprod(1.015 + 0.02 * np.cos(_n * 0.7)), index=MONTHS),
    "TP.BISPOLFAIZ.TUR": pd.Series(np.where(np.arange(len(MONTHS)) < 150, 10.0, 40.0)
                                   + (np.arange(len(MONTHS)) % 7), index=MONTHS),
}
DAYS = pd.bdate_range("2011-01-03", "2026-11-06")
FUNDING = pd.Series(10 + np.sin(np.arange(len(DAYS)) / 50.0), index=DAYS)


class FakeEVDS:
    def __init__(self, fail=(), today=None):
        self.fail = set(fail)
        self.calls = 0
        # BIS aylik politika serisi ~5 hafta gecikmeli: "bugun"den 65 gunden
        # yeni aylar henuz yayinlanmamis (hakem-2 B2 #1 olcumu: 2026-11-06'da son ay 2026-09)
        self.today = pd.Timestamp(today) if today else None

    def get_data(self, series, startdate, enddate):
        self.calls += 1
        code = series[0]
        if code in self.fail:
            raise RuntimeError("Connection error, please check your API Key or request.")
        a = pd.Timestamp(pd.to_datetime(startdate, format="%d-%m-%Y"))
        b = pd.Timestamp(pd.to_datetime(enddate, format="%d-%m-%Y"))
        if code == "TP.APIFON4":
            s = FUNDING[(FUNDING.index >= a) & (FUNDING.index <= b)]
            tarih = [d.strftime("%d-%m-%Y") for d in s.index]
        else:
            t = TRUTH[code]
            ts = t.index.to_timestamp()
            keep = (ts >= a.replace(day=1)) & (ts <= b)
            if code == "TP.BISPOLFAIZ.TUR" and self.today is not None:
                keep &= ts <= self.today - pd.Timedelta(days=65)
            s = t[keep]
            tarih = [f"{p.year}-{p.month}" for p in s.index]           # 'YYYY-M'
        df = pd.DataFrame({"Tarih": tarih, code.replace(".", "_"): s.to_numpy()})
        return df.tail(MacroDataFetcher.EVDS_MAX_ROWS).reset_index(drop=True)   # BASTAN kes


class FakeTicker:
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, start, end):
        idx = pd.bdate_range(start, pd.Timestamp(end) - pd.Timedelta(days=1))
        base = 1000.0 + (abs(hash(self.symbol)) % 50)
        return pd.DataFrame({"Close": base + np.arange(len(idx), dtype=float)}, index=idx)


mf.yf.Ticker = FakeTicker   # ag yok
# fetch_incremental delta icin yeni bir MacroDataFetcher kurar -> evdsAPI da sahte
FAKE_CFG = {"today": None, "fail": set()}
mf.evdsAPI = lambda key: FakeEVDS(fail=FAKE_CFG["fail"], today=FAKE_CFG["today"])
mf.EVDS_AVAILABLE = True


def make(tmp, start="2018-01-01", end=None, fail=(), strict=False, today=None):
    f = MacroDataFetcher.__new__(MacroDataFetcher)
    f.strict_data = strict
    f.api_key = "x"
    f.start_date_evds = f._convert_date_format(start, to_evds=True)
    f.end_date_evds = f._convert_date_format(end, to_evds=True) if end else "06-11-2026"
    f.start_date_yf = f._convert_date_format(start, to_evds=False)
    f.end_date_yf = f._convert_date_format(end, to_evds=False) if end else "2026-11-06"
    f.data_dir = tmp
    f.evds = FakeEVDS(fail=fail, today=today)
    f.data_quality = {}
    return f


# --------------------------------------------------------------------------
print("\nB2 — EVDS ayristirma")
d = MacroDataFetcher._parse_evds_dates(pd.Series(["2026-1", "2026-10", "05-11-2026"]))
check("aylik 'YYYY-M' ve gunluk 'DD-MM-YYYY' ayristiriliyor",
      list(d) == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-10-01"), pd.Timestamp("2026-11-05")])
try:
    MacroDataFetcher._parse_evds_dates(pd.Series(["2026/01/01"]))
    check("taninmayan bicim hata verir", False)
except ValueError:
    check("taninmayan bicim hata verir", True)

with tempfile.TemporaryDirectory() as tmp:
    f = make(tmp)
    s = f._evds_series("TP.APIFON4", start_evds="01-01-2011", end_evds="06-11-2026")
    check("parcali cekim tum gunluk seriyi getiriyor (1000 satir siniri)",
          s.index.equals(FUNDING.index) and np.allclose(s.to_numpy(), FUNDING.to_numpy()),
          f"{len(s)} / {len(FUNDING)}")
    check("alt cizgili sutun adi bulunuyor", s.name == "TP.APIFON4")
    m = f._evds_series("TP.TUKFIY2025.GENEL", start_evds="01-01-2017", end_evds="06-11-2026")
    check("aylik seri ayin 1'ine yerlesiyor", m.index[0] == pd.Timestamp("2017-01-01"))
    check("eski kodlar artik kullanilmiyor",
          MacroDataFetcher.EVDS_CODES["policy_rate"] != "TP.YSSK.A01"
          and MacroDataFetcher.EVDS_CODES["cpi_index"] != "TP.FG.J0"
          and MacroDataFetcher.EVDS_CODES["ppi_index"] != "TP.FG.G0")


# --------------------------------------------------------------------------
print("\nB3 — aciklanma tarihine hizalama")
monthly = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-01", "2024-02-01"]))
pit = MacroDataFetcher._monthly_point_in_time(monthly, 6)
check("Ocak degeri 6 Subat'tan itibaren", pit.index[0] == pd.Timestamp("2024-02-06"))
check("Subat degeri 6 Mart'tan itibaren", pit.index[1] == pd.Timestamp("2024-03-06"))

with tempfile.TemporaryDirectory() as tmp:
    df = make(tmp).fetch_macro_data(save=False)
    yoy = (TRUTH["TP.TUKFIY2025.GENEL"].pct_change(12) * 100)
    ok_cpi = True
    for p in pd.period_range("2019-01", "2026-09", freq="M"):
        rel = (p + 1).to_timestamp() + pd.Timedelta(days=5)        # ay m+1'in 6'si
        before = df.loc[:rel - pd.Timedelta(days=1), "cpi_inflation"]
        after = df.loc[rel:, "cpi_inflation"]
        if len(before) and np.isclose(before.iloc[-1], yoy[p]) and not np.isclose(yoy[p], yoy[p - 1]):
            ok_cpi = False
        if len(after) and not np.isclose(after.iloc[0], yoy[p]):
            ok_cpi = False
    check("hicbir ayin TUFE'si aciklanmadan once gorunmuyor, sonra gorunuyor", ok_cpi)
    pol = TRUTH["TP.BISPOLFAIZ.TUR"]
    check("politika: Mart degeri 1 Nisan'da, 31 Mart'ta degil",
          np.isclose(df.loc["2024-04-01", "policy_rate"], pol[pd.Period("2024-03", "M")])
          and np.isclose(df.loc["2024-03-29", "policy_rate"], pol[pd.Period("2024-02", "M")]))
    check("temiz cekimde kalite bayragi yok", df.attrs.get("data_quality") == {})


# --------------------------------------------------------------------------
print("\nB1 — bfill yok (kesme degismezligi)")
with tempfile.TemporaryDirectory() as tmp:
    full = make(tmp, end="2026-11-06").fetch_macro_data(save=False)
    cut = make(tmp, end="2023-06-30").fetch_macro_data(save=False)
    common = cut.index[cut.index <= pd.Timestamp("2023-06-29")]
    a = full.loc[common, list(MacroDataFetcher.REQUIRED_COLS)]
    b = cut.loc[common, list(MacroDataFetcher.REQUIRED_COLS)]
    check("T1 = 2023-06-30 ile T2 = 2026-11-06: ortak satirlar bit-es",
          a.equals(b), str((a - b).abs().max()))
    check("bitis tarihinden sonraya satir yazilmiyor (hakem-2 B2 #4)",
          cut.index.max() <= pd.Timestamp("2023-06-30"), str(cut.index.max()))
    first = full.index.min()
    check("bas bosluk atildi: ilk satirda zorunlu sutunlarin hepsi dolu",
          full.loc[first, list(MacroDataFetcher.REQUIRED_COLS)].notna().all())
    check("14 ay erken cekim: baslangic gunu atilmiyor (2018-01-01 dolu)",
          first == pd.Timestamp("2018-01-01"), str(first))

    # Dogrudan bfill kontrolu (hakem-2 B2 #3 — kesme degismezligi bfill'i
    # yakalamaz): ilk satirdaki deger, baslangictan ONCE bilinen son gozlem
    # olmali; baslangictan sonra gelen ilk gozlem degil.
    yoy_true = TRUTH["TP.TUKFIY2025.GENEL"].pct_change(12) * 100
    known_at_start = yoy_true[pd.Period("2017-11", "M")]    # 2017-12-06'dan beri gecerli
    first_after = yoy_true[pd.Period("2017-12", "M")]       # 2018-01-06'da gelecek
    check("ilk satirin TUFE'si baslangictan once bilinen deger (gelecekteki degil)",
          np.isclose(full.iloc[0]["cpi_inflation"], known_at_start)
          and not np.isclose(full.iloc[0]["cpi_inflation"], first_after))

    # Oz-test: kasitli mutasyonlar yukaridaki kontrolleri KIRMALI
    orig_pit = MacroDataFetcher.__dict__["_monthly_point_in_time"]     # staticmethod nesnesi
    p = pd.Period("2022-03", "M")
    mid = p.to_timestamp() + pd.Timedelta(days=14)                     # ay m'nin 15'i
    honest = np.isclose(full.loc[:mid, "cpi_inflation"].iloc[-1], yoy_true[p])
    try:
        MacroDataFetcher._monthly_point_in_time = staticmethod(        # ay m'yi ay m'nin 1'ine koy
            lambda m, day: pd.Series(m.dropna().to_numpy(),
                                     index=pd.DatetimeIndex(m.dropna().index), name=m.name))
        leak = make(tmp).fetch_macro_data(save=False)
        leaked = np.isclose(leak.loc[:mid, "cpi_inflation"].iloc[-1], yoy_true[p])
    finally:
        MacroDataFetcher._monthly_point_in_time = orig_pit
    check("oz-test: 'ay m degeri ay m'nin 1'inde' mutasyonu ayin 15'inde ay m'yi gosterir, "
          "dogru hat gostermez", leaked and not honest, f"leaked={leaked} honest={honest}")
    check("ilk satir baslangictan once degil", first >= pd.Timestamp("2018-01-01"))
    check("atilan bas satir sayisi kayitli", full.attrs.get("leading_rows_dropped", -1) >= 0)
    code_lines = [ln.split("#", 1)[0] for ln in open(mf.__file__, encoding="utf-8")]
    check("macro_fetcher'da bfill() cagrisi kalmadi (yorumlar haric)",
          not any(".bfill(" in ln and "`" not in ln for ln in code_lines))


# --------------------------------------------------------------------------
print("\nG.2 — hata yolu")
with tempfile.TemporaryDirectory() as tmp:
    df = make(tmp, fail={"TP.BISPOLFAIZ.TUR"}).fetch_macro_data(save=False)
    check("politika cekilemezse yedek + bayrak", df.attrs["data_quality"].get("policy_rate") == "fallback_constant_50.0")
    try:
        make(tmp, fail={"TP.BISPOLFAIZ.TUR"}, strict=True).fetch_macro_data(save=False)
        check("strict_data hata verir", False)
    except ValueError:
        check("strict_data hata verir", True)

    f = make(tmp, fail={"TP.BISPOLFAIZ.TUR"})
    f.fetch_macro_data(save=True)                       # bayrakli dosya
    f2 = make(tmp, end="2026-11-06")
    res = f2.fetch_incremental()
    check("bayrakli mevcut dosya TAM cekimle yenileniyor", res["mode"] == "full", str(res))
    q = json.load(open(os.path.join(tmp, "macro_data_quality.json"), encoding="utf-8"))
    check("yenilenen dosyada kalite bayragi temiz", q["data_quality"] == {}, str(q))

with tempfile.TemporaryDirectory() as tmp:
    # hakem-2 B2 #1: BIS gecikmesiyle kisa artimli pencere sabit 50'ye dusuyordu
    make(tmp, end="2026-10-31", today="2026-10-31").fetch_macro_data(save=True)
    inc = make(tmp, end="2026-11-06", today="2026-11-06")
    FAKE_CFG.update(today="2026-11-06", fail=set())
    res = inc.fetch_incremental()
    after = pd.read_csv(os.path.join(tmp, "macro_data.csv"), index_col=0, parse_dates=True)
    q = json.load(open(os.path.join(tmp, "macro_data_quality.json"), encoding="utf-8"))
    check("BIS gecikmesi altinda artimli cekim yedek kullanmiyor",
          res["mode"] == "incremental" and q["data_quality"] == {}, str(res))
    check("artimli cekimde politika faizi 50'ye dusmedi", not (after["policy_rate"] == 50.0).any())
    check("artimli cekim gun kaybetmiyor (11-02..11-06 is gunleri var)",
          all(pd.Timestamp(d) in after.index for d in pd.bdate_range("2026-11-02", "2026-11-06")))
    # mevcut dosyayi bir gun geri al ki artimli cekim yapilsin, sonra EVDS'i dusur
    trimmed = after[after.index < pd.Timestamp("2026-11-06")]
    trimmed.to_csv(os.path.join(tmp, "macro_data.csv"))
    after = trimmed
    FAKE_CFG.update(fail={"TP.BISPOLFAIZ.TUR"})
    res_bad = make(tmp).fetch_incremental()
    FAKE_CFG.update(today=None, fail=set())
    after2 = pd.read_csv(os.path.join(tmp, "macro_data.csv"), index_col=0, parse_dates=True)
    check("yedek degerli delta dosyaya yazilmiyor", res_bad["mode"] == "error"
          and after2.index.max() == after.index.max(), str(res_bad.get("mode")))


# --------------------------------------------------------------------------
print("\nA8 — risksiz faiz dosyasi")
from evaluation import risk_free as rfm  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    f = make(tmp)
    rf = f.fetch_risk_free(start_date="2011-01-01", save=True)
    path = os.path.join(tmp, MacroDataFetcher.RISK_FREE_FILE)
    check("risk_free.csv yazildi, gunluk seri tam", os.path.exists(path) and len(rf) == len(FUNDING))
    check("kaynak dosyasi seri kodlarini tasiyor",
          json.load(open(os.path.splitext(path)[0] + "_source.json"))["funding_rate"] == "TP.APIFON4")
    sess = pd.bdate_range("2014-01-06", periods=6)
    series = rfm.risk_free_series(sess, csv_path=path)
    exp = (1 + FUNDING.loc[sess[1]] / 100) ** (1 / 365) - 1      # Sali: Pazartesi orani, 1 gun
    check("donem basi oran t-1'deki fonlama maliyetinden", np.isclose(series.iloc[2], exp)
          if False else np.isclose(series.loc[sess[2]], exp), f"{series.loc[sess[2]]} vs {exp}")
    check("varsayilan sutun funding_rate", rfm.DEFAULT_COLUMN == "funding_rate")
    f_bad = make(tmp, fail={"TP.APIFON4"})
    try:
        f_bad.fetch_risk_free(save=False)
        check("fonlama serisi yoksa hata (yedek yok)", False)
    except Exception:
        check("fonlama serisi yoksa hata (yedek yok)", True)

print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
