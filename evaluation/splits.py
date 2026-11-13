"""Faz 9 / R1 — RL walk-forward bolme tanimi (TEK KAYNAK).

Neden (Faz 9 §9.2 #3, Faz 10 Y17 / Y21 / K20): egitim rotasi paneli ORANLA
(70/15/15) boluyordu — panel uzadikca sinirlar kayiyor, dogrulama bolumu hic
kullanilmiyor, HPO ise pencerelerini istekten aliyordu (panoda dogrulama
penceresi CSV'nin sonuna, yani rotanin TEST penceresine kadar uzaniyordu).
Burada bolme sabit TAKVIM TARIHLI ve surumludur; rota ve HPO pencereyi
yalnizca buradan alir. Tanimin ozeti (`digest`) meta dosyasina, manifeste ve
deneme kaydina girer: iki kaydin ayni bolmeyle uretilip uretilmedigi olculur.

Sema (tasarim varsayimlari; Faz 9 §9.11):
  * genisleyen pencere: her katin egitimi RL veri penceresinin basindan (K17)
  * kat = egitim | dogrulama | test, uc ardisik blok; aralilar YARI ACIK
    [baslangic, bitis)
  * test blogu bir takvim yili, dogrulama bir onceki takvim yili
  * purge: `purge_sessions` blok SONUNDAN atilan seans sayisi. Eski ortamda
    odul ufku bir adim ve ortam kendi penceresinin son gununde bitiyor;
    egitimin son gecisi dogrulamanin ilk gunune degmiyor -> 0. Bu VARSAYILMAZ:
    `tests/test_rl_walk_forward.py` kat sinirinda kesme degismezligini olcer;
    ufuk buyurse (Faz 10 K12, t+1 acilis) o test kirilir ve deger yukseltilir.
  * embargo yok: walk-forward'da egitim hicbir zaman degerlendirmeden sonra
    gelmez.

Bir katin test blogu panelde TAMAMLANMAMISSA (panelde blok bitisine esit ya
da sonraki bir tarih yoksa) `SplitCoverageError` — kisa test sessizce kabul
edilmez.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


# Panelin ilk tarihi egitim baslangicindan en fazla bu kadar sonra olabilir (baslangic
# tatil/hafta sonuna denk gelebilir; RL penceresi 2013-06-01 Cumartesi, ilk seans 06-03).
START_TOLERANCE_DAYS = 10


class SplitCoverageError(ValueError):
    """Panel istenen kati kapsamiyor (bos blok ya da tamamlanmamis test)."""


@dataclass(frozen=True)
class Fold:
    """Bir walk-forward kati. Tum sinirlar 'YYYY-MM-DD'; bitisler HARIC."""
    name: str
    train_start: str
    val_start: str      # egitim bitisi (haric) = dogrulama baslangici
    test_start: str     # dogrulama bitisi (haric) = test baslangici
    test_end: str       # test bitisi (haric)

    def __post_init__(self):
        t = [pd.Timestamp(x) for x in (self.train_start, self.val_start,
                                       self.test_start, self.test_end)]
        if not (t[0] < t[1] < t[2] < t[3]):
            raise ValueError(f"Kat '{self.name}': sinirlar artan sirada olmali "
                             f"({self.train_start} < {self.val_start} < "
                             f"{self.test_start} < {self.test_end})")


@dataclass(frozen=True)
class SplitSpec:
    name: str
    folds: Tuple[Fold, ...]
    purge_sessions: int = 0
    # False yalniz `ratio_spec` icin: test blogu panelin son gununde biter
    require_complete: bool = True

    def __post_init__(self):
        if self.purge_sessions < 0:
            raise ValueError("purge_sessions negatif olamaz")
        names = [f.name for f in self.folds]
        if not names or len(set(names)) != len(names):
            raise ValueError(f"Kat adlari bos olmamali ve tekil olmali: {names}")

    def fold(self, name) -> Fold:
        key = str(name)
        for f in self.folds:
            if f.name == key:
                return f
        raise KeyError(f"'{self.name}' bolmesinde '{key}' kati yok; "
                       f"katlar: {[f.name for f in self.folds]}")

    def fold_names(self) -> List[str]:
        return [f.name for f in self.folds]

    def to_dict(self) -> dict:
        return {'name': self.name, 'purge_sessions': int(self.purge_sessions),
                'require_complete': bool(self.require_complete),
                'folds': [asdict(f) for f in self.folds]}

    def digest(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


def _calendar_year_folds(train_start: str, test_years, last_test_end: Optional[str] = None
                         ) -> Tuple[Fold, ...]:
    folds = []
    years = list(test_years)
    for i, y in enumerate(years):
        end = f"{y + 1}-01-01"
        if last_test_end is not None and i == len(years) - 1:
            end = last_test_end
        folds.append(Fold(name=str(y), train_start=train_start,
                          val_start=f"{y - 1}-01-01", test_start=f"{y}-01-01", test_end=end))
    return tuple(folds)


# wf-v1 (2026-11-10, kullanici onayi): 6 kat, test = takvim yili 2021..2026,
# dogrulama = bir onceki yil, egitim 2013-06'dan (K17) genisleyen. 2026 kati
# 2026-11-01'de biter (tanim tarihinde tamamlanmis son ay). Tanim DEGISMEZ:
# yeni veriyle uzatmak yeni bir surumdur (wf-v2), cunku ozet degisir ve eski
# kayitlarla karistirilamaz.
WF_V1 = SplitSpec(
    name='wf-v1',
    folds=_calendar_year_folds('2013-06-01', range(2021, 2027), last_test_end='2026-11-01'),
    purge_sessions=0,
)

SPECS: Dict[str, SplitSpec] = {WF_V1.name: WF_V1}
DEFAULT_SPEC_NAME = WF_V1.name
# Tez kaydi sayilan tanimlarin ozetleri. Kayitta `split_from_spec` yalniz bunlardan
# biri kullanildiysa True olur; `register_spec` ile eklenen gecici tanim (test,
# olcum betigi) acik kalir.
CANONICAL_DIGESTS = {WF_V1.digest()}


def is_canonical(spec: SplitSpec) -> bool:
    return spec.digest() in CANONICAL_DIGESTS


def register_spec(spec: SplitSpec) -> SplitSpec:
    """Gecici bir tanimi ada bagla (test ve olcum betikleri). Kanonik YAPMAZ."""
    if spec.name in SPECS and SPECS[spec.name].digest() != spec.digest():
        raise ValueError(f"'{spec.name}' adi baska bir tanima bagli; tanim degistirilemez")
    SPECS[spec.name] = spec
    return spec


def ratio_spec(dates, name: str, train_ratio: float = 0.7, val_ratio: float = 0.15) -> SplitSpec:
    """`DataFetcher.split_data` ile AYNI sinirlari veren tek katli tanim.

    Yalniz test ve eski-yeni kiyas betikleri icin: sinirlar verilen tarih
    listesinin oranindan cikar (panel uzayinca kayar — Y17'nin ta kendisi) ve
    kanonik degildir; bununla uretilen kayit `split_from_spec=False` tasir.
    """
    d = sorted(pd.unique(pd.DatetimeIndex(pd.to_datetime(list(dates)))))
    n = len(d)
    tr, va = int(n * train_ratio), int(n * (train_ratio + val_ratio))
    if not (0 < tr < va < n):
        raise ValueError(f"Oranli bolme icin yeterli tarih yok (n={n})")
    day = lambda x: str(pd.Timestamp(x).date())
    fold = Fold(name='ratio', train_start=day(d[0]), val_start=day(d[tr]), test_start=day(d[va]),
                test_end=day(pd.Timestamp(d[-1]) + pd.Timedelta(days=1)))
    return SplitSpec(name=name, folds=(fold,), purge_sessions=0, require_complete=False)


def get_spec(name: Optional[str] = None) -> SplitSpec:
    key = name or DEFAULT_SPEC_NAME
    if key not in SPECS:
        raise KeyError(f"Bilinmeyen bolme tanimi '{key}'; tanimlilar: {sorted(SPECS)}")
    return SPECS[key]


def _naive_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    d = pd.DatetimeIndex(pd.to_datetime(df.index.get_level_values('date')))
    return d.tz_localize(None) if d.tz is not None else d


def _range(dates) -> Optional[List[str]]:
    return [str(pd.Timestamp(dates[0]).date()), str(pd.Timestamp(dates[-1]).date())] \
        if len(dates) else None


def fold_complete(df: pd.DataFrame, fold: Fold, spec: Optional[SplitSpec] = None) -> bool:
    """Katin test blogu panelde tamamlanmis mi (blok bitisine esit/sonraki tarih var mi)."""
    d = _naive_dates(df)
    if not len(d):
        return False
    if spec is not None and not spec.require_complete:
        return d.max() >= pd.Timestamp(fold.test_start)
    return d.max() >= pd.Timestamp(fold.test_end)


def fold_coverage(df: pd.DataFrame, spec: Optional[SplitSpec] = None) -> List[dict]:
    """Panelin tanimdaki her kati TARIH olarak kapsayip kapsamadigi (pano ve veri uclari).

    `complete` = egitim baslangici panelde (`fold_frames` ile ayni tolerans) VE test
    blogu tamamlanmis. Yalniz test bitisine bakmak 2018'de baslayan bir paneli her
    katta "tamam" gosterirdi, oysa hicbir kat kurulamaz. Veri kapisinin sonucu (hayalet
    gun, artefakt) ve sembol evreni burada olculmez; onlar kat kurulurken raporlanir.
    """
    spec = spec or get_spec()
    d = _naive_dates(df)
    out = []
    for f in spec.folds:
        starts_ok = bool(len(d)) and bool(
            d.min() <= pd.Timestamp(f.train_start) + pd.Timedelta(days=START_TOLERANCE_DAYS))
        ends_ok = fold_complete(df, f, spec)
        out.append({'name': f.name, 'train': [f.train_start, f.val_start],
                    'val': [f.val_start, f.test_start], 'test': [f.test_start, f.test_end],
                    'starts_ok': starts_ok, 'ends_ok': ends_ok,
                    'complete': starts_ok and ends_ok})
    return out


def latest_complete_fold(df: pd.DataFrame, spec: SplitSpec) -> Fold:
    """Panelin TAMAMEN kapsadigi en son kat (rota varsayilani)."""
    done = [f for f in spec.folds if fold_complete(df, f, spec)]
    if not done:
        raise SplitCoverageError(
            f"Panel '{spec.name}' bolmesinin hicbir katinin test blogunu tamamlamiyor "
            f"(panel sonu {_naive_dates(df).max().date() if len(df) else None})")
    return done[-1]


def fold_frames(df: pd.DataFrame, spec: SplitSpec, fold) -> dict:
    """Paneli katin egitim / dogrulama / test bloklarina bol.

    Args:
        df: MultiIndex (symbol, date) paneli (veri kapisindan gecmis).
        spec: bolme tanimi.
        fold: kat adi ya da `Fold`.

    Returns:
        {'train', 'val', 'test': DataFrame, 'report': dict}. Rapor blok
        araliklarini, seans sayilarini ve purge ile atilan tarihleri tasir.

    Raises:
        SplitCoverageError: panel egitim baslangicini kapsamiyor, bir blok bos
            ya da test blogu tamamlanmamis.
    """
    f = fold if isinstance(fold, Fold) else spec.fold(fold)
    d = _naive_dates(df)
    if not len(d):
        raise SplitCoverageError("Panel bos")
    t0, tv, tt, te = (pd.Timestamp(x) for x in (f.train_start, f.val_start, f.test_start, f.test_end))
    if spec.require_complete and d.max() < te:
        raise SplitCoverageError(
            f"Kat '{f.name}': test blogu {f.test_start}..{f.test_end} (haric) panelde "
            f"tamamlanmamis — panel sonu {d.max().date()}. Veriyi guncelleyin; kisa test "
            f"sessizce kabul edilmez.")
    if d.min() > t0 + pd.Timedelta(days=START_TOLERANCE_DAYS):
        raise SplitCoverageError(
            f"Kat '{f.name}': egitim {f.train_start}'de baslamali, panel {d.min().date()}'de "
            f"basliyor (RL veri penceresi bolme tanimiyla uyusmuyor).")

    masks = {'train': (d >= t0) & (d < tv), 'val': (d >= tv) & (d < tt), 'test': (d >= tt) & (d < te)}
    purged: Dict[str, List[str]] = {}
    p = int(spec.purge_sessions)
    if p > 0:
        for role in ('train', 'val'):
            sessions = sorted(pd.unique(d[masks[role]]))
            drop = sessions[-p:]
            masks[role] = masks[role] & ~d.isin(drop)
            purged[role] = [str(pd.Timestamp(x).date()) for x in drop]

    out: dict = {}
    report = {'spec': spec.name, 'spec_digest': spec.digest(), 'fold': f.name,
              'purge_sessions': p, 'purged_dates': purged, 'ranges': {}, 'n_sessions': {}}
    for role, mask in masks.items():
        part = df[mask]
        sessions = sorted(pd.unique(d[mask]))
        if not len(sessions):
            raise SplitCoverageError(f"Kat '{f.name}': {role} blogu bos")
        out[role] = part
        report['ranges'][role] = _range(sessions)
        report['n_sessions'][role] = int(len(sessions))
    out['report'] = report
    return out


def register_ratio_spec(dates, train_ratio: float = 0.7, val_ratio: float = 0.15) -> SplitSpec:
    """`ratio_spec`'i icerigine gore adlandirip kaydet (test ve olcum betikleri)."""
    d = sorted(pd.unique(pd.DatetimeIndex(pd.to_datetime(list(dates)))))
    key = hashlib.sha256(f"{d[0]}|{d[-1]}|{len(d)}|{train_ratio}|{val_ratio}".encode()).hexdigest()[:8]
    return register_spec(ratio_spec(d, f"ratio-{key}", train_ratio, val_ratio))
