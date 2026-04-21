# Veri Yönetimi Arayüzü — Geliştirme Özeti

**Tarih:** 25 Mart 2026  
**Kapsam:** `dashboard/pages/data.py` · `app/api/routes/trading.py` · `dashboard/api_client.py` · `dashboard/assets/custom.css`

---

## 1. Sorun Tespiti

Eski veri sayfası (`/dash/data`) aşağıdaki eksikliklere sahipti:

- Altın verisi indirilemiyordu
- Veri kaynakları seçimi yoktu (hepsi ya indirilir ya indirilmezdi)
- Artımlı (incremental) güncelleme desteği yoktu — her seferinde tüm veri yeniden çekiliyordu
- Yeni tasarım tarayıcıda görünmüyordu (Dash callback kayıt sorunu)
- `dbc.Checkbox` eski DBC versiyonlarında mevcut değildi → sayfa render edilemiyordu
- Tema sorunları: `dcc.DatePickerRange`, `dcc.Dropdown` ve çeşitli input'lar beyaz arka plan / gri metin gösteriyordu

---

## 2. Backend Değişiklikleri

### `app/api/routes/trading.py`

#### Yeni endpoint: `GET /api/trading/data/status`
Her veri kaynağının mevcut durumunu döndürür:
```json
{
  "status": {
    "bist_stocks": {
      "exists": true,
      "file": "raw_stock_data.csv",
      "last_date": "2026-03-24",
      "missing_days": 0,
      "symbols": ["AKBNK.IS", "THYAO.IS", ...],
      "label": "BIST-30 Hisseleri"
    },
    "gold": { ... },
    "macro": { ... }
  }
}
```

#### Yeni endpoint: `POST /api/trading/data/update`
Seçili kaynakları artımlı ya da tam olarak günceller.

**Request modeli (`DataUpdateRequest`):**
```python
class DataUpdateRequest(BaseModel):
    sources: List[str]            # ["bist_stocks", "gold", "macro"]
    mode: str = "incremental"     # "incremental" | "full"
    start_date: Optional[str] = None
    symbols: Optional[List[str]] = None  # BIST hisse listesi (None → tüm BIST-30)
```

- `symbols` gönderilmezse varsayılan olarak tüm BIST-30 (`get_symbols(phase=2)`) kullanılır
- Artımlı modda son kayıtlı tarihten itibaren yalnızca eksik günler çekilir

---

## 3. Frontend Değişiklikleri

### `dashboard/pages/data.py` — Tam yeniden tasarım

#### Layout değişiklikleri
- **`dbc.Checkbox`** kaldırıldı → **`dbc.Checklist` (switch=True)** ile değiştirildi (tüm DBC versiyonlarında çalışır)
- Kaynak seçim toggle'ları ve durum badge'leri yan yana iki sütunda gösterilir
- **BIST hisse seçim paneli** (`dbc.Collapse`): `bist_stocks` toggle'ı açıkken görünür, kapatınca gizlenir
  - Her hisse küçük **chip/pill** görünümünde (`bist-chip-list` CSS sınıfı)
  - "Tümü", "Phase-1 (5 hisse)", "Temizle" hızlı seçim butonları
  - `X / 31 seçili` sayacı
- **"Veri Kaynağı Özeti"** paneli:
  - `api.get_data_info()` çağrısı kaldırıldı
  - `data-status-store`'dan besleniyor → kaynak seçiminden **bağımsız**, her zaman tüm kaynakları gösterir
  - Her kaynak için: dosya adı, son tarih, durum badge'i, sembol sayısı

#### Callback mimarisi
| Callback | Tetikleyici | Çıktı |
|---|---|---|
| `refresh_status` | Sayfa yükü + "Durumu Yenile" butonu | Status store, badge'ler, checklist değerleri, dosya listesi |
| `update_info_panel` | Status store değişimi | Kaynak özet paneli |
| `toggle_bist_collapse` | Checklist değeri | BIST collapse açık/kapalı |
| `update_bist_count` | BIST sembol listesi | Sayaç metni |
| `bist_quick_select` | Tümü/Phase-1/Temizle butonları | BIST sembol listesi |
| `incremental_update` | Güncelle butonu | Sonuç mesajı |
| `full_download` | Yeniden İndir butonu | Sonuç mesajı |

#### Önemli düzeltme: `allow_duplicate` sorunu
Üç ayrı callback aynı `data-action-result` output'una yazıyordu. İlk callback (`incremental_update`) birincil output olarak tanımlandı, diğerleri `allow_duplicate=True` ile.

### `dashboard/api_client.py`

Eklenen fonksiyonlar:
```python
def get_data_status() -> Dict      # GET /api/trading/data/status
def update_data(payload: Dict) -> Dict  # POST /api/trading/data/update
```

HTTP hataları `logging.WARNING` seviyesinde loglanıyor — sessiz başarısızlık önlendi.

---

## 4. Tema Düzeltmeleri

### `dashboard/assets/custom.css` — Kapsamlı dark tema

#### `dcc.DatePickerRange`
Yeni Dash 2.x sürümünde `react-dates` yerine tamamen farklı sınıf adları kullanılıyor:

| Bileşen | CSS Sınıfı |
|---|---|
| Input wrapper | `.dash-datepicker-input-wrapper` |
| Metin inputları | `.dash-datepicker-input` |
| Ok ikonu | `.dash-datepicker-range-arrow` |
| Caret ikonu | `.dash-datepicker-caret-icon` |
| Takvim popup | `[data-radix-popper-content-wrapper]` (Radix UI) |

Eski `DateInput_*`, `DateRangePicker_*`, `CalendarDay_*` sınıfları artık geçersiz.

`-webkit-text-fill-color` özelliği eklendi — Chrome'da `color` override'ını etkisiz kılan sorun bu şekilde çözüldü.

#### `dcc.Dropdown`
Hem eski (`Select-*`) hem yeni (`[class*="-control"]` vb.) react-select sürümleri için override yazıldı.

#### Genel iyileştirmeler
- CSS değişkenleri (`--card`, `--blue-d` vb.) ile merkezi renk yönetimi
- Alert bileşenlerinde yarı saydam koyu arka plan (beyaz alert sorunu giderildi)
- Switch/Checklist için doğru checked durumu rengi

#### BIST chip selector
```css
.bist-chip-list .form-check-input { opacity: 0; /* ham checkbox gizlendi */ }
.bist-chip-list .form-check-label { /* pill görünümü */ }
.bist-chip-list .form-check-input:checked + .form-check-label {
  background-color: var(--blue-d); /* seçili = mavi dolu */
}
```

---

## 5. Bilinen Sorunlar / Notlar

- **EVDS API anahtarı** `.env` dosyasına `EVDS_API_KEY=<key>` olarak eklenmelidir. `app/core/config.py`'de `Settings` sınıfına bu alan eklendi.
- **CORS ayarları** `.env`'de JSON array formatında olmalıdır: `CORS_ORIGINS=["*"]`
- **Tarayıcı cache:** Sunucu yeniden başladıktan sonra **Ctrl+Shift+R** (hard refresh) zorunludur — Dash'ın JavaScript state'i sıfırlanır.
- Yeni endpoint'ler `suppress_callback_exceptions=True` gerektirir (zaten `app.py`'de mevcut).

---

## 6. Dosya Listesi

```
dashboard/
├── pages/
│   └── data.py          ← Tam yeniden tasarım
├── api_client.py        ← get_data_status, update_data eklendi
└── assets/
    └── custom.css       ← Kapsamlı dark tema overrides

app/api/routes/
└── trading.py           ← /data/status, /data/update endpoint'leri
                            DataUpdateRequest.symbols alanı
```
