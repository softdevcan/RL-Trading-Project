# Faz 8 — UI/UX: Aydınlık Tema + Bileşen Okunabilirliği

**Durum:** ✅ Uygulandı · **Tarih:** 2026-08-30

> Aşağıdaki plan uygulandı. Uygulama sırasında planın dışına çıkılan üç yer
> ve canlıda yakalanan üç kusur **"Uygulama notları"** başlığında (belgenin
> sonunda) kayıtlı — plan metni tasarım kararlarını, o bölüm gerçekte ne
> olduğunu anlatıyor.
>
> Doğrulama (tümü, son durum): `test_theme_contrast` 92/92,
> `test_account_profile` 84/84, `test_topbar` 39/39,
> `test_theme_preference` 31/31, `test_auth` 28/28,
> `test_workspace_isolation` 18/18.
>
> **Görsel kontrol iki ayrı turda yapıldı** — kapsamları farklı:
> Faz A–E kapanışında 10 sayfa × 2 tema; Faz F–G kapanışında 9 sayfa × 2 tema
> (headless Chrome/CDP, ekran görüntüsü + hesaplanmış stil ölçümü). İkinci tur
> üç kusur çıkardı, "G.6 — Görsel doğrulama" başlığında.
>
> **Faz F (2026-08-30)** — Hesabım sayfası panoda *bulunamıyordu*; kenar
> çubuğundaki tek giriş noktası düz metin görünümündeki bir addı. O tur
> "Faz F — Profil sayfası" başlığında.
>
> **Faz G (2026-08-30)** — Üst çubuk: kırıntı, bağlamsal eylem, sayfa araması,
> taşınan görünüm anahtarı ve bildirim zili. "Faz G — Üst çubuk" başlığında.

Amaç: panoya çalışır bir **aydınlık tema** kazandırmak ve bileşenleri
sadelik/okunabilirlik yönünde profesyonelleştirmek. Referans olarak kullanıcının
paylaştığı portföy uygulaması ekranı alındı — **kopyalanmıyor**, yalnızca yön
veriyor: yüksek beyaz alan, ince kenarlıklar, düşük renk gürültüsü, tek bir
vurgu rengi, kompakt filtre satırı, yumuşak gölge yerine sınır çizgisi.

---

## Context — ölçülen mevcut durum

| Bulgu | Değer | Sonuç |
|---|---|---|
| Bootstrap teması | `dbc.themes.DARKLY` (`dashboard/app.py:66`) | Derlenmiş **koyu** CSS; aydınlık tema fiziksel olarak mümkün değil |
| `custom.css` | 480 satır, neredeyse tamamı `!important` | `!important` yığınının sebebi DARKLY ile çekişme |
| Inline stile gömülü tema sabiti | **~550 kullanım** (`TEXT_MUTED` 131, `TEXT` 97, `CARD2` 75, `CARD` 69, aksanlar ~135) | Python'da hex'e çevrilip HTML `style=` içine yazılıyor → çalışma anında tema değişimine tepki veremez |
| Plotly | 63 `apply_dark_template`/`empty_figure` çağrısı, 14 `dcc.Graph`, ~28 satırda doğrudan hex aksan | Grafikler `var()` kabul etmez; ayrı palet gerekir |
| Tema tanımının kopya sayısı | **3** — `dashboard/theme.py`, `assets/custom.css` `:root`, `app/auth/templates/*.html` `:root` | Tek kaynak yok; renk değişimi 3 yerde elle |
| Sayfa başlığı kalıbı | 9 sayfada aynı `H4 + P` bloğu elle tekrar | Bileşenleşmemiş |

### Zaten var olan kontrast hatası (ölçüldü)

Mevcut koyu temanın aksan renkleri kart zemininde (`#1e293b`) WCAG AA'yı
geçmiyor — bu Faz 8'in çözdüğü **mevcut** bir kusur, yeni bir gereksinim değil:

```
FAIL 3.98  BLUE   #3b82f6 / #1e293b
FAIL 3.89  RED    #ef4444 / #1e293b
FAIL 3.70  PURPLE #a855f7 / #1e293b
PASS 6.42  GREEN  #22c55e
PASS 7.63  YELLOW #eab308
```

---

## Hedefler

1. Aydınlık / koyu / sistem — üç durumlu tema tercihi, sayfa yenilemeden
   anahtarlanabilir, **kullanıcı hesabına** kayıtlı (tarayıcıya değil).
2. Tema tek kaynaktan tanımlansın (tokens.css) — Python, CSS ve auth şablonları
   aynı değerleri okusun.
3. Her metin/zemin çifti **her iki temada** WCAG AA ≥ 4.5:1 (UI sınırları ≥ 3:1).
4. Bileşen sayısı azalsın, tekrar eden kalıplar tek bileşene insin.
5. Renk yalnızca **anlam** taşıdığında kullanılsın (kâr/zarar, durum); dekoratif
   renk kaldırılsın.

### Kapsam dışı

- Sayfa bilgi mimarisi / menü yapısı değişikliği (mevcut 9 sayfa aynı kalır).
- Yeni grafik türü, yeni API ucu, backend davranışı.
- Referans ekranın birebir taklidi (Kanban, üst arama çubuğu, bildirim vb.).

> ⚠️ Bu liste **Faz A–E için** geçerliydi. Faz F ve G onu bilinçli olarak
> aştı: altı yeni uç (`/api/account/*`), bir yeni sayfa yüzeyi (üst çubuk),
> arama ve bildirim geldi. Yani "yeni API ucu / üst arama çubuğu / bildirim
> kapsam dışı" satırları **artık geçerli değil** — gerekçeleri ilgili faz
> başlıklarında.

---

## Faz A — Token katmanı (temel; diğer her şey buna dayanır)

### A.1 — `DARKLY` → `BOOTSTRAP` taban

`dashboard/app.py:66` `external_stylesheets` içindeki `dbc.themes.DARKLY`
yerine `dbc.themes.BOOTSTRAP` (nötr taban). Böylece renk kararını tamamen
bizim token katmanımız verir.

**Yan fayda:** `custom.css`'teki `!important`ların önemli kısmı gereksizleşir;
A.2 sırasında temizlenir (dosyanın ~%30 küçülmesi bekleniyor).

**Risk:** DARKLY'nin bize bedava verdiği koyu varsayılanlar (modal, tooltip,
dropdown iç kısımları) kaybolur → A.2'de açıkça tanımlanmalı. Doğrulama listesi
Faz E'de.

### A.2 — `dashboard/assets/00-tokens.css` (yeni)

Dash `assets/` dosyalarını **alfabetik** yükler; `00-` öneki tokenların
`custom.css`'ten önce gelmesini garanti eder.

> ⚠️ Aşağıdaki isimler **plan anındaki** hâlleridir. Uygulamada hepsi `--rlt-`
> önekine alındı (`--muted` → `--rlt-muted`), çünkü Dash DataTable kendi
> `--muted`/`--border`/`--accent` tokenlarını tanımlayıp bunları gölgeliyordu.
> Gerekçe "Üçüncü parti çakışmaları" bölümünde; güncel liste
> `static/tokens.css`'te.

Üç durum için kaskad (bkz. B.1): aydınlık taban `:root`'ta durur, koyu blok
**iki kez** yazılır — bir kez damgasız `system` hâli için media sorgusunda,
bir kez açık seçim için `[data-theme="dark"]` olarak. `:not([data-theme="light"])`
koruması, kullanıcı aydınlığı açıkça seçtiğinde koyu bir işletim sistemi
temasının onu ezmesini engeller.

```css
:root {                          /* AYDINLIK — taban */
  --bg:            #f6f8fb;
  --surface:       #ffffff;
  --surface-2:     #f1f5f9;
  --surface-hover: #eef2f7;
  --border:        #e2e8f0;
  --border-strong: #8595a9;
  --text:          #0f172a;
  --muted:         #556275;
  --primary:       #2563eb;
  --profit:        #15803d;
  --loss:          #b91c1c;
  --warn:          #b45309;
  --info:          #0e7490;
  --accent:        #7e22ce;
  --orange:        #c2410c;
  --on-primary:    #ffffff;
  --shadow:        0 1px 2px rgba(15,23,42,.06), 0 1px 3px rgba(15,23,42,.04);
}

/* KOYU — 1) damgasız "system" hâli */
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { /* ...aynı koyu tokenlar... */ }
}

/* KOYU — 2) açıkça seçilmiş hâli */
:root[data-theme="dark"] {
  --bg:            #0f172a;
  --surface:       #1e293b;
  --surface-2:     #334155;
  --surface-hover: #3b4a5f;
  --border:        #334155;
  --border-strong: #64748b;
  --text:          #e2e8f0;
  --muted:         #a8b6c9;
  --primary:       #93c5fd;
  --profit:        #4ade80;
  --loss:          #fca5a5;
  --warn:          #fbbf24;
  --info:          #22d3ee;
  --accent:        #d8b4fe;
  --orange:        #fb923c;
  --on-primary:    #0f172a;
  --shadow:        0 1px 3px rgba(0,0,0,.35);
}
```

**Ölçülen kontrast** (her token, kendi temasının **en kötü** zemini üzerinde —
`bg` / `surface` / `surface-2` üçlüsünün minimumu):

| Token | Aydınlık | oran | Koyu | oran |
|---|---|---|---|---|
| `--text` | `#0f172a` | 16.30 | `#e2e8f0` | 8.40 |
| `--muted` | `#556275` | 5.65 | `#a8b6c9` | 5.03 |
| `--primary` | `#2563eb` | 4.72 | `#93c5fd` | 5.74 |
| `--profit` | `#15803d` | 4.58 | `#4ade80` | 5.94 |
| `--loss` | `#b91c1c` | 5.91 | `#fca5a5` | 5.46 |
| `--warn` | `#b45309` | 4.58 | `#fbbf24` | 6.20 |
| `--info` | `#0e7490` | 4.89 | `#22d3ee` | 5.73 |
| `--accent` | `#7e22ce` | 6.37 | `#d8b4fe` | 5.86 |
| `--orange` | `#c2410c` | 4.73 | `#fb923c` | 4.58 |

Hepsi AA (≥4.5). `--border-strong` metin değil, UI sınırı: aydınlık 3.06,
koyu 3.07 — AA non-text eşiği 3.0 sağlanıyor.

> Not: mevcut `#94a3b8` (koyu muted) `--surface-2` üzerinde 4.04 ile kalıyordu;
> `#a8b6c9` ile 5.03'e çıkıyor. Koyu tema **ölçülebilir biçimde iyileşiyor**,
> sadece "aydınlık ekleniyor" değil.

### A.3 — `dashboard/theme.py` yeniden yapılandırma

Kritik ayrım: **DOM'a giden renk `var()` olur, Plotly'ye giden renk hex kalır.**

```python
# --- DOM katmanı: inline style'da kullanılır, temaya kendiliğinden uyar ---
BG         = "var(--bg)"
CARD       = "var(--surface)"
CARD2      = "var(--surface-2)"
BORDER     = "var(--border)"
TEXT       = "var(--text)"
TEXT_MUTED = "var(--muted)"
GREEN  = "var(--profit)";  RED    = "var(--loss)"
BLUE   = "var(--primary)"; YELLOW = "var(--warn)"
PURPLE = "var(--accent)";  CYAN   = "var(--info)";  ORANGE = "var(--orange)"
```

Bu tek dosyalık değişiklik **~550 çağrı yerinin tamamını** dokunmadan temaya
duyarlı hale getirir: `{"color": TEXT}` → `{"color": "var(--text)"}` tarayıcıda
sorunsuz çalışır, `f"1px solid {CARD2}"` gibi f-string kullanımları da bozulmaz.

**İstisna — Plotly.** `go.Bar(marker_color=BLUE)` `var()` kabul etmez. Bu yüzden
ayrı hex palet:

```python
PLOT = {
  "light": {"bg": "#ffffff", "grid": "#e2e8f0", "text": "#0f172a",
            "muted": "#556275", "blue": "#2563eb", "green": "#15803d", ...},
  "dark":  {"bg": "#1e293b", "grid": "#334155", "text": "#e2e8f0",
            "muted": "#a8b6c9", "blue": "#93c5fd", "green": "#4ade80", ...},
}

def current_theme() -> str:      # çerezden okur (auth_context._from_cookie kalıbı)
def plot_palette() -> dict:      # PLOT[current_theme()]
def apply_theme_template(fig):   # apply_dark_template'in yerini alır
```

`apply_dark_template` **kaldırılmaz**, `apply_theme_template`'e delege eden bir
takma ad olarak kalır → 63 çağrı yeri tek seferde bozulmaz, kademeli geçilir.

Dokunulacak Plotly satırları: `marker_color=`, `line={"color": ...}`,
`fillcolor=`, `ALGO_COLORS` — grep ile **~28 satır**, sabiti `plot_palette()`
sözlüğünden okuyacak şekilde değiştirilir. `ALGO_COLORS` da temaya göre çözülen
bir fonksiyona (`algo_colors()`) dönüşür.

### A.4 — Auth şablonları aynı tokenları paylaşsın

`app/auth/templates/login.html` ve `change_password.html` içindeki kopya
`:root` blokları silinir; `<link rel="stylesheet" href="/static/tokens.css">`
ile aynı dosya okunur. Tek fiziksel kaynak: **`static/tokens.css`**;
`dashboard/assets/00-tokens.css` onu `@import` eder.

---

## Faz B — Tema tercihi (kullanıcı bazlı, 3 durumlu)

Tercih üç değer alır: **`light` · `dark` · `system`**. `system` gerçek bir
üçüncü durumdur, "koyu"nun eş anlamlısı değil: işletim sistemi teması
değiştiğinde pano da o an değişir.

### B.1 — Nerede saklanır

Tercih **kullanıcıya** ait, tarayıcıya değil — başka makineden giren aynı
temayı bulur. Veritabanı kaynak, çerez okuma önbelleği:

| Katman | Ne tutar | Neden |
|---|---|---|
| `users.theme` (DB) | `light` / `dark` / `system` | Kalıcı kaynak, cihazdan bağımsız |
| `theme` çerezi | aynı üç değerden biri | Sunucu her istekte DB'ye gitmesin |
| `theme_resolved` çerezi | `light` / `dark` | `system` seçiliyken **istemcinin çözdüğü** sonuç — Plotly figürünü sunucuda doğru palette üretmenin tek yolu |

Akış:

- **Giriş** — `POST /auth/login` yanıtı `user.theme`'i okur, iki çerezi de yazar
  (`app/auth/cookies.py` içine `set_theme_cookies`, `set_auth_cookies`'in
  yanına). Çerezleri silinmiş tarayıcı bir sonraki girişte tercihi geri alır.
- **Değişiklik** — `PATCH /auth/preferences {theme}` → DB + çerezler + denetim
  kaydı yok (kişisel tercih, audit gürültüsü olmasın).
- **Her istek** — sunucu yalnızca çerez okur; DB turu yok.
- **`system` seçiliyken** — istemcideki
  `matchMedia("(prefers-color-scheme: dark)")` dinleyicisi işletim sistemi
  tercihi değişince `theme_resolved`'ı günceller ve `data-theme` damgasını
  koyar/kaldırır.

Üç durumun DOM karşılığı:

| Tercih | `<html>` damgası | Hangi CSS bloğu kazanır |
|---|---|---|
| `light` | `data-theme="light"` | `:root` (aydınlık) |
| `dark` | `data-theme="dark"` | `:root[data-theme="dark"]` |
| `system` | damga **yok** | `@media (prefers-color-scheme: dark)` |

Bu yüzden `00-tokens.css` koyu bloğu **iki kez** yazılır: bir kez
`@media (prefers-color-scheme: dark)` içinde (damgasız hâl), bir kez
`:root[data-theme="dark"]` olarak (açık seçim). Aydınlık, `:root` tabanında
kalır ve `data-theme="light"` damgası koyu media sorgusunu yener.

### B.2 — Şema değişikliği ve göç (dikkat)

`User` modeline tek sütun:

```python
theme: Mapped[str] = mapped_column(String(8), default="system")
```

**Tuzak:** projede alembic yok; `app/auth/db.py::init_db()` yalnızca
`Base.metadata.create_all()` çağırıyor — bu **var olan** tabloya sütun eklemez.
Mevcut `data/auth/*.db` sessizce eski şemada kalır ve ilk sorguda
`no such column: users.theme` ile patlar. Bu yüzden `init_db()` içine
idempotent, additive bir adım gerekiyor:

```python
# PRAGMA table_info(users) ile kontrol; sütun yoksa ekle
ALTER TABLE users ADD COLUMN theme VARCHAR(8) NOT NULL DEFAULT 'system'
```

`User.to_dict()` ve `GET /auth/me` alanı döndürür.

### B.3 — Hesabım sayfası (`/dash/account`)

Yeni Dash sayfası; **her rol** erişir — viewer dahil, çünkü tema okuma
yetkisiyle ilgisi olmayan kişisel bir tercih. Kenar çubuğundaki kullanıcı
rozetinden açılır.

İçerik:

- **Görünüm** — üç seçenekli segment kontrolü: *Aydınlık · Koyu · Sistem*.
  Her seçeneğin altında o temanın küçük önizleme şeridi (zemin + kart + metin),
  böylece seçim yapmadan önce görülür.
- **Hesap** — e-posta, ad soyad, rol, son giriş (salt okunur).
- **Güvenlik** — mevcut `/change-password` sayfasına bağlantı.

> `/dash/users` **admin** sayfasıdır ve öyle kalır; tema kişisel tercih olduğu
> için oraya taşınmıyor. Adminin bir başkasının temasını değiştirmesi gerekirse
> ayrı bir istek olarak ele alınmalı.

### B.4 — Kenar çubuğu hızlı anahtarı

Hesabım'a gitmeden değiştirebilmek için kenar çubuğunda üç durumu sırayla
dolaşan tek düğme: ☀ Aydınlık → ☾ Koyu → ◐ Sistem. Aynı
`PATCH /auth/preferences` ucunu çağırır — iki yüzey de tek kaynağa yazar,
ayrışma olmaz. Uygulama anında (clientside), kayıt arka planda.

### B.5 — FOUC yok

`dash_app.index_string` özelleştirilir; `<head>` içine küçük bir **senkron**
script konur: `theme` çerezini okur, `system` ise `matchMedia` ile çözer,
`<body>` boyanmadan `document.documentElement.dataset.theme`'i ayarlar.
Böylece yanlış temayla tek kare çizim olmaz.

### B.6 — Halihazırda çizilmiş Plotly figürleri

Aynı clientside callback `document.querySelectorAll('.js-plotly-plot')` üzerinde
`Plotly.relayout(el, {...})` çağırarak `paper_bgcolor`, `plot_bgcolor`, font ve
eksen renklerini günceller. **Trace renkleri** (kâr yeşili vb.) `relayout` ile
değişmez; onlar bir sonraki callback yenilemesinde doğru palette gelir. Bu
kabul edilebilir: trace renkleri iki temada da AA geçiyor (A.2 tablosu).

---

## Faz C — Bileşen seti (sadelik + okunabilirlik)

Referanstan alınan ilkeler, somut kurallara çevrilmiş hali:

### C.1 — Tipografi ölçeği

Şu an her yerde 11/12/13/14/28px karışık. Tek ölçek:

| Rol | Boyut | Ağırlık | Renk |
|---|---|---|---|
| Sayfa başlığı | 20px | 600 | `--text` |
| Sayfa alt metni | 13px | 400 | `--muted` |
| Kart başlığı | 14px | 600 | `--text` |
| Gövde | 13px | 400 | `--text` |
| KPI değeri | 26px | 650, `tabular-nums` | duruma göre |
| Etiket/üst-bilgi | 11px, `0.06em` harf aralığı | 600 | `--muted` |

`letter-spacing` + `text-transform: uppercase` yalnızca **etiket** rolünde.
Şu an `metric_card` ve `.section-title` bunu ayrı ayrı tanımlıyor → tek sınıf.

### C.2 — `PageHeader` bileşeni (yeni)

9 sayfada elle tekrar eden `dbc.Row[H4 + P] + sağda rozet` kalıbı tek
bileşene iner: `create_page_header(title, subtitle, actions=None)`.
Referanstaki gibi başlık bloğu solda, eylemler sağda, altında ince ayırıcı.

### C.3 — `MetricCard` v2

Değişiklikler:
- Değer rengi **varsayılan olarak `--text`**; renk yalnızca değer bir yön
  taşıdığında (getiri +/−, drawdown). Şu an 5 kartın 5'i farklı renkte →
  görsel gürültü, hiyerarşi yok.
- İkon `--muted`, 14px, sabit; dekoratif renk yok.
- Sayılar `font-variant-numeric: tabular-nums` (sütunlar hizalanır).
- Kenarlık `--border`, gölge `--shadow` (koyu temada gölge neredeyse görünmez,
  aydınlıkta kartı zeminden ayırır).

### C.4 — Kart / bölüm

- Gölge yerine **ince kenarlık** birincil ayırıcı (referansın en belirgin
  özelliği).
- `border-radius` 8px tek değer (şu an 4/6/8/12/20 karışık).
- `card-header` yalnızca başlık + isteğe bağlı eylem; alt çizgi `--border`.
- İç boşluk 16px/18px yerine tek `--pad: 18px`.

### C.5 — Filtre satırı kalıbı

Referanstaki "Kişi / Öncelik / Deadline / Sırala" barı: etiket **üstte**, 11px
`--muted`, kontrol altta, kontroller tek satırda eşit aralıklı, kart içinde.
`data.py`, `prediction.py`, `hyperopt.py`, `academic.py` bu kalıba geçer →
`create_filter_bar(fields)` bileşeni.

### C.6 — Tablolar

- Zebra şerit yok; satır ayırıcı 1px `--border`.
- Sayısal sütunlar **sağa dayalı** + `tabular-nums`.
- Başlık satırı 11px uppercase `--muted`, zemin `--surface-2`.
- Satır yüksekliği 38px (şu an sıkışık).
- `style_header`/`style_cell`/`style_data` sözlükleri tek yerde:
  `dashboard/components/table.py::TABLE_STYLES`.

### C.7 — Durum ve boş durum

`create_state_block(kind, message)` — `loading` / `empty` / `error` üçlüsü tek
bileşen. Şu an her sayfa `html.P("Yukleniyor...", style={"color": TEXT_MUTED})`
kalıbını elle tekrar ediyor.

### C.8 — Kenar çubuğu

Referanstaki gibi bölüm başlığıyla gruplanır:
*Analiz* (Dashboard, Akademik, Modeller) / *İşlem* (Trading, Tahmin) /
*Sistem* (Eğitim, Veri, HiperParam, Kullanıcılar). Aktif öğe dolgu yerine
**sol kenar çubuğu + hafif zemin** (aydınlıkta tam dolgu ağır durur).

---

## Faz D — Sayfa geçişi

A ve B bittiğinde sayfalar zaten **çalışır** durumda olur (`var()` sayesinde).
Bu faz görsel cilalama; sayfa sayfa, birbirinden bağımsız:

| Sıra | Sayfa | Neden bu sırada | Yük |
|---|---|---|---|
| 1 | `home.py` | Vitrin; kalıpları burada oturt | Orta (28 style) |
| 2 | `training.py` | Az stil, hızlı kazanım | Düşük (21) |
| 3 | `models.py` | Grafik ağırlıklı, Plotly paletini test eder | Düşük (15) |
| 4 | `daily_trading.py` | Tablo kalıbı (C.6) burada oturur | Orta (28) |
| 5 | `academic.py` | 4 grafik | Orta (24) |
| 6 | `users.py` | Tablo + modal | Düşük (17) |
| 7 | `hyperopt.py` | Filtre barı ağır (15 Row) | Yüksek (38) |
| 8 | `data.py` | En çok kontrol, chip seçici | Yüksek (49) |
| 9 | `prediction.py` | **En ağır** — 89 style, 4 grafik, mum grafiği | En yüksek |

---

## Faz E — Doğrulama

### E.1 — Otomatik kontrast denetimi

`tests/test_theme_contrast.py` (standalone, `python` ile çalışır):
`static/tokens.css` ayrıştırılır, her `--text`/`--muted`/aksan tokenı her üç
zemin tokenıyla eşleştirilip WCAG oranı hesaplanır; **< 4.5 ise fail**
(sınır tokenları için < 3.0). Palet ileride değişirse regresyon anında yakalanır.

### E.2 — Kaçak hex denetimi

Aynı test dosyasında grep tabanlı kural: `dashboard/pages/*.py` içinde
`#rrggbb` **yasak** (Plotly paletini içeren `theme.py` muaf). Şu an 2 kaçak var
(`data.py`, `app.py`) — plan bitiminde 0 olmalı.

### E.3 — Tema tercihi testi

`tests/test_theme_preference.py` (standalone):

- Eski şemalı bir SQLite dosyası üretilip `init_db()` çağrılır → `theme`
  sütununun eklendiği ve mevcut satırların `system` aldığı doğrulanır
  (B.2 göçü). İkinci çağrı hata vermemeli (idempotent).
- `PATCH /auth/preferences` üç geçerli değeri kabul eder, dördüncüyü reddeder.
- Girişten sonra `theme` ve `theme_resolved` çerezleri set edilir.
- Viewer rolü kendi tercihini değiştirebilir; başkasının hesabına yazamaz.

### E.4 — Görsel doğrulama

10 sayfa (yeni `account` dahil) × 3 tema durumu. `system` için işletim
sistemi teması iki yönde de denenir. Kontrol edilenler: hiçbir yerde
açık-zemin/açık-yazı; DARKLY kaldırıldıktan sonra modal / dropdown / tooltip /
takvim iç kısımlarının boyanmış olması; grafik ekseni, legend ve hover etiketi;
sayfa açılışında yanlış temayla tek kare çizim olmaması (B.5).

### E.5 — Regresyon

- `python tests/test_auth.py` — login şablonu (A.4) ve login yanıtı (B.1) değişti.
- `python tests/test_workspace_isolation.py` — yeni `/auth/preferences` ucu
  RBAC matrisini bozmamalı.
- Tüm sayfalar 200 dönüyor, callback hatası yok (Faz 5 smoke test kalıbı).

---

## Riskler

| Risk | Etki | Önlem |
|---|---|---|
| DARKLY kaldırılınca bazı dbc iç kısımları koyu temada beyaz kalır | Yüksek — koyu tema regresyonu | E.3 görsel liste; `custom.css`'te modal/dropdown/tooltip/takvim blokları zaten var, tokenlara bağlanacak |
| Plotly `relayout` trace renklerini güncellemez | Düşük — anahtarlama sonrası grafik bir tur eski renkte | Trace renkleri iki temada da AA geçiyor; sonraki yenilemede düzelir |
| **`users` tablosuna sütun eklenmesi mevcut DB'ye uygulanmaz** — alembic yok, `create_all()` ALTER etmez | **Yüksek** — çalışan kurulum `no such column: users.theme` ile açılmaz | `init_db()` içinde `PRAGMA table_info` kontrollü additive ALTER (B.2) + eski şemayla açılan test (E.3) |
| `system` seçiliyken sunucunun Plotly paletini bilememesi | Orta — grafik ters temada çizilir | `theme_resolved` çerezi istemcide çözülüp yazılır (B.1); `matchMedia` dinleyicisi OS değişiminde günceller |
| Çerez tabanlı tema + WSGI mount | Orta | `auth_context._from_cookie()` aynı yolu zaten kullanıyor, kanıtlanmış kalıp |
| `var()` inline style'da eski tarayıcı | Yok sayılır | Hedef modern Chrome/Edge; `var()` uzun süredir destekli |
| 550 çağrı yerinin sessizce bozulması | Orta | E.2 kaçak hex denetimi + sayfa sayfa görsel kontrol |

---

## Efor tahmini

| Faz | İş | Tahmin |
|---|---|---|
| A | Token katmanı + theme.py + Plotly paleti + auth şablonları | ~1 gün |
| B | 3 durumlu tercih: DB sütunu + göç, `/auth/preferences`, Hesabım sayfası, kenar çubuğu anahtarı, FOUC, clientside relayout | ~1 gün |
| C | 6 bileşen (PageHeader, MetricCard v2, FilterBar, Table, StateBlock, Sidebar) | ~1 gün |
| D | 9 sayfa cilalama | ~1.5 gün |
| E | Testler + görsel doğrulama + düzeltmeler | ~0.5 gün |

**Toplam ~5 gün.** A+B tek başına teslim edilebilir: o noktada üç durumlu tema
tercihi hesaba kayıtlı olarak çalışır ve pano okunur; C/D olmadan sadece
"cilasız" olur.

---

## Kritik dosyalar

| Dosya | Değişiklik |
|---|---|
| `dashboard/app.py` | DARKLY→BOOTSTRAP, `index_string` (FOUC), `/dash/account` rotası |
| `app/auth/models.py` | `User.theme` sütunu + `to_dict()` (B.2) |
| `app/auth/db.py` | `init_db()` içine idempotent additive ALTER (B.2) |
| `app/auth/cookies.py` | `set_theme_cookies()` — `theme` + `theme_resolved` (B.1) |
| `app/auth/routes.py` | `PATCH /auth/preferences`; login yanıtı tema çerezlerini yazar |
| `dashboard/pages/account.py` | **YENİ** — Hesabım: 3 durumlu tema seçimi (B.3) |
| `dashboard/theme.py` | DOM `var()` katmanı + `PLOT` hex paleti + `apply_theme_template` |
| `static/tokens.css` | **YENİ** — tek kaynak token tanımı |
| `dashboard/assets/00-tokens.css` | **YENİ** — `@import "/static/tokens.css"` |
| `dashboard/assets/custom.css` | `!important` temizliği, tokenlara bağlanma |
| `dashboard/assets/theme-toggle.js` | **YENİ** — clientside anahtar + Plotly relayout |
| `dashboard/components/page_header.py` | **YENİ** (C.2) |
| `dashboard/components/filter_bar.py` | **YENİ** (C.5) |
| `dashboard/components/table.py` | **YENİ** (C.6) |
| `dashboard/components/state_block.py` | **YENİ** (C.7) |
| `dashboard/components/metric_card.py` | v2 (C.3) |
| `dashboard/components/sidebar.py` | Gruplama (C.8) + 3 durumlu hızlı anahtar (B.4) + Hesabım bağlantısı |
| `dashboard/pages/*.py` | 9 sayfa (Faz D) |
| `app/auth/templates/*.html` | Kopya `:root` silinir (A.4) |
| `tests/test_theme_contrast.py` | **YENİ** (E.1, E.2) |
| `tests/test_theme_preference.py` | **YENİ** (E.3) — göç, uç nokta, çerezler, RBAC |

### Faz F ve G'de eklenenler

| Dosya | Değişiklik |
|---|---|
| `app/api/routes/account.py` | **YENİ** — `/api/account/*`: profil, oturumlar, etkinlik, bildirimler (F.3, G.5) |
| `app/auth/service.py` | `list_sessions`, `revoke_other_sessions` (kayıt **siler**, bkz. F.4), `list_audit_for_user` |
| `app/api/routes/trading.py` | Eğitim durumuna `finished_ts` — zilin "yakında bitti" penceresi için (G.5) |
| `app/main.py` | `account_routes` kaydı |
| `dashboard/components/topbar.py` | **YENİ** — kırıntı, bağlamsal eylem, arama, zil, görünüm anahtarı (G) |
| `dashboard/components/sidebar.py` | Hesap satırı (F.1), görünüm anahtarı **çıkarıldı** (G.2), marka ikonu token kaçağı (G.3) |
| `dashboard/pages/account.py` | Profil düzenleme, hesap özeti, oturumlar, son etkinlik (F.2) |
| `dashboard/pages/home.py` | Boş portföy grafiği düzeltmesi (G.6) |
| `dashboard/theme.py` | `TOPBAR_HEIGHT` / `TOPBAR_STYLE`, `CONTENT_STYLE` üst boşluğu (G.1) |
| `dashboard/api_client.py` | `_request_raw` + hesap uçları sarmalayıcıları |
| `dashboard/assets/custom.css` | Üst çubuk, zil, hesap satırı; **devre dışı varyant** düzeltmesi (G.6) |
| `tests/test_account_profile.py` | **YENİ** — 84 kontrol (F.7) |
| `tests/test_topbar.py` | **YENİ** — 39 kontrol (G.4) |

---

## Uygulama notları

Planın dışına çıkılan yerler ve canlı doğrulamada yakalanan kusurlar.

### Plandan sapmalar

**1. Algoritma rozetleri inline stil yerine sınıf kullanıyor.**
Plan `ALGO_COLORS` sözlüğünü inline `backgroundColor` olarak vermeye devam
ediyordu. Çalışmadı: `dbc.Badge` kendi `bg-secondary` sınıfını basıyor ve
Bootstrap'in utility kuralları `!important` taşıdığı için inline renk sessizce
eziliyordu — rozet hem aydınlıkta hem koyuda yanlış renkteydi ve bu ancak
tarayıcıda hesaplanmış stile bakınca görüldü. Çözüm:
`theme.py::algo_badge_class()` + `custom.css`'te `.badge.algo-badge.algo-ppo`
gibi sınıf eşleşmeleri. Tanınmayan algoritma nötr `--surface-2` kalıyor.

**2. Bootstrap renk varyantlarının tamamı ezilmek zorunda kaldı.**
DARKLY kalkınca ezilmeyen her varyant Bootstrap'in kendi rengine düşüyor.
`.btn-outline-warning` tanımsızdı ve Bootstrap'in `#ffc107` sarısı beyaz zeminde
~1.6:1 kalıyordu — Veri sayfasındaki "Yeniden İndir (tam)" düğmesi okunmuyordu.
Eksik varyantlar (`btn-outline-warning/info`, `btn-info`, `btn-light/dark`,
`alert-primary/secondary`) eklendi ve `test_theme_contrast.py`'ye bir bekçi
kondu: sayfalarda kullanılan her `color="..."` varyantı için dört seçicinin
(`.btn-*`, `.btn-outline-*`, `.badge.bg-*`, `.alert-*`) tanımlı olması aranıyor.

**3. `PATCH /auth/preferences` kendi CSRF kontrolünü yapıyor.**
`AuthGateMiddleware` CSRF'i yalnızca `/api/*` yazmalarında doğruluyor; bu uç
`/auth/*` altında. SameSite=Lax zaten çapraz siteden PATCH'i kesiyor ama uç
durum değiştirdiği için ikinci katman olarak kontrol uca eklendi. Ayrıca gövde
elle ayrıştırılırken geçersiz değer 500 veriyordu; gövde FastAPI parametresine
çevrildi (artık 422).

### Canlı doğrulamada yakalananlar

- **`_metric_val` kart ölçeğini eziyordu.** `home.py`'deki yardımcı, değere
  kendi `fontSize`/`fontWeight`/`color`'ını basıyordu; layout'taki C.3
  değişikliği bu yüzden ekrana yansımıyordu, beş kart hâlâ beş ayrı renkteydi.
  Yardımcı `.metric-value` sınıfına ve `tone` parametresine geçirildi. Ton artık
  **biçimlenmiş metinden değil ham sayıdan** çıkarılıyor: `"394.3%"` işaretsiz
  ama pozitif — metne bakan bir çözüm bunu nötr sayardı.
- **Tema düğmesi etiketi güncellenmiyordu.** `theme-toggle.js` açılışta
  `#page-content`'i izliyordu; Dash arayüzü `DOMContentLoaded`'dan sonra React
  ile çizildiği için kenar çubuğu o an henüz yoktu. İzleme `document.body`
  alt ağacına alındı (rAF ile birleştirilmiş).
- **`empty_figure` artık eksenleri gizliyor** — boş grafik "veri yok" derken
  1'den 6'ya boş bir eksen göstermesin diye.

### Faz C ikinci turu — bileşenlerin sayfalara bağlanması

İlk turda bileşenler yazıldı ama ikisi hiçbir sayfaya bağlanmamıştı. İkinci
tur bağlama işi ve üç ciddi kusuru ortaya çıkardı.

**Yapılan bağlama**

- 42 kart inline stili kaldırıldı. `.card` kuralı zaten doğrusunu veriyordu;
  üstelik oradaki `border` yanlış tokenı (`--rlt-surface-2`) kullanıyordu.
- 43 kart başlığı `card-title-sm` sınıfına indi.
- `models.py` ve `users.py` ortak `TABLE_STYLES`'a geçti (zebra kalktı,
  sayısal sütunlar sağa dayandı); `users.py`'deki yerel stil sözlüğü silindi.
- 13 ad-hoc "… yok" satırı `create_state_block`'a döndü; `training.py`'deki
  kum saatli blok da bos duruma çevrildi (kum saati "bir şey oluyor" izlenimi
  veriyordu, oysa henüz hiçbir şey başlamamıştı).
- DARKLY döneminden kalma 12 `"color": CARD` dropdown hack'i temizlendi —
  token dünyasında bu "yazıyı kart zeminine boya" demek.
- Kart temizliğinden sonra ölü kalan ~50 tema importu budandı.

**C.5 (FilterBar) uygulanmadı.** Sayfalardaki kontrol blokları kenar sütununda
dikey formlar; yatay filtre satırı yok, `academic.py`'de hiç filtre yok.
Planın o maddesi referans ekrandaki Kanban filtre barına bakılarak yazılmıştı
ve bu panoda karşılığı çıkmadı. Bileşen duruyor, ihtiyaç olursa hazır.

### Üçüncü parti çakışmaları (bu turun asıl bulgusu)

**1. Dash DataTable token adlarımızı gölgeliyordu.**
DataTable kendi bundle'ında `--muted: #c8c8c8`, `--border`, `--accent`
tanımlıyor. Tablonun içinde bizim `var(--muted)`'ımız onların değerine
çözülüyordu: başlık `#c8c8c8`, zemin üzerinde **1.35:1**. Ne `style_header`
ne CSS kuralı düzeltiyordu — bu bir ad çakışmasıydı, öncelik sorunu değil.
Çözüm: **tüm tokenlar `--rlt-` önekine alındı** (514 kullanım). Test artık
hem çakışmayı hem öneksiz token kalmadığını denetliyor.

**2. Dash'in yeni bileşen DOM'u hiç temalanmamıştı.**
Bu Dash sürümünde `dcc.Dropdown` react-select değil `button.dash-dropdown`;
`dcc.Slider` `dash-slider-*`. CSS'imiz eski react-select seçicilerini
hedefliyordu, yani:
- açılır kutular koyu temada **beyaz** kalıyordu,
- slider işaretleri `rgba(0,9,38,.9)` ile koyu kartta **görünmüyordu**,
- slider yan kutusu beyaz zeminde `--rlt-text` ile **1.14:1**'di,
- tarih seçicinin dış kapsayıcısı beyazdı (altta 16px'lik şerit).

`dash-dropdown-*`, `dash-slider-*`, `dash-input-*`, `dash-options-*`,
`dash-spinner-*` aileleri temaya bağlandı. Test artık sayfalarda kullanılan
her `dcc.X` için karşılık gelen sınıf ailesinin `custom.css`'te olmasını
arıyor — Dash yükseltmesi yeni DOM getirirse yakalanır.

**3. DataTable `fontFamily: "inherit"` kabul etmiyor.** Tablolar monospace
kalıyordu; yığın açıkça verildi.

### Ölçülen sonuç

| | Önce | Sonra |
|---|---|---|
| Tema sayısı | 1 (koyu) | 3 durum (aydınlık / koyu / sistem), hesaba kayıtlı |
| Tema tanımının kopyası | 3 | 1 (`static/tokens.css`) |
| AA'yı geçmeyen renk | 4 (BLUE 3.98, RED 3.89, PURPLE 3.70, MUTED 4.04) | 0 (en düşük 4.58) |
| Sayfa kodunda kaçak hex | 2 | 0 (test bekçilik ediyor) |
| Kart inline stili | 42 | 0 (`.card` kuralı) |
| Tablonun kendi stil sözlüğü | 2 sayfa | 0 (`TABLE_STYLES`) |
| Temasız Dash bileşen ailesi | 5 | 0 (test bekçilik ediyor) |
| Token adı çakışması | 3 (`--muted`/`--border`/`--accent`) | 0 (`--rlt-` öneki) |
| Tema testi | yok | 115 kontrol (kontrast + tercih) |

---

## Faz F — Profil sayfası (2026-08-30)

B.3 bir "Hesabım" sayfası üretmişti, ama panoyu kullanan biri için **yoktu**:
kenar çubuğunun altındaki kullanıcı adı `textDecoration: none` ile duz metin
gibi duruyordu, hover'a kadar link olduğu belli olmuyordu ve menüde karşılığı
yoktu. Ayrıca sayfa yalnızca tema seçiciydi — B.3'ün söz verdiği "son giriş"
bile eksikti.

### F.1 — Giriş noktası

Kenar çubuğu alt bölümü `dbc.NavLink`'e çevrildi: baş harf avatarı + ad + rol
+ chevron, hover zemini ve `active="exact"` ile aktif sayfa vurgusu. Çıkış
bağlantısı ve tema anahtarı alta ayrı bir kontrol satırına indi. Tüm renkler
token; yeni sınıflar `custom.css`'te (`sidebar-footer`, `sidebar-account`,
`account-avatar`, …).

> **Tuzak:** `dbc.NavLink` yalnızca sayılı prop kabul ediyor
> (`active/href/target/className/style/...`). `title` verilince **tüm Dash
> ağacı** render edilemiyor ve `/dash/` 500 dönüyordu — `test_workspace_isolation`
> bunu 4 fail ile yakaladı. İpucu metni sarmalayan `html.Div`'e taşındı.

**Kenar çubuğu ad değişiminden sonra bayattı.** `create_sidebar()` yalnızca
`app.py::serve_layout` içinde, yani **tam sayfa yüklemesinde** çalışıyor;
`display_page` sadece `page-content`'i değiştiriyor. Ad kaydedildikten sonra
kenar çubuğu eski adı göstermeye devam ediyordu (tarayıcı yenilenene kadar).
Profil callback'i artık `sidebar-account-name` / `sidebar-account-avatar`
çıktılarını da yazıyor.

Bunun için hesap satırı **auth kapalıyken de** çiziliyor ("Misafir" ·
"Kimlik doğrulama kapalı"): o kimlikler her zaman DOM'da olsun ki callback
var olmayan bir bileşene yazmaya çalışmasın. Yan fayda — `AUTH_ENABLED=False`
modunda Hesabım sayfasına hiç giriş yolu yoktu, oysa oradaki görünüm tercihi
o modda da çalışıyor. Çıkış bağlantısı bu durumda gösterilmiyor.

### F.2 — Sayfa gerçek bir profil sayfasına çıktı

| Kart | İçerik |
|---|---|
| **Profil** | Avatar, ad soyad **düzenlenebilir**, e-posta (kilitli — giriş anahtarı), rol + açıklaması |
| **Hesap** | Son giriş, hesap açılışı, hesabı açan, çalışma alanı kullanımı (dosya + boyut) |
| **Görünüm** | Değişmedi — üç durumlu tema, clientside |
| **Güvenlik** | Parola değiştir + **aktif oturumlar** + "Diğer oturumları kapat" |
| **Son etkinlik** | Kendi denetim kaydı: girişler (başarılı/başarısız + sebep), çıkış, parola değişimi, oturum iptali |

Zaman damgaları `"30.08.2026 14:32 UTC"` olarak yazılıyor; kayıtlar naive-UTC
tutulduğu için etiketsiz göstermek "yerel saat" izlenimi verirdi.

### F.3 — `/api/account/*` (yeni)

`/auth/*` tarayıcının **doğrudan** çağırdığı yüzey (giriş formu, clientside
tema anahtarı) ve orada CSRF'i ucun kendisi doğrulamak zorunda. Profil/oturum
uçlarını pano callback'leri `api_client` üzerinden çağırdığı için `/api/*`
altına kondu: CSRF, RBAC ve çalışma alanı bağlamı mevcut kapıdan geliyor —
`/api/admin/*` ile aynı kalıp.

```
GET   /api/account/me                      hesap alanları + çalışma alanı özeti
PATCH /api/account/profile                 {full_name}
GET   /api/account/sessions                gruplanmış aktif oturumlar
POST  /api/account/sessions/revoke-others  bu tarayıcı hariç hepsini kapat
GET   /api/account/activity                kendi denetim kaydı (son N olay)
GET   /api/account/notifications           üst çubuk zili (bkz. G.5)
```

Hepsi `CurrentUser` — viewer dahil her rol. Hedef **her zaman** oturumdaki
kullanıcı; gövdeden kullanıcı kimliği alınmıyor, `role`/`is_active`/`email`
şemada yok. Böylece kendi rolünü yükseltme yolu yapısal olarak kapalı
(test bunu gövdeye o alanları koyarak doğruluyor).

### F.4 — Oturum kapatma: grace penceresi açığı (bulundu ve kapatıldı)

`rotate_session`, iptal edilmiş bir jti 30 sn içinde tekrar kullanılırsa bunu
"eşzamanlı yenileme yarışı" sayıp **yeni bir oturum veriyor**
(`REFRESH_REUSE_GRACE_SEC` — Dash'in paralel callback'leri için konmuş, doğru
bir önlem). Ama "diğer oturumları kapat" düğmesi kayıtları yalnızca
`revoked_at` ile işaretleseydi bu pencere iptali de geçersiz kılardı:
`/auth/refresh` CSRF istemiyor, dolayısıyla çalınmış bir refresh çerezini
elinde tutan taraf saniyede bir yenileyerek kapatmayı atlatabilirdi.

Çözüm: **kasıtlı iptalde kayıt silinir**, işaretlenmez → `record is None` →
`session_unknown` → 401; grace yolu hiç çalışmaz. `revoke_all_sessions`
(admin/parola/deaktivasyon yolları) davranışını korudu; asimetri
`revoke_other_sessions` docstring'inde gerekçesiyle yazılı. İz denetim
kaydında (`account.revoke_sessions`).

**Kalan sınır (bilinçli):** iptal refresh token'ını öldürür, ama access JWT
stateless — diğer taraf en fazla `ACCESS_TOKEN_EXPIRE_MINUTES` kadar (varsayılan
30 dk) okumaya devam edebilir. Bu, admin'in `revoke-sessions` ucunda da böyle;
değiştirmek access token'ı da DB'ye bağlamak demek.

### F.5 — Oturum listesi neden gruplu

Rotasyon her sessiz yenilemede yeni satır yazıp eskisini iptal ediyor, ama
grace penceresindeki eşzamanlı yenilemeler aynı tarayıcı için birden fazla
**geçerli** satır bırakabiliyor. Ham listelemek "3 aktif oturum" gibi yanlış
bir sayı üretirdi; satırlar `(ip, user_agent)` ile gruplanıyor, `tokens` alanı
kaç kayda karşılık geldiğini saklamadan veriyor. UA "Chrome · Windows" gibi
okunur bir etikete çevriliyor — tanınmayan UA uydurulmuyor, kırpılarak
gösteriliyor.

### F.6 — Son etkinlik: neyin gösterildiği, neyin gösterilmediği

`GET /api/account/activity` kullanıcının kendi denetim satırlarını döndürüyor.
Filtre **yalnızca `user_id`** üzerinden — yani "benim yaptığım / benim
oturumumda olan" olaylar: girişler (başarılı ve **başarısız, sebebiyle**),
çıkış, parola değişimi, profil güncellemesi, oturum iptali, jeton tekrar
kullanımı tespiti.

**`target` eşleşmesi kasıtlı olarak dışarıda.** Bir yöneticinin bu hesap
üzerinde yaptığı işlem (rol değişimi, parola sıfırlama, oturum kapatma) o
satırda `user_id` olarak **yöneticiyi** taşıyor; onları kullanıcı yüzeyine
almak yöneticinin kimliğini ve IP'sini yönetici olmayan bir ekrana
sızdırırdı. Yönetici tarafı zaten `/dash/users` → Denetim Kaydı'nda görünüyor.
Test bunu iki yönlü doğruluyor: admin kendi etkinliğinde işlemi **görüyor**,
hedef kullanıcı **görmüyor**.

Uç ham veri döndürüyor (`action`, `success`, ayrıştırılmış `detail`); ham
eylem kodunu okunur etikete çevirmek arayüzün işi (`ACTION_LABELS`,
`LOGIN_FAIL_REASONS`). Başarısız satır `--rlt-loss` ile işaretleniyor — renk
burada **anlam** taşıyor, C.3'ün dekoratif renk yasağıyla çelişmiyor.

### F.7 — Testler

`tests/test_account_profile.py` — **84 kontrol**: uçlar, doğrulama, CSRF,
viewer'ın kendi profilini yönetmesi, rol yükseltme denemesi, oturum gruplama,
iptalden sonra kapatılan oturumun grace penceresinden dönememesi, denetim
kaydı, `_device_label` birim kontrolleri.

Son bölüm **Dash callback'lerini gerçek HTTP yolundan** tetikliyor
(`/dash/_dash-update-component`): layout render testi callback *gövdesindeki*
hatayı yakalamıyor — Faz C'de budanmış bir import tam da böyle kaçmıştı.

`test_theme_contrast.py`'ye 10. denetim eklendi: tasarım sistemimizin kendi
önekli sınıfları (`sidebar-`, `account-`, `metric-`, …) `custom.css`'te tanımlı
olmalı. Bağlanmayan sınıf hata vermiyor, bileşeni sessizce biçimsiz bırakıyor.
Stil taşımayan iki kanca (`theme-label`, `sidebar-link`) gerekçesiyle muaf.

### Ölçülen sonuç (Faz F)

| | Önce | Sonra |
|---|---|---|
| Profil sayfasına görünür giriş | yok (düz metin ad) | avatar satırı + hover + aktif vurgu |
| Hesabım sayfasındaki kart | 3 (tema, salt-okunur hesap, parola linki) | 5 (profil düzenleme, hesap, tema, güvenlik+oturumlar, son etkinlik) |
| Kullanıcının kendi düzenleyebildiği alan | tema | tema + ad soyad |
| Kendi oturumlarını görme/kapatma | yok (yalnızca admin) | var |
| İptalin grace penceresiyle atlatılabilmesi | mümkündü | kapalı (kayıt siliniyor) |
| Hesabına yönelik başarısız giriş denemesini görme | yok | var (sebebiyle) |
| Hesap testi | yok | 84 kontrol |

---

## Faz G — Üst çubuk (2026-08-30)

Kenar çubuğu "nereye gidebilirim"i anlatıyor; eksik olan "neredeyim ve buradan
ne yapabilirim". Görünüm anahtarı da menünün dibinde, sık kullanılan bir
kontrol için yanlış yerdeydi.

### G.1 — Yerleşim

Kenar çubuğu **tam boy** kalır (marka üstte), üst çubuk **yalnızca içerik
alanını** kaplar (`left: SIDEBAR_WIDTH`). Böylece marka tek yerde durur ve
mevcut sabit yerleşim bozulmaz. Konum `theme.py::TOPBAR_STYLE`'dan (kenar
çubuğundaki kalıbın aynısı), görünüm `custom.css`'ten geliyor. Üst çubuk
`fixed` olduğu için `CONTENT_STYLE` üst boşluğu `calc(54px + 24px)` oldu.

### G.2 — İçerik

| Bölge | İçerik | Karar |
|---|---|---|
| Sol | Kırıntı: **grup › sayfa** | Sayfa başlığını **tekrarlamaz** — `PageHeader` zaten başlık + alt metni veriyor; buradaki katkı hangi grupta olduğun |
| Sağ | Bağlamsal eylem | Rastgele kısayol değil: belgelenen akışı (veri indir → eğit → analiz) izler; akışta karşılığı olmayan sayfada (Hesabım, Kullanıcılar) slot boş kalır |
| Sağ | Arama | Sayfalar üzerinde komut paleti; **role duyarlı** — Yönetim grubu yalnızca admin'e önerilir (rota zaten korumalı, ama öneri de yanıltmamalı) |
| Sağ | Görünüm anahtarı | Kenar çubuğundan **taşındı**, çoğaltılmadı |

Görünüm anahtarının tek kopya olması önemli: iki kopya kalsaydı birine
tıklandığında diğerinin etiketi güncellenmez ve anahtar tutarsız görünürdü
(`theme-toggle.js` `getElementById` kullanıyor, ilk kopyayı bulur). Test
DOM'da tam bir tane olduğunu doğruluyor.

Aramada seçim `url.pathname`'i yazıp kutuyu boşaltıyor — boşaltılmazsa aynı
sayfa ikinci kez seçilemez (değer değişmediği için callback tetiklenmez). Aynı
sayfa seçilirse `no_update` dönüyor, sayfa boşuna yeniden çizilmiyor.

### G.3 — Yol üstünde bulunan kusur

`--rlt-` önek göçünden kalma bir kaçak: kenar çubuğundaki marka ikonu
`style={"color": "var(--primary)"}` kullanıyordu. O token artık yok
(`--rlt-primary` var), dolayısıyla bildirim geçersiz sayılıp ikon miras
renge düşüyordu — sessiz bir kayıp. `E.2` kaçak-hex denetimi bunu yakalamıyor
çünkü ortada hex yok. `BLUE` sabitine çevrildi.

### G.4 — Testler

`tests/test_topbar.py` — **39 kontrol**: yerleşim, tek kopya görünüm anahtarı,
kırıntı (bilinmeyen rotada boş), bağlamsal eylem, role duyarlı arama, arama
yönlendirmesi ve `no_update` yolu; bildirim üretimi (süren/hatalı/yeni biten),
pencere dışında kalan koşumun düşmesi, damgasız bitişin gösterilmemesi,
**başka kullanıcının koşumunun sızmaması** ve zil callback'i.

İki **yapısal bekçi** var: kenar çubuğundaki her menü maddesinin kırıntı
karşılığı olmalı (menüye madde eklenip buraya eklenmezse üst çubuk o sayfada
sessizce boşalır), ve sonraki-adım haritası yalnızca var olan rotalara işaret
etmeli.

`test_theme_contrast`'ın sınıf denetimine `topbar` öneki eklendi.

### G.5 — Bildirimler

**Olay günlüğü değil, durum özeti.** Kalıcı bir bildirim tablosu yok;
`GET /api/account/notifications` zaten bellekte tutulan çalışma durumlarını
okuyor — RL eğitimi (`trading._training_states`) ve tahmin eğitimi
(`prediction_service._training_state`), ikisi de kullanıcı bazlı sözlükler.
Bu yüzden uç ucuz: diske ve DB'ye hiç gitmiyor. "Okundu" işareti de yok — bir
iş bittiğinde ya da düzeldiğinde satır kendiliğinden kayboluyor.

Üretilen satırlar:

| Durum | Tür | Nereye götürür |
|---|---|---|
| RL eğitimi sürüyor | info (`%43 · phase`) | Eğitim |
| RL eğitimi hata verdi | error (hata metni) | Eğitim |
| RL eğitimi **yakında** bitti | success | Modeller |
| Tahmin eğitimi sürüyor / hata / yakında bitti | info / error / success | Tahmin |

İki karar açıkça yazılı:

- **Veri tazeliği kasıtlı olarak dışarıda.** `/trading/data/status` paneli
  CSV'den okuyor; zilin arkasına koymak her yoklamada o maliyeti ödetirdi.
  Veri durumu kendi sayfasında kalıyor.
- **"Bitti" satırları 12 saatlik pencereyle sınırlı** (`NOTIFY_RECENT_SECONDS`).
  Aksi halde son koşumun sonucu günlerce zilde asılı kalırdı. Zaman damgası
  olmayan bir "completed" hiç gösterilmiyor: *"ne zaman bittiğini bilmiyorum"*
  ile *"az önce bitti"* aynı şey değil. Bunun için `trading.py` koşumun bittiği
  anı `finished_ts`'e yazıyor (`learn_end_ts` değerlendirme öncesini işaretler,
  o yüzden ayrı tutuldu).

Rozetin rengini **en ağır** tür belirler (hata > bilgi > başarı): kullanıcı
zili açmadan önce "bir şey mi bozuldu" sorusunun cevabını görmeli. Tür ayrıca
satırın **sol kenar şeridiyle** de işaretleniyor, yani renk tek başına bilgi
taşımıyor.

**Yoklama cadansı 60 sn** — panonun zaten var olan sürekli yoklayıcılarıyla
aynı (`prediction.py` 60 sn, `home.py` 30 sn), yeni bir yük sınıfı açmıyor.
Yan etkisi kayda değer: açık bir sekme sessiz yenilemeyi tetiklediği için
oturumu canlı tutar. Bu davranış `home.py`'nin yoklayıcısıyla zaten vardı;
üst çubuk onu **tüm sayfalara** yayıyor. Üst sınır refresh token'ın azami ömrü
(`REFRESH_TOKEN_EXPIRE_DAYS`).

### G.6 — Görsel doğrulama (F + G)

Container yeniden derlendi ve Faz F/G **ilk kez gerçekten görüldü**: 9 sayfa ×
2 tema, headless Chrome/CDP ile ekran görüntüsü **ve** hesaplanmış stil
ölçümü. Doğrulama `AUTH_ENABLED=False` ile **yerel** bir kopyada yapıldı
(port 8001) — kullanıcının veritabanına ve çalışan container'a dokunulmadı.

Yerleşim ölçümleri 18 kombinasyonun hepsinde aynı ve temiz çıktı: üst çubuk
`top=0 left=220 h=54 z=900`, içerik `78`den başlıyor (çakışma yok), yatay
kaydırma yok, `system` yolu damgasız çalışıyor, konsol hatası yok.

Üç kusur çıktı — üçü de **ancak tarayıcıda** görülebilirdi:

**1. Bildirim zili dolgulu mavi çıkıyordu**, sayfanın birincil eylemi gibi.
`dbc.DropdownMenu` toggle'a kendi renk varyantını basıyor (varsayılan
`color="primary"` → `btn btn-primary`). `.topbar-bell` onunla **aynı
özgüllükte** (0,1,0) olduğu için dosyada sonra gelen `.btn-primary`
kazanıyordu. Seçici `.topbar .topbar-bell.btn` yapıldı — hangi varyantın
bastığına bakmadan kapanıyor. (Faz C'deki `dbc.Badge` bulgusunun aynısı:
dbc bileşeni kendi varyant sınıfını getiriyor.)

**2. Devre dışı dolgulu düğmeler Bootstrap'in ham paletine düşüyordu.**
Bootstrap'in `.btn-primary:disabled` kuralı (0,2,0) bizim `.btn-primary`
kuralımızdan (0,1,0) özgül; ezilmediği için devre dışı düğme `#0d6efd`
oluyordu (hesaplanmış stille ölçüldü). Bu **yalnızca yeni kodda değil,
uygulama genelinde** geçerliydi — Faz A–E'nin varyant temizliği `:disabled`
durumunu atlamış. `primary/success/danger/warning/info/secondary` için
`:disabled` kuralları eklendi, `test_theme_contrast`'a bekçi kondu (86 → 92).

**3. Dashboard'daki portföy grafiği boş veriyle çizgisiz, mesajsız,
varsayılan −1..6 eksenli kalıyordu.** Sebep dallanma hatası: uç
`{"history": []}` dönüyor → `if history:` **doğru** ama `if records:` yanlış;
ikisinin arasında dal olmadığı için `empty_figure` hiç çağrılmıyordu.
Dallanma kayıt üzerine alındı; dört girdi (None / `{}` / boş kayıt / dolu)
için davranış doğrulandı.

> Ders: `test_theme_contrast`'ın varyant bekçisi bir seçicinin **var
> olduğunu** doğruluyor, kaskadı **kazandığını** değil. 1 ve 2 numaralı
> kusurlar tam bu boşluktan geçti. Yeni bir dbc bileşeni eklerken hesaplanmış
> stile bakmak hâlâ gerekli.

### G.7 — Parola değiştirme neden Hesabım'a gömülmedi

Hesabım'daki Güvenlik kartı `/change-password` sayfasına **bağlantı** veriyor;
formu sayfaya gömmek düşünüldü ve **kasıtlı olarak yapılmadı**.

Parola değişimi tüm oturumları iptal edip **yeni çerezleri tarayıcıya**
yazmak zorunda. Dash callback'i `api_client`'ın in-process ASGI çağrısından
geçiyor; iç yanıttaki `Set-Cookie` tarayıcıya ulaşmıyor. Gömülü form
çalışsaydı kullanıcı kendi parolasını değiştirdiği anda oturumdan düşerdi.
Çözmek ya ucun semantiğini bozmayı (çağıranın oturumunu ayakta bırakmak) ya
da `Set-Cookie`'yi Flask yanıtına elle taşımayı gerektirir; ikisi de mevcut
çözümden kötü. Ayrı Jinja sayfası doğru tasarım.

---

## Belge Güncelleme Notu

Faz kapanışında güncellendi: `CLAUDE.md` (Development Plan, proje yapısı, tests
listesi, "Tema (Faz 8)" bölümü ve Do-NOT maddeleri),
`docs/development/roadmap.md` (Faz 8 girdisi), `docs/README.md` (indeks).

Faz F sonrası güncellendi: `CLAUDE.md` (proje yapısı — `app/api/routes/account.py`,
tests listesi — `test_account_profile.py`, "Hesap ve profil (Faz 8/F)" bölümü),
`docs/README.md` (Faz 8 satırının açıklaması).

Faz G sonrası güncellendi: `CLAUDE.md` (proje yapısı — `topbar.py`, tests
listesi — `test_topbar.py`, bildirim ve kırıntı Do-NOT maddeleri),
`docs/development/roadmap.md` (Faz G girdisi). Bu belgede ayrıca beş drift
kapatıldı: başlıktaki doğrulama iddiasının kapsamı ayrıldı (A–E ve F–G iki
ayrı tur), "Kapsam dışı" listesinin F/G ile aşıldığı işaretlendi, F.3 uç
listesine `/notifications` eklendi, Faz F sonuç tablosundaki kart sayısı
4 → 5 düzeltildi, "Kritik dosyalar" tablosuna F/G dosyaları eklendi.

### Kalan açık iş

- **Zil yoklamasının oturumu canlı tutması** (G.5). Davranış belgelendi, karar
  verilmedi: kabul mü, cadans mı düşsün, yoksa zil yalnızca açılınca mı
  yoklasın.

---

## Faz H — Dar ekran ve ölü bileşen (2026-08-30)

### H.1 — Dar ekran: önce ölçüldü, sonra karar verildi

Varsayım "kenar çubuğu 220px sabit, dar ekranda düzen bozuluyor" idi.
**Ölçüm bunu çürüttü.** Dört genişlikte (1280 / 1024 / 820 / 640), üç sayfada
(`home`, `data`, `prediction`) CDP ile bakıldı: hiçbir kombinasyonda taşan
öğe, yatay kaydırma veya kırılma **yok** — Bootstrap ızgarası sütunları
yığıyor, tablolar kendi `overflow-x`'inde kalıyor. 640px'de sayfa okunur ve
kullanılır durumda.

Gerçek sorun bozulma değil **darlık**: 640px'de 220px'lik menü ekranın
%34'ünü yiyordu. Bu yüzden düzen değişmedi, yalnızca ≤820px'de kenar çubuğu
**64px ikon rayına** iniyor:

| Genişlik | Önce (kullanılabilir) | Sonra |
|---|---|---|
| 1280 / 1024 | 1012 / 756 | değişmedi (menü 220px) |
| 820 | 552 | **708** (+156) |
| 640 | 372 | **528** (+156) |

Uygulama detayları:

- Etiketler ayrı `nav-label` span'inde — CSS yalnızca yazıyı gizliyor, ikon ve
  tıklama alanı kalıyor. Aynısı marka yazısı (`brand-text`), hesap satırı
  metni ve çıkış etiketi için.
- Rayda tek ayırt edici ikon olduğu için **her menü maddesi `title` taşıyor**.
  `title` sarmalayan `html.Div`'de: `dbc.NavLink` o propu kabul etmiyor
  (F.1'deki aynı tuzak, `/dash/` 500 döndürüyordu).
- Menü boşlukları inline stilden `#sidebar .nav-link` kuralına taşındı, çünkü
  medya sorgusu **inline stili `!important` olmadan ezemez**. Konum/genişlik
  hâlâ `theme.py`'den inline geldiği için ray kuralları `!important` kullanıyor
  — gerekçe CSS'te yazılı.

Telefon boyu (≤480px) **hedef değil**: bu pano yoğun tablolar ve mum grafiği
gösteriyor, 640px altında okunabilirlik ızgarayla değil içerik tasarımıyla
çözülür. Ray onu da kullanılır kılıyor ama iddia edilen bir hedef değil.

### H.2 — `create_filter_bar` silindi

C.5'te planlanmış, Faz C'de "karşılığı çıkmadı" diye bağlanmamış, iki tur
boyunca **hiçbir sayfada kullanılmamıştı**. Sayfaların kontrol blokları yatay
filtre satırı değil, kenar sütununda dikey formlar (`prediction` 17,
`hyperopt` 18, `daily_trading` 13 `section-title`); beş ağır sayfayı bu
kalıba çevirmek görsel kazancı belirsiz, riski yüksek bir yeniden düzenleme
olurdu — üstelik o sayfalar görsel olarak yeni doğrulandı.

"İhtiyaç olursa hazır" ölü kodun birikme biçimidir; bileşen ve tek
kullanıcısı olan `.field-label` kuralı silindi. Gerekirse git geçmişinden
geri gelir.

### H.3 — Testler

`test_theme_contrast` sınıf bekçisine `nav-label` ve `brand-text` eklendi
(92/92). Kenar çubuğu DOM'u değiştiği için (menü maddeleri artık `title`
taşıyan bir Div'e sarılı) `test_workspace_isolation` 18/18 ve `test_topbar`
39/39 yeniden koşuldu — ikisi de kenar çubuğu yapısına bakıyor.
