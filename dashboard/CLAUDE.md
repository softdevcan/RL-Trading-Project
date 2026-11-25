# CLAUDE.md — dashboard/ (Dash pano, tema, UI)

Kok `CLAUDE.md`'den tasindi; dashboard/ altinda calisirken yuklenir.

UI degisikliginden sonra `python tests/test_theme_contrast.py` ve `python tests/test_dash_props.py` calistir.

### Tema (Faz 8)
- **Renk degeri yalnizca `static/tokens.css`'te.** Sayfa/bilesen kodunda hex yasak;
  `tests/test_theme_contrast.py` kacaklari yakalar (`dashboard/pages/account.py` muaf —
  iki temanin onizlemesini ayni anda gostermek zorunda).
- Uc durum: `light` / `dark` / `system`. `system` = DOM'da damga YOK, karari
  `@media (prefers-color-scheme)` verir. Bu yuzden koyu blok tokens.css'te **iki kez**
  yazilir; ikisi ayrisirsa test kalir.
- **Tokenlar `--rlt-` onekli.** Dash DataTable kendi bundle'inda `--muted`,
  `--border`, `--accent` tanimliyor ve oneksiz adlari tablo icinde golgeliyor
  (tablo basligi 1.35:1 cikiyordu). Yeni token eklerken oneki koru.
- **Yeni bir `dcc.*` bileseni eklerken sinif ailesini custom.css'e bagla.**
  Dash surumleri DOM'u degistiriyor (`dcc.Dropdown` artik `button.dash-dropdown`,
  react-select degil); baglanmayan aile Dash'in kendi renginde kalir.
  `test_theme_contrast.py` bunu denetliyor.
- Tercih **hesaba** bagli (`users.theme`), cerez yalnizca okuma onbellegi.
  `rlt_theme` = tercih, `rlt_theme_r` = istemcinin cozdugu sonuc (Plotly icin sart).
- **DOM sabiti Plotly'ye verilmez**: `TEXT`, `BLUE` vb. artik `var(--token)` dizesi.
  Grafiklerde `plot_palette()` / `plot_rgba()` / `apply_theme_template()` kullan.
- Detay: `docs/development/phase-8-ui-theming.md`

### Ust cubuk ve bildirimler (Faz 8/G)
- Bilesen: `dashboard/components/topbar.py`. Kenar cubugu tam boy kalir, ust
  cubuk YALNIZCA icerik alanini kaplar (`theme.py::TOPBAR_STYLE`,
  `left: SIDEBAR_WIDTH`); `CONTENT_STYLE` ust boslugu `calc(54px + 24px)`.
- **Gorunum anahtari ust cubukta**, kenar cubugunda degil ve TEK KOPYA.
  `theme-toggle.js` `getElementById` kullaniyor; iki kopya olursa birine
  tiklaninca digerinin etiketi guncellenmez. Test DOM'da tam bir tane
  oldugunu denetliyor.
- **Kenar cubugu menusune madde eklerken `topbar.ROUTE_INDEX`'i unutma** —
  aksi halde ust cubuktaki kirinti o sayfada sessizce bosalir
  (`tests/test_topbar.py` yapisal bekci). `NEXT_STEP` haritasi da yalnizca
  var olan rotalara isaret etmeli.
- Arama role duyarli: `viewer`/`user` icin Yonetim grubu onerilmez.
- **Bildirimler OLAY GUNLUGU DEGIL, DURUM OZETI**: `GET /api/account/notifications`
  bellekteki calisma durumlarindan uretir (`trading._training_states`,
  `prediction_service._training_state`) — kalici tablo, "okundu" isareti yok.
  Is bitince satir kendiliginden kaybolur.
- Veri tazeligi bilincli olarak KAPSAM DISI: `/trading/data/status` paneli
  CSV'den okuyor, 60 sn'de bir yoklanamaz.
- "Bitti" satirlari 12 saatlik pencereyle sinirli (`NOTIFY_RECENT_SECONDS`);
  zaman damgasi olmayan `completed` HIC gosterilmez. Bunun icin
  `trading.py` kosum bitisini `finished_ts`'e yazar.
- **Bilinen ve KABUL EDILMIS davranis:** zil 60 sn'de bir yokladigi icin acik
  bir sekme sessiz yenilemeyi tetikler ve oturumu canli tutar. Ust sinir
  refresh token'in azami omru (`REFRESH_TOKEN_EXPIRE_DAYS`). Degistirmek
  gerekirse: cadansi dusur ya da zili yalnizca acilinca yoklat (rozet sayaci
  gider).

### Dar ekran (Faz 8/H)
- <=820px'de kenar cubugu 64px **ikon rayina** iner; masaustu degismez.
  Olculdu: 1280/1024/820/640'ta tasma veya yatay kaydirma YOK — sorun bozulma
  degil darlikti (640px'de menu ekranin %34'u).
- Menu etiketleri `nav-label` span'inde ki CSS yalnizca yaziyi gizleyebilsin;
  ipucu icin her madde `title` tasiyan bir Div'e sarili (**dbc.NavLink `title`
  kabul etmiyor** — verilince tum Dash agaci render edilemez, `/dash/` 500).
- **Medya sorgusu inline stili `!important` olmadan ezemez.** Konum/genislik
  `theme.py`'den inline geldigi icin ray kurallari `!important` kullanir;
  menu bosluklari bu yuzden inline'dan `#sidebar .nav-link` kuralina tasindi.
- Telefon boyu (<=480px) HEDEF DEGIL ve oyle iddia edilmiyor.

### UI icin Do NOT
- Sayfa/bileşen koduna hex renk yazma — `static/tokens.css`'e token ekle, kontrastı ölç
- Yeni bir `dbc` bileşeni eklerken **hesaplanmış stile bak**: dbc kendi renk
  varyant sınıfını basıyor (`DropdownMenu` → `btn-primary`, `Badge` → `bg-secondary`)
  ve aynı özgüllükteki kendi kuralımızı kaskadda yenebiliyor. Test seçicinin
  *var olduğunu* doğruluyor, *kazandığını* değil
- **`className` her `dbc` bileşeninde yok.** Dash tanımadığı kwarg'ı yok saymaz,
  `TypeError` fırlatır — hata da çalışma anında, callback gövdesinde patlar.
  `dbc.Spinner` → `spinner_class_name`, `dbc.NavLink` → `title` kabul etmiyor.
  `tests/test_dash_props.py` tüm çağrıları gerçek prop listesiyle karşılaştırır
- Plotly'ye `TEXT`/`BLUE` gibi DOM sabitlerini verme (bunlar `var()` dizesi, grafik siyah çizer)

### Rehber sayfasi ve sozluk (K28)
- Icerik TEK KAYNAK `dashboard/content/guide.yaml` + `glossary.yaml` (imaja girer; `docs/` girmez). Sayfa
  `pages/guide.py` yalniz bunlari okur. `docs/sozluk.md` URETILIR: YAML degisince `python scripts/build_guide.py`
  kos; `tests/test_guide.py` bayatligi yakalar. Elle yazilan icerikte deney sonucu sayisi YOK: kart sayilari
  `snapshot.json`'dan (uretilir; kaynak JSON + alan). Sayfa olcum dosyasi acmaz (imajda `docs/` yok).
- Terim capasi `#terim-<id>`; sinif ailesi `guide-` (custom.css, tema testinde taranir).

### Ana sayfa (Faz 11 §3.14)
- "Öne çıkan model" bir SIRALAMA DEĞİL: M8 en iyi kolu seçmez. Öğrenen kolların en yükseği (boş kol hariç, negatif
  kontrol) aynı maliyetteki kural portföylerinin YANINDA gösterilir. İleri veri (126 seans dolmadan) "en iyi" seçiminde
  kullanılmaz; simülasyon kutusu ara görünüm uyarısını taşır.
- Sayılar `snapshot.json` `m8_ozet`'ten (build_guide üretir); sayfa ölçüm dosyası açmaz. M8 okuması değişirse
  `python scripts/build_guide.py`.
- Günlük işin saat damgaları saat dilimsiz; panoda `finished_trt` kullan (VDS UTC).

### Turkce metin ve bicim (Faz 11 §3.15)
- Ekranda gorunen metin Turkce karakterle yazilir; kimlik, className, value, API anahtari ASCII kalir.
- Para / yuzde / saat / is durumu `dashboard/components/fmt.py`'den (`money`, `signed_pct`, `dt_trt`, `status`).
  Sayfada `f"₺{x:,.2f}"` yazma. Kimlik kayitlari naive UTC: `dt_trt` TRT'ye cevirip "TRT" etiketiyle yazar.
- `<html lang="tr">` (app.py): CSS `text-transform: uppercase` i -> İ icin sart.
- Tek ajanli (Faz 1-9) sayfalar `create_page_header(note=LEGACY_NOTE)` tasir.
- `dcc.Dropdown` ic metinleri `components/dropdown_tr.py` varsayilanindan (sayi yer tutucusu `{num_selected}`);
  takvim ay/gun adlari `assets/datepicker-tr.js` (ceviri hedefi kaynak anahtarla CAKISMAMALI: "Sa" dersi).
- Dash 4 acilir liste/takvim DOM'u: `.dash-dropdown-*`, `.dash-options-list-*`, `td.dash-datepicker-calendar-date-*`.
  Yeni kural yazinca koyu temada hesaplanmis stile bak (beyaz arama kapsayicisi boyle bulundu).

### Veri sayfasi (Faz 11 §3.17)
- Veri haritasi `app/services/data_catalog.py` + `GET /api/trading/data/catalog` (ayni yanit `status`'u da tasir: tek istek).
  Yeni bir veri seti eklenirse katalogda grubuna ekle; "Bu sayfa: Guncelle" YALNIZ `/data/update`'in yazdiklari icin
  (test bunu denetler). M8 / ileri veri bu sayfadan guncellenmez.

### Uzun suren isler ve sayfa gezinmesi (ONEMLI)
- `display_page` yalnizca `page-content`'i degistirir; sayfa her gezinmede
  **yeniden uretilir**. `dcc.Interval(disabled=True)` + "yalnizca Baslat
  dugmesi acar" kalibi bu yuzden bozuk: baska sayfaya gidip donen kullanici
  surmekte olan isin ilerlemesini bir daha goremez.
- Kural: uzun suren bir isin sayfasi **acilista durumu sormali** ve yoklamayi
  ona gore acmali (`training.py::layout` -> `_status_block`,
  `hyperopt.py::layout` -> calisan study'yi listeden bulur).
- Ayrica calisan isin kimligini yalnizca `dcc.Store`'da tutma — store da
  sifirlanir; kimlik backend'den geri bulunabilmeli. Tahmin egitiminde
  `/prediction/train/status` SEMBOL istiyor, sayfa donuste neyi soracagini
  bilmiyordu; `GET /api/prediction/train/active` listeyi backend'den verir.
- Yoklamayi is bitince KAPAT (`disabled=True`), yoksa sayfa acik kaldigi
  surece bos yere sorgu atar.
