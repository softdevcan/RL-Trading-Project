# Sistem akış haritası

**Son güncelleme:** 2026-11-19 · `v2.5.1` · Faz 10 M4b pilotu: PİLOT_İŞARETİ, mimari-eş kontrolde gat'ın iletişimi bilgi taşıyor işareti (G 0,26, z 3,05; [M4 tasarımı](phase-10-m4-design.md) §6.4). Önceki: M4a bitti, gat KISMİ, attn GÖZLENMEDİ (§3.7). Önceki: Faz 10 M4 tasarımı v0.1: `attn`/`gat` eşit bütçe, GAT grafı `sector + mst` önerisi, M4a komşu sinyali ön kaydı taslağı ([M4 tasarımı](phase-10-m4-design.md)). Önceki: Faz 10 M5 ön kaydı v0.1: P5c okuma kuralı sonuçtan önce yazıldı; gerçek sinyal kolları sağlamlık sorusu, mekanizma iddiaları yalnız pozitif kontrolde ([M5 tasarımı](phase-10-m5-design.md) §2). Önceki: Faz 10 M3b Aşama 1 (20 bps kapısı) koşuldu, etiket KISMİ: tohum sıralama koşulu sağlanmadı, ön kayıt gereği Aşama 2 koşulmadı ([M3 tasarımı](phase-10-m3-design.md) §6.7). Kullanıcı kararı: M4'e referans yapılandırmayla (MAPPO, tüm yerel gözlemler, α = 0,25; varsayılan, kanıta dayanmaz) geçilir (§6.8). Önceki (2026-11-18) Faz 10 M3a boru hattı kapısı koşuldu, etiket GEÇTİ (maliyetsiz, gelecekten türetilmiş pozitif kontrol; tez bulgusu değil); sırada M3b ön kaydı ([M3 tasarımı](phase-10-m3-design.md) §5.2). Önceki (2026-11-17) Faz 10 P5: `pit_v2` (hedef O_{t+1} → O_{t+2}) kuruldu, kapı `pit_v1` ile yan yana; ikisi de geçmiyor (0/30). `pit_v1` dondurulmuş, bit-eş. Önceki (2026-11-16) Faz 10 M2: `marl/` çekirdeği + PettingZoo ortamı; hakem-rl + hakem-finans Kabul; gerçek 30 sembollü panelde gözlem sağlığı ölçüldü, taban açılışta satış engeli (ön kayıtlı kural) uygulandı. S-obs ufuk hizası ölçüldü: PC060 yeni ortamda pozitif kontrol değil, M3 kontrolü PC060-K12 (§9.1). Hakem-2 kapanışı Küçük revizyon, işlendi; satış engeli dönem bağımlı fiyat marjı (v0.7). M2 çekirdeği kapandı, sırada M3 ([M2 tasarımı](phase-10-m2-design.md) §9).

Bu belge sistemin **canlı haritasıdır**: hangi bileşen var, veri nereden nereye akıyor, hangi aşama bitti, hangisi sürüyor, hangisi planlı. Diyagramlar Mermaid metni olduğu için GitHub'da çizilir ve PR'da fark olarak okunur.

## Nasıl güncellenir

Bir PR bir bileşenin ya da aşamanın **durumunu** değiştiriyorsa aynı PR bu belgeyi de günceller:
1. Diyagramdaki düğümün sınıfını değiştir (`:::done`, `:::active`, `:::planned`, `:::decision`, `:::partial`).
2. [Durum tablosundaki](#durum-tablosu) satırı güncelle (durum, kanıt, sıradaki).
3. Yukarıdaki "Son güncelleme" satırını sürüm ve tarihle güncelle.

Yeni bir bileşen ya da aşama eklerken üçünü birden yap: düğüm, ok ve tablo satırı. Ayrıntı buraya yazılmaz, bağlantı verilir: plan §9 alt bölümü, test, ölçüm JSON'u. Bu belge özet ve harita; tek gerçek kaynak plan belgeleridir.

**Renk anahtarı:**

| Sınıf | Anlamı |
|---|---|
| `done` (koyu yeşil) | Bitti, testli; kanıtı plan §9'da |
| `partial` (koyu kırmızı) | Var, ama bilinen açığı beyanlı (sınırlılık) |
| `active` (koyu kahverengi) | Şu an üzerinde çalışılıyor |
| `decision` (mor) | Kullanıcı kararı bekliyor |
| `planned` (beyaz, kesik çizgi) | Planlı, başlamadı |

---

## 1. Sistem mimarisi: veri akışı

```mermaid
flowchart LR
  classDef done fill:#1b5e20,stroke:#0b3d10,color:#ffffff
  classDef partial fill:#b71c1c,stroke:#7f0000,color:#ffffff
  classDef active fill:#8a4b00,stroke:#5a3000,color:#ffffff
  classDef decision fill:#4a148c,stroke:#2a0060,color:#ffffff
  classDef planned fill:#ffffff,stroke:#1a1a1a,stroke-dasharray:5 4,color:#1a1a1a

  subgraph SRC["Kaynaklar"]
    YF["yfinance<br/>OHLCV 2005+"]:::done
    EVDS["TCMB EVDS<br/>makro + risksiz faiz"]:::done
    IDX["XU100 / XU030 / USDTRY / TÜFE"]:::done
    FUND["Temel oranlar<br/>tek anlık görüntü"]:::partial
    KAP["KAP VYK API + Borsa İst.<br/>endeks duyuruları"]:::partial
  end

  subgraph DATA["Veri katmanı"]
    PANEL["Panel 30 hisse<br/>data_fetcher"]:::done
    GATE["Veri kapısı V<br/>hayalet gün, bayat hücre"]:::done
    IND["Göstergeler<br/>ADX wilder, B7"]:::done
    UNI["Evren E0/E2 + üyelik E1<br/>E1 geçici, beyanlı"]:::partial
  end

  subgraph PRED["Tahmin katmanı"]
    FE["Özellik + seçim<br/>kat içi B8"]:::done
    ENS["Ensemble: 3 GBM + BiLSTM + TFT<br/>meta, TATS, kalibrasyon F"]:::done
    PIT["P tablosu: nokta-zamanlı<br/>3 GBM + lojistik meta"]:::done
    TRK["Canlı tahmin + takip<br/>Y28/Y29"]:::done
  end

  subgraph RL["RL katmanı"]
    SPLIT["wf-v1 katları + ölçek<br/>rl_folds, K25"]:::done
    ENV["TradingEnv tek ajan<br/>+ P4 tahmin bloğu"]:::done
    SB3["SB3 A2C/PPO/TD3/SAC<br/>doğrulamada seçim"]:::done
    HPO["HPO Optuna<br/>tohumlu sampler"]:::done
    MENV["MARL ortamı M2<br/>hisse başına ajan, CTDE<br/>marl/ çekirdek + env"]:::active
    DEC["Karar katmanı K<br/>canlı parite K13"]:::planned
  end

  subgraph EVAL["Değerlendirme"]
    REG["Deneme kaydı + protokol<br/>N, açık maddeler"]:::done
    STAT["DSR, LW, MCS, PBO, BH<br/>rliable"]:::done
    BASE["Baseline + birimler + rejim<br/>D, B6"]:::done
  end

  subgraph PROD["Ürün"]
    API["FastAPI /api<br/>RBAC, çalışma alanı"]:::done
    DASH["Dash pano /dash"]:::done
    LIVE["Günlük karar + kâğıt portföy"]:::partial
  end

  YF --> PANEL --> GATE --> IND
  EVDS --> BASE
  IDX --> BASE
  FUND --> ENV
  KAP --> UNI
  UNI -.-> PANEL
  IND --> FE --> ENS --> TRK
  IND --> PIT
  PIT --> ENV
  IND --> SPLIT --> ENV --> SB3 --> REG
  HPO --> SB3
  REG --> STAT
  BASE --> STAT
  PIT -.-> MENV
  SPLIT -.-> MENV
  MENV -.-> DEC
  SB3 --> LIVE
  API --> DASH
  API --> LIVE
```

**Okuma notları:**
- Kesik oklar henüz kurulmamış bağlantılar.
- **Tahmin RL'ye yalnız P tablosundan girer** (P4); canlı ensemble RL gözlemine bağlı değil.
- Temel oranlar ve E1 üyeliği `partial`: ilki nokta-zamanlı değil (`fundamentals_pit` açık), ikincisi geçici (§9.32).
- Günlük karar `partial`: canlı durum vektörü eğitim gözlemiyle tam eşleşmiyor (K13).

---

## 2. Geliştirme aşamaları

```mermaid
flowchart TB
  classDef done fill:#1b5e20,stroke:#0b3d10,color:#ffffff
  classDef partial fill:#b71c1c,stroke:#7f0000,color:#ffffff
  classDef active fill:#8a4b00,stroke:#5a3000,color:#ffffff
  classDef decision fill:#4a148c,stroke:#2a0060,color:#ffffff
  classDef planned fill:#ffffff,stroke:#1a1a1a,stroke-dasharray:5 4,color:#1a1a1a

  subgraph URUN["Ürün fazları (tamamlandı)"]
    F1["Faz 1 POC"]:::done
    F2["Faz 2 tahmin sistemi"]:::done
    F3["Faz 3 üretim iyileştirmeleri"]:::done
    F6["Faz 6 performans"]:::done
    F7["Faz 7 kimlik + çok kullanıcı"]:::done
    F8["Faz 8 UI/UX"]:::done
  end

  subgraph F9["Faz 9: araştırma geçerliliği"]
    P9A["A, C, V, B1-B3, B7-B11<br/>K18, R1, E0, E2<br/>§9.1-9.21"]:::done
    P9B["Kapanış 1-6: G1+G3, B6, F1-F4<br/>sampler tohumu, P, D<br/>§9.23-9.31"]:::done
    P9E["Kapanış 7: E1<br/>beyanla, §9.32"]:::partial
    P9M["Kapanış 8: Milestone 0' protokolü<br/>donduruldu m0-protocol-v1, §9.35"]:::done
    H2["Kapanış kapısı hakem-2<br/>Büyük → ikinci okuma Kabul"]:::done
  end

  subgraph M0["Milestone 0' koşumları"]
    GOLD["Golden yenileme K1, CPU<br/>v2.0.0, §9.36"]:::done
    PRE["Protokol m0p-v2 donduruldu<br/>m0-protocol-v2, 180 eğitim"]:::done
    FIN["Koşum 180/180 + hakem-2 Kabul<br/>ara karar, §9.37"]:::done
  end

  subgraph M2P["Makale 2"]
    B5["B5 nedensel ICEEMDAN"]:::planned
    K0K4["K0-K4 kolları<br/>ICEEMDAN x TATS"]:::planned
  end

  subgraph F10["Faz 10: MARL"]
    M1["M1 spike: BenchMARL seçildi<br/>7/7, §3.2.7"]:::done
    M2["M2 ortam + K karar katmanı<br/>çekirdek kapandı (hakem-2 Küçük rev., işlendi)"]:::active
    M34["M3/M4 MAPPO, iletişim<br/>M3a GEÇTİ, M3b Aşama 1 KISMİ"]:::active
    M5["M5 güvenli mesaj H2/H3"]:::planned
    M67["M6/M7 rejim, yorum"]:::planned
    M8["M8 ön kayıt"]:::planned
    FINAL["Nihai koşumlar"]:::planned
    K27["K27 pano: ajan türü seçimi<br/>+ karşılaştırma görünümü"]:::planned
  end

  subgraph TEZ["Tez"]
    T1["Milestone 1 literatür<br/>148 kaynak"]:::partial
    T4["Yazım, makaleler, savunma"]:::planned
  end

  F1 --> F2 --> F3 --> F6 --> F7 --> F8 --> P9A --> P9B --> P9E --> P9M --> H2
  H2 --> GOLD --> PRE --> FIN
  FIN --> K0K4
  B5 --> K0K4
  M1 --> M2
  FIN --> M2 --> M34 --> M5 --> M67 --> M8 --> FINAL
  M2 --> K27
  T1 --> T4
  FINAL --> T4
  K0K4 --> T4
```

---

## Durum tablosu

| Aşama / bileşen | Durum | Kanıt | Sıradaki |
|---|---|---|---|
| Faz 1–3, 6–8 | bitti | [roadmap.md](roadmap.md) "Tamamlanan Fazlar" | — |
| Faz 9 A, C, V, B, K18, R1, E0, E2 | bitti | plan §9.1–9.21 | — |
| Faz 9 kapanış 1–6 (G, B6, F, sampler, P, D) | bitti | plan §9.23–9.31; İ1–İ3 uzman okumaları | — |
| Faz 9 kapanış 7: E1 tarihsel üyelik | beyanla kapandı | plan §9.32; `test_universe` 36 | Üretim KAP erişimi olursa 2013–2022 dönem içi olaylar |
| P4: P tablosu → eski RL ortamı | bitti | plan §9.32; `test_pit_table` 31 | — |
| Rota DTF + M8 son nokta test kaydı + koşum betiği | bitti | plan §9.33; `test_rl_walk_forward` 98 | — |
| Faz 9 kapanış 8: Milestone 0' protokolü | **donduruldu** (`m0-protocol-v1`) | [milestone0-protocol.md](milestone0-protocol.md); plan §9.33–9.35 | — (değişiklik = m0p-v2) |
| Golden yenileme (K1) | bitti (CPU; `v2.0.0`) | plan §9.36; `test_prediction_regression` yeşil | — (yeniden yenileme yalnız bilinçli davranış değişikliğinde) |
| Milestone 0' koşumları | **bitti, ara karar** (180/180; Q2 boş, PC060 boş). Pozitif kontrol ölçümleri: (a) RL'siz değer MDE'yi aşıyor, (b) politika bloktan değer üretmiyor | plan §9.37–9.39; `measurements/phase10-m0/` | İleri veri; S-obs girdisi Faz 10 §3.2.7 |
| Makale 2 (B5 + K0–K4) | planlı | plan §4 "Makale 2 deney tasarımı", §9.14 | B5 nedensel ICEEMDAN, sonra kollar |
| Faz 10 M1 çerçeve spike'ı | **bitti**: BenchMARL + TorchRL seçildi (K9, 2026-11-16) | Faz 10 plan §3.2.7; `measurements/phase10-m1/` | M2 ortam (spike girdileri: nakit üst sınırı, eksik hücre maskesi, M4 parametre eşitliği) |
| Faz 10 M2 ortam + karar çekirdeği | **sürüyor**: `marl/` (muhasebe, projeksiyon, K5, `obs_spec` v1, K2) + PettingZoo ortamı; logit sınırı 5 + ln N; hakem-rl + hakem-finans Kabul; gerçek panel ölçüldü; taban açılışta satış engeli | [phase-10-m2-design.md](phase-10-m2-design.md); `test_marl_core` 61, `test_marl_env` 36; `measurements/phase10-m2/obs_health.json` | M3: BenchMARL görevi + PC060-K12 P tablosu; 2021+ öncesi fiyat marjı geçmişi doğrulaması (tasarım §2.5, §9) |
| Faz 10 P5: tahmin hedefi K12 yürütme ufkuna (`pit_v2`) | **bitti**: `target='open_to_open'`, purge h+1, bant dışı açılış maskesi; `pit_v1` bit-eş. Kapı: 0/30 bilgili (`pit_v1` gibi); havuzlu AUC 0,5044 | Faz 10 plan §3.1.3 "P5 durumu"; `test_pit_table` 45; `measurements/phase10-p5/` | M5 ön kaydı §2'de okuma kuralı yazıldı ([phase-10-m5-design.md](phase-10-m5-design.md), v0.1) |
| Faz 10 M3a boru hattı kapısı | **bitti, GEÇTİ**: A1 (PC060-K12) ↔ S1 (karıştırılmış), 3 tohum, maliyetsiz, 600 bin kare. Tez bulgusu değil | [phase-10-m3-design.md](phase-10-m3-design.md) §5.2; `test_marl_m3` 18; `measurements/phase10-m3/m3a_results.json` | — |
| Faz 10 M3b 20 bps kapısı | **Aşama 1 bitti, KISMİ**: A1c ↔ S1c, 3 tohum, 600 bin kare; sıralama koşulu sağlanmadı, Aşama 2 koşulmadı. Tez bulgusu değil | [phase-10-m3-design.md](phase-10-m3-design.md) §6.7; `measurements/phase10-m3/m3b_stage1.json` | M4 temeli referans yapılandırma (kullanıcı kararı 2026-11-19, §6.8) |
| Sistem rehberi ve sözlük (K28) | **planlı, onay bekliyor**: tek kaynak `dashboard/content/`, pano "Rehber" sayfası, R1–R4 | [system-guide-plan.md](system-guide-plan.md) | Kullanıcı onayı, sonra R1 (sözlük çekirdeği ~60 terim) |
| Faz 10 M4–M8, nihai koşumlar | planlı; M4 temeli referans yapılandırma; M4 tasarımı v0.1 (`attn`/`gat` kodlu, `test_marl_m4` 14, koşu yok); M5 ön kaydı v0.2 (yalnız §2 P5c okuma kuralı + kararlar) | Faz 10 plan §3, §4; [phase-10-m3-design.md](phase-10-m3-design.md) §6.8; [phase-10-m5-design.md](phase-10-m5-design.md) | M4a bitti (gat KISMİ, attn GÖZLENMEDİ); M4b pilotu PİLOT_İŞARETİ (§6.4); M5 mekanizma tabanı kullanıcı kararı ([phase-10-m4-design.md](phase-10-m4-design.md)) |
| K27 pano: ajan türü seçimi + karşılaştırma | planlı (karar 2026-11-15) | Faz 10 plan §6 K27 | M2 + karar katmanından (K1–K10) sonra |
| Canlı karar paritesi (K13) | açık | Faz 10 plan §6 | Karar katmanında |
| Y30 SHAP ucu, Y31 altın birimi | açık (ürün) | `prediction/CLAUDE.md` | Tez sonucu değil |
| Tez Milestone 1 (literatür) | kısmen | [roadmap.md](roadmap.md) Milestone 1 | Dört 2026 tam metni, YÖK taraması |
