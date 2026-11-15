# Sistem akış haritası

**Son güncelleme:** 2026-11-15 · `v2.0.0` · Golden CPU'da yenilendi (K1), Faz 10 kararları K6/K9/K12/K15/K24, Milestone 0' **m0p-v2 donduruldu** (`m0-protocol-v2`, 180 eğitim, §9.36). Sırada koşum (PPO, sonra A2C).

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
    MENV["MARL ortamı M2<br/>hisse başına ajan, CTDE"]:::planned
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
    FIN["Koşum ~2,4 sa 8 işçi<br/>+ hakem-2 sonuç okuması"]:::active
  end

  subgraph M2P["Makale 2"]
    B5["B5 nedensel ICEEMDAN"]:::planned
    K0K4["K0-K4 kolları<br/>ICEEMDAN x TATS"]:::planned
  end

  subgraph F10["Faz 10: MARL"]
    M1["M1 çerçeve spike'ı"]:::planned
    M2["M2 ortam + K karar katmanı"]:::planned
    M34["M3/M4 MAPPO, iletişim"]:::planned
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
| Milestone 0' koşumları | sürüyor (m0p-v2 **donduruldu**, `m0-protocol-v2`; v1 koşulmadan bırakıldı) | [milestone0-protocol-v2.md](milestone0-protocol-v2.md); plan §9.36 | PPO 90 + A2C 90 eğitim (~2,4 sa, 8 işçi) → hakem-2 sonuç okuması |
| Makale 2 (B5 + K0–K4) | planlı | plan §4 "Makale 2 deney tasarımı", §9.14 | B5 nedensel ICEEMDAN, sonra kollar |
| Faz 10 M1 çerçeve spike'ı | planlı (K9 kararlı) | Faz 10 plan §4 S1, K9 | BenchMARL/TorchRL önce, ayrı `venv-marl/`; Windows, bütçe |
| Faz 10 M2–M8, nihai koşumlar | planlı | Faz 10 plan §3, §4 | M1'den sonra |
| K27 pano: ajan türü seçimi + karşılaştırma | planlı (karar 2026-11-15) | Faz 10 plan §6 K27 | M2 + karar katmanından (K1–K10) sonra |
| Canlı karar paritesi (K13) | açık | Faz 10 plan §6 | Karar katmanında |
| Y30 SHAP ucu, Y31 altın birimi | açık (ürün) | `prediction/CLAUDE.md` | Tez sonucu değil |
| Tez Milestone 1 (literatür) | kısmen | [roadmap.md](roadmap.md) Milestone 1 | Dört 2026 tam metni, YÖK taraması |
