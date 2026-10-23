# Literatür Kaynak Tablosu — Seçilmiş Kaynakça

**Tarih:** 2026-10-07 · **Durum:** Seçim kuralı uygulandı; tezin ve makalelerin kaynak tabanı bu belgedir. Diğer belgeler (vision-and-scope §2 ve §9, doğrulama raporu, Faz 9 planı) buna göre güncellenecek.
**Kapsam:** 169 kayıt değerlendirildi → **148 seçildi**, 21 elendi (Ek A).

**Belgeyi okuma sırası:**
1. §2: seçilenlerin özeti ve kapsam matrisi. Matris, her tez/kod bileşeninin hangi kaynaklarla dayandırıldığını gösterir.
2. §3: konu gruplarına göre tam tablolar.
3. §4: dergi indekslerinin doğrulama durumu.
4. Ek A: elenen kayıtlar ve yerlerine kullanılacak kaynaklar.

---

## 1. Yöntem

### 1.1 Kaynak havuzu nasıl oluşturuldu

| Adım | Ne yapıldı | Sonuç |
|---|---|---|
| 1. Önceki tarama | Ekim 2026 literatür taraması (121 kaynak) + bağımsız doğrulama ([literature-review-verification.md](literature-review-verification.md)) | vision-and-scope §9'daki 73 kayıt |
| 2. Faz 9 planı | Değerlendirme yöntemlerinin özgün kaynakları (DM, HLN, Newey-West, MCS, FDR…) | +14 |
| 3. 2026 anahtar kelime taraması | arXiv + OpenAlex, 2026 Ocak–Eylül, G1–G6 sorguları | +9 (Hung & Huang, Öztürk, Yuan vd., Taşkıran…) |
| 4. **İleri atıf taraması** | 25 çekirdek makaleye 2025–2026'da atıf yapan 247 çalışma (OpenAlex `cites:`) başlıkla süzüldü, ilgililerin özeti okundu | +10 (Sattar vd., CAROC, Xiong vd., Pippas vd., Hu vd. TKDE…) |
| 5. **Kapsam denetimi** | Kodda kullanılan her yöntemin ve tezdeki her iddianın kaynağı arandı; eksikler eklendi | +63 (PPO, A2C, TD3, XGBoost, LightGBM, CatBoost, LSTM, stacking, SHAP, Optuna, Markowitz, Hamilton, Harvey-Liu-Zhu, hayatta kalma yanlılığı…) |

### 1.2 Veri kaynakları

| Alan | Kaynak | Not |
|---|---|---|
| Başlık, yazar, yıl, yayın yeri, ISSN | OpenAlex (DOI ya da arXiv kimliğiyle; kimliksizlerde başlık araması); arXiv API; Crossref | Şüpheli eşleşmeler elle düzeltildi (§5) |
| Atıf | Semantic Scholar · OpenAlex, 2026-10-07 | OpenAlex konferans makalelerinde atıfı eksik sayıyor (TarMAC: S2 582, OpenAlex 25). Karar eşiklerinde ikisinden büyük olan kullanıldı. Google Scholar genelde daha yüksek |
| Anahtar kelimeler | OpenAlex `keywords` (makale düzeyinde, otomatik) | Yazar anahtar kelimeleri ücretsiz bir API'de yok |
| SCIE / SSCI | Clarivate MJL (elle, 2026-10-07) + Wikidata P8875 | `[MJL]` elle kontrol edildi · `[WD]` Wikidata'da kayıtlı · `[doğrula]` henüz kontrol edilmedi (§4.2) |
| Konferans düzeyi | CORE sıralaması (bilgiye dayalı) | MARL'ın temel yöntemleri dergide değil A* konferanslarda |

### 1.3 Kademeler, roller ve seçim kuralı

| Kademe | Tanım |
|---|---|
| **K1** | SCIE / SSCI dergi |
| **K2** | Üst düzey hakemli konferans (CORE A*/A) |
| **K3** | Diğer hakemli: TR Dizin / ESCI / Scopus dergisi, LNCS/LNNS bildirisi, kitap, sıralamasız konferans, genişletilmiş özet |
| **K4** | Hakemsiz: arXiv/SSRN ön baskısı, tez, depo kaydı |

| Rol | Tanım |
|---|---|
| **Z** zorunlu | Özgünlük sınırını çizen ya da doğrudan öncül olan çalışma |
| **Y** yöntem | Uyguladığımız yöntemin, algoritmanın ya da yazılımın özgün kaynağı |
| **B** bağlam | Literatürü tanımlayan, karşılaştırma için anılan çalışma |
| **İ** ikincil | Konumlandırma ya da kapsam dışı gerekçesi |

**Seçim kuralı (uygulandı):**

1. **Z ve Y kademeden bağımsız tutulur.** Bunlar atılırsa ya özgünlük iddiası savunulamaz ya da kullanılan yöntem kaynaksız kalır.
2. **B rolü için K1 ya da K2 gerekir.** Ayrıca yayından en az 3 yıl geçmişse Semantic Scholar'da yılda en az 5 atıf aranır. Son iki yılın kayıtlarında atıf henüz birikmediği için ölçüt yayın yeridir.
3. **İstisnalar:**
   - Yılda 20'den fazla atıf almış K3/K4 kayıt kalır (yerleşik etki; ör. Jiang vd. 2017, FinRL).
   - BIST/Türkiye grubunda K1/K2 yeterlidir; yerel literatür dar.
4. **İ rolü elenir.** Gerekirse tezin "kapsam dışı" paragrafında toplu anılır.
5. **Çıktıya göre esneklik:** Makale 4 (Türkçe, ulusal dergi) için TR Dizin dergileri uygun kaynaktır; Ek A'daki TR Dizin kayıtları orada kullanılabilir.

> **Konferanslar hakkında:** SCIE/SSCI ölçütü dergiler içindir. YÖK ve doçentlik ölçütleri **bizim yayınlarımızın** nerede çıkacağıyla ilgilidir; kaynakçadaki A* konferans makaleleri bunları zayıflatmaz. TarMAC (ICML), MAPPO (NeurIPS) ve GAT (ICLR) gibi temel kaynakların dergi karşılığı yok.

