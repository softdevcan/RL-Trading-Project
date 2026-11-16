<!-- Kok CLAUDE.md'den tasindi (2026-11-14). Bu dizindeki bir dosyaya dokununca yuklenir. -->

# CLAUDE.md — tests/ (test komutlari)

Testler bagimsiz betik (`python tests/<ad>.py`, pytest degil). Parantezdeki sayi beklenen kontrol sayisi. RL/env testlerinde CPU icin `CUDA_VISIBLE_DEVICES=-1`.

## Running tests
```bash
python tests/test_env.py
python tests/test_ppo.py
python tests/test_all_algorithms.py
python tests/test_env_lookup_equivalence.py # RL env lookup cache bit-eslik (41 kontrol)
python tests/test_env_price_guards.py       # Gecersiz fiyat korumalari (26 kontrol)
python tests/test_training_eta.py           # Egitim suresi tahmini (50 kontrol)
python tests/test_training_status.py        # /train/status progress, sembol uyarisi,
                                           #        RL + tahmin egitimi durumunun sayfaya
                                           #        donunce geri gelmesi (19 kontrol)
python tests/test_daily_trading_tz.py     # Gunluk karar: CSV (tz'siz) + yfinance (tz'li)
                                           #        birlesimi, sessiz sembol dusmesi (18 kontrol)
python tests/test_paper_portfolio.py      # Kagit portfoy: ortalama maliyet,
                                           #        gerceklesmis/gerceklesmemis kar,
                                           #        cift uygulama bekcisi (43 kontrol)
python tests/test_panel_freshness.py      # Gunluk karar: tazelik olcutu (seans),
                                           #        taslak satir, kapsam esigi,
                                           #        panel onbellegi (21 kontrol)
python tests/test_trade_universe.py       # Gunluk karar: modelin sembol evreni model
                                           #        ADINDAN degil MODELDEN cozulur (22 kontrol)
python tests/test_dash_props.py             # dbc/dcc/html kwarg uyumu (4 kontrol)
python tests/test_auth.py                  # Faz 7: oturum akisi (29 kontrol)
python tests/test_workspace_isolation.py   # Faz 7: izolasyon + RBAC + CV deney dizini (20 kontrol)
python tests/test_prediction_regression.py # Faz 6: golden davranis dondurma; CPU'ya sabit (~1 dk), YESIL (K1 2026-11-15
                                           #    CPU'da yenilendi, §9.36). Bilincli degisiklikte --update (CPU); --gpu yalniz kiyas
python tests/test_tft_fast_vsn.py          # Faz 6: gruplanmis-VSN esdegerlik (12 kontrol)
python tests/test_train_batch_parallel.py  # Faz 6: batch paralellik + izolasyon (18 kontrol)
python tests/test_manifest_workspace.py    # Faz 6: manifest calisma alani cozumleme (13 kontrol)
python tests/test_hpo_resume.py            # Faz 6: HPO sqlite resume (12 kontrol)
python tests/test_cv_feature_selection.py  # Faz 9/B8: CV ozellik secimi kat ici; test y karisinca
                                           #          kume bit-es + mutasyon (13, < 1 dk)
python tests/test_tats_labels.py           # Faz 9/B10: TATS uyum girdileri HAM kapanis + gercek
                                           #          getiri; gunluk/haftalik, ICEEMDAN kolu, kisa OOF, mutasyon; K21: TATS'li/TATS'siz
                                           #          test + kaydet/yukle; Y25 gercek correct degerle; Y28 canli predict_next
                                           #          son satir = df.index[-1] gunluk/haftalik + mutasyon, stale_features bekcisi (43, ~1 dk)
python tests/test_dl_alignment.py          # Faz 9/B11: BiLSTM/TFT egitim<->cikarim penceresi ayni satir,
                                           #          yon etiketi ham hedeften (gunluk/haftalik), _fit -> ham y, target_type aktarimi, tohumlu mutasyon (18, ~1 dk)
python tests/test_prediction.py            # Tahmin zinciri (7 test; DL kucuk veride egitilmez, beyan);
                                           #          TEST 7 Y29: hedef tarih veriden, gercek h. seans, gecersiz kayit, mutasyon
python tests/test_macro_quality_flag.py    # Faz 6: makro kalite bayragi cache turu (14 kontrol)
python tests/test_macro_pit.py             # Faz 9/B1-B3: EVDS ayristirma, aciklanma
                                           #          gecikmesi, bfill yok (kesme
                                           #          degismezligi), artimli + BIS gecikmesi,
                                           #          mutasyon oz-testi, risksiz faiz (33, ag yok)
python tests/test_theme_contrast.py        # Faz 8: token kontrasti, kacak hex,
                                           #        ucuncu parti cakismasi, kendi
                                           #        CSS siniflarimiz, devre disi
                                           #        varyantlar (92 kontrol)
python tests/test_theme_preference.py      # Faz 8: 3 durumlu tema, sema gocu, CSRF (31 kontrol)
python tests/test_topbar.py                # Faz 8/G: ust cubuk, kirinti, arama,
                                           #          bildirimler (39 kontrol)
python tests/test_delete_artifacts.py      # Faz 8/I: model + optimizasyon kaydi
                                           #          silme, RBAC (32 kontrol)
python tests/test_account_profile.py       # Faz 8/F: profil ucu, oturum yonetimi,
                                           #          etkinlik kaydi, Dash callback
                                           #          smoke (84 kontrol)
python tests/test_eval_stats.py            # Faz 9/A: DSR makale ornegi, DM/LW boyut
                                           #          simulasyonu, MCS, rliable, PBO,
                                           #          BH, risksiz faiz (51 kontrol, ~45 sn)
python tests/test_data_integrity.py        # Faz 10/V: artefakt, hayalet gun, RL
                                           #          penceresi, gercek panel saglik (27)
python tests/test_env_missing_mode.py      # Faz 10/V: env eksik hucre modu (17)
python tests/test_rl_reproducibility.py    # Faz 9/C: tohum, deneme kaydi, veri
                                           #          anlik goruntusu, manifest, HPO sampler tohumu + mutasyon (43)
python tests/test_price_stats_window.py    # Faz 9/B9: val/test olcegi = egitim olcegi;
                                           #          env + rota + HPO uctan uca, mutasyon,
                                           #          main ile bit-es varsayilan (git show, 11
                                           #          yapilandirma), olcek kokeni, destek, AST
                                           #          bekcisi, sarmalayicili betikler (106)
python tests/test_indicator_warmup.py      # Faz 9/B7: gosterge isinma bfill'i — B0 kesme
                                           #          degismezligi, RL kapisi, olculen protokol (28)
python tests/test_k18_obs_health.py        # Faz 9/K18+K25: ADX wilder (2 bagimsiz referans),
                                           #          causal fiyat olcegi B0 + mutasyon, obs_health,
                                           #          main ile bit-es, rota uctan uca (109)
python tests/test_universe.py              # Faz 9/E: evren tek kaynak (E0, Y11) — panel sirasi, kod + pano
                                           #          listeleri; E2 KAP sektor gruplari + ham kopyadan bit-es yeniden uretim; E1 capa (bugunku XU030),
                                           #          duyuru ayristirici (3 bicim), geriye zincir + mutasyon, uyelik/maske/maruziyet SHA bagi,
                                           #          zincir + parse bit-es yeniden uretim, degismezler, 5 ceyreklik elle referans, dipnot, DOHOL tipi aralik, capa gunu; KAP VYK ayristirici + ceyreklik capraz denetim (36, ag yok)
python tests/test_reward_differential.py   # Faz 9/G1: Moody-Saffell D_t — denklem 13 (O(eta^2), eta ile dogrusal),
                                           #          denklem 16 sonlu fark, fazla getiri, isinma, 2 mutasyon; varsayilan
                                           #          'ema_level' main'deki reward_functions.py ile bit-es; nakitte taban; env + hpo_reward_mismatch tam odul tanimi (22)
python tests/test_regimes.py               # Faz 9/B6: nedensel XU100 rejim etiketi — B0 kesme degismezligi (sentetik +
                                           #          gercek), 2 mutasyon, eski Y3 metrigi yok, as-of, ozet; kayan tercil + USD kolu B0, episodes_min20 (17, ag yok)
python tests/test_calibration.py           # Faz 9/F1-F4: ECE (15 kutu) ve Brier elle, sicaklik T~3 bulur; ensemble
                                           #          legacy/calibrated, p_up_raw+p_up_cal, fiyat ayni, kaydet/yukle, test
                                           #          bozulunca yon modeli bit-es + arindirma 0 mutasyonu; AUC/BSS bilgi kapisi (18, ~1 dk)
python tests/test_pit_table.py            # Faz 9/P: PIT tahmin tablosu — B0 gunluk + haftalik (t sonrasi bozma), arindirmasiz
                                           #          taban ve meta mutasyonu B0'i kirar, K19 tarihleri, havuzlu T, acik maske,
                                           #          P2 kapsam, parquet + manifest, bootstrap p-degeri + BH; kapi v2 + I3 duyarlilik (capa kumesi bootstrap); P4 env adaptoru B0 + env uctan uca + mutasyon; Milestone 0' pozitif kontrol tablosu (33, ~1 dk; gercek panel)
python tests/test_baselines.py             # Faz 9/D1-D4: 1/N aylik + al-tut elle hesap, gec listeleme, maliyet izgarasi,
                                           #          reel (TUFE basamak) / USD (kur t-1) + ayni gun kur mutasyonu, A8 0,02 yok,
                                           #          gercek 2021 penceresi (19, ~10 sn)
python tests/test_rl_walk_forward.py       # Faz 9/R1 + P4 + M8 secim kurali: wf-v1 katlari, kat sinirinda B0 (purge 0
                                           #          kaniti), test secime sizmiyor, geri cagri
                                           #          bit-es (A2C/PPO/TD3/SAC), rota<->HPO ayni gozlem, kayit; rota P tablosuyla (DTF), argmax + son nokta iki test kaydi (98, ~1 dk)
python tests/test_marl_core.py            # Faz 10/M2 saf cekirdek: logit siniri 5+ln N, projeksiyon (500 rastgele), muhasebe
                                           #          elle hesap + maliyet bir kez + katki, K5 emir + apply_decisions gidis-donus,
                                           #          listeleme kilidi B0, satis engeli (taban acilis; maske B0, K5 bekcisi, v0.7 donem bagimli marj), gozlem B0 + mutasyon + K2 bit-es, panel (61, ~5 sn; ana venv)
venv-marl\Scripts\python tests/test_marl_env.py  # Faz 10/M2 PettingZoo ortami: parallel_api_test, acik hatalar, tohum,
                                           #          K2 60 adim, deger = prod(1+R), eksik hucre, held-out olcek, tam sayi <-> kesirli,
                                           #          listeleme kilidi, satis engeli + kontrol kolu, DSR yalniz alpha=0 + isinma bekcisi, TorchRL sarmalayici (36, ~10 sn; YALNIZ venv-marl)
venv-marl\Scripts\python tests/test_marl_m3.py  # Faz 10/M3 BenchMARL gorevi: tek grup, bolum sonu terminated=False +
                                           #          GAE bootstrap, MAPPO anahtar baglantisi, kisa MAPPO (iki elestirmen), tohum bit-es (12, ~1 dk)
python scripts/analyze_milestone0.py --self-test  # M0' m0p-v2 analizi sentetik kayitla: hucre/N sayimi, 2021 dogrulayici penceresi,
                                           #          sifir etkide red yok, PC060 tespiti, Q3 okuma kurali, eksik kayitta durur (9, ~5 sn)
```
