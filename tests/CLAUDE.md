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
python tests/test_prediction_regression.py # Faz 6: golden davranis dondurma (GPU'da rebaseline: --update)
                                           # !! B11'den beri BILINEN NEDENLE KIRMIZI (DL hizalamasi, K1 secenek 1);
                                           #    Milestone 0' ile bir kez yenilenir — o zamana kadar --update YAPMA
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
                                           #          anlik goruntusu, manifest (39)
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
                                           #          listeleri; E2 KAP sektor gruplari + ham kopyadan bit-es yeniden uretim; E1 capa (bugunku XU030) (13, ag yok)
python tests/test_rl_walk_forward.py       # Faz 9/R1: wf-v1 katlari, kat sinirinda B0 (purge 0
                                           #          kaniti), test secime sizmiyor, geri cagri
                                           #          bit-es (A2C/PPO/TD3/SAC), rota<->HPO ayni gozlem, kayit (90, < 1 dk)
```
