# CLAUDE.md - RL Trading Project

BIST-30 icin derin pekistirmeli ogrenme ile algoritmik islem + tahmin destekli gozlem; tez ve
makale calismasi (Ansari vd. 2024 replikasyonu -> Faz 10 MARL). Tamamlanan fazlar: 1 (POC),
2 (tahmin sistemi), 3 (uretim iyilestirmeleri), 6 (performans), 7 (kimlik + cok kullanici),
8 (UI/UX). **Suren: Faz 9 (arastirma gecerliligi)** — durum ve olcumler
`docs/development/phase-9-research-validity-plan.md` §9 (son alt bolum en guncel); Faz 10 plani
`docs/development/phase-10-marl-plan.md`. Dokuman indeksi `docs/README.md`.

Kullanici hangi dilde yazarsa o dilde yanit ver.

## Calistirma
- Sunucu: `python run_server.py` -> http://localhost:8000 (pano `/dash/`, API `/api/*`)
- Testler bagimsiz betik: `python tests/<ad>.py` (pytest degil). Hangi test neyi denetler ve
  beklenen kontrol sayilari: `tests/CLAUDE.md`. Bir degisiklikten sonra o alanin testini oradan bul.
- **Tahmin golden'i (`tests/test_prediction_regression.py`) BILINEN NEDENLE KIRMIZI** (B11 DL
  hizalamasi, Faz 9 §9.18; K1 secenek 1). Milestone 0' ile bir kez yenilenir — o zamana kadar
  `--update` YAPMA ve golden'i baska bir degisikligin kaniti sayma.
- `env/` = proje kodu (Gymnasium ortami). `venv/` = sanal ortam; okuma, degistirme.
- **Surum ve yayin:** tek kaynak `app/__init__.py` (`__version__`, SemVer: paket = MINOR). main korumali
  (yalniz PR). Birlesmeden sonra tag + `gh release --generate-notes`: `docs/development/versioning.md`.

## Alan kurallari nerede (o dizine dokununca otomatik yuklenir)
Bir dosyayi duzenlemeden once o alanin dosyasi baglamda degilse oku:
- **RL egitimi, HPO, rota (`app/api/routes/trading.py`), bolme, fiyat olcegi, veri kapisi** -> `env/CLAUDE.md`
- **Degerlendirme, deneme kaydi, protokol, risksiz faiz, tohum** -> `evaluation/CLAUDE.md`
- **Tahmin hatti (ensemble, TATS, DL, CV)** -> `prediction/CLAUDE.md`
- **Veri cekme (yfinance, EVDS makro)** -> `data/CLAUDE.md`
- **Kimlik/RBAC, calisma alani, hesap, yapit silme, ETA, kagit portfoy, gunluk karar** -> `app/CLAUDE.md`
- **Pano, tema, UI, uzun isler ve sayfa gezinmesi** -> `dashboard/CLAUDE.md`
- **Tez, literatur, kaynak** -> `docs/thesis/CLAUDE.md`

## Calisma alanlari (Faz 7)
Kullaniciya ait her yapit (model, sonuc, karar, manifest, deney kaydi) `workspaces/<user_id>/`
altina, `app/auth/workspace.py` cozumleyicisiyle yazilir (`models_dir()`, `results_dir()`,
`ws.use_workspace(user_id)`); piyasa verisi (`data/bist`, `data/macro`) ortaktir. Ayrinti `app/CLAUDE.md`.

## Do NOT
- Read or modify files inside `venv/`
- Add `models/`, `results/`, `logs/`, `workspaces/`, `data/auth/` to git
- `AUTH_ENABLED=False` ile sunucuya cikma — pano ve tum API herkese acik kalir
- Yeni yazma ucu eklerken `RequireWriter`/`RequireAdmin` bagimliligini atlama
  (Faz 8/I'de `/api/hyperopt/start`, Faz 9 §9.13'te `/analysis/generate-report` RBAC'siz yakalandi)
- Arka plan gorevine kullanici kimligini tasimayi unutma (`ws.use_workspace(user_id)`) —
  aksi halde dosyalar yanlis calisma alanina yazilir
- `models/`, `results/`, `data/live_trading` gibi yollari koda sabitleme;
  `app/auth/workspace.py` cozumleyicisini kullan
- Break existing state space structure when modifying `env/trading_env.py`
- Add hardcoded `macro_features=6` — global macro (VIX/US10Y/DXY) sadece prediction pipeline'a gider, RL state space'e eklenmez (trained model uyumluluğu)
- `use_atr_sizing` ve `use_kelly` varsayılan olarak False — mevcut eğitimli modeller bozulmaz
- `users` tablosuna sütun eklerken `app/auth/db.py::_ADDITIVE_COLUMNS`'a da ekle —
  alembic yok, `create_all()` var olan tabloyu değiştirmez

## Akademik gecerlilik (kodda da gecerli cekirdek; tez kurallari `docs/thesis/CLAUDE.md`)
- **"Leakage yok" DEME** — dogrulanmis sizinti kaynaklari var (`docs/development/roadmap.md` →
  "Bilinen Metodolojik Aciklar"; Faz 9 plani §4 B paketi).
- **Faz 9 ve Faz 10 planlari ONAYLANDI (2026-11-06)**, §8 hakem-2 eklemeleriyle. Kararlar: K1
  varsayilani cevir + golden'i Milestone 0' ile bir kez yenile; K17 RL penceresi 2013-06; K23 anahtar
  koddan kaldirildi (PR #3), yenilenmeyecek. Is planin paket SIRASIYLA yapilir; planda olmayan bir
  duzeltmeye girisme, plana ekletip yap. Tarihsel §9 alt bolumlerine dokunma; yeni durum yeni alt bolume.
- Kaynak ekleme: `/kaynak-ekle` skill'i. Metodolojik elestiri: **hakem paneli** —
  `hakem-finans` + `hakem-rl` (sonnet, dar kapsam, yalnizca alanina giren pakette) ve son
  kapi `hakem-2` (opus). Once ~5 bin token'lik inceleme paketi yaz, uzun olcumleri SEN yap
  ve JSON'u `docs/development/measurements/` altinda arsivle; duzeltme sonrasi ayni hakemi
  SendMessage ile surdur. Akis, paket sablonu ve maliyet kaydi: `docs/development/review-panel.md`.
- Hook'lar (`.claude/settings.json` → `.claude/hooks/guard.py`): uretilen literatur dosyalarini
  elle duzenleme engellenir; tez metnine yeni "sizinti yok" iddiasi onay ister;
  `env/trading_env.py` degisince env bekci testleri otomatik kosar.
