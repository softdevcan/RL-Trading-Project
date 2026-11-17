"""Faz 10 M2: cok ajanli portfoy ortami ve karar katmaninin SAF cekirdegi.

Tasarim: `docs/development/phase-10-m2-design.md` (tek gercek kaynak; bu paket onu uygular).

Paket iki katmandir ve bu ayrim bilinclidir (K9):
  - Saf moduller (`weights`, `accounting`, `orders`, `obs_spec`, `panel`) yalniz numpy (panel: pandas)
    kullanir. Hem ana `venv/` (canli karar, uretim) hem `venv-marl/` (egitim) bunlari import eder.
    Ortam ile canli karar AYNI fonksiyonu cagirir (K2, K5); parite bu sayede testle sinanir.
  - `env` modulu PettingZoo'ya baglidir ve YALNIZ `venv-marl/` icinde import edilir. Uretim kodu
    (`app/`, `dashboard/`) onu import etmez.

Bu dosya hicbir alt modulu import etmez: `import marl.orders` ana venv'de pettingzoo istemez.
`env/` paketi (eski tek ajan ortami) degismez; `env/__init__` `trading_env`'i hevesle yukledigi icin
yeni ortam oraya konmadi (plan §3.2'deki `env/marl_trading_env.py` yolundan sapma, tasarim §1).
"""
