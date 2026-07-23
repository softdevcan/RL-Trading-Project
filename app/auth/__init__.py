"""Kimlik dogrulama / yetkilendirme katmani (Faz 7).

Modul haritasi
--------------
db.py          SQLAlchemy engine + oturum fabrikasi + init_db()
models.py      User / SessionToken / AuditLog tablolari
security.py    Sifre hash'leme (bcrypt), JWT uretimi/dogrulamasi, CSRF
service.py     Kullanici CRUD + kimlik dogrulama is mantigi + audit
deps.py        FastAPI bagimliliklari (get_current_user, require_role)
middleware.py  /dash ve /api icin oturum kapisi + sessiz token yenileme
routes.py      /auth/* uc noktalari ve login sayfasi
workspace.py   Kullanici bazli dizin cozumleyici (hibrit izolasyon)
bootstrap.py   Ilk admin olusturma
"""
