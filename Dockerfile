# syntax=docker/dockerfile:1
#
# Uygulama imaji = bagimlilik temeli (docker/base.Dockerfile: Python, torch CPU, requirements.txt) + KOD.
# Temel imaj yalniz requirements.txt ya da base.Dockerfile degisince yeniden derlenir ve etiketi icerik ozetidir
# (scripts/deploy/base_tag.sh). Bu yuzden bir surum guncellemesinde sunucu yalniz birkac MB'lik kod katmanini indirir;
# ~3 GB'lik kutuphane katmani bir kez iner (docs/deployment/ci-cd.md §6).
#
# CI/Deploy BASE_IMAGE'i build-arg ile verir. Yerelde:
#   docker build -f docker/base.Dockerfile -t rltrading-base:local .
#   docker build --build-arg BASE_IMAGE=rltrading-base:local -t rltrading:local .
ARG BASE_IMAGE=rltrading-base:local
FROM ${BASE_IMAGE}

# Imajin kodunun commit'i (imajda git yok). M8 ileri veri cekim kaydi bunu yazar (Faz 11 plan §3.12).
# CI/Deploy build-arg ile verir; elle derlemede bos kalir. Yalniz kod katmanlarini etkiler.
ARG GIT_COMMIT=""
ENV GIT_COMMIT=${GIT_COMMIT}

WORKDIR /app

# Application source only — data/models/results/logs come in as volumes at
# runtime (see docker-compose.yml), never baked into the image.
COPY --chown=appuser:appuser . .

# Create the writable runtime dirs in case the volumes are empty on first run.
# workspaces/ holds the per-user models/results/decisions, data/auth/ the user
# database — both must be writable by appuser or the first login fails.
#
# data/ alt dizinleri TEK TEK acilir cunku compose onlari named volume olarak
# baglar (data/ bir Python PAKETI oldugu icin komple mount EDILEMEZ — mount
# image'daki data/*.py modullerini golgeler ve uygulama ModuleNotFoundError
# ile acilmaz). Named volume ilk olusurken mount noktasinin sahipligini
# image'dan devralir; dizin image'da yoksa root:root olusur ve appuser
# (uid 10001) yazamaz. Bu yuzden hepsi burada olusturulup chown edilir.
# data/marl: Faz 11 M8 ileri veri deposu (VDS'de bind mount, plan §3.12); kod imajdan, veri bagli dizinden.
# Kod zaten COPY --chown ile appuser'in; burada yalniz yeni dizinler (tum /app'i chown -R etmek katmani sisirirdi).
RUN mkdir -p models results logs workspaces data/marl \
        data/bist data/gold data/macro data/fundamental \
        data/predictions data/live_trading data/auth \
    && chown appuser:appuser models results logs workspaces data data/marl \
        data/bist data/gold data/macro data/fundamental \
        data/predictions data/live_trading data/auth

USER appuser

EXPOSE 8000

# Serving in prod: single worker (our predictor/model caches are in-process,
# so multiple workers would each hold a separate cache and waste RAM), no
# --reload. Bind to 0.0.0.0 so the container port is reachable.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Liveness: the app is healthy once /health answers 200.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
