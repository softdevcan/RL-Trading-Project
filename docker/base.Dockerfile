# syntax=docker/dockerfile:1
# Bagimlilik (temel) imaji: Python + torch CPU + requirements.txt. Uygulama imaji (../Dockerfile) bunun ustune yalniz
# kodu ekler. Etiket = requirements.txt + bu dosyanin SHA ozeti (scripts/deploy/base_tag.sh); yalniz bunlar degisince
# yeniden derlenir. Boylece sunucu kutuphane katmanini (~3 GB acik) BIR KEZ indirir, sonraki surumlerde yalniz birkac
# MB'lik kod katmani gelir (docs/deployment/ci-cd.md §6).
#
#   docker build -f docker/base.Dockerfile -t ghcr.io/softdevcan/rl-trading-base:$(bash scripts/deploy/base_tag.sh) .

# ── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

# Build deps needed by some scientific wheels (numba/llvmlite, catboost, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# torch CPU indeksinden (CUDA surumunu cekmesin). Pin requirements.txt ile ayni (torch==2.9.0).
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.9.0

COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# ── Stage 2: runtime temeli ─────────────────────────────────────────────────
FROM python:3.13-slim

LABEL org.opencontainers.image.source="https://github.com/softdevcan/RL-Trading-Project" \
      org.opencontainers.image.description="RL Trading bagimlilik temeli (Python 3.13, torch CPU, requirements.txt)"

# libgomp1: lightgbm/xgboost/numba OpenMP. curl: konteyner HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
COPY --from=builder /opt/venv /opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
