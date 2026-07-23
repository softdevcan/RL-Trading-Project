# syntax=docker/dockerfile:1

# ── Stage 1: builder ────────────────────────────────────────────────────────
# Compile/download all wheels into a virtualenv here so the runtime image
# doesn't carry build toolchains.
FROM python:3.13-slim AS builder

# Build deps needed by some scientific wheels (numba/llvmlite, catboost, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv keeps the runtime copy simple: just grab /opt/venv.
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install torch first from the CPU-only index so we never pull the multi-GB
# CUDA build. Pin matches requirements.txt (torch==2.9.0).
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.9.0

# Remaining deps. torch is already satisfied so pip skips it. The CPU index is
# passed again so any torch-adjacent resolution also stays CPU.
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

# libgomp1: OpenMP runtime required by lightgbm/xgboost/numba at import time.
# curl: used by the container HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the ready-built venv from the builder stage.
ENV VIRTUAL_ENV=/opt/venv
COPY --from=builder /opt/venv /opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Never buffer stdout/err (clean docker logs); no .pyc writes.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root user for security.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

# Application source only — data/models/results/logs come in as volumes at
# runtime (see docker-compose.yml), never baked into the image.
COPY --chown=appuser:appuser . .

# Create the writable runtime dirs in case the volumes are empty on first run.
# workspaces/ holds the per-user models/results/decisions, data/auth/ the user
# database — both must be writable by appuser or the first login fails.
RUN mkdir -p models results logs data/predictions workspaces data/auth \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Serving in prod: single worker (our predictor/model caches are in-process,
# so multiple workers would each hold a separate cache and waste RAM), no
# --reload. Bind to 0.0.0.0 so the container port is reachable.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Liveness: the app is healthy once /health answers 200.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
