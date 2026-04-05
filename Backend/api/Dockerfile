# ─────────────────────────────────────────────────────────
#  Arkon API — production Dockerfile
# ─────────────────────────────────────────────────────────
#  Build context must be the REPO ROOT so the governance
#  package is available:
#
#    docker build -f apps/api/Dockerfile -t arkon-api .
#
#  Run:
#    docker run --env-file apps/api/.env -p 8000:8000 arkon-api
# ─────────────────────────────────────────────────────────

# ── Stage 1: build ───────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed for some Python wheels (bcrypt, reportlab, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first for layer caching
COPY apps/api/requirements.txt  /build/api-requirements.txt
COPY packages/governance/pyproject.toml /build/gov-pyproject.toml

# Install all Python deps into a virtual-env
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /build/api-requirements.txt

# Copy the governance package and install it
COPY packages/governance /build/packages/governance
RUN pip install --no-cache-dir /build/packages/governance

# ── Stage 2: runtime ────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user
RUN groupadd --gid 1000 arkon && \
    useradd  --uid 1000 --gid arkon --create-home arkon

WORKDIR /app

# Copy the pre-built venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Copy the API source
COPY apps/api/ /app/

# Data & artifacts directories (mount a volume in production)
RUN mkdir -p /data /app/artifacts && \
    chown -R arkon:arkon /data /app/artifacts

# Default env vars (override at runtime via --env-file or -e)
ENV DATABASE_URL="sqlite:////data/arkon.db" \
    ARKON_ARTIFACTS_DIR="/app/artifacts" \
    CORS_ORIGINS="*"

USER arkon

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
