# ── Stage 1: build ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY pyproject.toml VERSION README.md /build/
COPY verifily_cli_v1/ /build/verifily_cli_v1/
COPY verifily_sdk/ /build/verifily_sdk/

RUN pip install --no-cache-dir ".[all]" && \
    pip install --no-cache-dir verifily_sdk/

# ── Stage 2: runtime ────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="Verifily Team"
LABEL description="Verifily ML data quality gate — API server"
LABEL version="1.0.0"

# Non-root user
RUN groupadd -r verifily && useradd -r -g verifily -d /app -s /sbin/nologin verifily

WORKDIR /app

# Copy installed packages and entry point from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/verifily /usr/local/bin/verifily
COPY --from=builder /build /app

# Persistence + workspace dirs
RUN mkdir -p /data /workspace && chown -R verifily:verifily /data /workspace /app

USER verifily

EXPOSE 8080

ENV VERIFILY_ENV=prod \
    VERIFILY_DATA_DIR=/data \
    VERIFILY_ALLOW_NONLOCAL=1

# Install curl for healthcheck (slim doesn't have it)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
USER verifily

# Health check using /health for liveness, /ready for readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8080/health >/dev/null || exit 1

CMD ["sh", "-c", "verifily serve --host 0.0.0.0 --port ${PORT:-8080} --allow-nonlocal"]
