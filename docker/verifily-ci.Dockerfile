# Verifily CI Docker Image
# Pre-built image for running Verifily in CI pipelines
# Includes: verifily CLI, common dependencies, and CI-optimized settings

FROM python:3.11-slim

LABEL maintainer="Verifily"
LABEL description="Verifily CLI for CI/CD pipelines"
LABEL version="1.0.0"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install Verifily CLI
RUN pip install --no-cache-dir \
    httpx \
    typer \
    pyyaml \
    pydantic

# Copy Verifily source and install
COPY . /opt/verifily
WORKDIR /opt/verifily
RUN pip install -e .

# Create non-root user for security
RUN useradd -m -u 1000 verifily
USER verifily

# Set CI-optimized environment variables
ENV VERIFILY_LOG_FORMAT=json
ENV VERIFILY_LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1
ENV VERIFILY_CI=true

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD verifily doctor || exit 1

# Default command shows version and help
CMD ["sh", "-c", "verifily --version && verifily --help"]
