# ── QUALIA Production Dockerfile ──────────────────────────────────────────
# Multi-stage build: keeps the final image lean by separating build deps
# from runtime deps. The final container is ~400MB vs ~1.2GB single-stage.

# Stage 1: Builder — installs all dependencies including compile-time ones
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system deps needed for compiling some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first so Docker cache doesn't bust on code changes
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# Stage 2: Runtime — clean Python image with only what's needed to RUN
FROM python:3.11-slim AS runtime

WORKDIR /app

# Non-root user for security (never run prod containers as root)
RUN groupadd -r qualia && useradd -r -g qualia qualia

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create persistent data directory and set ownership
RUN mkdir -p /data/qualia_memory && chown -R qualia:qualia /data /app

# Switch to non-root user
USER qualia

# Expose API port
EXPOSE 8000

# Health check — Docker/ECS/Kubernetes will restart the container if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the FastAPI server with Uvicorn
# --workers 1: one process per container (scale horizontally, not vertically)
# --proxy-headers: trust X-Forwarded-* from load balancer
CMD ["uvicorn", "api.qualia_api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--proxy-headers", \
     "--log-level", "info"]
