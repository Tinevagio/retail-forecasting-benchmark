# syntax=docker/dockerfile:1.7
# Multi-stage Dockerfile for the retail-forecasting-benchmark API.
#
# Stage 1 (builder): installs uv, resolves dependencies from uv.lock into a
#   virtual environment. This stage has Python build tools and pulls in
#   compilers when needed (e.g. for native LightGBM bindings).
#
# Stage 2 (runtime): copies just the venv and the source code. Final image
#   ships only what the API needs at runtime, no build tooling.
#
# Final image size target: ~400-500 MB (vs ~2 GB for a single-stage build).

# ---- Stage 1: builder ------------------------------------------------------

FROM python:3.11-slim AS builder

# Faster, more deterministic Python; do not buffer stdout (so docker logs
# show output immediately).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

# Install uv, our resolver. We use the official installer rather than pip
# because uv is significantly faster and the install is self-contained.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /build

# Copy ONLY the dependency descriptors first. This way Docker's layer cache
# only invalidates on dependency changes, not on every code change — making
# rebuilds during development much faster.
COPY pyproject.toml uv.lock README.md ./

# Resolve and install dependencies into /build/.venv. --frozen ensures we
# match uv.lock exactly (no surprise version drift between local and image).
RUN uv sync --frozen --no-dev --no-install-project

# Now copy the actual source code and install the package itself.
COPY src/ ./src/
RUN uv sync --frozen --no-dev


# ---- Stage 2: runtime ------------------------------------------------------

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PATH="/app/.venv/bin:$PATH"

# Create a non-root user. Running services as root inside containers is a
# common-but-bad practice; even if the container is "isolated", a process
# escaping the container would land as root on the host.
RUN groupadd --system app && useradd --system --gid app --no-create-home app

WORKDIR /app

# Copy the virtual environment and the source from the builder stage.
# The venv contains uv-resolved deps; src is our own package code.
COPY --from=builder --chown=app:app /build/.venv /app/.venv
COPY --from=builder --chown=app:app /build/src /app/src
COPY --from=builder --chown=app:app /build/pyproject.toml /app/pyproject.toml

# At runtime, the artifacts directory is expected to be mounted as a volume
# (see docker-compose.yml). The default location matches the API's default.
ENV FORECASTING_ARTIFACTS_DIR=/app/artifacts

# Drop privileges before running anything.
USER app

# Expose the API port. This is documentary only — Docker doesn't actually
# open the port without `-p` or a compose mapping.
EXPOSE 8000

# Healthcheck: docker can poll /health and report the container as
# unhealthy if it stops responding. Useful for orchestrators and for local
# `docker ps` introspection.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" \
        || exit 1

# Bind to 0.0.0.0 so Docker port forwarding works (127.0.0.1 only listens
# inside the container, unreachable from the host).
CMD ["uvicorn", "forecasting.serving.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
