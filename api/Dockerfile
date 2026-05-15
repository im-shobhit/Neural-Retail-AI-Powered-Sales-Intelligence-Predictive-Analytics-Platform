# ==========================================
# STAGE 1: BUILDER
# ==========================================
FROM python:3.12-slim as builder

WORKDIR /build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required for building psycopg2 and other C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev python3-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# ==========================================
# STAGE 2: PRODUCTION IMAGE
# ==========================================
FROM python:3.12-slim

# Create a non-root system user and group (NIST 800-53 AC-6 Compliance)
RUN addgroup --system appgroup && adduser --system --group appuser

WORKDIR /app

# Install ONLY runtime dependencies (no build tools = minimal attack surface)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 && \
    rm -rf /var/lib/apt/lists/*

# Copy pre-compiled wheels from the builder stage
COPY --from=builder /build/wheels /wheels
COPY --from=builder /build/requirements.txt .

# Install packages from wheels without reaching out to PyPI
RUN pip install --no-cache /wheels/*

# Copy application source code
COPY ./app ./app

# Change ownership to the non-root user
RUN chown -R appuser:appgroup /app

# Switch to the non-root user
USER appuser

EXPOSE 8000

# Run Uvicorn with 4 workers to maximize throughput and hit the < 1.5s P95 latency KPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]