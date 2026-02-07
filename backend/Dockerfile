# =============================================================================
# Backend Dockerfile - VDT GraphRec Pro
# =============================================================================
# Optimized for both development (volume-mounted libs) and production.
#
# Development mode (current): Uses volume-mounted /app/libs
# Production mode: Uncomment the builder stage and use multi-stage build
# =============================================================================

# --- Base Image (Slim Python) ---
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/libs \
    PATH="/app/libs/bin:${PATH}"

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# --- Development Stage (Current Pattern - Volume Mounted Libs) ---
FROM base as development

# Copy requirements for reference
COPY requirements.txt .

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application (single worker for dev)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# =============================================================================
# PRODUCTION BUILD
# =============================================================================

# --- Builder Stage ---
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
# Install into a local directory that can be copied
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Production Stage ---
FROM python:3.10-slim as production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/install/bin:${PATH}" \
    PYTHONPATH="/install/lib/python3.10/site-packages:/app/libs:${PYTHONPATH}"

WORKDIR /app

# Install runtime dependencies only (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /install

# Copy application code
COPY . .

# Create non-root user for security
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with Gunicorn for production (2 workers for student laptop is sufficient)
CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "-w", "2"]
