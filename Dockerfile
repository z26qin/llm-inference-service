# =============================================================================
# LLM Inference Service Dockerfile
# =============================================================================
# Multi-stage build for production-grade container with vLLM support
# Optimized for NVIDIA GPU inference

# -----------------------------------------------------------------------------
# Stage 1: Base image with CUDA support
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# Stage 2: Builder - Install dependencies
# -----------------------------------------------------------------------------
FROM base AS builder

WORKDIR /build

# Install build dependencies
RUN pip install --no-cache-dir hatchling

# Copy project files
COPY pyproject.toml .
COPY app/ ./app/

# Install the package and its dependencies
RUN pip install --no-cache-dir . \
    && pip install --no-cache-dir vllm>=0.3.0

# -----------------------------------------------------------------------------
# Stage 3: Production image
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS production

# Labels
LABEL maintainer="Aaron Qin <z26qin@github.com>"
LABEL description="Production-grade LLM Inference Service"
LABEL version="0.1.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create non-root user for security
RUN groupadd -r llmuser && useradd -r -g llmuser llmuser

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=llmuser:llmuser app/ ./app/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    # vLLM settings
    MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    TENSOR_PARALLEL_SIZE=1 \
    GPU_MEMORY_UTILIZATION=0.90 \
    MAX_NUM_SEQS=256 \
    # Server settings
    HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1 \
    # HuggingFace cache
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

# Create cache directory
RUN mkdir -p /app/.cache/huggingface && chown -R llmuser:llmuser /app/.cache

# Switch to non-root user
USER llmuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# -----------------------------------------------------------------------------
# Stage 4: Development image (optional)
# -----------------------------------------------------------------------------
FROM base AS development

WORKDIR /app

# Install development dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    DEBUG=true

# Expose port
EXPOSE 8000

# Run with reload for development
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
