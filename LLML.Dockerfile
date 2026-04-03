# ── Stage 1: build ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY LLML/requirements.txt ./

# Uncomment to enable CUDA GPU offload (requires nvidia/cuda base):
# ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

# libgomp is required at runtime by llama-cpp-python for OpenMP threading
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

WORKDIR /app

# Copy LLML source
COPY LLML/ ./LLML/

# Copy default config — override at runtime with:
#   -v ./ai-config.yaml:/app/ai-config.yaml
COPY ai-config.yaml ./ai-config.yaml

WORKDIR /app/LLML

# Mount .gguf model files here and reference them in ai-config.yaml
VOLUME ["/models"]

EXPOSE 3000

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py", "--config", "/app/ai-config.yaml", "--port", "3000"]
