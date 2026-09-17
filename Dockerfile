FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/

# Install CPU-only PyTorch separately.
RUN pip install --no-cache-dir \
    --default-timeout=1000 \
    --retries=10 \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install VoiceGuard and remaining dependencies.
RUN pip install --no-cache-dir \
    --disable-pip-version-check \
    --default-timeout=1000 \
    --retries=10 \
    -e .

EXPOSE 8000

CMD ["uvicorn", "voiceguard.api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]