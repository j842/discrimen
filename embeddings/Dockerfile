FROM python:3.12-slim

WORKDIR /app

# ca-certificates for the HuggingFace download on first start; curl for the
# healthcheck below. The slim base has neither.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Requirements ahead of the source: fastembed drags in onnxruntime and is by far
# the slowest layer, and main.py changes far more often than the pins do.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./

# fastembed defaults this to /tmp/fastembed_cache, which a container throws away
# on every recreate — a fresh model download (~65MB on disk for the default
# quantised bge-small ONNX) each time. Pinning it somewhere a volume can be
# mounted is the whole point, and a VOLUME on the conventional
# /root/.cache/fastembed does nothing at all without it.
ENV FASTEMBED_CACHE_PATH=/root/.cache/fastembed
VOLUME /root/.cache/fastembed

EXPOSE 8586

# The FastAPI startup hook warms the model before uvicorn starts serving, so on
# a cold cache nothing answers until the download finishes. start-period covers
# that; without it the container is marked unhealthy on first run and anything
# gating on health gives up long before the model lands.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${EMBEDDINGS_PORT:-8586}/health" > /dev/null || exit 1

CMD ["python", "main.py"]
