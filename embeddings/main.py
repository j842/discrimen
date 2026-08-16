"""llm-embeddings: OpenAI-compatible /v1/embeddings worker.

Backed by fastembed (ONNX Runtime, CPU). Self-registers with llm-router on
startup with feature flag `embeddings` so the router's /v1/embeddings handler
can find it.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import struct
import time
from typing import Iterable, List

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastembed import TextEmbedding
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("llm-embeddings")

PORT = int(os.environ.get("EMBEDDINGS_PORT", "8586"))
MODEL_NAME = os.environ.get("EMBEDDINGS_MODEL", "BAAI/bge-small-en-v1.5")
API_TOKENS = [t.strip() for t in os.environ.get("EMBEDDINGS_API_TOKENS", "").split(",") if t.strip()]

ROUTER_URL = os.environ.get("LLM_ROUTER_URL", "").rstrip("/")
ROUTER_WORKER_TOKEN = os.environ.get("ROUTER_WORKER_TOKEN", "")
WORKER_URL = os.environ.get("LLM_WORKER_URL", "")
WORKER_ID = os.environ.get("LLM_WORKER_ID", "llm-embeddings")
WORKER_MODEL = os.environ.get("LLM_WORKER_MODEL", "bge-small-en-v1.5")
WORKER_FEATURES = [
    f.strip() for f in os.environ.get("LLM_WORKER_FEATURES", "embeddings").split(",") if f.strip()
]
WORKER_TTL = int(os.environ.get("LLM_WORKER_TTL_SECONDS", "86400") or "86400")

REGISTER_INTERVAL = 60  # seconds between keepalive registrations

app = FastAPI(title="llm-embeddings", version="1.0.0")
_model: TextEmbedding | None = None


def _model_handle() -> TextEmbedding:
    """Lazy model load — first request takes a beat, subsequent are instant."""
    global _model
    if _model is None:
        log.info("loading embedding model %s", MODEL_NAME)
        _model = TextEmbedding(model_name=MODEL_NAME)
        log.info("model loaded")
    return _model


def _check_auth(auth_header: str | None) -> None:
    """Optional per-worker bearer-token check. Open if no tokens configured."""
    if not API_TOKENS:
        return
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(401, "unauthorized")
    if auth_header[len("Bearer "):] not in API_TOKENS:
        raise HTTPException(401, "unauthorized")


class EmbeddingsRequest(BaseModel):
    model: str | None = None
    input: str | List[str]
    encoding_format: str = "float"  # "float" | "base64"
    user: str | None = None
    dimensions: int | None = Field(default=None, description="Ignored; the underlying model fixes the dim.")


def _as_list(value: str | List[str]) -> List[str]:
    return [value] if isinstance(value, str) else list(value)


def _encode_base64(vec: Iterable[float]) -> str:
    """OpenAI's base64 encoding is little-endian float32 packed contiguously."""
    floats = list(vec)
    packed = struct.pack(f"<{len(floats)}f", *floats)
    return base64.b64encode(packed).decode("ascii")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME, "loaded": _model is not None}


@app.get("/v1/health")
def v1_health() -> dict:
    return health()


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingsRequest, authorization: str | None = Header(default=None)) -> dict:
    _check_auth(authorization)
    texts = _as_list(req.input)
    if not texts or any(not isinstance(t, str) or not t for t in texts):
        raise HTTPException(400, "`input` must be a non-empty string or list of non-empty strings")

    model = _model_handle()
    # fastembed.embed() is a generator over numpy arrays. Run in a worker
    # thread so we don't block the event loop on tokenisation + ONNX run.
    vectors = await asyncio.to_thread(lambda: [v.tolist() for v in model.embed(texts)])

    data = []
    for idx, vec in enumerate(vectors):
        payload = _encode_base64(vec) if req.encoding_format == "base64" else vec
        data.append({"object": "embedding", "embedding": payload, "index": idx})

    # Token usage: fastembed doesn't expose it cheaply; report rough byte count.
    rough_tokens = sum(max(1, len(t.split())) for t in texts)
    return {
        "object": "list",
        "data": data,
        "model": req.model or MODEL_NAME,
        "usage": {"prompt_tokens": rough_tokens, "total_tokens": rough_tokens},
    }


async def _register_with_router() -> bool:
    if not ROUTER_URL:
        log.info("LLM_ROUTER_URL not set — skipping router registration")
        return False
    if not WORKER_URL:
        log.warning("LLM_WORKER_URL not set — cannot register with router")
        return False
    payload = {
        "id": WORKER_ID,
        "url": WORKER_URL,
        "model": WORKER_MODEL,
        "features": WORKER_FEATURES,  # ["embeddings"] — how the router finds this worker
        "health_path": "/health",
        "ttl_seconds": WORKER_TTL,
        "api_key": "",
    }
    headers = {"Content-Type": "application/json"}
    if ROUTER_WORKER_TOKEN:
        headers["Authorization"] = f"Bearer {ROUTER_WORKER_TOKEN}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{ROUTER_URL}/backends/register", json=payload, headers=headers)
            if resp.status_code >= 400:
                log.warning("router registration failed: %s %s", resp.status_code, resp.text[:200])
                return False
            log.info("registered with router at %s as %s", ROUTER_URL, WORKER_ID)
            return True
    except Exception as exc:  # pragma: no cover - best-effort
        log.warning("router registration error: %s", exc)
        return False


async def _register_until_up() -> None:
    """Register, retrying fast until the router answers, then keep alive slowly.

    The two are separate loops because the failures mean different things. A
    failure at startup usually means the router is not up yet — routine when
    both come up together, and every second spent waiting is a second the
    router's auto-routing runs degraded, because it needs this worker to
    classify prompts at all. So back off quickly there. A failure later means
    the router went away, which the keepalive interval already covers.
    """
    delay = 1.0
    while not await _register_with_router():
        if not ROUTER_URL or not WORKER_URL:
            return  # nothing to register against; the warning is already logged
        await asyncio.sleep(delay)
        delay = min(delay * 2, REGISTER_INTERVAL)
    while True:
        await asyncio.sleep(REGISTER_INTERVAL)
        await _register_with_router()


@app.on_event("startup")
async def _startup() -> None:
    # Warm the model so the first request isn't penalised.
    await asyncio.to_thread(_model_handle)
    # Registration runs in the background rather than inline: it can block for
    # the retry ladder above, and the worker is ready to serve embeddings the
    # moment the model is warm.
    asyncio.create_task(_register_until_up())


if __name__ == "__main__":
    import uvicorn

    log.info("llm-embeddings starting on :%d (model=%s)", PORT, MODEL_NAME)
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")
