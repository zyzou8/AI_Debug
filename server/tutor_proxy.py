#!/usr/bin/env python
"""Origin-locked CORS proxy that puts the local Mistral Large behind the public demo site.

The demo site is static (GitHub Pages) and therefore cannot reach vLLM on
127.0.0.1:8000. This process is the only thing exposed through the tunnel: it
accepts a tutor prompt, forwards it to vLLM as a chat completion, and streams
the reply back as SSE.

Run:  bash /data/zbz5399/ai100-tutor/server/run_tutor_stack.sh
Solo: /data/zbz5399/envs/vllm-mistral/bin/python tutor_proxy.py
"""

from __future__ import annotations

import json
import os

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("TUTOR_MODEL", "mistral-large-2")
PORT = int(os.environ.get("TUTOR_PORT", "8800"))

# Only these origins may call the proxy. The browser enforces this; curl does
# not, so treat it as "keeps the public web off the GPU", not as authentication.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "TUTOR_ALLOWED_ORIGINS",
        # 8080, not 8000: vLLM already owns 8000, so a local copy of the site
        # has to be served from a different port.
        "https://zyzou8.github.io,http://localhost:8080,http://127.0.0.1:8080",
    ).split(",")
    if o.strip()
]

# Functional request bounds: a 123B model answering an unbounded prompt would
# hold the GPUs for minutes and stall every other student.
MAX_PROMPT_CHARS = int(os.environ.get("TUTOR_MAX_PROMPT_CHARS", "12000"))
MAX_TOKENS = int(os.environ.get("TUTOR_MAX_TOKENS", "500"))
UPSTREAM_TIMEOUT = float(os.environ.get("TUTOR_TIMEOUT", "120"))

app = FastAPI(title="AI 100 Tutor proxy", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    max_age=86400,
)

client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def _startup() -> None:
    global client
    client = httpx.AsyncClient(base_url=VLLM_URL, timeout=UPSTREAM_TIMEOUT)


@app.on_event("shutdown")
async def _shutdown() -> None:
    if client is not None:
        await client.aclose()


def _check_origin(origin: str | None) -> None:
    """Reject cross-origin callers the browser would have blocked anyway.

    CORSMiddleware only *omits* the allow header for a bad origin, which shows
    up in the console as an opaque failure. Rejecting here makes the cause
    visible in the proxy log and in the response body.
    """
    if origin is None:
        # Same-origin fetches and curl send no Origin header. Allowed so the
        # health check and local testing keep working.
        return
    if origin not in ALLOWED_ORIGINS:
        raise HTTPException(status_code=403, detail=f"origin not allowed: {origin}")


@app.get("/health")
async def health() -> JSONResponse:
    """Report whether vLLM is actually reachable, not just whether we are up."""
    assert client is not None
    try:
        r = await client.get("/v1/models", timeout=10)
        r.raise_for_status()
        models = [m["id"] for m in r.json().get("data", [])]
        return JSONResponse(
            {"ok": True, "model": MODEL, "upstream_models": models, "origins": ALLOWED_ORIGINS}
        )
    except Exception as exc:  # upstream down or restarting
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)


@app.post("/api/tutor")
async def tutor(request: Request, origin: str | None = Header(default=None)):
    _check_origin(origin)
    assert client is not None

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body must be JSON")

    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="missing 'prompt'")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"prompt too long ({len(prompt)} chars, max {MAX_PROMPT_CHARS})",
        )

    stream = bool(body.get("stream", True))
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        # Low but not greedy: the tutor should vary its phrasing across hint
        # levels without drifting off the "never give the fix" rules.
        "temperature": float(body.get("temperature", 0.3)),
        "top_p": 0.9,
        "max_tokens": min(int(body.get("max_tokens", MAX_TOKENS)), MAX_TOKENS),
        "stream": stream,
    }

    if not stream:
        r = await client.post("/v1/chat/completions", json=payload)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"upstream {r.status_code}: {r.text[:300]}")
        data = r.json()
        return JSONResponse({"text": data["choices"][0]["message"]["content"]})

    async def events():
        """Re-emit vLLM's SSE as {delta}/{done}/{error} frames.

        Translating here keeps the page from having to know the OpenAI chunk
        shape, and lets upstream errors surface as a readable message instead
        of a truncated stream.
        """
        try:
            async with client.stream("POST", "/v1/chat/completions", json=payload) as r:
                if r.status_code != 200:
                    detail = (await r.aread()).decode(errors="replace")[:300]
                    yield _sse({"error": f"upstream {r.status_code}: {detail}"})
                    return
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = line[6:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        delta = json.loads(chunk)["choices"][0]["delta"].get("content")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if delta:
                        yield _sse({"delta": delta})
            yield _sse({"done": True})
        except Exception as exc:
            yield _sse({"error": str(exc)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Without this the tunnel/proxy layer may buffer the whole reply,
            # defeating the point of streaming.
            "X-Accel-Buffering": "no",
        },
    )


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
