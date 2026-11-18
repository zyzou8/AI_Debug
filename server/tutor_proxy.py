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

# A tutoring session is multi-turn, so the whole transcript is resent each time.
# Trim it rather than reject it: a student deep in a conversation should never
# hit a wall. 60k chars is roughly 15k tokens, half the 32k context.
MAX_CONVO_CHARS = int(os.environ.get("TUTOR_MAX_CONVO_CHARS", "60000"))
MAX_TURNS = int(os.environ.get("TUTOR_MAX_TURNS", "40"))

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

    messages = _build_messages(body)
    stream = bool(body.get("stream", True))
    payload = {
        "model": MODEL,
        "messages": messages,
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


def _build_messages(body: dict) -> list[dict]:
    """Turn the page's payload into a chat transcript for vLLM.

    The tutoring rules belong in a `system` message: with them buried in the
    user turn the model treats them as discussable context and will hand over
    the corrected line when a student pushes for it.
    """
    system = (body.get("system") or "").strip()

    raw = body.get("messages")
    if raw is None:  # single-shot callers (curl, the old page)
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="missing 'messages' or 'prompt'")
        raw = [{"role": "user", "content": prompt}]
    if not isinstance(raw, list) or not raw:
        raise HTTPException(status_code=400, detail="'messages' must be a non-empty list")

    turns: list[dict] = []
    for m in raw[-MAX_TURNS:]:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        if len(content) > MAX_PROMPT_CHARS:
            raise HTTPException(
                status_code=413,
                detail=f"message too long ({len(content)} chars, max {MAX_PROMPT_CHARS})",
            )
        turns.append({"role": role, "content": content})

    if not turns:
        raise HTTPException(status_code=400, detail="no usable messages")
    if turns[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="the last message must be the student's")

    # Trim oldest-first when the transcript outgrows the budget, but never drop
    # turns[0]: that is the one carrying the assignment and the student's code,
    # so losing it is exactly the "tutor can't see my code" failure.
    def total() -> int:
        return len(system) + sum(len(t["content"]) for t in turns)

    while total() > MAX_CONVO_CHARS and len(turns) > 2:
        turns.pop(1)

    return ([{"role": "system", "content": system}] if system else []) + turns


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
