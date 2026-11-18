# AI 100 Debugging Lab

A Socratic debugging tutor for non-CS students, backed by **Mistral Large 2** running
locally on the lab machine (no third-party API, no student logins).

- **Site:** https://zyzou8.github.io/AI_Debug/
- **Model:** `casperhansen/mistral-large-instruct-2407-awq` (123B, AWQ 4-bit) served by vLLM

## How the pieces fit

```
browser (GitHub Pages, public)
   |  https, reads api-config.json for the address
   v
cloudflared quick tunnel
   |
   v
tutor_proxy.py  (127.0.0.1:8800)   <-- origin lock, size caps, SSE
   |
   v
vLLM  (127.0.0.1:8000)  mistral-large-2
```

GitHub Pages is static, so it cannot reach `127.0.0.1`. The tunnel is what makes the
local GPU reachable from the public page; `tutor_proxy.py` is the only thing exposed.

## Running it

vLLM is a separate long-lived service. Start it first if it isn't up:

```bash
GPUS=2,3 nohup bash /data/zbz5399/envs/serve_mistral.sh > /data/zbz5399/logs/vllm.log 2>&1 &
```

Then bring the public tutor online:

```bash
bash /data/zbz5399/AI_Debug/server/run_tutor_stack.sh
```

That starts the proxy, opens the tunnel, writes the new address into
`api-config.json`, and pushes it so the live site picks it up (~1 min for Pages to
redeploy). Stop everything with `run_tutor_stack.sh stop`.

### The address changes on every restart

A cloudflared *quick* tunnel gets a new random hostname each start. `api-config.json`
exists so that address lives in exactly one place: the page fetches it at load, so
students always use the same link. Re-running the script republishes it automatically.

If the tutor must stay up across restarts without re-pushing, move to a **named**
Cloudflare tunnel with your own domain — that hostname is permanent.

## Local development

```bash
cd /data/zbz5399/AI_Debug && python3 -m http.server 8080
# then open http://localhost:8080/?api=https://<tunnel>.trycloudflare.com
```

Port 8080, not 8000 — vLLM owns 8000. `?api=` overrides `api-config.json`, and
`http://localhost:8080` is already in the proxy's allowed origins.

## Access control — read this before sharing the link

The only protection is a **CORS origin lock**: browsers are refused unless the page
is served from `https://zyzou8.github.io`. That keeps the open web from casually
pointing a chat UI at your GPUs, but **it is not authentication** — `curl` ignores
CORS, so anyone who learns the tunnel hostname can send requests directly.

Mitigating factors: the hostname is random, changes on every restart, and only exists
while you leave the tunnel running. Still, treat the endpoint as public. If this goes
beyond a pilot, add a per-IP rate limit and a shared key in `tutor_proxy.py`.

Request bounds that are enforced: 12,000-char prompt cap, 500-token reply cap,
120 s upstream timeout.

## Files

| Path | Purpose |
|---|---|
| `index.html` | The whole student-facing app (tasks, hint ladder, free Q&A, research log) |
| `api-config.json` | Current tunnel address; rewritten by the launch script |
| `server/tutor_proxy.py` | Origin-locked CORS + SSE proxy in front of vLLM |
| `server/run_tutor_stack.sh` | Starts proxy + tunnel, publishes the address |
| `.nojekyll` | Stops Pages from running Jekyll over the files |

## Pedagogy

The prompt rules live in `index.html` (`TUTOR_RULES`, `HINT_RULES`, `QA_RULES`). The
tutor is instructed never to give the corrected code or line, to cover one step at a
time, and to end every reply with exactly one question. The hint ladder (think it
through → trace it → similar example) unlocks one level at a time, so students must
engage at the lower level before getting more direct help.

`index.html` keeps an in-memory `events` array (asks, replies, hint level, difficulty
ratings) as a research log. **It is not persisted anywhere** — to collect data for a
study, POST it to a backend.

## Credit

Adapted from the `education_proj` AI Studio browser-extension prototype, which drove a
Penn State AI Studio tab via DOM automation. This version replaces that bridge with a
direct call to the locally hosted model, so no extension or AI Studio login is needed.
