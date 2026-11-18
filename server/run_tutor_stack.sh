#!/usr/bin/env bash
# Bring the public tutor online:
#   vLLM (already running) <- tutor_proxy <- cloudflared tunnel <- GitHub Pages
#
#   bash /data/zbz5399/AI_Debug/server/run_tutor_stack.sh
#
# A cloudflared quick tunnel gets a NEW random hostname every start, so the last
# step writes that hostname into api-config.json and pushes it. The live page
# reads that file on load, which is why students never need a new link.
#
# Stop everything:  bash run_tutor_stack.sh stop
set -uo pipefail

REPO=/data/zbz5399/AI_Debug
SERVER="$REPO/server"
LOGS=/data/zbz5399/logs
VENV=/data/zbz5399/envs/vllm-mistral/bin/python
CLOUDFLARED=/data/zbz5399/cloudflared/cloudflared
PORT="${TUTOR_PORT:-8800}"
PUSH="${PUSH:-1}"   # PUSH=0 to test locally without touching GitHub

PROXY_PID="$LOGS/tutor-proxy.pid"
TUNNEL_PID="$LOGS/tutor-tunnel.pid"
PROXY_LOG="$LOGS/tutor-proxy.log"
TUNNEL_LOG="$LOGS/tutor-tunnel.log"

mkdir -p "$LOGS"

stop_all() {
  for f in "$TUNNEL_PID" "$PROXY_PID"; do
    [ -f "$f" ] || continue
    pid=$(cat "$f")
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null && echo "stopped pid $pid"; fi
    rm -f "$f"
  done
}

if [ "${1:-start}" = "stop" ]; then stop_all; exit 0; fi

# vLLM is a separate, long-lived service: this script never starts or stops it,
# it only refuses to publish a tunnel that would 503 on every request.
echo "==> checking vLLM on 127.0.0.1:8000"
if ! curl -sf -m 10 http://127.0.0.1:8000/v1/models >/dev/null; then
  echo "ERROR: vLLM is not responding. Start it first:" >&2
  echo "  GPUS=2,3 nohup bash /data/zbz5399/envs/serve_mistral.sh > $LOGS/vllm.log 2>&1 &" >&2
  exit 1
fi
echo "    vLLM OK"

stop_all

echo "==> starting tutor proxy on 127.0.0.1:$PORT"
TUTOR_PORT="$PORT" nohup "$VENV" "$SERVER/tutor_proxy.py" > "$PROXY_LOG" 2>&1 &
echo $! > "$PROXY_PID"

for _ in $(seq 1 30); do
  sleep 1
  curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null && break
done
if ! curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null; then
  echo "ERROR: proxy did not come up. Last lines of $PROXY_LOG:" >&2
  tail -20 "$PROXY_LOG" >&2
  exit 1
fi
echo "    proxy OK"

echo "==> starting cloudflared quick tunnel"
: > "$TUNNEL_LOG"
nohup "$CLOUDFLARED" tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" > "$TUNNEL_LOG" 2>&1 &
echo $! > "$TUNNEL_PID"

PUBLIC_URL=""
for _ in $(seq 1 40); do
  sleep 1
  PUBLIC_URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$TUNNEL_LOG" | head -1)
  [ -n "$PUBLIC_URL" ] && break
done
if [ -z "$PUBLIC_URL" ]; then
  echo "ERROR: no tunnel URL appeared. Last lines of $TUNNEL_LOG:" >&2
  tail -20 "$TUNNEL_LOG" >&2
  exit 1
fi
echo "    tunnel URL: $PUBLIC_URL"

echo "==> verifying the tunnel reaches the model"
for _ in $(seq 1 15); do
  sleep 2
  curl -sf -m 10 "$PUBLIC_URL/health" >/dev/null && break
done
if ! curl -sf -m 10 "$PUBLIC_URL/health" >/dev/null; then
  echo "ERROR: $PUBLIC_URL/health is not answering." >&2
  exit 1
fi
echo "    tunnel OK"

cat > "$REPO/api-config.json" <<JSON
{
  "apiBase": "$PUBLIC_URL",
  "note": "Written by server/run_tutor_stack.sh. A cloudflared quick tunnel changes hostname on every restart, so this file is the single place the address lives."
}
JSON
echo "==> wrote api-config.json"

if [ "$PUSH" = "1" ]; then
  echo "==> publishing new address to GitHub Pages"
  git -C "$REPO" add api-config.json
  if git -C "$REPO" diff --cached --quiet; then
    echo "    address unchanged, nothing to push"
  elif git -C "$REPO" commit -q -m "tutor endpoint: $PUBLIC_URL" && git -C "$REPO" push -q origin main; then
    echo "    pushed. Pages redeploys in ~1 minute."
  else
    echo "WARNING: push failed. The site will keep using the OLD address." >&2
    echo "         Fix the remote, then: git -C $REPO push origin main" >&2
  fi
else
  echo "==> PUSH=0, not pushing (test with: http://localhost:8080/?api=$PUBLIC_URL)"
fi

cat <<EOF

tutor is online
  public API : $PUBLIC_URL
  site       : https://zyzou8.github.io/AI_Debug/
  logs       : $PROXY_LOG
               $TUNNEL_LOG
  stop       : bash $SERVER/run_tutor_stack.sh stop

Leave this machine's processes running: closing the tunnel takes the site's
tutor offline (the page then shows the "tutor is offline" banner).
EOF
