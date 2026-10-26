#!/usr/bin/env bash
# Exploration (memory generation) with Qwen alone as the api planner; writes into MEMORY_DIR.
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
cd "$ROOT"
eval "$(grep -E '^export (INFINI_BASE_URL|INFINI_API_KEY)=' "$HOME/.bashrc")"
: "${INFINI_BASE_URL:?}" "${INFINI_API_KEY:?}" "${MEMORY_DIR:?}" "${SUITE:?}" "${TASK:?}"
v1() { local u="${1%/}"; case "$u" in */v1) echo "$u";; *) echo "$u/v1";; esac; }
unset ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy RPENT_ESCALATION
export PYTHONPATH="$ROOT" OPENAI_API_KEY="$INFINI_API_KEY"
export PI05_CHECKPOINT_PATH=/mnt/user_dir/hanjinchen/lijinhao/duanyun/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT
export SAM3_CHECKPOINT_PATH=/mnt/user_dir/hanjinchen/lijinhao/duanyun/RPent/checkpoints/sam3/sam3.pt
export RPENT_SHARED_PERCEPTION=auto RPENT_SHARED_SIM=1 RPENT_COERCE_TOOL_ARGS=1 PYTHONUNBUFFERED=1
exec python3 -m rpent.cli.main --robot libero --suite "$SUITE" --task "$TASK" --seed 0 --cuda-device 0 \
  --vla-endpoint http://127.0.0.1:51157 --sam3-endpoint http://127.0.0.1:56897 \
  --planner api --model openai-chat:qwen3.6-27b --base-url "$(v1 "$INFINI_BASE_URL")" --reasoning-effort none \
  --planner-timeout-s "${SESSION_TIMEOUT_S:-2400}" \
  --explore --explore-sessions "${EXPLORE_SESSIONS:-3}" --explore-attempts-per-session "${EXPLORE_ATTEMPTS:-5}" \
  --memory-dir "$MEMORY_DIR" --output-dir "$OUTPUT_DIR"
