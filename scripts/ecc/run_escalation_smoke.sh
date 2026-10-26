#!/usr/bin/env bash
# Smoke: Qwen-led api planner + GPT escalation + shared perception + arg coercion.
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
cd "$ROOT"
eval "$(grep -E '^export (RPENT_BASE_URL|RPENT_API_KEY|INFINI_BASE_URL|INFINI_API_KEY)=' "$HOME/.bashrc")"
: "${RPENT_BASE_URL:?}" "${RPENT_API_KEY:?}" "${INFINI_BASE_URL:?}" "${INFINI_API_KEY:?}"
v1() { local u="${1%/}"; case "$u" in */v1) echo "$u";; *) echo "$u/v1";; esac; }
unset ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy
export PYTHONPATH="$ROOT"
export OPENAI_API_KEY="$INFINI_API_KEY"
export RPENT_ESCALATION=1 RPENT_ESCALATION_MODEL=openai:gpt-6-astra \
  RPENT_ESCALATION_BASE_URL="$(v1 "$RPENT_BASE_URL")" RPENT_ESCALATION_KEY_ENV=RPENT_API_KEY \
  RPENT_ESCALATION_REASONING_EFFORT=low RPENT_ESCALATION_CHECKIN=${RPENT_ESCALATION_CHECKIN:-12} RPENT_ESCALATION_MAX_SHARE=0.2
export PI05_CHECKPOINT_PATH=/mnt/user_dir/hanjinchen/lijinhao/duanyun/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT
export SAM3_CHECKPOINT_PATH=/mnt/user_dir/hanjinchen/lijinhao/duanyun/RPent/checkpoints/sam3/sam3.pt
export RPENT_SHARED_PERCEPTION=auto RPENT_COERCE_TOOL_ARGS=1 PYTHONUNBUFFERED=1
# Host-wide simulator slot on by default (parallel batches time out without it); RPENT_SHARED_SIM=0 turns it off.
export RPENT_SHARED_SIM="${RPENT_SHARED_SIM:-1}"
EXTRA=()
[ -n "${VLA_ENDPOINT:-}" ] && EXTRA+=(--vla-endpoint "$VLA_ENDPOINT")
[ -n "${SAM3_ENDPOINT:-}" ] && EXTRA+=(--sam3-endpoint "$SAM3_ENDPOINT")
[ -n "${OUTPUT_DIR:-}" ] && EXTRA+=(--output-dir "$OUTPUT_DIR")
# PLANNER_MODE=gpt: GPT alone as the api planner (no Qwen, no escalation).
# PLANNER_MODE=qwen: Qwen alone as the api planner (no escalation).
# MEMORY_DIR overrides the local memory corpus (default: official GPT_6_astra_low).
if [ "${PLANNER_MODE:-escalation}" = codex ]; then
  # Official LIBERO-PRO backend: Codex SDK + GPT-6 Astra (low), isolated Codex home.
  unset RPENT_ESCALATION
  export CODEX_HOME=/mnt/user_dir/hanjinchen/lijinhao/duanyun/codex-home-rpent
  export CODEX_BASE_URL="${CODEX_BASE_URL_OVERRIDE:-$RPENT_BASE_URL}" CODEX_API_KEY="$RPENT_API_KEY"
  MODEL_ARGS=(--planner codex --model gpt-6-astra --reasoning-effort low)
elif [ "${PLANNER_MODE:-escalation}" = codex-escalation ] || [ "${PLANNER_MODE:-escalation}" = codex-qwen ]; then
  # codex-qwen: Qwen alone inside the Codex harness (no senior turns).
  [ "$PLANNER_MODE" = codex-qwen ] && export RPENT_ESCALATION_WORKER_ONLY=1
  # Codex harness with Qwen as the worker and GPT-6 Astra (Codex model) as the senior.
  export CODEX_HOME=/mnt/user_dir/hanjinchen/lijinhao/duanyun/codex-home-rpent
  export CODEX_BASE_URL="${CODEX_BASE_URL_OVERRIDE:-$RPENT_BASE_URL}" CODEX_API_KEY="$RPENT_API_KEY"
  export RPENT_ESCALATION_WORKER_MODEL=qwen3.6-27b RPENT_ESCALATION_WORKER_BASE_URL="$(v1 "$INFINI_BASE_URL")" \
    RPENT_ESCALATION_WORKER_KEY_ENV=INFINI_API_KEY
  MODEL_ARGS=(--planner codex --model gpt-6-astra --reasoning-effort low)
elif [ "${PLANNER_MODE:-escalation}" = gpt ]; then
  unset RPENT_ESCALATION
  export OPENAI_API_KEY="$RPENT_API_KEY"
  MODEL_ARGS=(--planner api --model openai:gpt-6-astra --base-url "$(v1 "${API_BASE_URL_OVERRIDE:-$RPENT_BASE_URL}")" --reasoning-effort low)
elif [ "${PLANNER_MODE:-escalation}" = qwen ]; then
  unset RPENT_ESCALATION
  MODEL_ARGS=(--planner api --model openai-chat:qwen3.6-27b --base-url "$(v1 "$INFINI_BASE_URL")" --reasoning-effort none)
else
  MODEL_ARGS=(--planner api --model openai-chat:qwen3.6-27b --base-url "$(v1 "$INFINI_BASE_URL")" --reasoning-effort none)
fi
exec python3 -m rpent.cli.main "${EXTRA[@]}" --robot libero --suite "${SUITE:-libero_10}" --task "${TASK:-0}" --seed "${SEED:-1}" \
  --cuda-device 0 "${MODEL_ARGS[@]}" --planner-timeout-s 1200 \
  --memory-profile local --memory-dir "${MEMORY_DIR:-$ROOT/memory/libero/GPT_6_astra_low}"
