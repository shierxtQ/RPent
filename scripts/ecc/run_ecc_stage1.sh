#!/usr/bin/env bash
# ECC stage 1 (ecc_rule.md): device-side task memory.
# Qwen3.6-27B alone (api, thinking off) explores every LIBERO-PRO cell at seed 0, 5 sessions x
# 3 attempts, 2400 s per session, starting from a copy of the cloud (GPT-6) memory. With
# RPENT_MEMORY_MERGE_REPLACE=1 a solved cell's own task-family write-up and audit/recipe replace
# the inherited ones (archived under _internal/replaced); global notes merge as usual.
# Usage: scripts/ecc/run_ecc_stage1.sh RUN_DIR MEMORY_DIR [CLOUD_MEMORY_DIR]
#   MEMORY_DIR is created as a copy of CLOUD_MEMORY_DIR (default memory/libero/GPT_6_astra_low)
#   when it does not exist yet; PARALLEL (default 8) sets the number of cells run at once.
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RUN_DIR="$(realpath -m "${1:?run dir}")"
MEMORY_DIR="$(realpath -m "${2:?memory dir}")"
CLOUD="${3:-$ROOT/memory/libero/GPT_6_astra_low}"
[ -e "$MEMORY_DIR" ] || cp -a "$CLOUD" "$MEMORY_DIR"
cd "$ROOT"
export RUNNER_SCRIPT=run_explore_qwen.sh MEMORY_DIR EXPLORE_SESSIONS=5 EXPLORE_ATTEMPTS=3 \
  SESSION_TIMEOUT_S=2400 RPENT_MEMORY_MERGE_REPLACE=1
exec python3 "$HERE/run_escalation_batch80.py" "$RUN_DIR" --parallel "${PARALLEL:-8}" --seeds 0 \
  --planner-label "ECC stage 1: api qwen3.6-27b explore from GPT memory, 5 sessions x 3 attempts"
