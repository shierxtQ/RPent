#!/bin/bash
# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# One RC-ECC hybrid evaluation episode: Qwen worker plus GPT-6 senior under one call rule, on one RoboCasa
# Target50 task (stage 2 rungs, stage 3 episodes, the final test).
#   hybrid_cell.sh SPLIT TASK SEED OUT_DIR MEMORY_DIR
# Started by the stage runners, which pass every machine value in ECC_RC_* variables from their config (nothing
# host-specific is in this file) plus RC_GPU, VLA_ENDPOINT, RC_RENDER=cpu and RC_LP_THREADS.
# The environment is the Target50 cell launcher's base arm (Qwen worker, GPT-6 senior at low effort, share cap,
# the same RPENT_* and RLDX_* values and CPU-render block, credentials sourced from a file with set -a), with:
#   - the call rule: ECC_RC_PLAN=0 exports RPENT_ESCALATION_PLAN=0 (the launcher's ABL_PLAN=0; with 1 it stays
#     unset, i.e. on), ECC_RC_CHECKIN is the launcher's CHECKIN argument, ECC_RC_ERRORS becomes
#     RPENT_ESCALATION_ERRORS (the launcher leaves it at RPent's default 2);
#   - optional cloud corrections: ECC_RC_CORRECTIONS names a file that must exist (RPent silently ignores a
#     missing one) and, with ECC_RC_CORRECTIONS_SHA256, must have that sha256; it becomes RPENT_CLOUD_CORRECTIONS;
#   - optional ECC_RC_ESCALATION=0 (final-test Qwen-alone arms only): no senior at all, as the launcher's
#     ABL_ESCALATION=0; unset or 1 keeps the hybrid planner;
#   - PYTHONPATH and RPENT_REPO_ROOT point at the ECC checkout, which is also the working directory;
#   - --memory-dir is a frozen corpus that must not be writable (evaluation reads memory read-only anyway),
#     --planner-timeout-s is ECC_RC_TIMEOUT_S, the RLDX server is always a shared endpoint.
# The ECC_RC_* inputs are removed from the environment before rpent starts.
set -euo pipefail
SPLIT=$1; TASK=$2; SEED=$3; OUT=$4; MEMORY_DIR=$5
for v in ECC_RC_CODE_DIR ECC_RC_VENV ECC_RC_CRED_FILE ECC_RC_WORKER_KEY_VAR ECC_RC_WORKER_BASE_URL \
         ECC_RC_WORKER_MODEL ECC_RC_WORKER_EFFORT ECC_RC_SENIOR_BASE_URL_VAR ECC_RC_SENIOR_KEY_VAR ECC_RC_SENIOR_MODEL \
         ECC_RC_SENIOR_EFFORT ECC_RC_MAX_SHARE ECC_RC_ALLOWED_GPUS ECC_RC_PLAN ECC_RC_CHECKIN ECC_RC_ERRORS \
         ECC_RC_TIMEOUT_S ECC_RC_MAX_TURNS ECC_RC_ASSETS ECC_RC_HF_HOME ECC_RC_LLVMPIPE_SCRIPT ECC_RC_MESA_EGL_VENDOR \
         ECC_RC_CUDA_LIB ECC_RC_TORCH_LIB ECC_RC_TORCH_ARCH VLA_ENDPOINT RC_GPU; do
  [ -n "${!v:-}" ] || { echo "hybrid_cell: $v is not set (start cells through the stage runners)" >&2; exit 2; }
done
case " $ECC_RC_ALLOWED_GPUS " in *" $RC_GPU "*) ;; *) echo "RC_GPU='$RC_GPU' is not one of: $ECC_RC_ALLOWED_GPUS" >&2; exit 2 ;; esac
# GPU (EGL) rendering has aborted the env server next to RLDX on GPU hosts: CPU rendering only.
[ "${RC_RENDER:-}" = cpu ] || { echo "hybrid_cell: RC_RENDER=cpu is required" >&2; exit 2; }
case "$SPLIT" in atomic|composite_seen|composite_unseen) ;; *) echo "bad split $SPLIT" >&2; exit 2 ;; esac
case "$ECC_RC_PLAN" in 0|1) ;; *) echo "hybrid_cell: ECC_RC_PLAN must be 0 or 1" >&2; exit 2 ;; esac
case "${ECC_RC_ESCALATION:-1}" in 0|1) ;; *) echo "hybrid_cell: ECC_RC_ESCALATION must be 0 or 1" >&2; exit 2 ;; esac
SENIOR=on; NOTE=""; [ "${ECC_RC_ESCALATION:-1}" = 0 ] && SENIOR=off NOTE=" senior=off"
for v in ECC_RC_CHECKIN ECC_RC_ERRORS ECC_RC_TIMEOUT_S ECC_RC_MAX_TURNS; do
  [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || { echo "hybrid_cell: $v must be a positive integer" >&2; exit 2; }
done
[ -f "$MEMORY_DIR/global/GLOBAL_MEMORY.md" ] || { echo "no memory corpus at $MEMORY_DIR" >&2; exit 2; }
[ ! -w "$MEMORY_DIR" ] && [ ! -w "$MEMORY_DIR/task-specific" ] || \
  { echo "hybrid_cell: memory dir $MEMORY_DIR is writable; pass the frozen read-only corpus" >&2; exit 2; }
[ -f "$ECC_RC_CRED_FILE" ] || { echo "hybrid_cell: credentials file not found" >&2; exit 2; }
CODE=$ECC_RC_CODE_DIR; VENV=$ECC_RC_VENV
CORRECTIONS=${ECC_RC_CORRECTIONS:-}
if [ -n "$CORRECTIONS" ]; then
  [ -f "$CORRECTIONS" ] || { echo "hybrid_cell: corrections file $CORRECTIONS not found" >&2; exit 2; }
  if [ -n "${ECC_RC_CORRECTIONS_SHA256:-}" ]; then
    got=$("$VENV/bin/python" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$CORRECTIONS")
    [ "$got" = "$ECC_RC_CORRECTIONS_SHA256" ] || { echo "hybrid_cell: corrections file sha256 $got, expected $ECC_RC_CORRECTIONS_SHA256" >&2; exit 2; }
  fi
fi
set -a; . "$ECC_RC_CRED_FILE"; set +a
export OPENAI_API_KEY="${!ECC_RC_WORKER_KEY_VAR}" OPENAI_BASE_URL="$ECC_RC_WORKER_BASE_URL"
export RPENT_ESCALATION=1 RPENT_ESCALATION_MODEL="$ECC_RC_SENIOR_MODEL" RPENT_ESCALATION_BASE_URL="${!ECC_RC_SENIOR_BASE_URL_VAR}" \
       RPENT_ESCALATION_KEY_ENV="$ECC_RC_SENIOR_KEY_VAR" RPENT_ESCALATION_REASONING_EFFORT="$ECC_RC_SENIOR_EFFORT" \
       RPENT_ESCALATION_CHECKIN="$ECC_RC_CHECKIN" RPENT_ESCALATION_MAX_SHARE="$ECC_RC_MAX_SHARE"
# Qwen sends integer tool arguments as JSON strings; coerce them and repair leading-zero numbers.
export RPENT_COERCE_TOOL_ARGS=1
# No generic cap on state-changing tools: an RLDX skill runs up to 40 chunks; the planner timeout bounds the run.
export RPENT_TOOL_TIMEOUT_S=0
# Resume up to 3 times after a reply without a tool call.
export RPENT_CONTINUE_NOTES=3
# The senior gateway rejects requests with more than 64-80 images: keep GPT-6's 40 newest.
export RPENT_ESCALATION_SENIOR_MAX_IMAGES=40
# The call rule. Plan on = RPENT_ESCALATION_PLAN unset, as in the Target50 launcher.
[ "$ECC_RC_PLAN" = 0 ] && export RPENT_ESCALATION_PLAN=0
export RPENT_ESCALATION_ERRORS="$ECC_RC_ERRORS"
[ -n "$CORRECTIONS" ] && export RPENT_CLOUD_CORRECTIONS="$CORRECTIONS"
# Qwen alone (final-test arms): no senior at all, as the launcher's ABL_ESCALATION=0.
if [ "$SENIOR" = off ]; then
  [ -z "$CORRECTIONS" ] || { echo "hybrid_cell: cloud corrections need the senior (ECC_RC_ESCALATION=0)" >&2; exit 2; }
  export RPENT_ESCALATION=0
fi
export RLDX_MAX_CHUNKS=40 RLDX_SETTLE_PATIENCE=999 RLDX_ACTION_STEPS_PER_CHUNK=8
unset RLDX_RESET_SEED
export ROBOCASA_ASSETS_PATH="$ECC_RC_ASSETS" HF_HOME="$ECC_RC_HF_HOME" HF_HUB_CACHE="$ECC_RC_HF_HOME/hub"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TORCH_CUDA_ARCH_LIST="$ECC_RC_TORCH_ARCH" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$RC_GPU" PYTHONDONTWRITEBYTECODE=1
# CPU rendering: env servers rasterize with Mesa llvmpipe; RLDX stays on $RC_GPU behind VLA_ENDPOINT.
RC_LLVMPIPE=$(bash "$ECC_RC_LLVMPIPE_SCRIPT") || { echo "no llvmpipe EGL device" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES= MUJOCO_EGL_DEVICE_ID="$RC_LLVMPIPE" LP_NUM_THREADS="${RC_LP_THREADS:-8}" \
       __EGL_VENDOR_LIBRARY_FILENAMES="$ECC_RC_MESA_EGL_VENDOR"
export LD_LIBRARY_PATH="$ECC_RC_TORCH_LIB:$ECC_RC_CUDA_LIB:${LD_LIBRARY_PATH:-}"
# Run this checkout's rpent and robots with the read-only venv; refuse to start otherwise.
export PYTHONPATH="$CODE" RPENT_REPO_ROOT="$CODE"
"$VENV/bin/python" - "$CODE" <<'PY' || { echo "hybrid_cell: rpent/robots do not resolve to $CODE" >&2; exit 3; }
import importlib.util, os, sys
if sys.path and sys.path[0] == "":
    del sys.path[0]
code = os.path.realpath(sys.argv[1]) + os.sep
for name in ("rpent", "robots"):
    spec = importlib.util.find_spec(name)
    origin = os.path.realpath(spec.origin) if spec and spec.origin else ""
    if not origin.startswith(code):
        sys.exit(f"{name} resolves to {origin or 'nothing'}")
PY
TIMEOUT=$ECC_RC_TIMEOUT_S; TURNS=$ECC_RC_MAX_TURNS; MODEL=$ECC_RC_WORKER_MODEL; EFFORT=$ECC_RC_WORKER_EFFORT
echo "hybrid_cell: $TASK seed $SEED plan=$ECC_RC_PLAN checkin=$ECC_RC_CHECKIN errors=$ECC_RC_ERRORS" \
     "corrections=${CORRECTIONS:-none} timeout=${TIMEOUT}s turns=$TURNS$NOTE"
mkdir -p "$OUT"
cd "$CODE"
unset "${!ECC_RC_@}"
exec "$VENV/bin/rpent" --robot robocasa --task-name "$TASK" --split target --seed "$SEED" \
  --vla-endpoint "$VLA_ENDPOINT" --cuda-device "$RC_GPU" \
  --planner api --model "$MODEL" --reasoning-effort "$EFFORT" \
  --max-turns "$TURNS" --planner-timeout-s "$TIMEOUT" \
  --memory-profile local --memory-dir "$MEMORY_DIR" --output-dir "$OUT"
