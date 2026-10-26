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
# One RC-ECC stage-1 exploration cell: Qwen alone runs RPent --explore on one RoboCasa Target50 task.
#   explore_cell.sh SPLIT TASK SEED OUT_DIR MEMORY_DIR
# Started by stage1.py, which passes every machine value in ECC_RC_* variables from its config (nothing
# host-specific is in this file) plus RC_GPU, VLA_ENDPOINT, RC_RENDER=cpu and RC_LP_THREADS.
# The environment is the Target50 cell launcher's Qwen-alone arm (ABL_ESCALATION=0): the same RPENT_* and
# RLDX_* values, the same CPU-render block, credentials sourced from a file with set -a. Differences, by design:
#   - PYTHONPATH and RPENT_REPO_ROOT point at the ECC checkout, which is also the working directory;
#   - --explore with SESSIONS x ATTEMPTS, a per-session planner timeout, --no-auto-merge-memory (the runner
#     publishes), --memory-dir is the task's staging corpus, the RLDX server is always a shared endpoint;
#   - RPENT_EXPLORE_LIST_MEMORY (stage1.list_memory): the explore prompt lists the staged cloud files of the
#     task, which RoboCasa's explore prompt does not name otherwise (evaluation lists them).
# The ECC_RC_* inputs are removed from the environment before rpent starts.
set -euo pipefail
SPLIT=$1; TASK=$2; SEED=$3; OUT=$4; MEMORY_DIR=$5
for v in ECC_RC_CODE_DIR ECC_RC_VENV ECC_RC_CRED_FILE ECC_RC_WORKER_KEY_VAR ECC_RC_WORKER_BASE_URL \
         ECC_RC_WORKER_MODEL ECC_RC_WORKER_EFFORT ECC_RC_SENIOR_BASE_URL_VAR ECC_RC_SENIOR_KEY_VAR ECC_RC_SENIOR_MODEL \
         ECC_RC_SENIOR_EFFORT ECC_RC_CHECKIN ECC_RC_MAX_SHARE ECC_RC_ALLOWED_GPUS ECC_RC_SESSIONS ECC_RC_ATTEMPTS \
         ECC_RC_SESSION_TIMEOUT_S ECC_RC_MAX_TURNS ECC_RC_ASSETS ECC_RC_HF_HOME ECC_RC_LLVMPIPE_SCRIPT \
         ECC_RC_MESA_EGL_VENDOR ECC_RC_CUDA_LIB ECC_RC_TORCH_LIB ECC_RC_TORCH_ARCH ECC_RC_LIST_MEMORY VLA_ENDPOINT \
         RC_GPU; do
  [ -n "${!v:-}" ] || { echo "explore_cell: $v is not set (start cells through stage1.py)" >&2; exit 2; }
done
case " $ECC_RC_ALLOWED_GPUS " in *" $RC_GPU "*) ;; *) echo "RC_GPU='$RC_GPU' is not one of: $ECC_RC_ALLOWED_GPUS" >&2; exit 2 ;; esac
# GPU (EGL) rendering has aborted the env server next to RLDX on GPU hosts: CPU rendering only.
[ "${RC_RENDER:-}" = cpu ] || { echo "explore_cell: RC_RENDER=cpu is required" >&2; exit 2; }
case "$SPLIT" in atomic|composite_seen|composite_unseen) ;; *) echo "bad split $SPLIT" >&2; exit 2 ;; esac
[ -f "$MEMORY_DIR/global/GLOBAL_MEMORY.md" ] || { echo "no staging corpus at $MEMORY_DIR" >&2; exit 2; }
[ -f "$ECC_RC_CRED_FILE" ] || { echo "explore_cell: credentials file not found" >&2; exit 2; }
CODE=$ECC_RC_CODE_DIR; VENV=$ECC_RC_VENV
set -a; . "$ECC_RC_CRED_FILE"; set +a
export OPENAI_API_KEY="${!ECC_RC_WORKER_KEY_VAR}" OPENAI_BASE_URL="$ECC_RC_WORKER_BASE_URL"
export RPENT_ESCALATION=1 RPENT_ESCALATION_MODEL="$ECC_RC_SENIOR_MODEL" RPENT_ESCALATION_BASE_URL="${!ECC_RC_SENIOR_BASE_URL_VAR}" \
       RPENT_ESCALATION_KEY_ENV="$ECC_RC_SENIOR_KEY_VAR" RPENT_ESCALATION_REASONING_EFFORT="$ECC_RC_SENIOR_EFFORT" \
       RPENT_ESCALATION_CHECKIN="$ECC_RC_CHECKIN" RPENT_ESCALATION_MAX_SHARE="$ECC_RC_MAX_SHARE"
# Qwen sends integer tool arguments as JSON strings; coerce them and repair leading-zero numbers.
export RPENT_COERCE_TOOL_ARGS=1
# No generic cap on state-changing tools: an RLDX skill runs up to 40 chunks; the session timeout bounds the run.
export RPENT_TOOL_TIMEOUT_S=0
# Resume up to 3 times after a reply without a tool call.
export RPENT_CONTINUE_NOTES=3
export RPENT_ESCALATION_SENIOR_MAX_IMAGES=40
# Qwen alone: no senior at all (the launcher's ABL_ESCALATION=0).
export RPENT_ESCALATION=0
# Stage 1 starts from the cloud memory: name the staged files of this task in the explore prompt.
export RPENT_EXPLORE_LIST_MEMORY="$ECC_RC_LIST_MEMORY"
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
"$VENV/bin/python" - "$CODE" <<'PY' || { echo "explore_cell: rpent/robots do not resolve to $CODE" >&2; exit 3; }
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
SESSIONS=$ECC_RC_SESSIONS; ATTEMPTS=$ECC_RC_ATTEMPTS; TIMEOUT=$ECC_RC_SESSION_TIMEOUT_S; TURNS=$ECC_RC_MAX_TURNS
MODEL=$ECC_RC_WORKER_MODEL; EFFORT=$ECC_RC_WORKER_EFFORT
mkdir -p "$OUT"
cd "$CODE"
unset "${!ECC_RC_@}"
exec "$VENV/bin/rpent" --robot robocasa --task-name "$TASK" --split target --seed "$SEED" \
  --vla-endpoint "$VLA_ENDPOINT" --cuda-device "$RC_GPU" \
  --planner api --model "$MODEL" --reasoning-effort "$EFFORT" \
  --max-turns "$TURNS" --planner-timeout-s "$TIMEOUT" \
  --memory-profile local --memory-dir "$MEMORY_DIR" --output-dir "$OUT" \
  --explore --explore-sessions "$SESSIONS" --explore-attempts-per-session "$ATTEMPTS" --no-auto-merge-memory
