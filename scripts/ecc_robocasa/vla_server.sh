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
# One shared RLDX vla_server on RC_GPU for RC-ECC cells (episodes attach with --vla-endpoint; RPC sessions keep
# each episode's action history apart). Started and stopped by stage1.py after it checked that the GPU is free.
#   vla_server.sh PORT LOG
# The environment is the Target50 shared-server launcher's: the RLDX_* runtime values, HF and asset paths
# (HF_HUB_OFFLINE comes from the runner's episode environment), with this checkout's code.
set -euo pipefail
PORT=$1; LOG=$2
for v in ECC_RC_CODE_DIR ECC_RC_VENV ECC_RC_ALLOWED_GPUS ECC_RC_ASSETS ECC_RC_HF_HOME ECC_RC_CKPT ECC_RC_CUDA_LIB \
         ECC_RC_TORCH_LIB ECC_RC_TORCH_ARCH RC_GPU; do
  [ -n "${!v:-}" ] || { echo "vla_server: $v is not set (start servers through stage1.py)" >&2; exit 2; }
done
case " $ECC_RC_ALLOWED_GPUS " in *" $RC_GPU "*) ;; *) echo "RC_GPU='$RC_GPU' is not one of: $ECC_RC_ALLOWED_GPUS" >&2; exit 2 ;; esac
CODE=$ECC_RC_CODE_DIR; VENV=$ECC_RC_VENV; CKPT=$ECC_RC_CKPT
export HF_HOME="$ECC_RC_HF_HOME" HF_HUB_CACHE="$ECC_RC_HF_HOME/hub" ROBOCASA_ASSETS_PATH="$ECC_RC_ASSETS"
export RLDX_MAX_CHUNKS=40 RLDX_SETTLE_PATIENCE=999 RLDX_ACTION_STEPS_PER_CHUNK=8
unset RLDX_RESET_SEED
export TORCH_CUDA_ARCH_LIST="$ECC_RC_TORCH_ARCH" CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$ECC_RC_TORCH_LIB:$ECC_RC_CUDA_LIB:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$CODE" RPENT_REPO_ROOT="$CODE"
cd "$CODE"
unset "${!ECC_RC_@}"
exec "$VENV/bin/python" robots/robocasa/vla_server.py --transport http --host 127.0.0.1 --port "$PORT" \
  --cuda-device "$RC_GPU" --model-path "$CKPT" > "$LOG" 2>&1
