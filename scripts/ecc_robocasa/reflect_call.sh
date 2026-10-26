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
# One RC-ECC stage-3 reflection call (GPT-6, text only) between two episodes of a task.
#   reflect_call.sh REQUEST.json REPLY.json
# Started by stage3.py with ECC_RC_CODE_DIR, ECC_RC_VENV, ECC_RC_CRED_FILE and the NAMES of the senior's base-URL
# and key variables (ECC_RC_SENIOR_BASE_URL_VAR, ECC_RC_SENIOR_KEY_VAR), as the cell launchers get them. Like the
# cell launchers, it sources the credentials file with set -a; the runner never reads it. The client gets the base URL
# and the name of the key variable, builds the senior's model chain and writes REPLY.json; nothing secret is printed.
set -euo pipefail
REQUEST=$1; REPLY=$2
for v in ECC_RC_CODE_DIR ECC_RC_VENV ECC_RC_CRED_FILE ECC_RC_SENIOR_BASE_URL_VAR ECC_RC_SENIOR_KEY_VAR; do
  [ -n "${!v:-}" ] || { echo "reflect_call: $v is not set (start reflections through stage3.py)" >&2; exit 2; }
done
[ -f "$REQUEST" ] || { echo "reflect_call: no request file $REQUEST" >&2; exit 2; }
[ -f "$ECC_RC_CRED_FILE" ] || { echo "reflect_call: credentials file not found" >&2; exit 2; }
CODE=$ECC_RC_CODE_DIR; VENV=$ECC_RC_VENV
set -a; . "$ECC_RC_CRED_FILE"; set +a
[ -n "${!ECC_RC_SENIOR_BASE_URL_VAR:-}" ] || { echo "reflect_call: the senior base-URL variable is empty" >&2; exit 2; }
[ -n "${!ECC_RC_SENIOR_KEY_VAR:-}" ] || { echo "reflect_call: the senior key variable is empty" >&2; exit 2; }
export ECC_RC_REFLECT_BASE_URL="${!ECC_RC_SENIOR_BASE_URL_VAR}" ECC_RC_REFLECT_KEY_ENV="$ECC_RC_SENIOR_KEY_VAR"
export PYTHONPATH="$CODE" RPENT_REPO_ROOT="$CODE" PYTHONDONTWRITEBYTECODE=1
cd "$CODE"
for v in "${!ECC_RC_@}"; do
  case "$v" in ECC_RC_REFLECT_*) ;; *) unset "$v" ;; esac
done
exec "$VENV/bin/python" -m scripts.ecc_robocasa.reflect_client "$REQUEST" "$REPLY"
