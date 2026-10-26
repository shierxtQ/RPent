#!/usr/bin/env bash
# One episode with the task's ECC call rule (RULES_DIR/<suite>_t<task>.json from run_ecc_stage2.py):
# api Qwen worker + GPT escalation configured by the rule's GPT plan / K / N.
set -euo pipefail
: "${RULES_DIR:?}" "${SUITE:?}" "${TASK:?}"
rule="$RULES_DIR/${SUITE#libero_}_t${TASK}.json"
eval "$(python3 - "$rule" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))["rule"]
print(f"export RPENT_ESCALATION_PLAN={1 if r['gpt_plan'] else 0} "
      f"RPENT_ESCALATION_CHECKIN={int(r['checkin_K'])} RPENT_ESCALATION_ERRORS={int(r['errors_N'])}")
EOF
)"
# CORR_DIR: per-task cloud correction memory (ECC stage 3), appended to GPT senior notes.
[ -n "${CORR_DIR:-}" ] && export RPENT_CLOUD_CORRECTIONS="$CORR_DIR/${SUITE#libero_}_t${TASK}.md"
unset PLANNER_MODE
exec "$(dirname "$0")/run_escalation_smoke.sh"
