#!/usr/bin/env python3
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

"""RC-ECC final test on RoboCasa Target50: named arms on held-out seeds, paired by cell.

Each arm of ``[final_test.arms.<name>]`` is one planner configuration:

- ``planner``: ``hybrid`` (Qwen worker, GPT-6 senior), ``qwen`` (Qwen alone) or
  ``gpt6`` (the senior's model alone on every turn: ``senior.model`` at
  ``senior.reasoning_effort``, through the senior's base URL and key, with no
  escalation wrapper; ``max_images``, default 40, caps the images of every
  request, and ``reasoning_ids``, default false, sends no ids of earlier
  response items, as the senior's requests);
- ``memory``: ``s1`` (the frozen stage-1 corpus), ``cloud`` (``paths.cloud_corpus``)
  or an absolute corpus path;
- ``rule`` (hybrid only): ``s2`` (each task's stage-2 rule) or ``fixed`` with
  ``plan``, ``checkin`` (K) and ``errors`` (N);
- ``corrections`` (hybrid only): ``none`` or ``s3`` (the task's stage-3
  ``final/<Task>.md``, appended to the senior's notes only).

A cell is (task, seed, repeat). Every run arm plays every cell; the arms of a cell
are launched one after another in a seeded random order, so they meet the same
provider and RLDX conditions. All inputs (memory corpora, rules, corrections) are
copied read-only into the run root and hashed when the run is created, and the
episodes use the copies. Episodes start through ``hybrid_cell.sh`` with the
environment of :func:`hybrid_episode.hybrid_env` (``ECC_RC_ESCALATION=0`` for a
Qwen-alone arm, also ``ECC_RC_SENIOR_ALONE=1`` for a GPT-6-alone arm). After
every valid episode the escalation log is checked with
:func:`hybrid_episode.check_escalation`, as in stages 2 and 3 (config row: plan,
K, N, corrections; replayed plan turn, check-ins and share cap;
``corrections_chars`` on every senior row); a Qwen-alone or GPT-6-alone arm must
leave no escalation log, and a GPT-6-alone episode's ``result.json`` and
``run.log`` must show its model, effort, image cap and reasoning-id setting. A
mismatch makes that launch an infrastructure error and stops new launches
(``final_test.evidence_policy = "stop"``). A GPT-6-alone episode's calls and
tokens come from RPent's transcript stats (``run.log`` usage lines after a
planner timeout) and are priced at the senior's list price, so its GPT-6 calls
per episode count every returned request, as the hybrid arms count the
senior's returned requests.

``final_test.share_own_gpus`` (off by default) lets the runner start a server on
a GPU that holds only this machine's ECC processes (another run's RLDX servers
and episodes, started from ``paths.venv``) when its free memory covers this run's
servers and slots on it plus ``servers.fit_margin_mb``; GPUs held by anything
else are refused as before. A refused server is tried again every
``servers.retry_busy_s`` (at most ``servers.max_refusals`` times), and the runner
waits for it instead of exiting when no server could start.

``score`` reports per arm the task-weighted success by split (Target50: a split's
rate is the mean of its tasks' rates, overall is the mean over all tasks), GPT-6
calls per episode by reason, how often the error trigger fired (and how often
2, 3 or 4 errored tool results came in a row), and list-price $ per episode and
per success. ``--pair A:B`` adds paired tests on the cells scored in both arms:
exact McNemar, a sign test over tasks, and a task-clustered sign-flip
permutation test of the task-weighted difference with a bootstrap interval.

Dry run by default (``plan``); ``run --go`` launches. ``example-config`` prints
a config section.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import stat
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ecc_robocasa import common as C  # noqa: E402
from scripts.ecc_robocasa import hybrid_episode as HE  # noqa: E402
from scripts.ecc_robocasa import stage1 as S1  # noqa: E402

STAGE = "final_test"
SPLIT = S1.SPLIT  # RoboCasa environment split of every Target50 cell
#: The launcher switch a Qwen-alone arm needs (the Target50 launcher's ABL_ESCALATION=0).
QWEN_SWITCH = "ECC_RC_ESCALATION"
#: The launcher switch a GPT-6-alone arm needs (with QWEN_SWITCH=0).
ALONE_SWITCH = "ECC_RC_SENIOR_ALONE"
#: Images per request of a GPT-6-alone arm: the senior's cap (the gateway rejects 64-80).
GPT6_MAX_IMAGES = 40
#: Planner types without the escalation wrapper.
ALONE_PLANNERS = ("qwen", "gpt6")
#: What RPent logs when the image cap and the reasoning-id switch are on (rpent/planner/api_loop.py).
IMAGE_CAP_LOG = "history images: at most {n} per request (RPENT_MAX_HISTORY_IMAGES)"
NO_IDS_LOG = "earlier response item ids are not sent back (RPENT_SEND_REASONING_IDS=0)"

FT_DEFAULTS: dict[str, Any] = {
    "seeds": [1, 2, 3],
    "repeats": 1,
    "tasks": [],  # empty: all 50 Target50 tasks in manifest order
    "run_arms": [],  # empty: every arm in [final_test.arms]
    "order_seed": 20261027,
    "cell_order": "seed_major",  # or "shuffled"
    "timeout_s_by_split": {},  # default: the Target50 manifest (atomic 1800 s, composite 3600 s)
    "infra_retries": 2,
    "render_retries": 6,
    "evidence_policy": "stop",  # stop | warn
    "require_config_row": True,  # the escalation log's config row is the only record of N
    "launcher": [],  # episode launcher argv prefix (default: hybrid_cell.sh, as stages 2-3)
    # Frozen stage products (verified with their stage's verify()); s2_rules / s3_corrections
    # name bare folders instead (not verified, e.g. rule files of the original LIBERO runner).
    "products": {"s1": "", "s2": "", "s3": "", "s2_rules": "", "s3_corrections": ""},
    "s2_rule_name": "{task}.json",  # file names in a bare s2_rules / s3_corrections folder
    "s3_corrections_name": "{task}.md",
    "allow_missing_corrections": False,
    "allow_off_ladder_rules": False,
    # The S2 rules must come from the S1 corpus, the S3 corrections from both (their manifests).
    "allow_product_mismatch": False,
    "endpoints": [],  # replaces [[endpoints]] for this stage when set
    "scheduler": {},  # overrides of [scheduler] for this stage
    # Start servers on GPUs that hold only this machine's ECC processes when their memory fits.
    "share_own_gpus": False,
    "gpu_apps_cmd": [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits",
    ],
    "gpu_uuid_cmd": ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
    "own_process_prefixes": [],  # executables that count as this machine's ECC processes (empty: paths.venv)
    "pairs": [],  # default --pair list, e.g. [["s3_rules_corr", "s2_rules"]]
    "permutations": 100000,
    "bootstrap": 10000,
    "stats_seed": 1,
    "arms": {},
}
#: final_test keys that define the experiment: the run keeps the values it was created with.
FT_EXPERIMENT_KEYS = (
    "seeds",
    "repeats",
    "tasks",
    "run_arms",
    "order_seed",
    "cell_order",
    "timeout_s_by_split",
    "s2_rule_name",
    "s3_corrections_name",
    "allow_missing_corrections",
    "allow_off_ladder_rules",
    "allow_product_mismatch",
    "arms",
)
ARM_KEYS = {
    "planner",
    "memory",
    "rule",
    "plan",
    "checkin",
    "errors",
    "corrections",
    "note",
    "max_images",
    "reasoning_ids",
}

EXAMPLE_CONFIG = """\
# RC-ECC final test (scripts/ecc_robocasa/final_test.py). Add to the S2/S3 config, outside the repository.
[final_test]
seeds = [1, 2, 3]                  # held-out Target50 seeds (S1 explored seed 0, S2 seed 11, S3 seed 12)
repeats = 1                        # episodes per (task, seed) per arm
tasks = []                         # empty: all 50 Target50 tasks
run_arms = ["s2_rules", "s3_rules_corr"]   # which defined arms run (empty: all defined)
order_seed = 20261027              # seeded arm order within every cell
cell_order = "seed_major"          # seed 1 of every task first, then seed 2, ... ("shuffled": seeded)
# timeout_s_by_split = { atomic = 1800, composite_seen = 3600, composite_unseen = 3600 }   # default: Target50
infra_retries = 2                  # relaunches after an invalid result (never counted as a failure)
render_retries = 6
evidence_policy = "stop"           # a launch whose escalation log does not match its arm stops new launches
# share_own_gpus = true            # claim a GPU busy only with this machine's ECC processes once this run's plan fits
pairs = [["s3_rules_corr", "s2_rules"]]
# launcher = ["bash", "/path/to/ecc-code/scripts/ecc_robocasa/hybrid_cell.sh"]   # the default

[final_test.products]              # frozen products; each must pass its stage's verify
s1 = "/path/to/ecc/runs/s1_full/product"    # corpus/ (memory "s1")
s2 = "/path/to/ecc/runs/s2_full/product"    # rules/<Task>.json (rule "s2"), searched on this S1 corpus
s3 = "/path/to/ecc/runs/s3_full/product"    # final/<Task>.md (corrections "s3"), built under these S2 rules

[final_test.scheduler]             # overrides of [scheduler] for this stage
cost_ceiling_usd = 1500

[final_test.arms.s2_rules]         # stage-2 rule, stage-1 memory, no corrections
planner = "hybrid"
memory = "s1"
rule = "s2"
corrections = "none"

[final_test.arms.s3_rules_corr]    # stage-2 rule + stage-3 corrections for the senior
planner = "hybrid"
memory = "s1"
rule = "s2"
corrections = "s3"

[final_test.arms.base_hybrid]      # the Target50 base arm: GPT-6 plan, check-in every 5, cloud memory
planner = "hybrid"
memory = "cloud"
rule = "fixed"
plan = true
checkin = 5
errors = 2

[final_test.arms.qwen_only]        # Qwen alone with the cloud memory (the Target50 Qwen-only arm)
planner = "qwen"
memory = "cloud"

# [final_test.arms.gpt6_only]      # GPT-6 (senior.model at senior.reasoning_effort) on every turn, cloud memory
# planner = "gpt6"
# memory = "cloud"
# max_images = 40                  # images per request (the gateway rejects 64-80); the senior's cap
# reasoning_ids = false            # send no ids of earlier response items, as the senior's requests

[final_test.arms.gpt6_plan_k12]    # ladder rung 2 for every task
planner = "hybrid"
memory = "s1"
rule = "fixed"
plan = true
checkin = 12
errors = 4
"""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def load_config(path: Path | str) -> dict[str, Any]:
    """The shared config (``common.load_config``) with a validated ``final_test`` section."""
    cfg = C.load_config(path)
    cfg["final_test"] = C._deep_merge(FT_DEFAULTS, cfg.get("final_test") or {})
    products = cfg["final_test"]["products"]
    for key, value in products.items():
        if isinstance(value, str) and value.startswith("~"):
            products[key] = str(Path(value).expanduser())
    validate_final_test(cfg)
    return cfg


def validate_arm(name: str, arm: Mapping[str, Any]) -> None:
    """Refuse an arm whose keys do not describe exactly one planner configuration."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,39}", name) or "__" in name:
        raise ValueError(
            f"arm name {name!r}: letters, digits, '_', '.', '-' (no '__'), at most 40"
        )
    unknown = set(arm) - ARM_KEYS
    if unknown:
        raise ValueError(f"arm {name}: unknown keys {sorted(unknown)}")
    planner = arm.get("planner")
    if planner not in ("hybrid", *ALONE_PLANNERS):
        raise ValueError(f"arm {name}: planner must be 'hybrid', 'qwen' or 'gpt6'")
    alone_keys = sorted(k for k in ("max_images", "reasoning_ids") if k in arm)
    if planner != "gpt6" and alone_keys:
        raise ValueError(f"arm {name}: {alone_keys} apply to a GPT-6-alone arm only")
    images = arm.get("max_images", GPT6_MAX_IMAGES)
    if not isinstance(images, int) or isinstance(images, bool) or images < 1:
        raise ValueError(f"arm {name}: max_images must be a positive integer")
    if not isinstance(arm.get("reasoning_ids", False), bool):
        raise ValueError(f"arm {name}: reasoning_ids must be true or false")
    memory = arm.get("memory")
    if not isinstance(memory, str) or not (
        memory in ("s1", "cloud") or os.path.isabs(memory)
    ):
        raise ValueError(
            f"arm {name}: memory must be 's1', 'cloud' or an absolute corpus path"
        )
    corrections = arm.get("corrections", "none")
    if corrections not in ("none", "s3"):
        raise ValueError(f"arm {name}: corrections must be 'none' or 's3'")
    fixed = sorted(k for k in ("plan", "checkin", "errors") if k in arm)
    if planner in ALONE_PLANNERS:
        if "rule" in arm or fixed or corrections != "none":
            who = "Qwen-alone" if planner == "qwen" else "GPT-6-alone"
            raise ValueError(
                f"arm {name}: a {who} arm has no senior: no rule, plan/checkin/errors or corrections"
            )
        return
    if arm.get("rule") == "s2":
        if fixed:
            raise ValueError(
                f"arm {name}: rule 's2' takes plan, K and N from each task's rule; drop {fixed}"
            )
    elif arm.get("rule") == "fixed":
        if fixed != ["checkin", "errors", "plan"]:
            raise ValueError(f"arm {name}: rule 'fixed' needs plan, checkin and errors")
        HE.Rule(
            plan=arm["plan"], checkin=arm["checkin"], errors=arm["errors"]
        )  # validates
    else:
        raise ValueError(f"arm {name}: a hybrid arm needs rule 's2' or 'fixed'")


def validate_final_test(cfg: Mapping[str, Any]) -> None:
    ft = cfg["final_test"]
    seeds = ft["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or not all(
            isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds
        )
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(
            "final_test.seeds must be a list of distinct nonnegative integers"
        )
    if 0 in seeds:
        raise ValueError(
            "final_test.seeds must not contain 0: stage 1 explored seed 0's scenes"
        )
    if (
        not isinstance(ft["repeats"], int)
        or isinstance(ft["repeats"], bool)
        or ft["repeats"] < 1
    ):
        raise ValueError("final_test.repeats must be a positive integer")
    if ft["cell_order"] not in ("seed_major", "shuffled"):
        raise ValueError("final_test.cell_order must be 'seed_major' or 'shuffled'")
    if ft["evidence_policy"] not in ("stop", "warn"):
        raise ValueError("final_test.evidence_policy must be 'stop' or 'warn'")
    for key in ("infra_retries", "render_retries"):
        if not isinstance(ft[key], int) or ft[key] < 0:
            raise ValueError(f"final_test.{key} must be a nonnegative integer")
    if not isinstance(ft["share_own_gpus"], bool):
        raise ValueError("final_test.share_own_gpus must be true or false")
    for key in ("gpu_apps_cmd", "gpu_uuid_cmd", "own_process_prefixes"):
        if not isinstance(ft[key], list) or not all(
            isinstance(v, str) and v for v in ft[key]
        ):
            raise ValueError(f"final_test.{key} must be a list of strings")
    unknown_splits = set(ft["timeout_s_by_split"]) - set(C.SPLITS)
    if unknown_splits:
        raise ValueError(
            f"final_test.timeout_s_by_split: unknown splits {sorted(unknown_splits)}"
        )
    arms = ft["arms"]
    if not isinstance(arms, Mapping):
        raise ValueError("final_test.arms must be a table of arm tables")
    for name, arm in arms.items():
        validate_arm(name, arm)
    unknown = [a for a in ft["run_arms"] if a not in arms]
    if unknown:
        raise ValueError(f"final_test.run_arms names undefined arms {unknown}")
    for pair in ft["pairs"]:
        if not (
            isinstance(pair, (list, tuple)) and len(pair) == 2 and pair[0] != pair[1]
        ) or not all(a in arms for a in pair):
            raise ValueError(
                f"final_test.pairs: {pair!r} is not two different defined arms"
            )


def stage_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The config this stage runs with: ``final_test.scheduler`` and ``.endpoints`` applied."""
    out = copy.deepcopy(dict(cfg))
    ft = out["final_test"]
    out["scheduler"] = C._deep_merge(out["scheduler"], ft["scheduler"])
    if ft["endpoints"]:
        out["endpoints"] = copy.deepcopy(ft["endpoints"])
    C.validate_config(out)
    return out


def cell_argv(cfg: Mapping[str, Any]) -> list[str]:
    code = cfg["paths"].get("code_dir") or str(REPO_ROOT)
    argv = cfg["final_test"]["launcher"] or [
        "bash",
        "{code_dir}/scripts/ecc_robocasa/hybrid_cell.sh",
    ]
    return [str(item).replace("{code_dir}", code) for item in argv]


def launcher_script(cfg: Mapping[str, Any]) -> Path:
    """The launcher's script: the first ``.sh``/``.py`` argument (after an interpreter), else argv[0]."""
    argv = cell_argv(cfg)
    return Path(next((a for a in argv if Path(a).suffix in (".sh", ".py")), argv[0]))


def launcher_runs_qwen_alone(cfg: Mapping[str, Any]) -> bool:
    """Whether the launcher honours ``ECC_RC_ESCALATION=0`` (needed by Qwen-alone arms)."""
    try:
        return QWEN_SWITCH in launcher_script(cfg).read_text(errors="replace")
    except OSError:
        return False


def launcher_runs_gpt6_alone(cfg: Mapping[str, Any]) -> bool:
    """Whether the launcher honours ``ECC_RC_SENIOR_ALONE=1`` (needed by GPT-6-alone arms)."""
    try:
        text = launcher_script(cfg).read_text(errors="replace")
    except OSError:
        return False
    return QWEN_SWITCH in text and ALONE_SWITCH in text


def unsupported_alone_arms(cfg: Mapping[str, Any], arms: list[str]) -> list[str]:
    """Run arms whose planner switch the launcher lacks (it would run them as hybrid episodes)."""
    ft = cfg["final_test"]
    out = []
    for arm in arms:
        planner = ft["arms"][arm]["planner"]
        if (planner == "qwen" and not launcher_runs_qwen_alone(cfg)) or (
            planner == "gpt6" and not launcher_runs_gpt6_alone(cfg)
        ):
            out.append(arm)
    return out


def config_row_supported() -> bool:
    """Whether the escalation wrapper this runner imports writes the config row (stage 2's check)."""
    from scripts.ecc_robocasa import stage2 as S2

    return S2.config_row_supported()


def experiment(cfg: Mapping[str, Any]) -> dict[str, Any]:
    ft = cfg["final_test"]
    return {
        "worker": cfg["worker"],
        "senior": cfg["senior"],
        "episode": cfg["episode"],
        "final_test": {k: copy.deepcopy(ft[k]) for k in FT_EXPERIMENT_KEYS},
    }


def run_tasks(cfg: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``final_test.tasks`` (empty: all) as ``(split, task)`` in Target50 order."""
    all_tasks = C.load_target50(cfg["paths"]["code_dir"])
    wanted = list(cfg["final_test"]["tasks"])
    unknown = sorted(set(wanted) - {task for _, task in all_tasks})
    if unknown:
        raise SystemExit(f"unknown tasks: {unknown}")
    return [row for row in all_tasks if not wanted or row[1] in wanted]


def apply_cli(cfg: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """The config with the command line's experiment choices written in (they reach the manifest)."""
    cfg = copy.deepcopy(dict(cfg))
    ft = cfg["final_test"]
    if getattr(args, "seeds", None):
        ft["seeds"] = [int(s) for s in args.seeds.split(",") if s]
    if getattr(args, "repeats", 0):
        ft["repeats"] = int(args.repeats)
    arms = [
        a
        for a in (
            args.arms.split(",") if getattr(args, "arms", None) else ft["run_arms"]
        )
        if a
    ]
    ft["run_arms"] = arms or list(ft["arms"])
    if not ft["run_arms"]:
        raise SystemExit("no arms: define [final_test.arms.<name>]")
    if len(set(ft["run_arms"])) != len(ft["run_arms"]):
        raise SystemExit(f"arms repeated: {ft['run_arms']}")
    if getattr(args, "tasks", None):
        ft["tasks"] = [t for t in args.tasks.split(",") if t]
    tasks = run_tasks(cfg)
    if getattr(args, "limit", 0):
        tasks = tasks[: args.limit]
    ft["tasks"] = [task for _, task in tasks]
    validate_final_test(cfg)
    return cfg


# ---------------------------------------------------------------------------
# inputs: products, rules, corrections, read-only snapshots
# ---------------------------------------------------------------------------


def parse_rule(doc: Any, *, allow_off_ladder: bool = False) -> dict[str, Any]:
    """(plan, K, N) and rung of one stage-2 rule file (``rule.gpt_plan/checkin_K/errors_N``)."""
    if not isinstance(doc, Mapping) or not isinstance(doc.get("rule"), Mapping):
        raise ValueError('no "rule" object')
    try:
        rule = HE.Rule.from_json(doc["rule"])
    except KeyError as exc:
        raise ValueError(f"the rule has no {exc}") from None
    rung = HE.LEADER_LADDER.index(rule) + 1 if rule in HE.LEADER_LADDER else None
    if rung is None and not allow_off_ladder:
        raise ValueError(f"{rule} is not a rung of the stage-2 ladder")
    named = doc.get("config")
    if named is not None and rung is not None and named != rule.config_name(rung):
        raise ValueError(
            f"config name {named!r} disagrees with the rule ({rule.config_name(rung)})"
        )
    if doc.get("rung") is not None and rung is not None and doc["rung"] != rung:
        raise ValueError(f"rung {doc['rung']!r} disagrees with the rule (rung {rung})")
    return {
        "plan": rule.plan,
        "checkin": rule.checkin,
        "errors": rule.errors,
        "rung": rung,
    }


def corrections_info(raw: bytes) -> dict[str, Any]:
    """What RPent derives from a corrections file: the length and hash of its stripped text."""
    text = raw.decode("utf-8").strip()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "chars": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text
        else None,
    }


def memory_label(value: str) -> str:
    return (
        value
        if value in ("s1", "cloud")
        else "path_" + hashlib.sha256(value.encode()).hexdigest()[:10]
    )


def memory_source(cfg: Mapping[str, Any], value: str) -> dict[str, Any]:
    """Where an arm's memory corpus comes from, and its tree hash."""
    if value == "s1":
        product = cfg["final_test"]["products"]["s1"]
        if not product:
            raise SystemExit("an arm uses memory 's1': set final_test.products.s1")
        product = Path(product)
        check = S1.verify(product)
        if not check.get("ok"):
            raise SystemExit(f"the stage-1 product {product} does not verify: {check}")
        doc = C.read_json(product / "PRODUCT.json") or {}
        return {
            "kind": "s1",
            "source": str(product / doc["corpus"]),
            "product": str(product),
            "product_json_sha256": C.sha256_file(product / "PRODUCT.json"),
            "tree_sha256": doc["corpus_tree_sha256"],
        }
    source = Path(cfg["paths"]["cloud_corpus"] if value == "cloud" else value)
    if not (source / "global" / "GLOBAL_MEMORY.md").is_file():
        raise SystemExit(f"memory {value!r}: no corpus at {source}")
    return {
        "kind": value if value == "cloud" else "path",
        "source": str(source),
        "tree_sha256": C.tree_hash(C.tree_files(source)),
    }


def _product_source(cfg: Mapping[str, Any], stage: str) -> dict[str, Any]:
    """Where an arm's stage-2 rules or stage-3 corrections come from.

    ``products.s2`` / ``.s3`` name a frozen product, verified by its stage's
    ``verify``; ``products.s2_rules`` / ``.s3_corrections`` name a bare folder.
    """
    ft = cfg["final_test"]
    products = ft["products"]
    product_key, folder_key, name_key, what = {
        "s2": ("s2", "s2_rules", "s2_rule_name", "rule 's2'"),
        "s3": ("s3", "s3_corrections", "s3_corrections_name", "corrections 's3'"),
    }[stage]
    if products[product_key] and products[folder_key]:
        raise SystemExit(
            f"set final_test.products.{product_key} or .{folder_key}, not both"
        )
    if products[folder_key]:
        return {
            "kind": "folder",
            "folder": str(Path(products[folder_key])),
            "name": ft[name_key],
            "doc": None,
        }
    if not products[product_key]:
        raise SystemExit(
            f"an arm uses {what}: set final_test.products.{product_key} (the frozen product)"
        )
    product = Path(products[product_key])
    if stage == "s2":
        from scripts.ecc_robocasa import stage2 as S2

        check, doc_name, sub = S2.verify(product), "RULES.json", "rules"
        name = "{task}.json"
    else:
        from scripts.ecc_robocasa import stage3 as S3

        check, doc_name, sub = S3.verify(product), "PRODUCT.json", "final"
        name = "{task}.md"
    if not check.get("ok"):
        raise SystemExit(f"the {stage} product {product} does not verify: {check}")
    doc = C.read_json(product / doc_name) or {}
    return {
        "kind": "product",
        "product": str(product),
        "product_json_sha256": C.sha256_file(product / doc_name),
        "folder": str(product / doc[sub]),
        "name": name,
        "doc": doc,
    }


def product_chain(
    inputs: Mapping[str, Any],
    s2_doc: Mapping[str, Any] | None,
    s3_doc: Mapping[str, Any] | None,
) -> list[str]:
    """Whether the products were built on each other: S2 on this S1 corpus, S3 on both."""
    s1 = (inputs["memory"].get("s1") or {}).get("tree_sha256")
    problems = []
    if s2_doc is not None and s1:
        used = (s2_doc.get("s1_product") or {}).get("corpus_tree_sha256")
        if used != s1:
            problems.append(
                f"the S2 rules were searched on S1 corpus {str(used)[:12]}, not {s1[:12]}"
            )
    if s3_doc is not None:
        inputs_s3 = s3_doc.get("inputs") or {}
        used = (inputs_s3.get("s1") or {}).get("corpus_tree_sha256")
        if s1 and used != s1:
            problems.append(
                f"the S3 corrections were built on S1 corpus {str(used)[:12]}, not {s1[:12]}"
            )
        rule_files = (inputs_s3.get("rules") or {}).get("files") or {}
        differ = sorted(
            t
            for t, r in inputs["rules"].items()
            if t in rule_files and rule_files[t] != r["sha256"]
        )
        missing = sorted(t for t in inputs["rules"] if t not in rule_files)
        if inputs["rules"] and (differ or missing):
            problems.append(
                f"the S3 corrections were built under other S2 rules for {len(differ) + len(missing)} task(s): "
                f"{(differ + missing)[:6]}"
            )
    return problems


def read_inputs(
    cfg: Mapping[str, Any], arms: list[str], tasks: list[tuple[str, str]]
) -> dict[str, Any]:
    """Every input the run arms read, from their sources (nothing is written)."""
    ft = cfg["final_test"]
    defs = [ft["arms"][name] for name in arms]
    inputs: dict[str, Any] = {
        "memory": {},
        "rules": {},
        "corrections": {},
        "sources": {},
    }
    for value in sorted({d["memory"] for d in defs}):
        inputs["memory"][memory_label(value)] = {
            "arm_value": value,
            **memory_source(cfg, value),
        }
    s2_doc = s3_doc = None
    if any(d.get("rule") == "s2" for d in defs):
        source = _product_source(cfg, "s2")
        s2_doc = source.pop("doc")
        inputs["sources"]["s2"] = source
        problems = []
        for _, task in tasks:
            path = Path(source["folder"]) / source["name"].format(task=task)
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
                rule = parse_rule(doc, allow_off_ladder=ft["allow_off_ladder_rules"])
            except (OSError, ValueError) as exc:
                problems.append(f"{task}: {exc}")
                continue
            inputs["rules"][task] = {
                **rule,
                "source": str(path),
                "sha256": C.sha256_file(path),
                "solved_in_search": doc.get("solved_in_search"),
                "rule_source": doc.get("rule_source"),
            }
        if problems:
            raise SystemExit(
                f"stage-2 rules in {source['folder']}: " + "; ".join(problems[:10])
            )
    if any(d.get("corrections") == "s3" for d in defs):
        source = _product_source(cfg, "s3")
        s3_doc = source.pop("doc")
        inputs["sources"]["s3"] = source
        missing = []
        for _, task in tasks:
            path = Path(source["folder"]) / source["name"].format(task=task)
            if path.is_file():
                inputs["corrections"][task] = {
                    "source": str(path),
                    **corrections_info(path.read_bytes()),
                }
            elif ft["allow_missing_corrections"]:
                inputs["corrections"][task] = {
                    "source": str(path),
                    "source_missing": True,
                    **corrections_info(b""),
                }
            else:
                missing.append(task)
        if missing:
            raise SystemExit(
                f"stage-3 corrections missing for {len(missing)} task(s) in {source['folder']} (RPent would "
                f"silently run them without corrections; allow_missing_corrections runs them empty): {missing[:10]}"
            )
    chain = product_chain(inputs, s2_doc, s3_doc)
    if chain and not ft["allow_product_mismatch"]:
        raise SystemExit(
            "the stage products do not chain (allow_product_mismatch accepts it): "
            + "; ".join(chain)
        )
    inputs["chain_problems"] = chain
    return inputs


def inputs_dir(layout: C.RunLayout) -> Path:
    return layout.root / "inputs"


def inputs_record(layout: C.RunLayout) -> Path:
    return layout.root / "inputs.json"


def snapshot_inputs(
    layout: C.RunLayout, inputs: dict[str, Any], tasks: list[tuple[str, str]]
) -> dict[str, Any]:
    """Copy every input into ``<run>/inputs`` read-only and check the copies' hashes."""
    base = inputs_dir(layout)
    if base.exists():  # a creation that did not finish
        S1._make_writable(base)
        shutil.rmtree(base)
    for label, info in inputs["memory"].items():
        dest = base / "memory" / C.safe_name(label)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(info["source"], dest, symlinks=False)
        if C.tree_hash(C.tree_files(dest)) != info["tree_sha256"]:
            raise SystemExit(f"memory {label}: the copy's hash differs from the source")
        validation = S1.validate_corpus(dest, tasks)
        if validation["errors"]:
            raise SystemExit(
                f"memory {label}: tasks do not load: {json.dumps(validation['errors'])[:1500]}"
            )
        info.update(snapshot=str(dest), selected=validation["selected"])
    for kind, suffix in (("rules", "json"), ("corrections", "md")):
        for task, info in inputs[kind].items():
            dest = base / kind / f"{C.safe_name(task)}.{suffix}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if info.get("source_missing"):
                dest.write_bytes(b"")
            else:
                shutil.copyfile(info["source"], dest)
            if C.sha256_file(dest) != info["sha256"]:
                raise SystemExit(
                    f"{kind} {task}: the copy's hash differs from the source"
                )
            info["snapshot"] = str(dest)
    S1._make_read_only(base)
    inputs["tree_sha256"] = C.tree_hash(C.tree_files(base))
    C.atomic_write_json(inputs_record(layout), inputs)
    inputs_record(layout).chmod(0o444)
    return inputs


def verify_snapshot(layout: C.RunLayout, inputs: Mapping[str, Any]) -> list[str]:
    """Problems with ``<run>/inputs`` against the hash recorded when the run was created."""
    base = inputs_dir(layout)
    if not base.is_dir():
        return ["inputs/ is missing"]
    problems = []
    if C.tree_hash(C.tree_files(base)) != inputs.get("tree_sha256"):
        problems.append("inputs/ changed since the run was created")
    writable = [p for p in base.rglob("*") if p.stat().st_mode & stat.S_IWUSR]
    if writable:
        problems.append(f"{len(writable)} inputs/ entries are writable")
    return problems


# ---------------------------------------------------------------------------
# cells, jobs, knobs
# ---------------------------------------------------------------------------


def cell_name(task: str, seed: int, rep: int) -> str:
    return f"{task}_s{seed}_r{rep}"


def job_name(arm: str, task: str, seed: int, rep: int) -> str:
    return f"{arm}__{cell_name(task, seed, rep)}"


def _key(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def arm_order(
    order_seed: int, arms: list[str], task: str, seed: int, rep: int
) -> list[str]:
    """The seeded order of the arms within one cell (sha256 keys: the same on every platform)."""
    return sorted(arms, key=lambda arm: _key(order_seed, task, seed, rep, arm))


def build_jobs(
    cfg: Mapping[str, Any], arms: list[str], tasks: list[tuple[str, str]]
) -> list[C.Job]:
    """One job per (cell, arm); cells in ``cell_order``, the arms of a cell next to each other."""
    ft = cfg["final_test"]
    cells = [
        (split, task, seed, rep)
        for rep in range(1, ft["repeats"] + 1)
        for seed in ft["seeds"]
        for split, task in tasks
    ]
    if ft["cell_order"] == "shuffled":
        cells.sort(key=lambda c: _key(ft["order_seed"], "cell", c[1], c[2], c[3]))
    jobs = []
    for index, (split, task, seed, rep) in enumerate(cells):
        for position, arm in enumerate(
            arm_order(ft["order_seed"], arms, task, seed, rep)
        ):
            jobs.append(
                C.Job(
                    job_id=job_name(arm, task, seed, rep),
                    task=task,
                    split=split,
                    seed=seed,
                    order=len(jobs),
                    meta={
                        "arm": arm,
                        "rep": rep,
                        "cell": cell_name(task, seed, rep),
                        "cell_index": index,
                        "arm_position": position,
                    },
                )
            )
    return jobs


def timeouts(cfg: Mapping[str, Any]) -> dict[str, int]:
    """Planner timeout per split: the Target50 manifest unless the config overrides it."""
    manifest = json.loads(
        (Path(cfg["paths"]["code_dir"]) / C.TARGET50_MANIFEST).read_text()
    )
    out = {s: int(manifest["splits"][s]["timeout_seconds"]) for s in C.SPLITS}
    out.update({s: int(v) for s, v in cfg["final_test"]["timeout_s_by_split"].items()})
    return out


def resolve_knobs(
    cfg: Mapping[str, Any],
    inputs: Mapping[str, Any],
    job: C.Job,
    timeout_s: Mapping[str, int],
) -> dict[str, Any]:
    """What one episode runs with."""
    arm = cfg["final_test"]["arms"][job.meta["arm"]]
    label = memory_label(arm["memory"])
    knobs: dict[str, Any] = {
        "arm": job.meta["arm"],
        "planner": arm["planner"],
        "escalation": arm["planner"] == "hybrid",
        "memory": label,
        "memory_dir": inputs["memory"][label]["snapshot"],
        "timeout_s": int(timeout_s[job.split]),
        "plan": None,
        "checkin": None,
        "errors": None,
        "rung": None,
        "rule_source": None,
        "corrections": None,
    }
    if arm["planner"] == "gpt6":
        # The only planner: the senior's model, effort and route.
        knobs.update(
            model=str(cfg["senior"]["model"]),
            reasoning_effort=str(cfg["senior"]["reasoning_effort"]),
            max_images=int(arm.get("max_images", GPT6_MAX_IMAGES)),
            reasoning_ids=bool(arm.get("reasoning_ids", False)),
        )
    if arm["planner"] == "hybrid":
        if arm["rule"] == "s2":
            rule = inputs["rules"][job.task]
            source = f"s2 {rule['sha256'][:16]}"
        else:
            rule = {
                "plan": arm["plan"],
                "checkin": arm["checkin"],
                "errors": arm["errors"],
            }
            source = "fixed"
        triple = HE.Rule(
            plan=rule["plan"], checkin=rule["checkin"], errors=rule["errors"]
        )
        knobs.update(
            plan=triple.plan,
            checkin=triple.checkin,
            errors=triple.errors,
            rung=HE.LEADER_LADDER.index(triple) + 1
            if triple in HE.LEADER_LADDER
            else None,
            rule_source=source,
        )
        if arm.get("corrections", "none") == "s3":
            info = inputs["corrections"][job.task]
            knobs["corrections"] = {
                k: info[k] for k in ("sha256", "bytes", "chars", "text_sha256")
            } | {"file": info["snapshot"]}
    return knobs


def knobs_rule(knobs: Mapping[str, Any]) -> HE.Rule:
    return HE.Rule(plan=knobs["plan"], checkin=knobs["checkin"], errors=knobs["errors"])


def cell_env(
    cfg: Mapping[str, Any],
    job: C.Job,
    n: int,
    endpoint: C.Endpoint,
    knobs: Mapping[str, Any],
    parent: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The launcher's environment (``hybrid_episode.hybrid_env``); it sources the credentials itself."""
    corr = knobs["corrections"]
    if knobs["escalation"]:
        rule = knobs_rule(knobs)
    else:  # Qwen or GPT-6 alone: the routing values are unused; the launcher turns the senior off
        rule = HE.Rule(plan=True, checkin=int(cfg["senior"]["checkin"]), errors=2)
    env = HE.hybrid_env(
        cfg,
        split=job.split,
        try_n=n,
        endpoint=endpoint,
        rule=rule,
        timeout_s=int(knobs["timeout_s"]),
        corrections=Path(corr["file"]) if corr else None,
        corrections_sha256=corr["sha256"] if corr else "",
        parent=parent,
    )
    env[QWEN_SWITCH] = "1" if knobs["escalation"] else "0"
    if knobs["planner"] == "gpt6":
        env.update(
            {
                ALONE_SWITCH: "1",
                "ECC_RC_MAX_IMAGES": str(knobs["max_images"]),
                "ECC_RC_REASONING_IDS": "1" if knobs["reasoning_ids"] else "0",
            }
        )
    return env


# ---------------------------------------------------------------------------
# evidence and error runs
# ---------------------------------------------------------------------------


def tool_category(content: Any) -> str:
    """How the escalation trigger sees one transcript tool message.

    ``err``/``ok``: a tool return whose parsed object has / lacks a top-level
    ``error``; ``retry``: pydantic-ai's argument-validation retry (a list of error
    details), which the trigger skips; ``unknown``: plain text (a retry prompt
    such as an unknown tool name, or a result the trigger reads as ``{}``).
    """
    value = content
    if isinstance(content, str):
        try:
            value = json.loads(content)
        except ValueError:
            return "unknown"
    if isinstance(value, dict):
        return "err" if "error" in value else "ok"
    if isinstance(value, list) and value:
        if all(isinstance(v, dict) and {"type", "loc", "msg"} <= set(v) for v in value):
            return "retry"
        for item in value:
            if isinstance(item, dict):
                return "err" if "error" in item else "ok"
    return "unknown"


def error_runs(try_dir: Path) -> dict[str, Any] | None:
    """Errored tool results and their longest run in a row (retries skipped), from the transcript.

    The senior resets the trigger's count, so a run of length L means N <= L could
    have fired at most; this is the counterfactual behind the N rungs. None when
    the transcript holds no turn (a planner timeout keeps only the first message).
    """
    for path in sorted(try_dir.glob("transcript_*.json"))[-1:]:
        doc = C.read_json(path)
        messages = doc.get("messages") if isinstance(doc, Mapping) else None
        if not isinstance(messages, list) or not any(
            isinstance(m, Mapping) and m.get("role") == "assistant" for m in messages
        ):
            return None
        errored = best = run = 0
        for message in messages:
            if not isinstance(message, Mapping) or message.get("role") != "tool":
                continue
            category = tool_category(message.get("content"))
            if category == "retry":
                continue
            run = run + 1 if category == "err" else 0
            errored += category == "err"
            best = max(best, run)
        return {"errored_results": errored, "max_run": best}
    return None


def qwen_alone_evidence(try_dir: Path) -> dict[str, Any]:
    """A Qwen-alone launch must not build the escalation wrapper (no escalation.jsonl)."""
    problems = []
    if (try_dir / "escalation.jsonl").exists():
        problems.append(
            "escalation.jsonl exists, but this arm runs Qwen alone (the launcher kept the senior)"
        )
    return {
        "ok": not problems,
        "problems": problems,
        "evidence": {
            "log": (try_dir / "escalation.jsonl").exists(),
            "config_row": None,
        },
        "counts": {
            "requests": 0,
            "senior_calls": 0,
            "senior_by_reason": dict.fromkeys(HE.SENIOR_REASONS, 0),
            "errors_trigger": 0,
        },
    }


def senior_alone_usage(try_dir: Path, pricing: Mapping[str, float]) -> dict[str, Any]:
    """Tokens and list-price USD of a GPT-6-alone launch, which writes no escalation log.

    RPent's transcript stats count every returned request, a last reply without a
    tool call included; after a planner timeout the transcript keeps no stats and
    the last ``[usage]`` line of ``run.log`` is used (it misses only such a last
    reply). Every request is a GPT-6 request, priced as the senior's: cached
    input at the cached price.
    """
    stats: Mapping[str, Any] = {}
    for path in sorted(try_dir.glob("transcript_*.json"))[-1:]:
        doc = C.read_json(path, {})
        if isinstance(doc, Mapping) and isinstance(doc.get("stats"), Mapping):
            if "requests" in doc["stats"]:
                stats = doc["stats"]
    if stats:
        source = "transcript"
        tokens_in = int(stats.get("total_input_tokens") or 0)
        cached = int(stats.get("cache_read_tokens") or 0)
        tokens_out = int(stats.get("total_output_tokens") or 0)
        requests = int(stats.get("requests") or 0)
    else:
        source = "run.log"
        log = C.usage_from_run_log(try_dir / "run.log")
        tokens_in, tokens_out, requests = (
            log["input_tokens"],
            log["output_tokens"],
            log["requests"],
        )
        cached = sum(int(s.get("cache_read_tokens") or 0) for s in log["sessions"])
    usd = (
        (tokens_in - cached) * pricing["senior_input_per_mtok"]
        + cached * pricing["senior_cached_input_per_mtok"]
        + tokens_out * pricing["senior_output_per_mtok"]
    ) / 1e6
    return {
        "source": source,
        "planner": "gpt6",
        "worker_in": 0,
        "worker_out": 0,
        "worker_calls": 0,
        "senior_in": tokens_in,
        "senior_cached": cached,
        "senior_out": tokens_out,
        "senior_calls": requests,
        "worker_usd": 0.0,
        "senior_usd": usd,
        "usd": usd,
    }


def senior_alone_evidence(
    try_dir: Path, knobs: Mapping[str, Any], pricing: Mapping[str, float]
) -> dict[str, Any]:
    """A GPT-6-alone launch: no escalation wrapper, RPent ran the senior's model with the image cap.

    With a valid result, ``result.json`` must name the api planner with the
    senior's model and effort, and ``run.log`` must show the image cap (and, with
    ``reasoning_ids`` off, the reasoning-id switch) RPent logs when it builds the
    planner. Every returned request counts as a GPT-6 call (reason ``planner``).
    """
    log = try_dir / "escalation.jsonl"
    problems = []
    if log.exists():
        problems.append(
            "escalation.jsonl exists, but this arm runs GPT-6 alone (the launcher kept the escalation wrapper)"
        )
    planner: Mapping[str, Any] = {}
    if C.read_outcome(try_dir).get("valid"):
        result = C.read_json(try_dir / "result.json", {}) or {}
        if isinstance(result.get("planner"), Mapping):
            planner = result["planner"]
        for key, want in (
            ("backend", "api"),
            ("model", knobs["model"]),
            ("reasoning_effort", knobs["reasoning_effort"]),
        ):
            if planner.get(key) != want:
                problems.append(
                    f"result.json planner {key}={planner.get(key)!r}, expected {want!r}"
                )
        try:
            text = (try_dir / "run.log").read_text(errors="replace")
        except OSError:
            text = ""
        if IMAGE_CAP_LOG.format(n=knobs["max_images"]) not in text:
            problems.append(
                f"run.log does not show the {knobs['max_images']}-image cap (RPENT_MAX_HISTORY_IMAGES)"
            )
        if not knobs["reasoning_ids"] and NO_IDS_LOG not in text:
            problems.append(
                "run.log does not show that response item ids are off (RPENT_SEND_REASONING_IDS)"
            )
    usage = senior_alone_usage(try_dir, pricing)
    calls = int(usage["senior_calls"])
    return {
        "ok": not problems,
        "problems": problems,
        "evidence": {
            "log": log.exists(),
            "config_row": None,
            "planner": dict(planner),
            "usage_source": usage["source"],
        },
        "counts": {
            "requests": calls,
            "senior_calls": calls,
            "senior_by_reason": {
                **dict.fromkeys(HE.SENIOR_REASONS, 0),
                "planner": calls,
            },
            "errors_trigger": 0,
        },
    }


def is_alone_usage(usage: Any) -> bool:
    """Whether a launch's recorded usage was priced as a GPT-6-alone launch."""
    return isinstance(usage, Mapping) and usage.get("planner") == "gpt6"


# ---------------------------------------------------------------------------
# GPUs shared with this machine's other ECC runs (final_test.share_own_gpus)
# ---------------------------------------------------------------------------


def gpu_processes(cfg: Mapping[str, Any]) -> dict[int, list[tuple[int, int]]]:
    """``{gpu index: [(pid, used MiB), ...]}`` of the compute processes nvidia-smi lists."""
    ft = cfg["final_test"]

    def query(argv: list[str]) -> list[list[str]]:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            raise RuntimeError(f"{argv[0]} failed: {out.stderr.strip()[:300]}")
        return [
            [p.strip() for p in line.split(",")]
            for line in out.stdout.splitlines()
            if line.strip()
        ]

    index = {
        row[1]: int(row[0])
        for row in query(ft["gpu_uuid_cmd"])
        if len(row) >= 2 and row[0].isdigit()
    }
    procs: dict[int, list[tuple[int, int]]] = {}
    for row in query(ft["gpu_apps_cmd"]):
        if len(row) < 3 or row[0] not in index or not row[1].isdigit():
            continue
        used = int(float(re.sub(r"[^0-9.]", "", row[2]) or 0))
        procs.setdefault(index[row[0]], []).append((int(row[1]), used))
    return procs


def own_process(cfg: Mapping[str, Any], pid: int) -> bool:
    """Whether ``pid``'s executable (argv[0], as started) lies under ``own_process_prefixes``.

    The default prefix is ``paths.venv``: this machine's RLDX servers and episode
    env servers run the venv's Python. A process whose command line cannot be read
    (another user's or container's) is not ours.
    """
    prefixes = list(cfg["final_test"]["own_process_prefixes"]) or [
        str(cfg["paths"].get("venv") or "")
    ]
    argv = C._cmdline(pid) or []
    if not argv:
        return False
    bases = {b.rstrip(os.sep) for p in prefixes if p for b in (p, os.path.realpath(p))}
    return any(argv[0] == b or argv[0].startswith(b + os.sep) for b in bases)


def shared_gpu_room(
    cfg: Mapping[str, Any],
    gpu: int,
    need_mb: int,
    usage: Mapping[int, Mapping[str, Any]],
    procs: Mapping[int, list[tuple[int, int]]],
) -> tuple[bool, str]:
    """Whether a busy ``gpu`` holds only this machine's ECC processes and ``need_mb`` more MiB fit.

    Memory that no listed process accounts for (another container's) must stay
    below ``gpus.max_used_mb``, as on a free GPU; utilization is not checked
    (the other run's servers keep the GPU busy).
    """
    if gpu not in cfg["gpus"]["allowed"]:
        return False, f"GPU {gpu} is not allowed ({cfg['gpus']['allowed']})"
    row = usage.get(gpu)
    if row is None or "used_mb" not in row or not row.get("total_mb"):
        return False, f"GPU {gpu} memory total/used not reported by nvidia-smi"
    mine = procs.get(gpu, [])
    foreign = [pid for pid, _ in mine if not own_process(cfg, pid)]
    if foreign:
        return (
            False,
            f"GPU {gpu} runs {len(foreign)} process(es) not started from this machine's ECC venv",
        )
    used = max(row["used_mb"], row.get("csv_used_mb", 0))
    unaccounted = used - sum(mb for _, mb in mine)
    if unaccounted >= int(cfg["gpus"]["max_used_mb"]):
        return False, f"GPU {gpu}: {unaccounted} MiB used by processes not visible here"
    room = row["total_mb"] - used
    if room < need_mb:
        return (
            False,
            f"GPU {gpu} shared with our other runs: {room} MiB free, {need_mb} MiB needed",
        )
    return True, (
        f"GPU {gpu} shared with our other runs ({len(mine)} processes); {room} MiB free >= {need_mb} MiB"
    )


class FinalServerPool(C.ServerPool):
    """``common.ServerPool``; with ``final_test.share_own_gpus`` a GPU busy only with this
    machine's ECC processes is claimed when this run's servers and slots on it fit.

    The plan of a newly claimed GPU is every endpoint of this run on it (server_mb
    per server, cell_mb per slot) plus ``servers.fit_margin_mb``; the servers.max_gpus
    limit and the port check apply as before, and a GPU this run already holds keeps
    the base rule (room for one more server).
    """

    def _start(
        self,
        ep: C.Endpoint,
        usage: Mapping[int, Mapping[str, Any]] | None,
        forced: bool,
    ) -> str:
        cfg = self.cfg
        if forced or not cfg["final_test"]["share_own_gpus"] or usage is None:
            return super()._start(ep, usage, forced)
        held = self.held_gpus(exclude=ep)
        max_gpus = int(cfg["servers"].get("max_gpus") or 0)
        if ep.gpu in held or (max_gpus and len(held) >= max_gpus):
            return super()._start(ep, usage, forced)
        if C.gpu_free(cfg, ep.gpu, usage)[0]:
            return super()._start(ep, usage, forced)
        servers = cfg["servers"]
        need = int(servers["fit_margin_mb"]) + sum(
            int(servers["server_mb"]) + e.slots * int(servers["cell_mb"])
            for e in self.endpoints
            if e.gpu == ep.gpu
        )
        try:
            ok, reason = shared_gpu_room(cfg, ep.gpu, need, usage, gpu_processes(cfg))
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            ok, reason = False, f"GPU process query failed: {exc}"
        self.log(
            {
                "event": "server_shared_gpu" if ok else "server_refused",
                "endpoint": ep.name,
                "reason": reason,
            }
        )
        if not ok:
            ep.up, ep.note = False, reason
            return "refused"
        return super()._start(ep, usage, True)


# ---------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------


class FinalTest(C.Stage):
    name = STAGE

    def __init__(
        self,
        cfg: Mapping[str, Any],
        layout: C.RunLayout,
        jobs: list[C.Job],
        inputs: Mapping[str, Any],
    ):
        self.cfg = cfg
        self.layout = layout
        self._jobs = jobs
        self.inputs = inputs
        self.code_dir = Path(cfg["paths"]["code_dir"])
        limits = timeouts(cfg)
        self.knobs = {
            job.job_id: resolve_knobs(cfg, inputs, job, limits) for job in jobs
        }
        self.by_id = {job.job_id: job for job in jobs}

    @property
    def cells_dir(self) -> Path:
        return self.layout.root / "cells"

    def jobs(self) -> list[C.Job]:
        return list(self._jobs)

    def job_dir(self, job: C.Job) -> Path:
        return self.cells_dir / C.safe_name(job.job_id)

    def try_dir(self, job: C.Job, n: int) -> Path:
        return self.job_dir(job) / f"try{n}"

    def record_path(self, job: C.Job) -> Path:
        return self.job_dir(job) / "episode.json"

    def budget_launches(self) -> int:
        return len(self._jobs)

    def alone(self, job_id: str) -> bool:
        """Whether the job runs GPT-6 alone (its usage is priced as the senior's)."""
        return self.knobs[job_id]["planner"] == "gpt6"

    def try_usage(self, job_id: str, entry: Mapping[str, Any]) -> dict[str, Any] | None:
        """Usage of one launch: as recorded, except GPT-6-alone launches, priced from their files.

        ``common.Scheduler`` records every launch with ``common.episode_usage``,
        which prices a launch without an escalation log at the worker's price.
        """
        usage = entry.get("usage")
        if not self.alone(job_id) or is_alone_usage(usage):
            return usage
        if not Path(entry["dir"]).exists():
            return usage
        return senior_alone_usage(Path(entry["dir"]), self.cfg["pricing"])

    def total_cost(self, states: Mapping[str, Mapping[str, Any]]) -> float:
        """``common.total_cost`` with GPT-6-alone launches priced as the senior's."""
        usd = 0.0
        for job_id, state in states.items():
            for entry in state["tries"]:
                usage = self.try_usage(job_id, entry)
                if (
                    usage is None
                    and entry.get("status") in ("launching", "running")
                    and Path(entry["dir"]).exists()
                ):
                    usage = C.episode_usage(Path(entry["dir"]), self.cfg["pricing"])
                usd += float((usage or {}).get("usd") or 0.0)
        return usd

    def expected(self, job: C.Job) -> dict[str, Any] | None:
        """What the escalation log must show (``hybrid_episode.expected_routing``); None for Qwen alone."""
        knobs = self.knobs[job.job_id]
        if not knobs["escalation"]:
            return None
        corr = knobs["corrections"]
        text = Path(corr["file"]).read_bytes().decode("utf-8") if corr else ""
        return HE.expected_routing(self.cfg, knobs_rule(knobs), corrections_text=text)

    def prepare_try(self, job: C.Job, n: int) -> None:
        """Refuse to launch on an input that is not what the run recorded.

        RPent ignores a missing corrections file, so the file's presence, sha256 and
        size are checked here and logged with the launch.
        """
        knobs = self.knobs[job.job_id]
        if not (Path(knobs["memory_dir"]) / "global" / "GLOBAL_MEMORY.md").is_file():
            raise RuntimeError(f"memory snapshot {knobs['memory_dir']} is missing")
        row: dict[str, Any] = {
            "event": "inputs",
            "job": job.job_id,
            "try": n,
            "memory": knobs["memory"],
        }
        corr = knobs["corrections"]
        if corr:
            path = Path(corr["file"])
            if not path.is_file():
                raise RuntimeError(f"corrections file {path} is missing")
            digest, size = C.sha256_file(path), path.stat().st_size
            if (digest, size) != (corr["sha256"], corr["bytes"]):
                raise RuntimeError(
                    f"corrections file {path} changed: sha256 {digest[:12]} size {size}, recorded "
                    f"{corr['sha256'][:12]} size {corr['bytes']}"
                )
            row["corrections"] = {
                "sha256": digest,
                "bytes": size,
                "chars": corr["chars"],
            }
        self.job_dir(job).mkdir(parents=True, exist_ok=True)
        C.atomic_write_json(
            self.job_dir(job) / f"try{n}.expect.json",
            {"knobs": knobs, "expected": self.expected(job), "at": C.now_iso()},
        )
        C.append_jsonl(self.layout.progress, row)

    def launch(
        self, job: C.Job, n: int, endpoint: C.Endpoint
    ) -> tuple[list[str], dict[str, str]]:
        knobs = self.knobs[job.job_id]
        argv = [
            *cell_argv(self.cfg),
            job.split,
            job.task,
            str(job.seed),
            str(self.try_dir(job, n)),
            str(knobs["memory_dir"]),
        ]
        return argv, cell_env(self.cfg, job, n, endpoint, knobs)

    def evidence(self, job: C.Job, n: int) -> dict[str, Any]:
        """Check launch ``n`` of ``job`` against its arm; the result is kept next to the launch.

        As in stage 2: the escalation log (``hybrid_episode.check_escalation``; a
        launch without a valid result may have no log), and the memory snapshot it
        read is unchanged.
        """
        try_dir = self.try_dir(job, n)
        expected = self.expected(job)
        if self.alone(job.job_id):
            result = senior_alone_evidence(
                try_dir, self.knobs[job.job_id], self.cfg["pricing"]
            )
        elif expected is None:
            result = qwen_alone_evidence(try_dir)
        else:
            result = HE.check_escalation(
                try_dir / "escalation.jsonl",
                expected,
                require_config_row=bool(self.cfg["final_test"]["require_config_row"]),
                result_valid=bool(C.read_outcome(try_dir).get("valid")),
            )
        label = self.knobs[job.job_id]["memory"]
        if (
            C.tree_hash(C.tree_files(self.knobs[job.job_id]["memory_dir"]))
            != self.inputs["memory"][label]["tree_sha256"]
        ):
            result["problems"].append(
                f"the memory snapshot {label} changed during the run"
            )
            result["ok"] = False
        C.atomic_write_json(self.job_dir(job) / f"try{n}.evidence.json", result)
        return result

    def config_row_problems(self, job: C.Job, try_dir: Path) -> list[str] | None:
        """Problems of a running launch's config row; None while the log has no config row yet."""
        log = try_dir / "escalation.jsonl"
        expected = self.expected(job)
        if expected is None:
            return (
                ["escalation.jsonl appeared, but this arm runs Qwen alone"]
                if log.exists()
                else None
            )
        rows, _ = HE.read_rows(log)
        config = next(
            (r["config"] for r in rows if isinstance(r.get("config"), dict)), None
        )
        return None if config is None else HE.config_row_problems(config, expected)

    def on_final(self, job: C.Job, state: dict[str, Any]) -> None:
        """Write the job's episode record (an error is logged; the next runner writes it)."""
        if self.record_path(job).exists():
            return
        try:
            record = self.episode_record(job, state)
        except Exception as exc:  # noqa: BLE001 - logged; the next runner writes it
            C.append_jsonl(
                self.layout.progress,
                {
                    "event": "record_error",
                    "job": job.job_id,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return
        C.atomic_write_json(self.record_path(job), record)

    def episode_record(self, job: C.Job, state: Mapping[str, Any]) -> dict[str, Any]:
        """Outcome, routing evidence, GPT-6 calls, cost and file-tool access of the scored launch."""
        final = state["final"]
        last = state["tries"][final["try"] - 1]
        try_dir = Path(last["dir"])
        knobs = self.knobs[job.job_id]
        valid = final["class"] == "valid"
        check: dict[str, Any] = {}
        if valid:
            check = C.read_json(
                self.job_dir(job) / f"try{final['try']}.evidence.json"
            ) or self.evidence(job, final["try"])
        usages = [self.try_usage(job.job_id, t) for t in state["tries"]]
        episode = (
            HE.episode_record(
                try_dir,
                self.cfg["pricing"],
                check,
                usage=usages[final["try"] - 1] if self.alone(job.job_id) else None,
            )
            if try_dir.exists()
            else {}
        )
        outcome = C.read_outcome(try_dir) if try_dir.exists() else {}
        memory = (
            outcome.get("memory") if isinstance(outcome.get("memory"), Mapping) else {}
        )
        shared = [
            self.layout.root,
            Path(self.cfg["paths"].get("runs_dir") or self.layout.root.parent),
            Path(self.cfg["paths"]["cloud_corpus"]),
            *(Path(v) for v in self.cfg["final_test"]["products"].values() if v),
        ]
        return {
            "job_id": job.job_id,
            "arm": job.meta["arm"],
            "cell": job.meta["cell"],
            "task": job.task,
            "split": job.split,
            "seed": job.seed,
            "rep": job.meta["rep"],
            "arm_position": job.meta["arm_position"],
            "final": dict(final),
            "class": final["class"],
            "valid": valid,
            "success": bool(valid and final.get("success")),
            "evidence_ok": bool(check.get("ok")) if valid else None,
            "knobs": knobs,
            "episode": episode,
            "usage": usages[final["try"] - 1],
            "usd": round(float((usages[final["try"] - 1] or {}).get("usd") or 0.0), 6),
            "usd_all_tries": round(
                sum(float((u or {}).get("usd") or 0.0) for u in usages), 6
            ),
            "error_runs": error_runs(try_dir) if try_dir.exists() else None,
            "tries": [
                {
                    "try": t["n"],
                    "class": t.get("class"),
                    "cause": t.get("cause"),
                    "endpoint": t.get("endpoint"),
                    "termination_reason": (t.get("outcome") or {}).get(
                        "termination_reason"
                    ),
                }
                for t in state["tries"]
            ],
            "memory_audit": {
                k: memory.get(k)
                for k in ("audit_status", "selected_files", "read_files")
            },
            "file_tool_audit": HE.file_tool_audit(
                try_dir,
                own=self.job_dir(job),
                memory=Path(knobs["memory_dir"]),
                code_dir=self.code_dir,
                shared=shared,
            )
            if try_dir.exists()
            else None,
            "final_try_dir": str(try_dir),
            "recorded_at": C.now_iso(),
        }


class FinalScheduler(C.Scheduler):
    """``common.Scheduler`` with the routing check of stage 2.

    A running launch's config row is checked as soon as it appears (a wrongly
    configured launch is stopped at once); every finished launch the runner did
    not stop is checked in full. A mismatch makes the launch an infrastructure
    error and writes STOP (``final_test.evidence_policy = "stop"``), so it is
    relaunched, after the cause is fixed and STOP removed, and never scored.
    """

    stage: FinalTest

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._config_checked: set[str] = set()
        self._usage_priced = 0.0

    def save(self, state: dict[str, Any]) -> None:
        """Save a job's state; finished GPT-6-alone launches keep their usage priced as the senior's.

        ``common.Scheduler.finish_try`` records ``common.episode_usage`` (the
        worker's price for a launch without an escalation log) and saves before it
        calls ``on_final``, so the state, the record and the cost guard all get the
        senior's price.
        """
        if self.stage.alone(state["job_id"]):
            for entry in state["tries"]:
                if entry.get("status") == "done" and not is_alone_usage(
                    entry.get("usage")
                ):
                    usage = self.stage.try_usage(state["job_id"], entry)
                    if usage is not None:
                        entry["usage"] = usage
        super().save(state)

    def _price_running_alone(self) -> None:
        """Give running GPT-6-alone launches a usage snapshot priced as the senior's.

        The cost guard (``common.total_cost``) prices a running launch from its
        files at the worker's price unless the launch carries a usage, which it then
        takes as it is; ``finish_try`` replaces the snapshot when the launch ends.
        """
        if time.time() - self._usage_priced < float(self.sched["cost_check_s"]):
            return
        self._usage_priced = time.time()
        for job_id in self.jobs:
            if not self.stage.alone(job_id):
                continue
            state = self.state(job_id)
            last = state["tries"][-1] if state["tries"] else None
            if not last or last.get("status") != "running":
                continue
            if Path(last["dir"]).exists():
                last["usage"] = {
                    **senior_alone_usage(Path(last["dir"]), self.cfg["pricing"]),
                    "partial": True,
                }
                self.save(state)

    def _mismatch(
        self, job_id: str, last: Mapping[str, Any], problems: list[str], when: str
    ) -> None:
        self.log(
            {
                "event": "evidence_mismatch",
                "job": job_id,
                "try": last.get("n"),
                "when": when,
                "problems": problems[:5],
            }
        )
        if self.cfg["final_test"]["evidence_policy"] != "stop":
            return
        marker = C.infra_marker(last["dir"])
        if not marker.exists():
            C.atomic_write_text(
                marker, f"evidence_mismatch ({when}): {'; '.join(problems[:3])[:600]}\n"
            )
        C.write_stop(
            self.layout,
            "escalation evidence mismatch",
            job=job_id,
            launch=last.get("n"),
            problems=problems[:5],
        )

    def poll(self) -> None:
        super().poll()
        for job_id, last in self.running(self.all_states()):
            if last["dir"] in self._config_checked or last.get("status") != "running":
                continue
            try:
                problems = self.stage.config_row_problems(
                    self.jobs[job_id], Path(last["dir"])
                )
            except Exception as exc:  # noqa: BLE001 - checked again when the launch ends
                self.log(
                    {
                        "event": "config_row_unreadable",
                        "job": job_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if problems is None:
                continue
            self._config_checked.add(last["dir"])
            if problems:
                self._mismatch(job_id, last, problems, "config row of a running launch")
                if (
                    self.cfg["final_test"]["evidence_policy"] == "stop"
                    and last.get("pid")
                    and C.pid_alive(int(last["pid"]))
                ):
                    C.kill_group(
                        int(last["pid"]), grace_s=float(self.sched["kill_grace_s"])
                    )
        self._price_running_alone()

    def finish_try(
        self, job_id: str, state: dict[str, Any], exit_code: int | None
    ) -> None:
        last = state["tries"][-1]
        try_dir = Path(last["dir"])
        if try_dir.exists() and not C.killed_marker(try_dir).exists():
            key = f"{job_id}#{last['n']}"
            try:  # noise frames first: the base class goes on from this count
                _, self.render_checked[key] = C.check_render(
                    try_dir, self.render_checked.get(key, 0)
                )
            except ImportError:
                pass
            try:
                check = self.stage.evidence(self.jobs[job_id], last["n"])
            except Exception as exc:  # noqa: BLE001 - an unreadable log is a mismatch, not a crash
                check = {
                    "ok": False,
                    "problems": [f"evidence check failed: {type(exc).__name__}: {exc}"],
                    "counts": {},
                }
            last["evidence"] = {"ok": check["ok"], "problems": check["problems"][:20]}
            if not check["ok"]:
                self._mismatch(job_id, last, check["problems"], "finished launch")
        super().finish_try(job_id, state, exit_code)


# ---------------------------------------------------------------------------
# reading a run
# ---------------------------------------------------------------------------


def make_manifest(
    cfg: Mapping[str, Any],
    run_id: str,
    tasks: list[tuple[str, str]],
    jobs: list[C.Job],
    inputs: Mapping[str, Any],
    knobs: Mapping[str, Any],
) -> dict[str, Any]:
    code = C.code_identity(cfg["paths"]["code_dir"])
    ft = cfg["final_test"]
    return {
        "schema": "rc-ecc-run/1",
        "stage": STAGE,
        "run_id": run_id,
        "created_at": C.now_iso(),
        "config_path": cfg["_path"],
        "config_sha256": C.config_hash(cfg),
        "experiment": experiment(cfg),
        "experiment_sha256": C.canonical_hash(experiment(cfg)),
        "knobs_sha256": C.canonical_hash(knobs),
        "inputs_tree_sha256": inputs.get("tree_sha256"),
        "code": code,
        "split": SPLIT,
        "run_arms": ft["run_arms"],
        "arms": {a: ft["arms"][a] for a in ft["run_arms"]},
        "seeds": ft["seeds"],
        "repeats": ft["repeats"],
        "timeout_s_by_split": timeouts(cfg),
        "tasks": [{"split": s, "task": t} for s, t in tasks],
        "jobs": [
            {
                "job_id": j.job_id,
                "arm": j.meta["arm"],
                "task": j.task,
                "split": j.split,
                "seed": j.seed,
                "rep": j.meta["rep"],
                "order": j.order,
            }
            for j in jobs
        ],
        "launcher": cell_argv(cfg),
        "endpoints": cfg["endpoints"],
        "scheduler": cfg["scheduler"],
        "history": [
            {
                "at": C.now_iso(),
                "event": "created",
                "code_tree_sha256": code["tree_sha256"],
            }
        ],
    }


def run_layout(cfg: Mapping[str, Any], args: argparse.Namespace) -> C.RunLayout:
    return S1.run_layout(cfg, args)


def load_run(
    cfg: Mapping[str, Any], args: argparse.Namespace
) -> tuple[C.RunLayout, dict[str, Any], FinalTest, dict[str, Any]]:
    """An existing run: its manifest, inputs and stage.

    The run keeps the final_test values it was created with (the manifest's
    ``experiment``); the config supplies the machine values, guards and endpoints.
    """
    layout = run_layout(cfg, args)
    manifest = C.read_json(layout.manifest)
    if manifest is None:
        raise SystemExit(f"no run at {layout.root}")
    if manifest.get("stage") != STAGE:
        raise SystemExit(f"{layout.root} is a {manifest.get('stage')} run, not {STAGE}")
    inputs = C.read_json(inputs_record(layout))
    if inputs is None:
        raise SystemExit(f"{layout.root}: inputs.json is missing")
    run_cfg = copy.deepcopy(dict(cfg))
    for key in FT_EXPERIMENT_KEYS:
        run_cfg["final_test"][key] = copy.deepcopy(
            manifest["experiment"]["final_test"][key]
        )
    run_cfg = stage_cfg(run_cfg)
    jobs = build_jobs(run_cfg, run_cfg["final_test"]["run_arms"], run_tasks(run_cfg))
    if [j.job_id for j in jobs] != [j["job_id"] for j in manifest["jobs"]]:
        raise SystemExit(
            "the manifest's jobs cannot be rebuilt from its arms, seeds and tasks"
        )
    return layout, manifest, FinalTest(run_cfg, layout, jobs, inputs), run_cfg


def job_states(layout: C.RunLayout, stage: FinalTest) -> dict[str, dict[str, Any]]:
    return {
        job.job_id: C.read_json(layout.job_state(job.job_id))
        or {
            "job_id": job.job_id,
            "task": job.task,
            "split": job.split,
            "seed": job.seed,
            "tries": [],
            "final": None,
        }
        for job in stage.jobs()
    }


def episode_records(layout: C.RunLayout, stage: FinalTest) -> list[dict[str, Any]]:
    """Every final job's record (written by ``on_final``; rebuilt when missing)."""
    out = []
    states = job_states(layout, stage)
    for job in stage.jobs():
        state = states[job.job_id]
        if state.get("final"):
            out.append(
                C.read_json(stage.record_path(job)) or stage.episode_record(job, state)
            )
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided sign test; with discordant cell counts it is the exact McNemar test."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def task_weighted(outcomes: Mapping[tuple[str, str], list[bool]]) -> dict[str, Any]:
    """Target50 rates (%): per split the mean of its tasks' rates; overall the mean over all tasks."""
    per_task = {key: sum(v) / len(v) for key, v in outcomes.items() if v}
    out: dict[str, Any] = {}
    for split in C.SPLITS:
        rates = [r for (s, _), r in per_task.items() if s == split]
        out[split] = 100 * statistics.fmean(rates) if rates else None
        out[f"{split}_tasks"] = len(rates)
    out["overall"] = 100 * statistics.fmean(per_task.values()) if per_task else None
    out["tasks"] = len(per_task)
    return out


def sign_flip_test(
    diffs: list[float], *, permutations: int, seed: int
) -> dict[str, Any]:
    """Two-sided task-clustered permutation test of mean(d_t) = 0, flipping each task's sign.

    ``d_t`` is a task's mean paired difference (arm A minus arm B over its cells),
    so swapping the arms of all of a task's cells flips its sign. Exact over all
    sign patterns when at most 16 tasks differ; otherwise Monte Carlo with
    ``permutations`` draws and p = (1 + hits) / (1 + draws).
    """
    if not diffs:
        return {"stat": None, "p": None, "method": "none", "tasks_differing": 0}
    stat = statistics.fmean(diffs)
    nonzero = [d for d in diffs if abs(d) > 1e-12]
    m, total = len(nonzero), len(diffs)
    target = abs(stat) - 1e-12
    if m == 0:
        return {"stat": stat, "p": 1.0, "method": "exact", "tasks_differing": 0}
    if m <= 16:
        hits = sum(
            1
            for signs in itertools.product((1, -1), repeat=m)
            if abs(sum(s * d for s, d in zip(signs, nonzero)) / total) >= target
        )
        return {"stat": stat, "p": hits / 2**m, "method": "exact", "tasks_differing": m}
    rng = random.Random(seed)
    hits = sum(
        1
        for _ in range(permutations)
        if abs(sum(d if rng.random() < 0.5 else -d for d in nonzero) / total) >= target
    )
    return {
        "stat": stat,
        "p": (1 + hits) / (1 + permutations),
        "method": f"monte_carlo_{permutations}",
        "tasks_differing": m,
    }


def bootstrap_ci(diffs: list[float], *, draws: int, seed: int) -> list[float] | None:
    """95 % percentile interval of mean(d_t), resampling tasks with replacement."""
    if len(diffs) < 2 or draws <= 0:
        return None
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(draws)
    )
    return [means[int(0.025 * (draws - 1))], means[int(math.ceil(0.975 * (draws - 1)))]]


def scored(record: Mapping[str, Any], include_mismatch: bool = False) -> bool:
    """A record counts when its final launch was valid and its escalation log matched the arm."""
    return bool(record.get("valid")) and (
        include_mismatch or record.get("evidence_ok") is not False
    )


def paired(
    records: list[Mapping[str, Any]],
    a: str,
    b: str,
    *,
    permutations: int,
    bootstrap: int,
    seed: int,
    include_mismatch: bool = False,
) -> dict[str, Any]:
    """Arm ``a`` against arm ``b`` on the cells scored in both."""
    by = {
        (r["arm"], r["task"], r["seed"], r["rep"]): r
        for r in records
        if r["arm"] in (a, b) and scored(r, include_mismatch)
    }
    cells = sorted({k[1:] for k in by if k[0] == a} & {k[1:] for k in by if k[0] == b})
    split_of = {r["task"]: r["split"] for r in records}
    out: dict[str, Any] = {"a": a, "b": b, "cells": len(cells)}
    for scope in (*C.SPLITS, "all"):
        cs = [c for c in cells if scope == "all" or split_of[c[0]] == scope]
        win_a = [c for c in cs if by[(a, *c)]["success"] and not by[(b, *c)]["success"]]
        win_b = [c for c in cs if by[(b, *c)]["success"] and not by[(a, *c)]["success"]]
        per_task: dict[str, list[float]] = {}
        for c in cs:
            per_task.setdefault(c[0], []).append(
                float(by[(a, *c)]["success"]) - float(by[(b, *c)]["success"])
            )
        diffs = [statistics.fmean(v) for v in per_task.values()]
        up = sum(1 for d in diffs if d > 1e-12)
        down = sum(1 for d in diffs if d < -1e-12)
        ci = bootstrap_ci(diffs, draws=bootstrap, seed=seed)
        out[scope] = {
            "cells": len(cs),
            "a_success": sum(1 for c in cs if by[(a, *c)]["success"]),
            "b_success": sum(1 for c in cs if by[(b, *c)]["success"]),
            "a_only": len(win_a),
            "b_only": len(win_b),
            "mcnemar_p": binom_two_sided(len(win_a), len(win_b)),
            "tasks": len(diffs),
            "tasks_a_better": up,
            "tasks_b_better": down,
            "sign_test_p": binom_two_sided(up, down),
            "task_weighted_diff_pct": 100 * statistics.fmean(diffs) if diffs else None,
            "permutation": sign_flip_test(diffs, permutations=permutations, seed=seed),
            "bootstrap_ci_pct": [100 * x for x in ci] if ci else None,
        }
    return out


def arm_summary(
    records: list[Mapping[str, Any]],
    arm: str,
    include_mismatch: bool = False,
    invalid_as_failure: bool = False,
) -> dict[str, Any]:
    """One arm's success, GPT-6 calls, error trigger and cost over its scored episodes.

    Rates count the scored episodes only (as the Target50 ablation scorer does);
    with ``invalid_as_failure`` an episode that stayed invalid after its retries
    counts as a failed cell in the rates (a leaderboard-style denominator).
    """
    mine = [r for r in records if r["arm"] == arm]
    valid = [r for r in mine if scored(r, include_mismatch)]
    outcomes: dict[tuple[str, str], list[bool]] = {}
    for r in valid:
        outcomes.setdefault((r["split"], r["task"]), []).append(bool(r["success"]))
    if invalid_as_failure:
        for r in mine:
            if not r.get("valid"):
                outcomes.setdefault((r["split"], r["task"]), []).append(False)
    n = len(valid)
    successes = sum(1 for r in valid if r["success"])
    reasons: dict[str, int] = {}
    senior = worker = errors_turns = errors_episodes = 0
    usd = usd_worker = usd_senior = 0.0
    runs = []
    for r in valid:
        ep = r.get("episode") or {}
        for reason, count in (ep.get("gpt_calls_by_reason") or {}).items():
            reasons[reason] = reasons.get(reason, 0) + int(count or 0)
        senior += int(ep.get("gpt_calls") or 0)
        worker += int((ep.get("qwen") or {}).get("calls") or 0)
        errors_turns += int(ep.get("errors_trigger") or 0)
        errors_episodes += int((ep.get("errors_trigger") or 0) > 0)
        usd += float(r.get("usd") or 0.0)
        usd_worker += float((ep.get("qwen") or {}).get("usd") or 0.0)
        usd_senior += float((ep.get("gpt6") or {}).get("usd") or 0.0)
        if r.get("error_runs"):
            runs.append(r["error_runs"]["max_run"])
    audit_keys = ("cross_reads", "cross_writes", "memory_writes", "code_dir_writes")
    return {
        "arm": arm,
        "final": len(mine),
        "scored": n,
        "successes": successes,
        "invalid": sum(1 for r in mine if not r.get("valid")),
        "evidence_mismatch": sum(
            1 for r in mine if r.get("valid") and r.get("evidence_ok") is False
        ),
        "task_weighted_pct": task_weighted(outcomes),
        "gpt6_calls_per_episode": senior / n if n else None,
        "gpt6_calls_per_episode_by_reason": {
            k: v / n for k, v in sorted(reasons.items()) if v
        }
        if n
        else {},
        "qwen_calls_per_episode": worker / n if n else None,
        "errors_trigger": {
            "turns": errors_turns,
            "episodes": errors_episodes,
            "episodes_with_errored_run_ge": {
                str(k): sum(1 for x in runs if x >= k) for k in (2, 3, 4)
            },
            "episodes_with_transcript": len(runs),
        },
        "usd_per_episode": usd / n if n else None,
        "usd_per_success": usd / successes if successes else None,
        "usd_scored": usd,
        "usd_qwen": usd_worker,
        "usd_gpt6": usd_senior,
        "usd_all_launches": sum(float(r.get("usd_all_tries") or 0.0) for r in mine),
        "flagged_file_access": sorted(
            r["job_id"]
            for r in mine
            if any((r.get("file_tool_audit") or {}).get(k) for k in audit_keys)
        ),
    }


def score(
    records: list[Mapping[str, Any]],
    arms: list[str],
    pairs: list[tuple[str, str]],
    *,
    permutations: int,
    bootstrap: int,
    seed: int,
    include_mismatch: bool = False,
    invalid_as_failure: bool = False,
) -> dict[str, Any]:
    return {
        "invalid_as_failure": invalid_as_failure,
        "arms": {
            arm: arm_summary(records, arm, include_mismatch, invalid_as_failure)
            for arm in arms
        },
        "pairs": [
            paired(
                records,
                a,
                b,
                permutations=permutations,
                bootstrap=bootstrap,
                seed=seed,
                include_mismatch=include_mismatch,
            )
            for a, b in pairs
        ],
    }


def _num(value: float | None, fmt: str = "{:.1f}") -> str:
    return "-" if value is None else fmt.format(value)


def format_score(doc: Mapping[str, Any]) -> str:
    lines = [
        "per arm: task-weighted success % (Target50: a split = mean of its tasks' rates; overall = mean over tasks; "
        + (
            "invalid episodes count as failures)"
            if doc.get("invalid_as_failure")
            else "scored episodes only)"
        ),
        f"  {'arm':<18} {'scored':>6} {'succ':>5} {'atomic':>7} {'seen':>6} {'unseen':>7} {'overall':>8} | "
        f"{'GPT-6/ep':>8} {'Qwen/ep':>7} | {'$/ep':>6} {'$/succ':>7} {'$ all':>8} | invalid mismatch",
    ]
    for arm, s in doc["arms"].items():
        tw = s["task_weighted_pct"]
        lines.append(
            f"  {arm:<18} {s['scored']:>6} {s['successes']:>5} {_num(tw['atomic']):>7} "
            f"{_num(tw['composite_seen']):>6} {_num(tw['composite_unseen']):>7} {_num(tw['overall']):>8} | "
            f"{_num(s['gpt6_calls_per_episode'], '{:.2f}'):>8} {_num(s['qwen_calls_per_episode']):>7} | "
            f"{_num(s['usd_per_episode'], '{:.2f}'):>6} {_num(s['usd_per_success'], '{:.2f}'):>7} "
            f"{s['usd_all_launches']:>8.2f} | {s['invalid']:>7} {s['evidence_mismatch']:>8}"
        )
    lines.append("GPT-6 calls per episode by reason; the N-error trigger")
    for arm, s in doc["arms"].items():
        reasons = (
            ", ".join(
                f"{k} {v:.2f}" for k, v in s["gpt6_calls_per_episode_by_reason"].items()
            )
            or "none"
        )
        e = s["errors_trigger"]
        ge = e["episodes_with_errored_run_ge"]
        lines.append(
            f"  {arm:<18} {reasons}; errors turns {e['turns']} in {e['episodes']} episodes; "
            f"episodes with >= 2/3/4 errored tool results in a row: {ge['2']}/{ge['3']}/{ge['4']} "
            f"of {e['episodes_with_transcript']} with a transcript"
        )
        if s["flagged_file_access"]:
            lines.append(
                f"    file-tool access outside the task: {', '.join(s['flagged_file_access'][:8])}"
            )
    for p in doc["pairs"]:
        lines.append(f"{p['a']} vs {p['b']}: {p['cells']} cells scored in both")
        for scope in (*C.SPLITS, "all"):
            q = p[scope]
            if not q["cells"]:
                continue
            perm, ci = q["permutation"], q["bootstrap_ci_pct"]
            lines.append(
                f"  {scope:<17} {q['cells']:>4} cells: solved {q['a_success']} vs {q['b_success']}; only "
                f"{p['a']} {q['a_only']}, only {p['b']} {q['b_only']}, McNemar p = {q['mcnemar_p']:.4f}; tasks "
                f"better {q['tasks_a_better']}/{q['tasks_b_better']}, sign p = {q['sign_test_p']:.4f}; "
                f"task-weighted diff {_num(q['task_weighted_diff_pct'])} pts, permutation p = "
                f"{_num(perm['p'], '{:.4f}')} ({perm['method']})"
                + (f", 95% CI [{ci[0]:.1f}, {ci[1]:.1f}]" if ci else "")
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def _probe_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Inputs as the dry run shows them (snapshot paths are placeholders)."""
    return {
        "memory": {
            label: {**info, "snapshot": f"<run>/inputs/memory/{label}"}
            for label, info in inputs["memory"].items()
        },
        "rules": dict(inputs["rules"]),
        "corrections": {
            t: {**c, "snapshot": f"<run>/inputs/corrections/{t}.md"}
            for t, c in inputs["corrections"].items()
        },
    }


def print_plan(cfg: Mapping[str, Any], args: argparse.Namespace) -> int:
    """The dry run: every input, arm and job --go would use; creates and starts nothing."""
    C.require(cfg, "paths.code_dir", "paths.cloud_corpus", "paths.runs_dir")
    run_cfg = stage_cfg(apply_cli(cfg, args))
    ft = run_cfg["final_test"]
    tasks, arms = run_tasks(run_cfg), ft["run_arms"]
    print(
        f"RC-ECC final test plan (dry run; add --go to launch)\n  config   {run_cfg['_path']}"
    )
    print(
        f"  config sha256 {C.config_hash(run_cfg)[:16]}  experiment sha256 {C.canonical_hash(experiment(run_cfg))[:16]}"
    )
    code = C.code_identity(run_cfg["paths"]["code_dir"])
    print(
        f"  code     {run_cfg['paths']['code_dir']}  commit {code['commit']} ({code['commit_source']})  "
        f"tree {code['tree_sha256'][:16]} ({code['files']} files)"
    )
    inputs = read_inputs(run_cfg, arms, tasks)
    for label, info in inputs["memory"].items():
        print(f"  memory {label}: {info['source']}  tree {info['tree_sha256'][:16]}")
    if inputs["rules"]:
        rungs: dict[str, int] = {}
        for rule in inputs["rules"].values():
            rungs[str(rule["rung"])] = rungs.get(str(rule["rung"]), 0) + 1
        print(
            f"  s2 rules {inputs['sources']['s2']['folder']} ({inputs['sources']['s2']['kind']}): "
            f"{len(inputs['rules'])} tasks; per rung "
            + ", ".join(f"{k}: {v}" for k, v in sorted(rungs.items()))
        )
    if inputs["corrections"]:
        chars = [c["chars"] for c in inputs["corrections"].values()]
        print(
            f"  s3 corrections {inputs['sources']['s3']['folder']} ({inputs['sources']['s3']['kind']}): "
            f"{len(chars)} tasks, {chars.count(0)} empty, "
            f"chars median {statistics.median(chars):.0f}, max {max(chars)}"
        )
    for problem in inputs["chain_problems"]:
        print(f"  WARNING  {problem} (accepted: allow_product_mismatch)")
    jobs = build_jobs(run_cfg, arms, tasks)
    by_split: dict[str, int] = {}
    for split, _ in tasks:
        by_split[split] = by_split.get(split, 0) + 1
    print(
        f"  cells    {len(tasks)} tasks ({', '.join(f'{k} {v}' for k, v in by_split.items())}) x seeds {ft['seeds']} x "
        f"{ft['repeats']} repeat(s) = {len(jobs) // len(arms)} cells; {len(arms)} arms; {len(jobs)} episodes; "
        f"order {ft['cell_order']} (seed {ft['order_seed']})"
    )
    print(f"  timeouts {timeouts(run_cfg)} s, {run_cfg['episode']['max_turns']} turns")
    for arm in arms:
        print(f"  arm {arm}: {json.dumps(ft['arms'][arm], sort_keys=True)}")
    first = {
        a: sum(1 for j in jobs if j.meta["arm"] == a and j.meta["arm_position"] == 0)
        for a in arms
    }
    print(f"  first in its cell: {first}")
    sched = run_cfg["scheduler"]
    ceiling = C.launch_ceiling(
        sched, len(jobs), len(jobs), ft["infra_retries"], ft["render_retries"]
    )
    print(
        f"  guards   infra {ft['infra_retries']} / render {ft['render_retries']} retries per episode; launch ceiling "
        f"{ceiling}; infra breaker {sched['infra_breaker']}; cost ceiling "
        f"{'$%.0f' % sched['cost_ceiling_usd'] if sched['cost_ceiling_usd'] else 'off'}; evidence {ft['evidence_policy']}"
    )
    script = launcher_script(run_cfg)
    print(
        f"  launcher {' '.join(cell_argv(run_cfg))} ({'exists' if script.is_file() else 'MISSING'})"
    )
    alone_arms = [a for a in arms if ft["arms"][a]["planner"] in ALONE_PLANNERS]
    missing = unsupported_alone_arms(run_cfg, arms)
    if missing:
        print(
            f"  WARNING  arms {missing} run one planner alone, but the launcher lacks the "
            f"{QWEN_SWITCH} / {ALONE_SWITCH} switch: --go refuses"
        )
    if len(alone_arms) < len(arms) and ft["require_config_row"]:
        try:
            supported = config_row_supported()
        except Exception as exc:  # noqa: BLE001 - dry run reports and goes on
            supported = f"not checked ({type(exc).__name__})"
        print(f"  config row required; this code writes it: {supported}")
    if args.run_id or getattr(args, "run_root", None):
        layout = run_layout(run_cfg, args)
        print(
            f"  run root {layout.root} ({'exists: --go resumes it' if layout.manifest.exists() else 'new'})"
        )
    else:
        layout = C.RunLayout(Path(run_cfg["paths"]["runs_dir"]) / "<run-id>")
    try:
        usage = C.gpu_usage(run_cfg)
    except Exception as exc:  # noqa: BLE001 - dry run reports and goes on
        usage = None
        print(f"  GPUs     query failed: {exc}")
    procs = None
    if ft["share_own_gpus"] and usage is not None:
        try:
            procs = gpu_processes(run_cfg)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            print(f"  GPUs     process query failed: {exc}")
    servers = run_cfg["servers"]
    for ep in C.endpoints_from_config(run_cfg):
        free = (
            C.gpu_free(run_cfg, ep.gpu, usage)[1]
            if usage is not None
            else "not checked"
        )
        if procs is not None and not C.gpu_free(run_cfg, ep.gpu, usage)[0]:
            need = int(servers["fit_margin_mb"]) + sum(
                int(servers["server_mb"]) + e.slots * int(servers["cell_mb"])
                for e in C.endpoints_from_config(run_cfg)
                if e.gpu == ep.gpu
            )
            free += (
                "; share_own_gpus: "
                + shared_gpu_room(run_cfg, ep.gpu, need, usage, procs)[1]
            )
        print(f"  endpoint {ep.name}: port {ep.port}, {ep.slots} slots; {free}")
    cred = Path(run_cfg["credentials"].get("file") or "")
    if cred.name:
        print(
            f"  credentials file {cred} ({'exists' if cred.is_file() else 'MISSING'}; never read by the runner)"
        )
    stage = FinalTest(run_cfg, layout, jobs, _probe_inputs(inputs))
    print("  first episodes:")
    for job in jobs[: max(2, len(arms))]:
        k = stage.knobs[job.job_id]
        corr = k["corrections"]
        if k["planner"] == "gpt6":
            print(
                f"    {job.job_id}: GPT-6 alone ({k['model']}, effort {k['reasoning_effort']}, at most "
                f"{k['max_images']} images per request, reasoning ids {'sent' if k['reasoning_ids'] else 'off'}), "
                f"memory {k['memory']}, timeout {k['timeout_s']} s"
            )
            continue
        print(
            f"    {job.job_id}: senior {'on' if k['escalation'] else 'off'}, plan {k['plan']}, K {k['checkin']}, "
            f"N {k['errors']} (rung {k['rung']}, {k['rule_source']}), memory {k['memory']}, corrections "
            f"{'%d chars' % corr['chars'] if corr else 'none'}, timeout {k['timeout_s']} s"
        )
    eps = C.endpoints_from_config(run_cfg)
    argv, env = stage.launch(
        jobs[0], 1, eps[0] if eps else C.Endpoint("gpuN", -1, 0, 0)
    )
    print(f"  first launch: {' '.join(argv)}")
    for key in sorted(k for k in env if k.startswith(("ECC_RC_", "RC_", "VLA_"))):
        print(f"    {key}={env[key]}")
    return 0


def cmd_run(cfg: Mapping[str, Any], args: argparse.Namespace) -> int:
    if not args.go:
        return print_plan(cfg, args)
    C.require(
        cfg,
        "paths.code_dir",
        "paths.venv",
        "paths.cloud_corpus",
        "paths.runs_dir",
        "paths.assets",
        "paths.hf_home",
        "paths.rldx_checkpoint",
        "paths.llvmpipe_index_script",
        "credentials.file",
        "credentials.worker_key_var",
        "credentials.senior_base_url_var",
        "credentials.senior_key_var",
        "worker.base_url",
        "gpus.allowed",
    )
    if not Path(cfg["credentials"]["file"]).is_file():
        raise SystemExit("credentials file missing")
    try:
        import numpy  # noqa: F401
        import PIL  # noqa: F401
    except ImportError:
        raise SystemExit(
            "run the runner with the venv's Python: the render-noise check needs numpy and Pillow"
        )
    layout = run_layout(cfg, args)
    resume = layout.manifest.exists()
    if resume:
        layout, manifest, stage, run_cfg = load_run(cfg, args)
    else:
        run_cfg = stage_cfg(apply_cli(cfg, args))
    ft = run_cfg["final_test"]
    if not launcher_script(run_cfg).is_file():
        raise SystemExit(
            f"the cell launcher {launcher_script(run_cfg)} does not exist (final_test.launcher)"
        )
    alone_arms = [
        a for a in ft["run_arms"] if ft["arms"][a]["planner"] in ALONE_PLANNERS
    ]
    missing = unsupported_alone_arms(run_cfg, ft["run_arms"])
    if missing:
        raise SystemExit(
            f"arms {missing} run one planner alone, but {launcher_script(run_cfg)} lacks the {QWEN_SWITCH} / "
            f"{ALONE_SWITCH} switch (it would run them as hybrid episodes)"
        )
    if (
        len(alone_arms) < len(ft["run_arms"])
        and ft["require_config_row"]
        and not config_row_supported()
    ):
        raise SystemExit(
            "this code's escalation wrapper writes no config row, so N cannot be checked; use a code dir "
            "with it or set final_test.require_config_row = false"
        )
    lock = C.acquire_lock(layout)  # noqa: F841 - held for the process lifetime
    if resume:
        code = C.code_identity(run_cfg["paths"]["code_dir"])
        problems = []
        if C.canonical_hash(experiment(run_cfg)) != manifest["experiment_sha256"]:
            problems.append(
                "worker/senior/episode settings differ from the run's manifest"
            )
        asked = experiment(stage_cfg(apply_cli(cfg, args)))["final_test"]
        changed = sorted(
            k
            for k in FT_EXPERIMENT_KEYS
            if asked[k] != manifest["experiment"]["final_test"][k]
        )
        if changed:
            problems.append(
                f"final_test {', '.join(changed)} differ from the run's manifest (a new "
                "run needs a new run id; resume with the config and --arms/--tasks/"
                "--seeds/--repeats/--limit it was created with)"
            )
        if (
            code["tree_sha256"] != manifest["code"]["tree_sha256"]
            and not args.allow_code_change
        ):
            problems.append(
                "code differs from the manifest (pass --allow-code-change to accept and record it)"
            )
        if C.canonical_hash(stage.knobs) != manifest["knobs_sha256"]:
            problems.append("the episodes' knobs differ from the manifest")
        problems += verify_snapshot(layout, stage.inputs)
        if problems:
            raise SystemExit("refusing to resume: " + "; ".join(problems))
        if code["tree_sha256"] != manifest["code"]["tree_sha256"]:
            manifest["history"].append(
                {"at": C.now_iso(), "event": "code_changed", "code": code}
            )
        manifest["history"].append(
            {"at": C.now_iso(), "event": "resumed", "pid": os.getpid()}
        )
        C.atomic_write_json(layout.manifest, manifest)
    else:
        tasks, arms = run_tasks(run_cfg), ft["run_arms"]
        env = {
            **S1.base_env(run_cfg),
            "PYTHONPATH": run_cfg["paths"]["code_dir"],
            "RPENT_REPO_ROOT": run_cfg["paths"]["code_dir"],
        }
        origin = C.import_origin(
            C.python_for(run_cfg), run_cfg["paths"]["code_dir"], env
        )
        if not origin.get("ok"):
            raise SystemExit(
                f"episode processes would not import rpent/robots from the code dir: {origin}"
            )
        mine = S1.runner_origin(run_cfg["paths"]["code_dir"])
        if not mine["ok"]:
            raise SystemExit(f"the runner does not import from the code dir: {mine}")
        inputs = snapshot_inputs(layout, read_inputs(run_cfg, arms, tasks), tasks)
        stage = FinalTest(run_cfg, layout, build_jobs(run_cfg, arms, tasks), inputs)
        manifest = make_manifest(
            run_cfg, layout.root.name, tasks, stage.jobs(), inputs, stage.knobs
        )
        manifest.update(import_origin=origin, runner_import_origin=mine)
        C.atomic_write_json(
            layout.root / "config.snapshot.json",
            {k: v for k, v in run_cfg.items() if k != "_path"},
        )
        C.atomic_write_json(layout.manifest, manifest)
    servers = FinalServerPool(
        run_cfg,
        layout,
        S1.base_env(run_cfg),
        lambda row: C.append_jsonl(layout.progress, row),
    )
    scheduler = FinalScheduler(
        run_cfg,
        layout,
        stage,
        servers,
        infra_retries=ft["infra_retries"],
        render_retries=ft["render_retries"],
        max_slots=args.max_slots,
    )
    C.append_jsonl(
        layout.progress,
        {
            "event": "start",
            "pid": os.getpid(),
            "jobs": len(stage.jobs()),
            "max_slots": scheduler.max_slots,
        },
    )
    if args.concurrency:
        C.atomic_write_text(layout.concurrency, f"{args.concurrency}\n")
    for job in stage.jobs():  # records an earlier runner did not write
        state = scheduler.state(job.job_id)
        if state["final"]:
            stage.on_final(job, state)
    states = scheduler.all_states()
    pending = not all(s["final"] for s in states.values())
    up = servers.ensure(force_gpus=args.force_gpu or ()) if pending else []
    print(
        f"endpoints up: {[(ep.name, ep.url, ep.slots) for ep in up]}; "
        f"down: {[(ep.name, ep.note) for ep in servers.endpoints if not ep.up]}",
        flush=True,
    )
    result = "no_endpoint" if pending else "done"
    # With share_own_gpus a server refused for lack of room is tried again every
    # servers.retry_busy_s (at most servers.max_refusals times) inside the run loop,
    # so the runner waits for room instead of exiting with no endpoint up.
    waiting = bool(
        ft["share_own_gpus"] and pending and not up and servers.restart_after
    )
    if waiting:
        print(
            f"no endpoint up yet: waiting for GPU room (a refused server is tried again every "
            f"{run_cfg['servers']['retry_busy_s']} s, at most {run_cfg['servers']['max_refusals']} times)",
            flush=True,
        )
    stage.cells_dir.mkdir(parents=True, exist_ok=True)
    try:
        if up or scheduler.running(states) or waiting:
            # Episodes cannot list cells/ while they run, so another arm's folder is not discoverable.
            stage.cells_dir.chmod(0o311)
            result = scheduler.run()
    finally:
        still = scheduler.running(scheduler.all_states())
        if not still:
            stage.cells_dir.chmod(0o755)
            if run_cfg["servers"]["stop_on_exit"]:
                servers.stop_all()
        C.append_jsonl(
            layout.progress, {"event": "end", "result": result, "running": len(still)}
        )
    print(f"runner finished: {result}", flush=True)
    return 0 if result in ("done", "stopped") else 1


def cmd_status(cfg: Mapping[str, Any], args: argparse.Namespace) -> int:
    """Progress bar, per-split and per-arm overview, endpoints, guards, evidence and cost."""
    layout, manifest, stage, run_cfg = load_run(cfg, args)
    states = job_states(layout, stage)
    for state in states.values():
        if state["tries"] and state["tries"][-1].get("status") in (
            "launching",
            "running",
        ):
            pid = state["tries"][-1].get("pid")
            state["tries"][-1]["alive"] = bool(pid and C.pid_alive(pid))
    total, final = len(states), sum(1 for s in states.values() if s["final"])
    print(
        f"RC-ECC final test {manifest['run_id']}  {C.progress_bar(final, total)} episodes final"
    )
    rows = C.status_rows(states)
    print(
        f"  {'split':<17} {'final':>9} {'running':>8} {'queued':>7} {'solved':>7} {'retries':>8} {'exhausted':>10}"
    )
    for split in [s for s in C.SPLITS if s in rows]:
        r = rows[split]
        print(
            f"  {split:<17} {r['final']:>4}/{r['total']:<4} {r['running']:>8} {r['queued']:>7} {r['solved']:>7} "
            f"{r['infra_tries'] + r['render_tries']:>8} {r['infra_exhausted']:>10}"
        )
    mismatches: dict[str, Any] = {}
    try:
        for line in layout.progress.read_text().splitlines():
            row = json.loads(line)
            if row.get("event") == "evidence_mismatch":
                mismatches[row["job"]] = row
    except (OSError, ValueError):
        pass
    print(f"  {'arm':<17} {'final':>9} {'running':>8} {'solved':>7} {'mismatch':>9}")
    for arm in manifest["run_arms"]:
        mine = {j: s for j, s in states.items() if stage.by_id[j].meta["arm"] == arm}
        print(
            f"  {arm:<17} {sum(1 for s in mine.values() if s['final']):>4}/{len(mine):<4} "
            f"{sum(1 for s in mine.values() if s['tries'] and s['tries'][-1].get('status') == 'running'):>8} "
            f"{sum(1 for s in mine.values() if (s['final'] or {}).get('success')):>7} "
            f"{sum(1 for j in mine if j in mismatches):>9}"
        )
    holder = C.lock_holder(layout)
    stop = C.read_json(layout.stop) if layout.stop.exists() else None
    print(
        f"  runner: {'pid %s' % holder if holder else 'not running'}; stop file: {stop['reason'] if stop else 'absent'}"
    )
    load: dict[str, int] = {}
    for s in states.values():
        if s["tries"] and s["tries"][-1].get("status") == "running":
            load[s["tries"][-1].get("endpoint")] = (
                load.get(s["tries"][-1].get("endpoint"), 0) + 1
            )
    shown = []
    for ep in C.endpoints_from_config(run_cfg):
        info = C.read_json(layout.servers_dir / f"{ep.name}.json", {}) or {}
        alive = bool(
            info.get("pid") and C.pid_alive(info["pid"]) and not info.get("stopped_at")
        )
        shown.append(
            f"{ep.name} {'up' if alive else 'down'} {load.get(ep.name, 0)}/{ep.slots}"
        )
    print("  endpoints: " + (", ".join(shown) or "none configured"))
    ft = run_cfg["final_test"]
    ceiling = C.launch_ceiling(
        run_cfg["scheduler"], total, total, ft["infra_retries"], ft["render_retries"]
    )
    print(f"  launches: {C.counted_launches(states)} of ceiling {ceiling}")
    stale = [
        j
        for j, s in states.items()
        if s["tries"] and s["tries"][-1].get("alive") is False
    ]
    if stale:
        print(
            f"  NOTE: {len(stale)} episode(s) marked running have no live process; the next --go scores them"
        )
    if mismatches:
        print(f"  evidence mismatches: {', '.join(sorted(mismatches)[:6])}")
    print(f"  cost so far (list price, all launches): ${stage.total_cost(states):.2f}")
    return 0


def cmd_stop(cfg: Mapping[str, Any], args: argparse.Namespace) -> int:
    layout, _, stage, run_cfg = load_run(cfg, args)
    C.write_stop(layout, "operator", now=bool(args.now))
    print(f"wrote {layout.stop}: no new launches; the runner drains and exits")
    grace = float(run_cfg["scheduler"]["kill_grace_s"])
    if args.now:
        killed = C.kill_running(layout, [j.job_id for j in stage.jobs()], grace_s=grace)
        print(
            f"stopped {killed} running episode(s); they are relaunched (not counted) on the next --go"
        )
    running = []
    for job in stage.jobs():
        last = ((C.read_json(layout.job_state(job.job_id)) or {}).get("tries") or [{}])[
            -1
        ]
        if last.get("status") == "running" and C.pid_alive(last.get("pid")):
            running.append(job.job_id)
    if C.lock_holder(layout) is None and not running:
        C.ServerPool(
            run_cfg, layout, {}, lambda row: C.append_jsonl(layout.progress, row)
        ).stop_all()
        print("no runner and no running episode: stopped this run's RLDX servers")
    return 0


def cmd_score(cfg: Mapping[str, Any], args: argparse.Namespace) -> int:
    layout, manifest, stage, run_cfg = load_run(cfg, args)
    ft = run_cfg["final_test"]
    arms = manifest["run_arms"]
    pairs = (
        [tuple(p.split(":", 1)) for p in args.pair]
        if args.pair
        else [tuple(p) for p in ft["pairs"]]
    )
    if not pairs and len(arms) == 2:
        pairs = [(arms[0], arms[1])]
    for a, b in pairs:
        if a not in arms or b not in arms or a == b:
            raise SystemExit(f"--pair {a}:{b}: two different run arms of {arms}")
    records = episode_records(layout, stage)
    doc = score(
        records,
        arms,
        pairs,
        permutations=int(ft["permutations"]),
        bootstrap=int(ft["bootstrap"]),
        seed=int(ft["stats_seed"]),
        include_mismatch=args.include_mismatch,
        invalid_as_failure=args.invalid_as_failure,
    )
    doc.update(
        run_id=manifest["run_id"],
        episodes_final=len(records),
        episodes_total=len(stage.jobs()),
    )
    print(
        f"RC-ECC final test {manifest['run_id']}  {C.progress_bar(len(records), len(stage.jobs()))} episodes final"
    )
    print(format_score(doc))
    if args.json:
        C.atomic_write_json(Path(args.json), doc)
        print(f"wrote {args.json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")

    def common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--config", help="TOML/JSON config outside the repo (or ECC_RC_CONFIG)"
        )
        p.add_argument("--run-id", help="run root name under paths.runs_dir")
        p.add_argument("--run-root", help="explicit run root (instead of --run-id)")

    for name in ("plan", "run"):
        p = sub.add_parser(name, help="dry run (default) / launch with --go")
        common_args(p)
        p.add_argument(
            "--arms",
            help="comma list of defined arms (default: final_test.run_arms, else all)",
        )
        p.add_argument(
            "--tasks",
            help="comma list of Target50 tasks (default: final_test.tasks, else all 50)",
        )
        p.add_argument(
            "--seeds", help="comma list of seeds (default: final_test.seeds)"
        )
        p.add_argument(
            "--repeats", type=int, default=0, help="episodes per (task, seed) per arm"
        )
        p.add_argument(
            "--limit", type=int, default=0, help="first N tasks only (smoke)"
        )
        p.add_argument(
            "--max-slots",
            type=int,
            default=0,
            help="lower the concurrent-episode ceiling",
        )
        p.add_argument(
            "--force-gpu",
            type=int,
            action="append",
            help="start a server on this GPU without the free check",
        )
        p.add_argument(
            "--concurrency",
            type=int,
            default=0,
            help="write <run>/concurrency before launching",
        )
        p.add_argument(
            "--go", action="store_true", help="really launch (otherwise print the plan)"
        )
        p.add_argument(
            "--allow-code-change",
            action="store_true",
            help="resume although the code tree changed",
        )
    p = sub.add_parser("status", help="progress bar, per-split and per-arm overview")
    common_args(p)
    p = sub.add_parser(
        "stop", help="write the stop file (drain); --now also stops running episodes"
    )
    common_args(p)
    p.add_argument("--now", action="store_true")
    p = sub.add_parser(
        "score", help="per-arm success, GPT-6 calls and cost; paired tests"
    )
    common_args(p)
    p.add_argument(
        "--pair",
        action="append",
        help="A:B, repeatable (default: final_test.pairs, or the two run arms)",
    )
    p.add_argument("--json", help="also write the score document to this file")
    p.add_argument(
        "--include-mismatch",
        action="store_true",
        help="also score episodes whose escalation log did not match their arm (evidence_policy = warn)",
    )
    p.add_argument(
        "--invalid-as-failure",
        action="store_true",
        help="count episodes still invalid after their retries as failed cells in the rates (pairs stay valid-only)",
    )
    sub.add_parser("example-config", help="print a [final_test] config section")
    args = ap.parse_args(argv)
    if args.cmd is None:
        ap.print_help()
        return 2
    if args.cmd == "example-config":
        print(EXAMPLE_CONFIG, end="")
        return 0
    if args.cmd == "plan":
        args.cmd, args.go = "run", False
    cfg = load_config(C.config_path(args.config))
    return {"run": cmd_run, "status": cmd_status, "stop": cmd_stop, "score": cmd_score}[
        args.cmd
    ](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
