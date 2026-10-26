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

"""Shared pieces of the RC-ECC RoboCasa runners (stage 1 now, stages 2-3 later).

- configuration (a TOML or JSON file kept outside the repository) and its hashes;
- run layout, manifests, atomic writes and tree hashes;
- GPU checks and the shared RLDX servers (one per GPU);
- an episode's outcome, read the way the Target50 batch runner reads it;
- token usage and list-price cost from the episode logs;
- a resumable slot scheduler with retries, a stop file and runaway guards;
- the status overview (progress bar and per-split table).

Nothing here reads the credentials file; only its path is handed to the cell
launcher, which sources it.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

SPLITS = ("atomic", "composite_seen", "composite_unseen")
TARGET50_MANIFEST = "robots/robocasa/eval/target50_v2.json"
#: Written into an episode folder whose camera frames are per-pixel noise.
RENDER_MARK = "render_corrupt"
#: Rendered kitchens have a mean absolute horizontal gradient below ~9, noise above 80.
RENDER_GRADIENT_LIMIT = 40.0
STOP_FILE = "STOP"

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """Local time with offset, seconds precision."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write_text(path: Path | str, text: str) -> Path:
    """Write ``text`` through a temporary file and ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def atomic_write_json(path: Path | str, value: Any) -> Path:
    """Write one JSON document atomically."""
    return atomic_write_text(
        path, json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n"
    )


def read_json(path: Path | str, default: Any = None) -> Any:
    """JSON content of ``path``, or ``default`` when missing or unparsable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def append_jsonl(path: Path | str, row: Mapping[str, Any]) -> None:
    """Append one JSON line (flushed) to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": now_iso(), **row}, default=str) + "\n")
        handle.flush()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_files(root: Path | str, *, exclude: Iterable[str] = ()) -> dict[str, str]:
    """``{relative posix path: sha256}`` of every regular file under ``root``."""
    root = Path(root)
    skipped = set(exclude)
    files = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"symlink in hashed tree: {path}")
        if path.is_file() and not any(
            rel == item or rel.startswith(item.rstrip("/") + "/") for item in skipped
        ):
            files[rel] = sha256_file(path)
    return files


def tree_hash(files: Mapping[str, str]) -> str:
    """One sha256 over sorted ``path  digest`` lines (a ``sha256sum``-style listing)."""
    listing = "".join(f"{digest}  {name}\n" for name, digest in sorted(files.items()))
    return hashlib.sha256(listing.encode()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def safe_name(text: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", text):
        raise ValueError(f"not a safe file name: {text!r}")
    return text


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "paths": {
        # Machine values (code_dir, venv, cloud_corpus, runs_dir, assets, hf_home,
        # rldx_checkpoint, llvmpipe_index_script) have no default.
        "gpu_status_csv": "",
        # Extra folders whose file-tool access by a cell is flagged as cross-task.
        "audit_watch": [],
        "mesa_egl_vendor_json": "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
        "cuda_lib": "/usr/local/cuda/lib64",
        "torch_lib": "",
    },
    "credentials": {},
    "worker": {"model": "openai-chat:qwen3.6-27b", "reasoning_effort": "none"},
    # Exported exactly as the Target50 launcher's Qwen-alone arm does; with
    # RPENT_ESCALATION=0 no senior model is built.
    "senior": {
        "model": "openai:gpt-6-astra",
        "reasoning_effort": "low",
        "checkin": 5,
        "max_share": 0.2,
    },
    "episode": {
        "max_turns": 100,
        "lp_threads": 8,
        "torch_cuda_arch_list": "9.0",
        "env": {},
    },
    "stage1": {
        "seed": 0,
        "sessions": 5,
        "attempts_per_session": 3,
        "session_timeout_s": 2400,
        "session_timeout_s_by_split": {},
        "tasks": [],
        "infra_retries": 2,
        "render_retries": 6,
        "publish_impl": "auto",
        # The explore prompt lists the task's staged cloud files (RPENT_EXPLORE_LIST_MEMORY).
        "list_memory": True,
    },
    "gpus": {
        "allowed": [],
        "max_used_mb": 2048,
        "max_util_pct": 5,
        "csv_max_age_s": 600,
        # Planned GPU memory per GPU: servers x servers.server_mb + slots x
        # servers.cell_mb must stay below this (checked when the config loads).
        "plan_max_mb": 72000,
        "query_cmd": [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu,memory.total",
            "--format=csv,noheader,nounits",
        ],
    },
    "endpoints": [],
    "servers": {
        "mode": "managed",
        "host": "127.0.0.1",
        "cmd": ["bash", "{code_dir}/scripts/ecc_robocasa/vla_server.sh"],
        "ready_pattern": "policy loaded",
        "ready_timeout_s": 900,
        "stop_on_exit": True,
        "max_restarts": 3,
        # A server start refused by the GPU or port check is tried again every
        # retry_busy_s, at most max_refusals times (a GPU freed later is used then).
        "retry_busy_s": 300,
        "max_refusals": 30,
        # Servers start on at most this many GPUs at once (0: every free allowed GPU).
        # A GPU is claimed only when it is free; further servers on a GPU this run
        # already holds only need room for one more server (server_mb + fit_margin_mb).
        "max_gpus": 0,
        # GPU memory one RLDX server holds after its first predictions (measured
        # 17.1 GB), and the idle CUDA context of one cell's env server on RC_GPU
        # (measured 0.52 GB, also with CPU rendering).
        "server_mb": 17500,
        "cell_mb": 600,
        "fit_margin_mb": 2048,
        # Added to every server's environment (over episode.env): the default
        # torch/OpenMP pools (112 threads here) made each busy server burn 26-35
        # cores and slowed its predictions about 4x; 4 threads restored full speed.
        "env": {
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
        },
    },
    "launch": {"cell": ["bash", "{code_dir}/scripts/ecc_robocasa/explore_cell.sh"]},
    "scheduler": {
        "max_slots": 1,
        "launch_stagger_s": 20,
        "poll_s": 10,
        "min_available_mb": 7000,
        "infra_breaker": 5,
        "max_total_launches": 0,
        "cost_ceiling_usd": 0.0,
        "cost_check_s": 60,
        "kill_grace_s": 30,
    },
    # USD per million tokens, list price (the Target50 cost script's values).
    "pricing": {
        "worker_input_per_mtok": 3 / 7.1,
        "worker_output_per_mtok": 18 / 7.1,
        "senior_input_per_mtok": 10.0,
        "senior_cached_input_per_mtok": 1.0,
        "senior_output_per_mtok": 50.0,
    },
}

#: Sections that define the experiment; a resumed run must not change them.
EXPERIMENT_SECTIONS = ("worker", "senior", "episode", "stage1")


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib  # Python >= 3.11
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 venv
        import tomli as tomllib
    with path.open("rb") as handle:
        return tomllib.load(handle)


def config_path(explicit: str | None) -> Path:
    value = explicit or os.environ.get("ECC_RC_CONFIG")
    if not value:
        raise SystemExit("no config: pass --config or set ECC_RC_CONFIG")
    return Path(value).expanduser().resolve()


def load_config(path: Path | str) -> dict[str, Any]:
    """Read a TOML or JSON config, apply defaults and validate the structure."""
    path = Path(path)
    raw = read_json(path) if path.suffix == ".json" else _load_toml(path)
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} is not a mapping")
    cfg = _deep_merge(DEFAULTS, raw)
    cfg["_path"] = str(path)
    for section, key in (
        ("credentials", "file"),
        *(("paths", k) for k in cfg["paths"]),
    ):
        value = cfg[section].get(key)
        if isinstance(value, str) and value.startswith("~"):
            cfg[section][key] = str(Path(value).expanduser())
    code_dir = cfg["paths"].get("code_dir")
    if code_dir:
        cfg["paths"]["code_dir"] = str(Path(code_dir).expanduser().resolve())
        for section, key in (("launch", "cell"), ("servers", "cmd")):
            cfg[section][key] = [
                str(item).replace("{code_dir}", cfg["paths"]["code_dir"])
                for item in cfg[section][key]
            ]
    validate_config(cfg)
    return cfg


def validate_config(cfg: Mapping[str, Any]) -> None:
    allowed = cfg["gpus"]["allowed"]
    if not isinstance(allowed, list) or not all(isinstance(g, int) for g in allowed):
        raise ValueError("gpus.allowed must be a list of GPU indices")
    seen_ports, seen_names = set(), set()
    planned: dict[int, int] = {}
    servers = cfg["servers"]
    for ep in cfg["endpoints"]:
        gpu, port, slots = ep.get("gpu"), ep.get("port"), ep.get("slots", 1)
        if gpu not in allowed:
            raise ValueError(f"endpoint GPU {gpu} is not in gpus.allowed {allowed}")
        if not isinstance(port, int) or port in seen_ports:
            raise ValueError(f"endpoint port {port!r} missing or repeated")
        if not isinstance(slots, int) or slots < 0:
            raise ValueError(f"endpoint slots {slots!r} must be a nonnegative integer")
        seen_ports.add(port)
        planned[gpu] = (
            planned.get(gpu, 0)
            + int(servers["server_mb"])
            + slots * int(servers["cell_mb"])
        )
    for name in (ep.name for ep in endpoints_from_config(cfg)):
        if name in seen_names:
            raise ValueError(f"endpoint name {name!r} repeated")
        seen_names.add(name)
    limit = int(cfg["gpus"]["plan_max_mb"])
    over = {gpu: mb for gpu, mb in planned.items() if mb > limit}
    if over:
        raise ValueError(
            f"planned GPU memory {over} MiB exceeds gpus.plan_max_mb {limit}: "
            "fewer servers or slots per GPU (servers.server_mb per server, "
            "servers.cell_mb per slot)"
        )
    if cfg["servers"]["mode"] not in {"managed", "external"}:
        raise ValueError("servers.mode must be 'managed' or 'external'")
    s1 = cfg["stage1"]
    if s1["seed"] != 0:
        raise ValueError(
            "stage1.seed must be 0: the RoboCasa local loader reads only <Task>_target_s0.* "
            "(robots/robocasa/memory.py local_task_files), so other seeds publish nothing"
        )
    for key in ("sessions", "attempts_per_session", "session_timeout_s"):
        if not isinstance(s1[key], int) or s1[key] < 1:
            raise ValueError(f"stage1.{key} must be a positive integer")
    if (
        not isinstance(cfg["scheduler"]["max_slots"], int)
        or cfg["scheduler"]["max_slots"] < 1
    ):
        raise ValueError("scheduler.max_slots must be a positive integer")


def require(cfg: Mapping[str, Any], *keys: str) -> None:
    """Fail unless every ``section.key`` is set (machine values without defaults)."""
    missing = []
    for dotted in keys:
        node: Any = cfg
        for part in dotted.split("."):
            node = node.get(part) if isinstance(node, Mapping) else None
        if node in (None, "", []):
            missing.append(dotted)
    if missing:
        raise SystemExit(f"config {cfg.get('_path')}: missing {', '.join(missing)}")


def experiment_hash(cfg: Mapping[str, Any]) -> str:
    """Hash of the sections that define the experiment (not slots or paths)."""
    return canonical_hash({key: cfg[key] for key in EXPERIMENT_SECTIONS})


def config_hash(cfg: Mapping[str, Any]) -> str:
    return canonical_hash({k: v for k, v in cfg.items() if not k.startswith("_")})


# ---------------------------------------------------------------------------
# tasks, code identity
# ---------------------------------------------------------------------------


def load_target50(code_dir: Path | str) -> list[tuple[str, str]]:
    """``[(split, task), ...]`` in the manifest's split and task order."""
    manifest = json.loads((Path(code_dir) / TARGET50_MANIFEST).read_text())
    rows = []
    for split in SPLITS:
        rows.extend((split, task) for task in manifest["splits"][split]["tasks"])
    names = [task for _, task in rows]
    if len(names) != len(set(names)):
        raise ValueError("Target50 task names are not unique across splits")
    return rows


def code_identity(code_dir: Path | str) -> dict[str, Any]:
    """Commit (when known) and a content hash of the code the episodes import."""
    code_dir = Path(code_dir)
    files = {}
    for top in ("rpent", "robots", "scripts/ecc_robocasa"):
        base = code_dir / top
        if base.is_dir():
            for path in sorted(base.rglob("*")):
                if (
                    path.is_file()
                    and "__pycache__" not in path.parts
                    and path.suffix != ".pyc"
                ):
                    files[path.relative_to(code_dir).as_posix()] = sha256_file(path)
    commit, source = None, None
    try:
        out = subprocess.run(
            ["git", "-C", str(code_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        top = subprocess.run(
            ["git", "-C", str(code_dir), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        same_repo = top.returncode == 0 and os.path.realpath(
            top.stdout.strip()
        ) == os.path.realpath(code_dir)
        if out.returncode == 0 and same_repo:
            commit, source = out.stdout.strip(), "git"
    except (OSError, subprocess.TimeoutExpired):
        pass
    if commit is None and (code_dir / "CODE_COMMIT").is_file():
        commit = (code_dir / "CODE_COMMIT").read_text().strip() or None
        source = "CODE_COMMIT file"
    dirty = None
    if (code_dir / "CODE_DIRTY").is_file():
        dirty = (code_dir / "CODE_DIRTY").read_text().strip()
    return {
        "commit": commit,
        "commit_source": source,
        "dirty_listing": dirty,
        "tree_sha256": tree_hash(files),
        "files": len(files),
    }


def import_origin(
    python: str, code_dir: Path | str, env: Mapping[str, str]
) -> dict[str, Any]:
    """Where ``rpent`` and ``robots`` resolve for an episode process (no import side effects)."""
    script = (
        "import importlib.util, json, os, sys\n"
        "if sys.path and sys.path[0] == '': del sys.path[0]\n"
        "out = {}\n"
        "for name in ('rpent', 'robots'):\n"
        "    spec = importlib.util.find_spec(name)\n"
        "    out[name] = os.path.realpath(spec.origin) if spec and spec.origin else None\n"
        "print(json.dumps(out))\n"
    )
    proc = subprocess.run(
        [python, "-c", script],
        capture_output=True,
        text=True,
        env=dict(env),
        cwd="/",
        timeout=120,
    )
    try:
        found = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"ok": False, "error": (proc.stderr or proc.stdout)[-500:]}
    root = os.path.realpath(str(code_dir)) + os.sep
    ok = all(isinstance(v, str) and v.startswith(root) for v in found.values())
    return {"ok": ok, **found}


# ---------------------------------------------------------------------------
# processes, ports, memory
# ---------------------------------------------------------------------------


def _cmdline(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        if stat.rsplit(")", 1)[1].split()[0] == "Z":
            return False
    except (OSError, IndexError):
        pass
    return True


def pids_with_arg(value: str) -> list[int]:
    """Live processes that have ``value`` as one of their arguments."""
    found = []
    proc = Path("/proc")
    if proc.is_dir():
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            args = _cmdline(int(entry.name))
            if args and value in args[1:] and pid_alive(int(entry.name)):
                found.append(int(entry.name))
        return sorted(found)
    out = subprocess.run(["ps", "-axo", "pid=,args="], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        pid, _, args = line.strip().partition(" ")
        if pid.isdigit() and (f" {value} " in f" {args} ") and int(pid) != os.getpid():
            found.append(int(pid))
    return sorted(found)


def cmdline_has(pid: int, *needles: str) -> bool:
    args = _cmdline(pid)
    if args is None:
        if not pid_alive(pid):
            return False
        out = subprocess.run(
            ["ps", "-o", "args=", "-p", str(pid)], capture_output=True, text=True
        )
        text = out.stdout
    else:
        text = " ".join(args)
    return all(needle in text for needle in needles)


def kill_group(pid: int, grace_s: float = 30.0) -> None:
    """SIGTERM the process group led by ``pid``, then SIGKILL after ``grace_s``."""
    try:
        pgid = os.getpgid(pid)
    except OSError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.time() + grace_s
    while time.time() < deadline and pid_alive(pid):
        time.sleep(0.2)
    if pid_alive(pid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass


def port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def available_mb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 1 << 30


# ---------------------------------------------------------------------------
# GPUs
# ---------------------------------------------------------------------------


def gpu_usage(cfg: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """Live ``{gpu: {used_mb, util}}`` from nvidia-smi and, when fresh, the platform CSV."""
    gpus = cfg["gpus"]
    out = subprocess.run(gpus["query_cmd"], capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"GPU query failed: {out.stderr.strip()[:300]}")
    usage: dict[int, dict[str, Any]] = {}
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            usage[int(parts[0])] = {
                "used_mb": int(float(parts[1])),
                "util": int(float(parts[2])),
            }
            if len(parts) >= 4 and re.fullmatch(r"[0-9.]+", parts[3]):
                usage[int(parts[0])]["total_mb"] = int(float(parts[3]))
    csv_path = cfg["paths"].get("gpu_status_csv") or ""
    if csv_path and Path(csv_path).is_file():
        age = time.time() - Path(csv_path).stat().st_mtime
        if age < gpus["csv_max_age_s"]:
            lines = Path(csv_path).read_text().splitlines()
            for line in lines[1:]:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 8 and parts[1].isdigit():
                    idx = int(parts[1])
                    used = int(re.sub(r"[^0-9]", "", parts[7]) or 0)
                    util = int(re.sub(r"[^0-9]", "", parts[5]) or 0)
                    usage.setdefault(idx, {})["csv_used_mb"] = used
                    usage[idx]["csv_util"] = util
        else:
            for value in usage.values():
                value["csv_stale_s"] = round(age)
    return usage


def gpu_free(
    cfg: Mapping[str, Any],
    gpu: int,
    usage: Mapping[int, Mapping[str, Any]] | None = None,
) -> tuple[bool, str]:
    """Whether ``gpu`` is allowed and idle now (nvidia-smi and the platform CSV)."""
    gpus = cfg["gpus"]
    if gpu not in gpus["allowed"]:
        return False, f"GPU {gpu} is not allowed ({gpus['allowed']})"
    try:
        usage = usage if usage is not None else gpu_usage(cfg)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return False, f"GPU query failed: {exc}"
    row = usage.get(gpu)
    if row is None or "used_mb" not in row:
        return False, f"GPU {gpu} not reported by nvidia-smi"
    limit_mb, limit_util = gpus["max_used_mb"], gpus["max_util_pct"]
    if row["used_mb"] >= limit_mb or row["util"] > limit_util:
        return (
            False,
            f"GPU {gpu} busy per nvidia-smi ({row['used_mb']} MiB, {row['util']} %)",
        )
    if "csv_used_mb" in row and (
        row["csv_used_mb"] >= limit_mb or row["csv_util"] > limit_util
    ):
        return (
            False,
            f"GPU {gpu} busy per the platform CSV ({row['csv_used_mb']} MiB, {row['csv_util']} %)",
        )
    note = "" if "csv_used_mb" in row else " (no fresh platform CSV row)"
    return True, f"GPU {gpu} free ({row['used_mb']} MiB, {row['util']} %){note}"


def gpu_fits(
    cfg: Mapping[str, Any],
    gpu: int,
    need_mb: int,
    usage: Mapping[int, Mapping[str, Any]] | None = None,
) -> tuple[bool, str]:
    """Whether ``need_mb`` more MiB fit on ``gpu`` now (a GPU this run already holds).

    Utilization is not checked: this run's own servers keep the GPU busy.
    """
    if gpu not in cfg["gpus"]["allowed"]:
        return False, f"GPU {gpu} is not allowed ({cfg['gpus']['allowed']})"
    try:
        usage = usage if usage is not None else gpu_usage(cfg)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return False, f"GPU query failed: {exc}"
    row = usage.get(gpu)
    if row is None or "used_mb" not in row or not row.get("total_mb"):
        return False, f"GPU {gpu} memory total/used not reported by nvidia-smi"
    used = max(row["used_mb"], row.get("csv_used_mb", 0))
    room = row["total_mb"] - used
    if room < need_mb:
        return False, f"GPU {gpu} has {room} MiB free, {need_mb} MiB needed"
    return True, f"GPU {gpu} held by this run; {room} MiB free >= {need_mb} MiB"


# ---------------------------------------------------------------------------
# run layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunLayout:
    """Paths of one run root; every stage keeps the same top-level shape."""

    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def progress(self) -> Path:
        return self.root / "progress.jsonl"

    @property
    def stop(self) -> Path:
        return self.root / STOP_FILE

    @property
    def concurrency(self) -> Path:
        return self.root / "concurrency"

    @property
    def lock(self) -> Path:
        return self.root / "runner.lock"

    @property
    def jobs_dir(self) -> Path:
        return self.root / "state" / "jobs"

    @property
    def servers_dir(self) -> Path:
        return self.root / "servers"

    @property
    def tasks_dir(self) -> Path:
        return self.root / "tasks"

    @property
    def product(self) -> Path:
        return self.root / "product"

    def task_dir(self, task: str) -> Path:
        return self.tasks_dir / safe_name(task)

    def job_state(self, job_id: str) -> Path:
        return self.jobs_dir / f"{safe_name(job_id)}.json"


def write_stop(layout: RunLayout, reason: str, **extra: Any) -> None:
    if not layout.stop.exists():
        atomic_write_json(layout.stop, {"reason": reason, "at": now_iso(), **extra})


# ---------------------------------------------------------------------------
# episode outcome, render check
# ---------------------------------------------------------------------------


def _frame_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), "") if path.stem.isdigit() else (1 << 30, path.name)


def agentview_frames(try_dir: Path) -> list[Path]:
    """Agentview frames in recording order (evaluation folder or explore sessions).

    Frames are ``NN.png`` by step, so they sort numerically (``100.png`` after ``99.png``).
    """
    frames = sorted((try_dir / "agentview.png").glob("*.png"), key=_frame_key)
    for session in sorted((try_dir / "sessions").glob("session_*")):
        frames.extend(sorted((session / "agentview.png").glob("*.png"), key=_frame_key))
    return frames


def frame_gradient(path: Path) -> float:
    import numpy as np
    from PIL import Image

    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    return float(np.abs(a[:, 1:] - a[:, :-1]).mean())


def check_render(try_dir: Path, checked: int = 0) -> tuple[bool, int]:
    """(corrupt, frames checked): marks ``try_dir`` when a new frame is noise."""
    frames = agentview_frames(try_dir)
    for frame in frames[checked:]:
        try:
            gradient = frame_gradient(frame)
        except (OSError, ValueError, SyntaxError):
            return False, checked  # still being written
        checked += 1
        if gradient > RENDER_GRADIENT_LIMIT:
            (try_dir / RENDER_MARK).write_text(
                f"{frame.relative_to(try_dir)} gradient {gradient:.1f}\n"
            )
            return True, checked
    return False, checked


def read_outcome(try_dir: Path) -> dict[str, Any]:
    """RPent's ``result.json``; render noise makes it invalid (Target50 batch runner rules).

    A missing or unreadable ``result.json`` is ``no_result`` (an infrastructure
    error); a render mark wins over both, as in the Target50 ablation runner.
    """
    result = read_json(try_dir / "result.json")
    if not isinstance(result, dict):
        result = {"valid": False, "success": None, "termination_reason": "no_result"}
    if (try_dir / RENDER_MARK).exists():
        result = dict(result, valid=False, termination_reason="render_corrupt")
    return result


def killed_marker(try_dir: Path | str) -> Path:
    """Marker written next to a try folder when the operator stops it (not a failure)."""
    return Path(f"{try_dir}.killed")


def infra_marker(try_dir: Path | str) -> Path:
    """Marker written next to a try folder the runner stopped for an infrastructure cause."""
    return Path(f"{try_dir}.infra")


# ---------------------------------------------------------------------------
# usage and cost
# ---------------------------------------------------------------------------

_USAGE = re.compile(
    r"\[usage\] in=(\d+) out=(\d+) cache_read=(\d+) cache_write=(\d+) requests=(\d+)"
)
_HANDOFF = "=== handing off to agent"


def usage_from_run_log(path: Path) -> dict[str, Any]:
    """Planner tokens per session: the last cumulative ``[usage]`` line before each handoff."""
    sessions: list[dict[str, int]] = []
    current: dict[str, int] | None = None
    try:
        handle = path.open(errors="replace")
    except OSError:
        return {"sessions": [], "input_tokens": 0, "output_tokens": 0, "requests": 0}
    with handle:
        for line in handle:
            if _HANDOFF in line:
                if current is not None:
                    sessions.append(current)
                current = None
                continue
            match = _USAGE.search(line)
            if match:
                current = {
                    "input_tokens": int(match.group(1)),
                    "output_tokens": int(match.group(2)),
                    "cache_read_tokens": int(match.group(3)),
                    "requests": int(match.group(5)),
                }
    if current is not None:
        sessions.append(current)
    return {
        "sessions": sessions,
        "input_tokens": sum(s["input_tokens"] for s in sessions),
        "output_tokens": sum(s["output_tokens"] for s in sessions),
        "requests": sum(s["requests"] for s in sessions),
    }


def usage_from_escalation(path: Path) -> dict[str, int]:
    """Worker and senior tokens from ``escalation.jsonl`` (errored rows skipped)."""
    totals = {
        "worker_in": 0,
        "worker_out": 0,
        "worker_calls": 0,
        "senior_in": 0,
        "senior_cached": 0,
        "senior_out": 0,
        "senior_calls": 0,
    }
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return totals
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if "input_tokens" not in row or "error" in row:
            continue
        if row.get("role") == "worker":
            totals["worker_calls"] += 1
            totals["worker_in"] += row["input_tokens"] or 0
            totals["worker_out"] += row.get("output_tokens") or 0
        else:
            totals["senior_calls"] += 1
            totals["senior_in"] += row["input_tokens"] or 0
            totals["senior_cached"] += row.get("cached_input_tokens") or 0
            totals["senior_out"] += row.get("output_tokens") or 0
    return totals


def episode_usage(try_dir: Path, pricing: Mapping[str, float]) -> dict[str, Any]:
    """Tokens and list-price USD of one launch (escalation log if present, else run.log)."""
    escalation = try_dir / "escalation.jsonl"
    if escalation.is_file():
        u = usage_from_escalation(escalation)
        worker_usd = (
            u["worker_in"] * pricing["worker_input_per_mtok"]
            + u["worker_out"] * pricing["worker_output_per_mtok"]
        ) / 1e6
        senior_usd = (
            (u["senior_in"] - u["senior_cached"]) * pricing["senior_input_per_mtok"]
            + u["senior_cached"] * pricing["senior_cached_input_per_mtok"]
            + u["senior_out"] * pricing["senior_output_per_mtok"]
        ) / 1e6
        return {
            "source": "escalation.jsonl",
            **u,
            "worker_usd": worker_usd,
            "senior_usd": senior_usd,
            "usd": worker_usd + senior_usd,
        }
    u = usage_from_run_log(try_dir / "run.log")
    worker_usd = (
        u["input_tokens"] * pricing["worker_input_per_mtok"]
        + u["output_tokens"] * pricing["worker_output_per_mtok"]
    ) / 1e6
    return {
        "source": "run.log",
        "worker_in": u["input_tokens"],
        "worker_out": u["output_tokens"],
        "worker_calls": u["requests"],
        "sessions": len(u["sessions"]),
        "worker_usd": worker_usd,
        "senior_usd": 0.0,
        "usd": worker_usd,
    }


# ---------------------------------------------------------------------------
# shared RLDX servers
# ---------------------------------------------------------------------------


@dataclass
class Endpoint:
    name: str
    gpu: int
    port: int
    slots: int
    host: str = "127.0.0.1"
    up: bool = False
    note: str = ""

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


def endpoints_from_config(cfg: Mapping[str, Any]) -> list[Endpoint]:
    """One endpoint per ``[[endpoints]]`` entry: ``gpu<N>``, or ``gpu<N>_<port>`` on a shared GPU."""
    host = cfg["servers"]["host"]
    per_gpu: dict[Any, int] = {}
    for ep in cfg["endpoints"]:
        per_gpu[ep["gpu"]] = per_gpu.get(ep["gpu"], 0) + 1
    return [
        Endpoint(
            name=str(
                ep.get("name")
                or (
                    f"gpu{ep['gpu']}"
                    if per_gpu[ep["gpu"]] == 1
                    else f"gpu{ep['gpu']}_{ep['port']}"
                )
            ),
            gpu=int(ep["gpu"]),
            port=int(ep["port"]),
            slots=int(ep.get("slots", 1)),
            host=host,
        )
        for ep in cfg["endpoints"]
    ]


class ServerPool:
    """Shared RLDX vla_servers (one per endpoint; several may share a GPU), started or
    adopted by the runner.

    A GPU is claimed only when it is free (and, with ``servers.max_gpus``, while fewer
    GPUs are held); more servers on a GPU this run holds need only room for one server.
    A managed server that dies is started again (at most ``servers.max_restarts``
    times per endpoint, at most once a minute, and only while its GPU passes the same
    check); the scheduler stops the cells that were attached to it.
    """

    def __init__(
        self,
        cfg: Mapping[str, Any],
        layout: RunLayout,
        env: Mapping[str, str],
        log: Callable[..., None],
    ):
        self.cfg = cfg
        self.layout = layout
        self.env = dict(env)
        self.log = log
        self.endpoints = endpoints_from_config(cfg)
        self.mode = cfg["servers"]["mode"]
        #: name -> (proc, log path, deadline) of servers started and not ready yet
        self.starting: dict[str, tuple[subprocess.Popen, Path, float]] = {}
        self.restarts: dict[str, int] = {}
        self.refusals: dict[str, int] = {}
        self.restart_after: dict[str, float] = {}
        self.external_misses: dict[str, int] = {}
        #: Endpoints whose server this pool started or adopted (a later start is a restart).
        self.launched: set[str] = set()

    def _state_path(self, ep: Endpoint) -> Path:
        return self.layout.servers_dir / f"{ep.name}.json"

    def _ours(self, ep: Endpoint) -> int | None:
        state = read_json(self._state_path(ep), {})
        pid = state.get("pid")
        if (
            pid
            and not state.get("stopped_at")
            and pid_alive(pid)
            and cmdline_has(pid, str(ep.port))
            and state.get("port") == ep.port
        ):
            return pid
        return None

    def held_gpus(self, exclude: Endpoint | None = None) -> set[int]:
        """GPUs on which this run has a live or starting server (other than ``exclude``'s)."""
        return {
            ep.gpu
            for ep in self.endpoints
            if ep is not exclude and (ep.name in self.starting or self._ours(ep))
        }

    def _start(
        self, ep: Endpoint, usage: Mapping[int, Mapping[str, Any]] | None, forced: bool
    ) -> str:
        """Start ``ep``'s server after the GPU checks.

        Returns ``started``, ``refused`` (busy GPU or port: retried), ``deferred``
        (``servers.max_gpus`` GPUs already held: retried, not counted as a refusal)
        or ``skipped`` (0 slots or a GPU that is not allowed: never started).
        """
        if port_open(ep.host, ep.port):
            ep.up, ep.note = False, f"port {ep.port} is in use by another process"
            self.log(
                {"event": "server_refused", "endpoint": ep.name, "reason": ep.note}
            )
            return "refused"
        if ep.slots == 0:
            ep.note = "0 slots; not started"
            return "skipped"
        if ep.gpu not in self.cfg["gpus"]["allowed"]:
            ep.note = f"GPU {ep.gpu} not allowed"
            return "skipped"
        if not forced:
            servers = self.cfg["servers"]
            held = self.held_gpus(exclude=ep)
            max_gpus = int(servers.get("max_gpus") or 0)
            if ep.gpu not in held and max_gpus and len(held) >= max_gpus:
                ep.up, ep.note = False, f"servers.max_gpus={max_gpus} GPUs already held"
                return "deferred"
            if usage is None:
                free, reason = False, "GPU query failed"
            elif ep.gpu in held:
                need = int(servers["server_mb"]) + int(servers["fit_margin_mb"])
                free, reason = gpu_fits(self.cfg, ep.gpu, need, usage)
            else:
                free, reason = gpu_free(self.cfg, ep.gpu, usage)
            if not free:
                ep.up, ep.note = False, reason
                self.log(
                    {"event": "server_refused", "endpoint": ep.name, "reason": reason}
                )
                return "refused"
        log_path = self.layout.servers_dir / f"{ep.name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            log_path.rename(log_path.with_name(f"{log_path.name}.{int(time.time())}"))
        server_env = {
            str(k): str(v) for k, v in (self.cfg["servers"].get("env") or {}).items()
        }
        proc = subprocess.Popen(
            [*self.cfg["servers"]["cmd"], str(ep.port), str(log_path)],
            env=dict(self.env, **server_env, RC_GPU=str(ep.gpu)),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        atomic_write_json(
            self._state_path(ep),
            {
                "pid": proc.pid,
                "port": ep.port,
                "gpu": ep.gpu,
                "log": str(log_path),
                "started_at": now_iso(),
                "cmd": self.cfg["servers"]["cmd"],
            },
        )
        self.log(
            {
                "event": "server_start",
                "endpoint": ep.name,
                "pid": proc.pid,
                "gpu": ep.gpu,
                "port": ep.port,
            }
        )
        self.starting[ep.name] = (
            proc,
            log_path,
            time.time() + float(self.cfg["servers"]["ready_timeout_s"]),
        )
        ep.up, ep.note = False, f"starting pid {proc.pid}"
        self.launched.add(ep.name)
        return "started"

    def _usage(self) -> Mapping[int, Mapping[str, Any]] | None:
        try:
            return gpu_usage(self.cfg)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            self.log({"event": "gpu_query_failed", "error": str(exc)})
            return None

    def ensure(
        self, force_gpus: Iterable[int] = (), wait: bool = True
    ) -> list[Endpoint]:
        """Adopt or start every endpoint's server; returns the endpoints that are up."""
        if self.mode == "external":
            for ep in self.endpoints:
                ep.up = port_open(ep.host, ep.port, timeout=5.0)
                ep.note = "external" if ep.up else "external server not answering"
            return [ep for ep in self.endpoints if ep.up]
        forced = set(force_gpus)
        usage = self._usage()
        for ep in self.endpoints:
            pid = self._ours(ep)
            if pid:
                ep.up = port_open(ep.host, ep.port, timeout=5.0)
                ep.note = f"adopted pid {pid}"
                self.launched.add(ep.name)
                self.log(
                    {
                        "event": "server_adopted",
                        "endpoint": ep.name,
                        "pid": pid,
                        "up": ep.up,
                    }
                )
                continue
            if self._start(ep, usage, ep.gpu in forced) in ("refused", "deferred"):
                self.schedule_restart(
                    ep.name, float(self.cfg["servers"]["retry_busy_s"])
                )
        while wait and self.starting:
            self.tick()
            if self.starting:
                time.sleep(1.0)
        return [ep for ep in self.endpoints if ep.up]

    def tick(self) -> None:
        """Advance servers that are starting: ready, exited, or timed out."""
        pattern = self.cfg["servers"]["ready_pattern"]
        by_name = {ep.name: ep for ep in self.endpoints}
        for name, (proc, log_path, deadline) in list(self.starting.items()):
            ep = by_name[name]
            if proc.poll() is not None:
                ep.up, ep.note = (
                    False,
                    f"server exited ({proc.returncode}); see {log_path}",
                )
                self.log(
                    {
                        "event": "server_failed",
                        "endpoint": name,
                        "exit": proc.returncode,
                    }
                )
                del self.starting[name]
                # Tried again like a server that died (restart limit, GPU re-checked).
                self.schedule_restart(name, float(self.cfg["servers"]["retry_busy_s"]))
                continue
            try:
                ready = pattern in log_path.read_text(errors="replace")
            except OSError:
                ready = False
            if ready and port_open(ep.host, ep.port):
                ep.up, ep.note = True, f"started pid {proc.pid}"
                self.log({"event": "server_ready", "endpoint": name, "pid": proc.pid})
                del self.starting[name]
            elif time.time() > deadline:
                ep.up, ep.note = False, "not ready in time; stopped"
                kill_group(
                    proc.pid, grace_s=float(self.cfg["scheduler"]["kill_grace_s"])
                )
                self.log({"event": "server_timeout", "endpoint": name})
                del self.starting[name]
                self.schedule_restart(name, float(self.cfg["servers"]["retry_busy_s"]))

    def check(self) -> list[str]:
        """Mark endpoints whose server died as down (returns their names); restart managed ones."""
        self.tick()
        down = []
        for ep in self.endpoints:
            if not ep.up:
                continue
            if self.mode == "external":
                # A loaded server may miss one connect; three in a row is down.
                misses = (
                    0
                    if port_open(ep.host, ep.port, timeout=5.0)
                    else self.external_misses.get(ep.name, 0) + 1
                )
                self.external_misses[ep.name] = misses
                alive = misses < 3
            else:
                alive = bool(self._ours(ep))
            if not alive:
                ep.up, ep.note = False, "server stopped"
                down.append(ep.name)
                self.schedule_restart(ep.name)
                self.log({"event": "server_down", "endpoint": ep.name})
        if self.mode == "managed":
            due = [
                ep
                for ep in self.endpoints
                if not ep.up
                and ep.name not in self.starting
                and ep.name in self.restart_after
                and time.time() >= self.restart_after[ep.name]
            ]
            usage = self._usage() if due else None
            for ep in due:
                if self.restarts.get(ep.name, 0) >= int(
                    self.cfg["servers"]["max_restarts"]
                ):
                    del self.restart_after[ep.name]
                    self.log(
                        {
                            "event": "server_given_up",
                            "endpoint": ep.name,
                            "why": "restart limit",
                        }
                    )
                    continue
                restart = ep.name in self.launched
                outcome = self._start(ep, usage, forced=False)
                if outcome == "started":
                    # A first start of an endpoint refused earlier is not a restart.
                    if restart:
                        self.restarts[ep.name] = self.restarts.get(ep.name, 0) + 1
                    del self.restart_after[ep.name]
                    self.refusals.pop(ep.name, None)
                    continue
                if outcome == "skipped":
                    del self.restart_after[ep.name]
                    continue
                if outcome == "deferred":  # enough GPUs held; check again later
                    self.restart_after[ep.name] = time.time() + float(
                        self.cfg["servers"]["retry_busy_s"]
                    )
                    continue
                # GPU taken (by another user, or our stopped cells still exiting): retry later.
                self.refusals[ep.name] = self.refusals.get(ep.name, 0) + 1
                if self.refusals[ep.name] >= int(self.cfg["servers"]["max_refusals"]):
                    del self.restart_after[ep.name]
                    self.log(
                        {
                            "event": "server_given_up",
                            "endpoint": ep.name,
                            "why": ep.note,
                        }
                    )
                else:
                    self.restart_after[ep.name] = time.time() + float(
                        self.cfg["servers"]["retry_busy_s"]
                    )
        return down

    def schedule_restart(self, name: str, delay_s: float = 30.0) -> None:
        """Start ``name``'s server again after ``delay_s`` (managed mode; GPU re-checked then)."""
        if self.mode == "managed":
            self.restart_after.setdefault(name, time.time() + delay_s)

    def stop_all(self) -> None:
        if self.mode != "managed":
            return
        for name, (proc, _, _) in list(self.starting.items()):
            kill_group(proc.pid, grace_s=float(self.cfg["scheduler"]["kill_grace_s"]))
            del self.starting[name]
        for ep in self.endpoints:
            pid = self._ours(ep)
            if pid:
                kill_group(pid, grace_s=float(self.cfg["scheduler"]["kill_grace_s"]))
                self.log({"event": "server_stop", "endpoint": ep.name, "pid": pid})
            ep.up = False
            state_path = self._state_path(ep)
            if state_path.exists():
                state = read_json(state_path, {})
                atomic_write_json(state_path, {**state, "stopped_at": now_iso()})


# ---------------------------------------------------------------------------
# scheduler
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """One unit of work (stage 1: one task's exploration cell)."""

    job_id: str
    task: str
    split: str
    seed: int
    order: int
    meta: dict[str, Any] = field(default_factory=dict)


class Stage:
    """What a stage provides to the scheduler (stage 1 implements it; 2-3 will too)."""

    name = "stage"

    def jobs(self) -> list[Job]:  # pragma: no cover - interface
        raise NotImplementedError

    def try_dir(self, job: Job, n: int) -> Path:  # pragma: no cover - interface
        raise NotImplementedError

    def prepare_try(self, job: Job, n: int) -> None:
        """Called right before try ``n`` launches (e.g. reset a staging corpus)."""

    def launch(
        self, job: Job, n: int, endpoint: Endpoint
    ) -> tuple[list[str], dict[str, str]]:  # pragma: no cover
        raise NotImplementedError

    def on_final(self, job: Job, state: dict[str, Any]) -> None:
        """Called once when a job's last try is scored."""

    def budget_launches(self) -> int:
        """Launches the stage's budget allows before any infrastructure retry."""
        return len(self.jobs())


#: Try classes: valid (scored), infra (invalid result), render (noise frames), killed (by us).
RETRYABLE = ("infra", "render")


def classify(
    outcome: Mapping[str, Any], killed: bool, forced_infra: bool = False
) -> str:
    """valid | infra | render | killed. A runner-stopped try (``forced_infra``) is infra even
    when RPent then wrote a valid result: its RLDX server was gone for part of the run."""
    if killed:
        return "killed"
    if forced_infra:
        return "infra"
    if outcome.get("termination_reason") == "render_corrupt":
        return "render"
    if not outcome.get("valid"):
        return "infra"
    return "valid"


class Scheduler:
    """Run every job's tries over the endpoints' slots, resumably.

    State per job lives in ``state/jobs/<job>.json`` and is written atomically
    at every transition; events go to ``progress.jsonl``. Cells run in their own
    session (process group), so a killed runner leaves them running; a new
    runner adopts live cells by their output folder and scores dead ones from
    their files, so nothing is launched twice and no result is lost.
    """

    def __init__(
        self,
        cfg: Mapping[str, Any],
        layout: RunLayout,
        stage: Stage,
        servers: ServerPool,
        *,
        infra_retries: int,
        render_retries: int,
        max_slots: int | None = None,
    ):
        self.cfg = cfg
        self.layout = layout
        self.stage = stage
        self.servers = servers
        self.sched = cfg["scheduler"]
        self.infra_retries = infra_retries
        self.render_retries = render_retries
        self.max_slots = min(
            int(max_slots or self.sched["max_slots"]), int(self.sched["max_slots"])
        )
        self.jobs = {job.job_id: job for job in stage.jobs()}
        self.children: dict[int, subprocess.Popen] = {}
        self.render_checked: dict[str, int] = {}
        self.last_launch = 0.0
        self.last_cost_check = 0.0
        self.cost_usd = 0.0
        self.started = time.time()

    # -- persistence -------------------------------------------------------

    def log(self, row: Mapping[str, Any]) -> None:
        append_jsonl(self.layout.progress, row)

    def state(self, job_id: str) -> dict[str, Any]:
        job = self.jobs[job_id]
        return read_json(self.layout.job_state(job_id)) or {
            "job_id": job_id,
            "task": job.task,
            "split": job.split,
            "seed": job.seed,
            "tries": [],
            "final": None,
        }

    def save(self, state: dict[str, Any]) -> None:
        atomic_write_json(self.layout.job_state(state["job_id"]), state)

    def all_states(self) -> dict[str, dict[str, Any]]:
        return {job_id: self.state(job_id) for job_id in self.jobs}

    # -- counting ----------------------------------------------------------

    @staticmethod
    def class_count(state: Mapping[str, Any], name: str) -> int:
        return sum(1 for t in state["tries"] if t.get("class") == name)

    def launches_so_far(self, states: Mapping[str, Mapping[str, Any]]) -> int:
        return counted_launches(states)

    def launch_ceiling(self) -> int:
        return launch_ceiling(
            self.sched,
            self.stage.budget_launches(),
            len(self.jobs),
            self.infra_retries,
            self.render_retries,
        )

    # -- recovery ----------------------------------------------------------

    def recover(self) -> None:
        """Adopt live tries from an earlier runner; score dead ones."""
        for job_id in self.jobs:
            state = self.state(job_id)
            if state["final"] or not state["tries"]:
                continue
            last = state["tries"][-1]
            if last.get("status") not in ("launching", "running"):
                continue
            try_dir = Path(last["dir"])
            live = pids_with_arg(str(try_dir))
            if live:
                last.update(
                    status="running", pid=last.get("pid") or live[0], adopted=True
                )
                self.save(state)
                self.log(
                    {
                        "event": "adopt",
                        "job": job_id,
                        "try": last["n"],
                        "pid": last["pid"],
                    }
                )
                continue
            if last.get("status") == "launching" and not try_dir.exists():
                state["tries"].pop()  # never started: not a launch
                self.save(state)
                self.log({"event": "unlaunched", "job": job_id, "try": last["n"]})
                continue
            self.finish_try(job_id, state, exit_code=None)

    # -- running tries -----------------------------------------------------

    def running(
        self, states: Mapping[str, Mapping[str, Any]]
    ) -> list[tuple[str, dict[str, Any]]]:
        out = []
        for job_id, state in states.items():
            if state["tries"] and state["tries"][-1].get("status") in (
                "launching",
                "running",
            ):
                out.append((job_id, state["tries"][-1]))
        return out

    def poll(self) -> None:
        states = self.all_states()
        for job_id, last in self.running(states):
            pid = last.get("pid")
            proc = self.children.get(pid) if pid else None
            if proc is not None:
                code = proc.poll()
                if code is None:
                    self._render_watch(job_id, last)
                    continue
                del self.children[pid]
                self.finish_try(job_id, states[job_id], exit_code=code)
            elif pid and pid_alive(pid) and cmdline_has(pid, last["dir"]):
                self._render_watch(job_id, last)
            else:
                self.finish_try(job_id, states[job_id], exit_code=None)

    def _render_watch(self, job_id: str, last: dict[str, Any]) -> None:
        key = f"{job_id}#{last['n']}"
        try:
            corrupt, self.render_checked[key] = check_render(
                Path(last["dir"]), self.render_checked.get(key, 0)
            )
        except ImportError:
            return  # no numpy/Pillow in this interpreter
        if corrupt:
            self.log({"event": "render_corrupt", "job": job_id, "try": last["n"]})
            kill_group(int(last["pid"]), grace_s=float(self.sched["kill_grace_s"]))

    def finish_try(
        self, job_id: str, state: dict[str, Any], exit_code: int | None
    ) -> None:
        last = state["tries"][-1]
        try_dir = Path(last["dir"])
        key = f"{job_id}#{last['n']}"
        if try_dir.exists():
            try:
                check_render(try_dir, self.render_checked.pop(key, 0))
            except ImportError:
                pass
        outcome = read_outcome(try_dir)
        forced = infra_marker(try_dir)
        cls = classify(
            outcome,
            killed=killed_marker(try_dir).exists(),
            forced_infra=forced.exists(),
        )
        if forced.exists():
            last["cause"] = forced.read_text().strip()[:200]
        usage = (
            episode_usage(try_dir, self.cfg["pricing"]) if try_dir.exists() else None
        )
        last.update(
            status="done",
            ended_at=now_iso(),
            ended_ts=time.time(),
            exit=exit_code,
            outcome={
                k: outcome.get(k)
                for k in ("valid", "success", "termination_reason", "elapsed_s")
            },
            usage=usage,
        )
        last["class"] = cls
        self.log(
            {
                "event": "done",
                "job": job_id,
                "try": last["n"],
                "class": cls,
                "termination": outcome.get("termination_reason"),
                "exit": exit_code,
                **({"success": outcome.get("success")} if cls == "valid" else {}),
            }
        )
        final = None
        if cls == "valid":
            final = {
                "class": "valid",
                "try": last["n"],
                "success": bool(outcome.get("success")),
            }
        elif cls == "infra" and self.class_count(state, "infra") > self.infra_retries:
            final = {"class": "infra_exhausted", "try": last["n"], "success": False}
        elif (
            cls == "render" and self.class_count(state, "render") > self.render_retries
        ):
            final = {"class": "render_exhausted", "try": last["n"], "success": False}
        if final:
            state["final"] = {**final, "at": now_iso()}
        self.save(state)
        if final:
            self.log({"event": "final", "job": job_id, **final})
            self.stage.on_final(self.jobs[job_id], state)

    # -- guards ------------------------------------------------------------

    def guard(self, states: Mapping[str, Mapping[str, Any]]) -> str | None:
        """Reason to stop launching, or None."""
        if self.layout.stop.exists():
            return str((read_json(self.layout.stop) or {}).get("reason") or "stop file")
        if self.launches_so_far(states) >= self.launch_ceiling():
            write_stop(self.layout, "launch ceiling", ceiling=self.launch_ceiling())
            return "launch ceiling"
        breaker = int(self.sched.get("infra_breaker") or 0)
        if breaker > 0:
            # Only tries this runner saw end (a restart after a fix starts a fresh count);
            # cells stopped because their RLDX server died are no evidence of a systematic fault.
            done = sorted(
                (
                    t
                    for s in states.values()
                    for t in s["tries"]
                    if t.get("status") == "done"
                    and t.get("class") != "killed"
                    and not str(t.get("cause") or "").startswith("endpoint_down")
                    and float(t.get("ended_ts") or 0) >= self.started
                ),
                key=lambda t: float(t.get("ended_ts") or 0),
            )
            tail = done[-breaker:]
            if len(tail) == breaker and all(t.get("class") == "infra" for t in tail):
                write_stop(self.layout, "infrastructure breaker", consecutive=breaker)
                return "infrastructure breaker"
        ceiling = float(self.sched.get("cost_ceiling_usd") or 0)
        if ceiling > 0:
            if time.time() - self.last_cost_check >= float(self.sched["cost_check_s"]):
                self.cost_usd = total_cost(states, self.cfg["pricing"])
                self.last_cost_check = time.time()
            if self.cost_usd >= ceiling:
                write_stop(
                    self.layout,
                    "cost ceiling",
                    usd=round(self.cost_usd, 2),
                    ceiling=ceiling,
                )
                return "cost ceiling"
        return None

    def slot_limit(self) -> int:
        try:
            live = int(self.layout.concurrency.read_text().strip())
        except (OSError, ValueError):
            live = self.max_slots
        return max(0, min(live, self.max_slots))

    # -- launching ---------------------------------------------------------

    def next_job(self, states: Mapping[str, Mapping[str, Any]]) -> str | None:
        busy = {job_id for job_id, _ in self.running(states)}
        candidates = [
            self.jobs[j] for j, s in states.items() if not s["final"] and j not in busy
        ]

        # Infrastructure retries first (they keep their place), render retries last.
        def key(job: Job) -> tuple[int, int]:
            tries = states[job.job_id]["tries"]
            last_cls = tries[-1].get("class") if tries else None
            return (
                2 if last_cls == "render" else 0 if last_cls == "infra" else 1,
                job.order,
            )

        candidates.sort(key=key)
        return candidates[0].job_id if candidates else None

    def pick_endpoint(self, states: Mapping[str, Mapping[str, Any]]) -> Endpoint | None:
        load: dict[str, int] = {}
        for _, last in self.running(states):
            load[last.get("endpoint")] = load.get(last.get("endpoint"), 0) + 1
        best = None
        for ep in self.servers.endpoints:
            if not ep.up or load.get(ep.name, 0) >= ep.slots:
                continue
            free = ep.slots - load.get(ep.name, 0)
            if best is None or free > best[0]:
                best = (free, ep)
        return best[1] if best else None

    def launch(self, job_id: str, endpoint: Endpoint) -> None:
        job = self.jobs[job_id]
        state = self.state(job_id)
        n = len(state["tries"]) + 1
        try_dir = self.stage.try_dir(job, n)
        if try_dir.exists() or pids_with_arg(str(try_dir)):
            raise RuntimeError(
                f"{try_dir} already exists; refusing to launch {job_id} try {n} twice"
            )
        self.stage.prepare_try(job, n)
        argv, env = self.stage.launch(job, n, endpoint)
        entry = {
            "n": n,
            "dir": str(try_dir),
            "endpoint": endpoint.name,
            "gpu": endpoint.gpu,
            "status": "launching",
            "launched_at": now_iso(),
        }
        state["tries"].append(entry)
        self.save(state)
        console = try_dir.with_name(f"{try_dir.name}.console.log")
        console.parent.mkdir(parents=True, exist_ok=True)
        with console.open("a") as out:
            proc = subprocess.Popen(
                argv,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.children[proc.pid] = proc
        entry.update(status="running", pid=proc.pid)
        self.save(state)
        self.last_launch = time.time()
        self.log(
            {
                "event": "launch",
                "job": job_id,
                "task": job.task,
                "split": job.split,
                "try": n,
                "endpoint": endpoint.name,
                "pid": proc.pid,
            }
        )

    # -- main loop ---------------------------------------------------------

    def run(self) -> str:
        """Run until every job is final (``done``), or drained after a stop (``stopped``)."""
        self.recover()
        # Cells adopted on an endpoint whose server is gone cannot finish properly.
        for ep in self.servers.endpoints:
            if not ep.up and any(
                t.get("endpoint") == ep.name for _, t in self.running(self.all_states())
            ):
                self.stop_endpoint_cells(ep.name)
                self.servers.schedule_restart(ep.name)
        reason = None
        while True:
            self.poll()
            for name in self.servers.check():
                self.stop_endpoint_cells(name)
            states = self.all_states()
            running = self.running(states)
            if all(s["final"] for s in states.values()):
                return "done"
            reason = self.guard(states)
            if reason:
                if not running:
                    self.log(
                        {
                            "event": "stopped",
                            "reason": reason,
                            "pending": sum(
                                1 for s in states.values() if not s["final"]
                            ),
                        }
                    )
                    return "stopped"
            elif (
                len(running) < self.slot_limit()
                and time.time() - self.last_launch
                >= float(self.sched["launch_stagger_s"])
                and available_mb() >= int(self.sched["min_available_mb"])
            ):
                job_id = self.next_job(states)
                endpoint = self.pick_endpoint(states) if job_id else None
                if job_id and endpoint:
                    try:
                        self.launch(job_id, endpoint)
                    except RuntimeError as exc:
                        # Inconsistent state: stop launching, keep the running cells.
                        write_stop(
                            self.layout, "launch refused", job=job_id, error=str(exc)
                        )
                        self.log(
                            {
                                "event": "launch_refused",
                                "job": job_id,
                                "error": str(exc),
                            }
                        )
                    continue
                if (
                    job_id
                    and not running
                    and not self.servers.starting
                    and not self.servers.restart_after
                    and not any(ep.up for ep in self.servers.endpoints)
                ):
                    self.log({"event": "stopped", "reason": "no endpoint up"})
                    return "no_endpoint"
            time.sleep(float(self.sched["poll_s"]))

    def stop_endpoint_cells(self, name: str) -> None:
        """Stop the cells attached to a dead RLDX server: infrastructure, retried."""
        for job_id, last in self.running(self.all_states()):
            if last.get("endpoint") != name:
                continue
            atomic_write_text(
                infra_marker(last["dir"]), f"endpoint_down {name} {now_iso()}\n"
            )
            self.log(
                {
                    "event": "endpoint_down_stop",
                    "job": job_id,
                    "try": last["n"],
                    "endpoint": name,
                }
            )
            if last.get("pid") and pid_alive(int(last["pid"])):
                kill_group(int(last["pid"]), grace_s=float(self.sched["kill_grace_s"]))

    def kill_running(self) -> int:
        """Stop every running try (marked ``killed``: not counted, relaunched on the next run)."""
        return kill_running(
            self.layout, self.jobs, grace_s=float(self.sched["kill_grace_s"])
        )


def kill_running(layout: RunLayout, job_ids: Iterable[str], grace_s: float) -> int:
    """Mark and stop every running try of ``job_ids``; a runner scores them as ``killed``."""
    killed = 0
    for job_id in job_ids:
        state = read_json(layout.job_state(job_id)) or {"tries": []}
        last = state["tries"][-1] if state["tries"] else None
        if not last or last.get("status") not in ("launching", "running"):
            continue
        atomic_write_text(killed_marker(last["dir"]), now_iso() + "\n")
        if last.get("pid") and pid_alive(int(last["pid"])):
            kill_group(int(last["pid"]), grace_s=grace_s)
            killed += 1
    return killed


def counted_launches(states: Mapping[str, Mapping[str, Any]]) -> int:
    """Launches counted against the ceiling: every try except those the operator stopped."""
    return sum(
        1
        for s in states.values()
        for t in s["tries"]
        if (t.get("pid") or t.get("class")) and t.get("class") != "killed"
    )


def launch_ceiling(
    sched: Mapping[str, Any],
    budget_launches: int,
    jobs: int,
    infra_retries: int,
    render_retries: int = 0,
) -> int:
    """Hard ceiling on launches: the budget, every job's infrastructure retries, and one
    render retry per job (render retries beyond that share the pool).

    ``scheduler.max_total_launches`` (when > 0) replaces the computed value.
    """
    configured = int(sched.get("max_total_launches") or 0)
    if configured > 0:
        return configured
    return budget_launches + jobs * infra_retries + (jobs if render_retries > 0 else 0)


def total_cost(
    states: Mapping[str, Mapping[str, Any]], pricing: Mapping[str, float]
) -> float:
    """List-price USD over every try (finished tries use their recorded usage)."""
    usd = 0.0
    for state in states.values():
        for t in state["tries"]:
            usage = t.get("usage")
            if (
                usage is None
                and t.get("status") in ("launching", "running")
                and Path(t["dir"]).exists()
            ):
                usage = episode_usage(Path(t["dir"]), pricing)
            usd += float((usage or {}).get("usd") or 0.0)
    return usd


def acquire_lock(layout: RunLayout):
    """Hold ``runner.lock`` for the process lifetime; exits if another runner holds it."""
    layout.root.mkdir(parents=True, exist_ok=True)
    handle = layout.lock.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"another runner holds {layout.lock}")
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def lock_holder(layout: RunLayout) -> int | None:
    """PID of the runner holding the lock, or None."""
    if not layout.lock.exists():
        return None
    with layout.lock.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.seek(0)
            text = handle.read().strip()
            return int(text) if text.isdigit() else -1
        fcntl.flock(handle, fcntl.LOCK_UN)
    return None


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def progress_bar(done: int, total: int, width: int = 20) -> str:
    filled = 0 if total <= 0 else int(round(width * done / total))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {done}/{total}"


def status_rows(states: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Per split: final, running, queued, solved, infra/render retries, unfinished infra."""
    rows: dict[str, dict[str, int]] = {}
    for state in states.values():
        row = rows.setdefault(
            state["split"],
            {
                "total": 0,
                "final": 0,
                "running": 0,
                "queued": 0,
                "solved": 0,
                "infra_tries": 0,
                "render_tries": 0,
                "infra_exhausted": 0,
            },
        )
        row["total"] += 1
        tries = state["tries"]
        row["infra_tries"] += sum(1 for t in tries if t.get("class") == "infra")
        row["render_tries"] += sum(1 for t in tries if t.get("class") == "render")
        final = state.get("final")
        if final:
            row["final"] += 1
            row["solved"] += int(bool(final.get("success")))
            row["infra_exhausted"] += int(
                final.get("class") in ("infra_exhausted", "render_exhausted")
            )
        elif tries and tries[-1].get("status") in ("launching", "running"):
            row["running"] += 1
        else:
            row["queued"] += 1
    return rows


def format_status(
    title: str,
    states: Mapping[str, Mapping[str, Any]],
    *,
    cost_usd: float,
    extra_lines: Iterable[str] = (),
) -> str:
    rows = status_rows(states)
    total = sum(r["total"] for r in rows.values())
    final = sum(r["final"] for r in rows.values())
    lines = [f"{title}  {progress_bar(final, total)} tasks final"]
    lines.append(
        f"  {'split':<17} {'final':>7} {'running':>8} {'queued':>7} {'solved':>7} {'infra-retry':>12} {'infra-exhausted':>16}"
    )
    for split in [s for s in SPLITS if s in rows] + sorted(set(rows) - set(SPLITS)):
        r = rows[split]
        retries = r["infra_tries"] + r["render_tries"]
        lines.append(
            f"  {split:<17} {r['final']:>3}/{r['total']:<3} {r['running']:>8} {r['queued']:>7} "
            f"{r['solved']:>7} {retries:>12} {r['infra_exhausted']:>16}"
        )
    lines.append(f"  cost so far (list price, all launches): ${cost_usd:.2f}")
    lines.extend(f"  {line}" for line in extra_lines)
    return "\n".join(lines)


def ceil_div(a: int, b: int) -> int:
    return int(math.ceil(a / b)) if b else 0


def python_for(cfg: Mapping[str, Any]) -> str:
    venv = cfg["paths"].get("venv")
    return str(Path(venv) / "bin" / "python") if venv else sys.executable
