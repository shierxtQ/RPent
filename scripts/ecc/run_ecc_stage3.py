#!/usr/bin/env python3
"""ECC stage 3 (ecc_rule.md): per-task cloud correction memory.

Each task runs with its stage-2 call rule and the stage-1 task memory (read-only). Its
correction memory (CORR_DIR/<tag>.md) is appended to every GPT-6 senior note; Qwen never
sees it. After every episode GPT-6 reflects offline on its own interventions (trigger,
situation, guidance, what followed, outcome) and edits the entries:

- an entry counts as used only when the reflection reports that its situation arose and the
  guidance applied its fix ("applied"); merely being shown does not count;
- entries applied in a successful episode, and entries written or revised from one, become
  verified;
- an entry applied in two failed episodes without any success is dropped; a revised entry starts
  its failure count again;
- episodes whose transcript holds no planner turn (e.g. a planner timeout before the transcript
  was written) are not reflected on and leave the memory untouched;
- at most MAX_ENTRIES entries.

A task runs at most 5 sessions x 3 attempts = 15 independent episodes and stops at its first
success. The kept memory (FINAL_DIR/<tag>.md) holds only verified entries.
Resumable: episodes with a result.json are not rerun; reflections with reflection.json neither.

Usage:
  python3 scripts/ecc/run_ecc_stage3.py RUN_DIR CORR_DIR RULES_DIR MEMORY_DIR [--parallel 8] [--seed 3] [--only suite:task,...]
  python3 scripts/ecc/run_ecc_stage3.py --reflect-only EPISODE_DIR CORR_JSON   (dry test on a finished episode)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from run_ecc_stage2 import gpt_calls
from run_escalation_batch80 import INFRA_RETRIES, ROOT, TASKS, outcome, prune_npz, write_json

SESSIONS, ATTEMPTS = 5, 3
MAX_ENTRIES = 8
FIELD_CHARS = 500
FAILS_TO_DROP = 2
REFLECT_MODEL = "gpt-6-astra"
CONTEXT_BEFORE, CONTEXT_AFTER = 3, 6


# -- correction memory ------------------------------------------------------------


def load_entries(path: Path) -> list[dict]:
    return json.loads(path.read_text())["entries"] if path.exists() else []


def render(entries: list[dict]) -> str:
    lines = []
    for e in entries:
        lines.append(
            f"- [{e['id']}] ({e['status']}; trigger: {e['trigger']}) situation: {e['situation']} | "
            f"diagnosis: {e['diagnosis']} | fix: {e['fix']}"
        )
    return "\n".join(lines)


def save_entries(json_path: Path, md_path: Path, tag: str, entries: list[dict], **extra) -> None:
    write_json(json_path, {"task": tag, "entries": entries, "updated_at": datetime.now(timezone.utc), **extra})
    md_path.write_text(render(entries) + ("\n" if entries else ""), encoding="utf-8")


# -- episode digest -----------------------------------------------------------------


def _short(value, n: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = re.sub(r"\s+", " ", text)
    return text if len(text) <= n else text[:n] + "..."


def _turn(msg: dict, results: dict[str, str], n_text: int, n_arg: int, n_res: int) -> str:
    parts = []
    for p in msg.get("content") or []:
        if p.get("type") == "text" and p.get("text", "").strip():
            parts.append(f"  says: {_short(p['text'], n_text)}")
        elif p.get("type") == "tool_use":
            res = results.get(p.get("id"), "")
            parts.append(f"  call {p['name']}({_short(p.get('input'), n_arg)}) -> {_short(res, n_res)}")
    return "\n".join(parts)


def digest(out: Path) -> dict:
    """Senior interventions of one episode, with the turns around them."""
    transcripts = sorted(out.glob("transcript_*.json"))
    tr = json.loads(transcripts[-1].read_text()) if transcripts else {"messages": []}
    msgs = tr.get("messages") or []
    results = {m.get("tool_call_id"): m.get("content") for m in msgs if m.get("role") == "tool"}
    turns = [m for m in msgs if m.get("role") == "assistant"]
    rows = []
    esc = out / "escalation.jsonl"
    if esc.exists():
        rows = [json.loads(line) for line in esc.open() if line.strip()]
    seniors = [r for r in rows if r.get("role") == "senior" and "error" not in r]
    blocks = []
    for r in seniors:
        i = int(r["request"]) - 1  # request k is the k-th assistant message
        if not 0 <= i < len(turns):
            continue
        before = "\n".join(
            f" worker turn {j + 1}:\n" + _turn(turns[j], results, 200, 160, 200)
            for j in range(max(0, i - CONTEXT_BEFORE), i)
        )
        after = "\n".join(
            f" turn {j + 1}:\n" + _turn(turns[j], results, 160, 120, 160)
            for j in range(i + 1, min(len(turns), i + 1 + CONTEXT_AFTER))
        )
        blocks.append(
            f"### Intervention at turn {i + 1} (trigger: {r['reason']})\n"
            f"Before:\n{before or ' (episode start)'}\n"
            f"Your turn:\n{_turn(turns[i], results, 1500, 300, 300)}\n"
            f"After:\n{after or ' (episode ended)'}"
        )
    finish = tr.get("finish") or {}
    return {"interventions": len(seniors), "shown": len(blocks), "turns": len(turns), "finish": finish,
            "text": "\n\n".join(blocks)}


def task_language(out: Path) -> str:
    for audit in out.glob("*_s[0-9]*.json"):
        if audit.name.startswith("transcript_"):
            continue
        try:
            lang = json.loads(audit.read_text()).get("task_language")
        except ValueError:
            continue
        if lang:
            return lang
    return ""


# -- reflection ---------------------------------------------------------------------

PROMPT = """You are GPT-6, the senior (cloud) planner that a small on-device worker model (Qwen) \
calls for help while it controls a robot arm in LIBERO-PRO. You keep a correction memory for \
this one task: short entries of failure situations you diagnosed and the fix that works. It is \
shown only to you at every later intervention.

Task: {language}
Call rule of this task: {rule}
Episode outcome: {outcome} after {turns} turns; final summary: {summary}

Your interventions in this episode, with the worker turns around them:
{interventions}

Current correction memory (JSON):
{entries}

Edit the memory from this episode. Keep entries concrete: the observable situation (what the \
worker was doing, what the state/images showed), the root cause, and the exact corrective action \
(numbers, tool, order). Do not restate generic task strategy that the task memory already covers; \
record what went wrong for the worker and how to correct it. If the episode failed, say what your \
guidance missed and revise or remove entries that did not help. Prefer revising an existing entry \
over adding a near duplicate. Each field at most two sentences. At most {max_entries} entries in total.
In "applied", list the ids of existing entries whose situation actually arose in this episode and \
whose fix your guidance applied; leave out entries that were not relevant.

Reply with JSON only:
{{"applied": ["id", ...],
 "add": [{{"trigger": "plan|checkin|errors|stall|giveup|any", "situation": "...", "diagnosis": "...", "fix": "..."}}],
 "revise": [{{"id": "...", "situation": "...", "diagnosis": "...", "fix": "..."}}],
 "remove": ["id", ...],
 "note": "one sentence on what this episode taught"}}"""


def call_gpt(prompt: str) -> tuple[str, dict]:
    from openai import OpenAI

    base = os.environ["RPENT_BASE_URL"].rstrip("/")
    client = OpenAI(base_url=base if base.endswith("/v1") else base + "/v1", api_key=os.environ["RPENT_API_KEY"])
    last = None
    for attempt in range(4):
        try:
            resp = client.responses.create(model=REFLECT_MODEL, input=prompt, reasoning={"effort": "low"})
            usage = resp.usage
            return resp.output_text, {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}
        except Exception as exc:  # network / provider errors: back off and retry
            last = exc
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"reflection failed: {type(last).__name__}: {last}")


def parse_edits(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("no JSON object in reflection reply")
    edits = json.loads(match.group(0))
    return {k: edits.get(k) or [] for k in ("applied", "add", "revise", "remove")} | {"note": edits.get("note", "")}


def apply(entries: list[dict], edits: dict, episode: str, success: bool, used: set[str]) -> list[dict]:
    entries = [dict(e) for e in entries]
    # Outcome bookkeeping for entries the senior applied in this episode.
    for e in entries:
        if e["id"] in used:
            e["used_in"] = e.get("used_in", []) + [episode]
            if success:
                e["successes"] = e.get("successes", 0) + 1
                e["status"] = "verified"
            else:
                e["failures"] = e.get("failures", 0) + 1
    remove = set(edits["remove"])
    revised = {str(r.get("id")) for r in edits["revise"]}
    entries = [
        e for e in entries
        if e["id"] not in remove
        and (e["id"] in revised or not (e.get("failures", 0) >= FAILS_TO_DROP and not e.get("successes")))
    ]
    by_id = {e["id"]: e for e in entries}
    for r in edits["revise"]:
        e = by_id.get(str(r.get("id")))
        if e is None:
            continue
        for k in ("situation", "diagnosis", "fix"):
            if r.get(k):
                e[k] = str(r[k])[:FIELD_CHARS]
        e["revised_in"] = e.get("revised_in", []) + [episode]
        e["failures"] = 0  # the revised fix has not failed yet
        if success:
            e["status"] = "verified"
    stem, next_id = episode.rsplit("/", 1)[-1], 1  # ids name the writing episode: s2_a1.3
    for a in edits["add"]:
        if len(entries) >= MAX_ENTRIES:
            break
        if not all(a.get(k) for k in ("situation", "diagnosis", "fix")):
            continue
        entries.append({
            "id": f"{stem}.{next_id}", "trigger": str(a.get("trigger") or "any"),
            **{k: str(a[k])[:FIELD_CHARS] for k in ("situation", "diagnosis", "fix")},
            "status": "verified" if success else "candidate", "written_in": episode,
            "used_in": [], "successes": 0, "failures": 0,
        })
        next_id += 1
    return entries


def reflect(out: Path, entries: list[dict], rule: dict, success: bool, episode: str) -> tuple[list[dict], dict]:
    cached = out / "reflection.json"
    if cached.exists():
        rec = json.loads(cached.read_text())
        return rec["entries_after"], rec
    d = digest(out)
    rec = {"episode": episode, "success": success, "interventions": d["interventions"],
           "entries_before": entries, "used": []}
    none = {"applied": [], "add": [], "revise": [], "remove": []}
    if d["interventions"] == 0:
        rec |= {"skipped": "no senior intervention", "edits": none, "entries_after": entries}
    elif d["shown"] == 0:
        rec |= {"skipped": f"no planner turn in the transcript ({d['turns']} turns)", "edits": none,
                "entries_after": entries}
    else:
        prompt = PROMPT.format(
            language=task_language(out) or "(see task memory)", rule=json.dumps(rule),
            outcome="SUCCESS" if success else "FAILURE", turns=d["turns"],
            summary=_short((d["finish"] or {}).get("summary", ""), 600), interventions=d["text"][:40000],
            entries=json.dumps([{k: e[k] for k in ("id", "status", "trigger", "situation", "diagnosis", "fix")}
                                for e in entries], ensure_ascii=False, indent=1),
            max_entries=MAX_ENTRIES)
        reply, usage = call_gpt(prompt)
        try:
            edits = parse_edits(reply)
        except ValueError as exc:
            edits = none | {"note": f"unparsed reply: {exc}"}
        used = {str(i) for i in edits["applied"]} & {e["id"] for e in entries}
        rec |= {"prompt_chars": len(prompt), "usage": usage, "reply": reply, "edits": edits, "used": sorted(used),
                "entries_after": apply(entries, edits, episode, success, used)}
    rec["at"] = datetime.now(timezone.utc)
    write_json(cached, rec)
    return rec["entries_after"], rec


# -- harness --------------------------------------------------------------------------


def run_episode(out: Path, suite: str, task: int, seed: int, env_extra: dict, label: str) -> dict:
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    for attempt in range(INFRA_RETRIES + 1):
        out.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "SUITE": suite, "TASK": str(task), "SEED": str(seed), "OUTPUT_DIR": str(out),
               "VLA_ENDPOINT": "http://127.0.0.1:51157", "SAM3_ENDPOINT": "http://127.0.0.1:56897",
               "RPENT_SHARED_SIM": "1", **env_extra}
        env.pop("PLANNER_MODE", None)
        started = time.time()
        with open(out / f"stdout.attempt{attempt}.log", "w") as log:
            proc = subprocess.Popen([str(ROOT / "run_ecc_rules.sh")], env=env, stdout=log,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            write_json(out / "launch.json", {"suite": suite, "task": task, "seed": seed, "pid": proc.pid,
                                             "attempt": attempt, "started_at": datetime.now(timezone.utc)})
            print(f"start {label} attempt={attempt} pid={proc.pid}", flush=True)
            rc = proc.wait()
        res = {"suite": suite, "task": task, "seed": seed, "returncode": rc, "attempt": attempt,
               "wall_s": round(time.time() - started, 1), **outcome(out)}
        if res["started"] or attempt == INFRA_RETRIES:
            break
        print(f"infra-retry {label}: {res['last_error']}", flush=True)
        for stale in out.glob("transcript_*.json"):
            stale.rename(stale.with_suffix(f".attempt{attempt}.json"))
        time.sleep(10)
    res["gpt_calls"] = gpt_calls(out)
    write_json(out / "result.json", res)
    prune_npz(out)
    return res


def run_task(args, index: int, suite: str, task: int, slots: threading.Semaphore) -> None:
    tag = f"{suite.replace('libero_', '')}_t{task}"
    final_json = args.corr_dir / "final" / f"{tag}.json"
    if final_json.exists():
        return
    work_json, work_md = args.corr_dir / "work" / f"{tag}.json", args.corr_dir / "work" / f"{tag}.md"
    rule = json.loads((args.rules_dir / f"{tag}.json").read_text())["rule"]
    task_dir = args.run_dir / f"{index:02d}_{suite}_task{task}_seed{args.seed}"
    history = []
    with slots:
        entries = load_entries(work_json)
        if not work_md.exists():
            save_entries(work_json, work_md, tag, entries)
        for n in range(SESSIONS * ATTEMPTS):
            name = f"s{n // ATTEMPTS + 1}_a{n % ATTEMPTS + 1}"
            out = task_dir / name
            out.mkdir(parents=True, exist_ok=True)
            if not (out / "corrections_used.md").exists():
                shutil.copy2(work_md, out / "corrections_used.md")
            entries = json.loads((out / "reflection.json").read_text())["entries_before"] \
                if (out / "reflection.json").exists() else entries
            res = run_episode(out, suite, task, args.seed,
                              {"RULES_DIR": str(args.rules_dir), "MEMORY_DIR": args.memory_dir,
                               "RPENT_CLOUD_CORRECTIONS": str(out / "corrections_used.md")},
                              f"{index:02d} {tag} {name}")
            try:
                entries, rec = reflect(out, entries, rule, res["success"], f"{tag}/{name}")
            except Exception as exc:  # keep the episode; the memory stays as it was
                print(f"reflect-error {index:02d} {tag} {name}: {exc}", flush=True)
                rec = {"error": str(exc)[:300]}
            save_entries(work_json, work_md, tag, entries)
            history.append({"episode": name, "success": res["success"], "gpt_calls": res["gpt_calls"],
                            "turns": res.get("turns"), "finish_status": res.get("finish_status"),
                            "last_error": res.get("last_error"), "reflected": "usage" in rec,
                            "entries": len(entries), "verified": sum(e["status"] == "verified" for e in entries)})
            print(f"done  {index:02d} {tag} {name} success={res['success']} gpt_calls={res['gpt_calls']} "
                  f"entries={len(entries)} verified={history[-1]['verified']}", flush=True)
            if res["success"]:
                break
    kept = [e for e in entries if e["status"] == "verified"]
    final_json.parent.mkdir(parents=True, exist_ok=True)
    save_entries(final_json, args.corr_dir / "final" / f"{tag}.md", tag, kept,
                 solved=any(h["success"] for h in history), episodes=len(history),
                 gpt_calls_in_episodes=sum(h["gpt_calls"] for h in history),
                 reflections=sum(h["reflected"] for h in history), seed=args.seed, history=history)
    write_json(task_dir / "stage3.json", json.loads(final_json.read_text()))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--reflect-only":
        out, corr = Path(sys.argv[2]), Path(sys.argv[3])
        res = json.loads((out / "result.json").read_text())
        rule = {"note": "dry run"}
        entries, rec = reflect(out, load_entries(corr), rule, res["success"], f"dry/{out.name}")
        print(json.dumps({k: rec.get(k) for k in ("interventions", "usage", "edits")}, indent=1, ensure_ascii=False))
        print(render(entries))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("corr_dir", type=Path)
    ap.add_argument("rules_dir", type=Path)
    ap.add_argument("memory_dir")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--only", default=None, help="comma list of suite:task")
    args = ap.parse_args()
    only = None
    if args.only:
        only = {(s.split(":")[0], int(s.split(":")[1])) for s in args.only.split(",")}
    for sub in ("work", "final"):
        (args.corr_dir / sub).mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.run_dir / "batch.json", {
        "stage": "ECC stage 3 cloud correction memory", "seed": args.seed, "parallel": args.parallel,
        "sessions": SESSIONS, "attempts": ATTEMPTS, "max_entries": MAX_ENTRIES, "memory_dir": args.memory_dir,
        "rules_dir": str(args.rules_dir), "corr_dir": str(args.corr_dir), "reflect_model": REFLECT_MODEL,
        "started_at": datetime.now(timezone.utc), "only": sorted(only) if only else None})
    slots = threading.Semaphore(args.parallel)
    threads = []
    for index, (suite, task) in enumerate(TASKS, 1):
        if only is not None and (suite, task) not in only:
            continue
        th = threading.Thread(target=run_task, args=(args, index, suite, task, slots))
        th.start()
        threads.append(th)
        time.sleep(0.05)
    for th in threads:
        th.join()
    finals = [json.loads(p.read_text()) for p in sorted((args.corr_dir / "final").glob("*.json"))]
    write_json(args.run_dir / "summary.json", {
        "tasks": len(finals), "solved": sum(f["solved"] for f in finals),
        "episodes": sum(f["episodes"] for f in finals),
        "gpt_calls_in_episodes": sum(f["gpt_calls_in_episodes"] for f in finals),
        "reflections": sum(f["reflections"] for f in finals),
        "verified_entries": sum(len(f["entries"]) for f in finals),
        "tasks_with_entries": sum(bool(f["entries"]) for f in finals)})
    print((args.run_dir / "summary.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
