"""Orchestrator — runs the evolution loop.

This module is the rollback guardian and is IMMUTABLE infrastructure:
the evolving agent must never edit it. Responsibilities:

- load configuration from config.yaml
- snapshot generations into artifacts/gen_N/
- run candidate agents in Docker subprocesses (never imported directly)
- proxy LLM calls from the sandbox so cost tracking is ground truth
- smoke-test candidates and gate accept/reject by val split score
- enforce stopping criteria (plateau or generation cap)
- log every step to artifacts/runs/<run_id>/log.jsonl
"""

import argparse
import datetime as _dt
import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable, Optional

import yaml


# --- config -----------------------------------------------------------------

def load_config(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# --- snapshot management ----------------------------------------------------

def bootstrap_gen0(config: dict, project_root: Path) -> Path:
    """Create artifacts/gen_0/ from the live agent/ + tools/ trees if missing.

    Refuses to bootstrap from an empty/missing source tree to prevent silent
    gen_0 generation from nothing.
    """
    gen0 = project_root / config["paths"]["artifacts_dir"] / "gen_0"
    if gen0.exists():
        return gen0

    agent_src = project_root / config["paths"]["agent_dir"]
    tools_src = project_root / config["paths"]["tools_dir"]

    missing = [str(p.relative_to(project_root))
               for p in (agent_src, tools_src) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot bootstrap gen_0: missing source dirs: {', '.join(missing)}. "
            "Both agent/ and tools/ must exist with seed code before the "
            "orchestrator runs."
        )

    gen0.parent.mkdir(parents=True, exist_ok=True)
    gen0.mkdir()
    shutil.copytree(agent_src, gen0 / "agent")
    shutil.copytree(tools_src, gen0 / "tools")
    return gen0


def copy_snapshot(src_dir: Path, dst_dir: Path) -> None:
    if dst_dir.exists():
        raise FileExistsError(f"snapshot already exists: {dst_dir}")
    shutil.copytree(src_dir, dst_dir)


# --- dataset loading --------------------------------------------------------

def _load_split(config: dict, project_root: Path, split_name: str) -> list:
    p = project_root / config["paths"]["data_dir"] / f"{split_name}.jsonl"
    if not p.exists():
        raise FileNotFoundError(
            f"split file not found: {p}. Build the dataset (python -m eval.dataset) "
            "to create train/val/test splits before running the orchestrator."
        )
    tasks = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


# --- LLM proxy --------------------------------------------------------------

def _serialize_tool_calls(tc_list) -> Optional[list]:
    """Convert OpenAI ToolCall SDK objects into JSON-friendly dicts."""
    if not tc_list:
        return None
    out = []
    for tc in tc_list:
        out.append({
            "id": tc.id,
            "type": getattr(tc, "type", "function"),
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            },
        })
    return out


def make_llm_handler(
    client,
    default_model: str,
    timeout_s: int = 60,
    llm_logger: Optional["LLMCallLogger"] = None,
) -> Callable[[dict], dict]:
    """Return a callable that turns an llm_request envelope into a response.

    The handler is the ground truth for token tracking — it sees every LLM
    call's token counts directly from the OpenAI response. If llm_logger is
    provided, every call (success or failure) is appended to llm_calls.jsonl.

    Supports OpenAI tool calling: if `tools` is present in the envelope it is
    forwarded to the API and `tool_calls` are returned in the response envelope.
    """
    def handle(env: dict) -> dict:
        model = env.get("model") or default_model
        call_id = str(uuid.uuid4())
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        t0 = time.monotonic()

        request_dict = {"model": model, "messages": env["messages"]}
        if env.get("response_format"):
            request_dict["response_format"] = env["response_format"]
        if env.get("tools"):
            request_dict["tools"] = env["tools"]
        if env.get("tool_choice"):
            request_dict["tool_choice"] = env["tool_choice"]

        kwargs = {"model": model, "messages": env["messages"], "timeout": timeout_s}
        if env.get("response_format"):
            kwargs["response_format"] = env["response_format"]
        if env.get("tools"):
            kwargs["tools"] = env["tools"]
        if env.get("tool_choice"):
            kwargs["tool_choice"] = env["tool_choice"]

        error = None
        response_dict = None
        usage = None
        try:
            response = client.chat.completions.create(**kwargs)
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": (
                    response.usage.total_tokens
                    if hasattr(response.usage, "total_tokens") and response.usage.total_tokens is not None
                    else response.usage.prompt_tokens + response.usage.completion_tokens
                ),
                "model": response.model,
            }
            msg = response.choices[0].message
            content = getattr(msg, "content", None)
            tool_calls = _serialize_tool_calls(getattr(msg, "tool_calls", None))
            response_dict = {"content": content, "tool_calls": tool_calls}
            result = {
                "_kind": "llm_response",
                "id": env["id"],
                "ok": True,
                "content": content,
                "tool_calls": tool_calls,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "model": response.model,
                },
            }
        except Exception as e:
            error = f"{e.__class__.__name__}: {e}"
            result = {
                "_kind": "llm_response",
                "id": env["id"],
                "ok": False,
                "error": error,
            }

        if llm_logger is not None:
            duration_ms = int((time.monotonic() - t0) * 1000)
            llm_logger.log(
                call_id=call_id,
                timestamp_iso=ts,
                purpose=env.get("purpose"),
                task_id=env.get("task_id"),
                step_in_task=env.get("step_in_task", 0),
                request=request_dict,
                response=response_dict,
                usage=usage,
                duration_ms=duration_ms,
                error=error,
            )

        return result
    return handle


# --- docker invocation ------------------------------------------------------

def _docker_run_cmd(gen_dir: Path, work_dir: Path, config: dict, container_name: str) -> list:
    sb = config["sandbox"]
    image = sb.get("image", "stem-agent-sandbox")
    cmd = [
        "docker", "run", "--rm", "-i",
        "--name", container_name,
        "--network", "none" if not sb.get("network") else "bridge",
        "--memory", f"{sb['memory_mb']}m",
        "--cpus", str(sb["cpu_cores"]),
        "-v", f"{gen_dir.resolve()}:/agent:ro",
        "-v", f"{work_dir.resolve()}:/work",
    ]
    cmd.append(image)
    return cmd


def _parse_runner_markers(stderr_lines: list) -> dict:
    out = {}
    for line in stderr_lines:
        if not line.startswith("RUNNER:"):
            continue
        rest = line[len("RUNNER:"):]
        parts = rest.split()
        if not parts:
            continue
        kind = parts[0]
        fields = {}
        for p in parts[1:]:
            if "=" in p:
                k, v = p.split("=", 1)
                fields[k] = v
        out[kind] = fields
    return out


def _run_task_protocol(
    send_line: Callable[[str], None],
    recv_line: Callable[[float], Optional[str]],
    task: dict,
    llm_handler: Callable[[dict], dict],
    timeout_s: float,
    log_event: Optional[Callable[..., None]] = None,
) -> dict:
    """Drive the line-delimited JSON RPC protocol with the given streams.

    Returns the final result dict (possibly with `error`). Tracks the
    orchestrator-side LLM usage as ground truth and overrides any
    `usage` field returned by the agent.
    """
    deadline = time.monotonic() + timeout_s
    task_usage = {"prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0}
    task_id = task.get("id") if isinstance(task, dict) else None

    try:
        send_line(json.dumps({"_kind": "task", "task": task}))
    except (BrokenPipeError, OSError) as e:
        return {"error": f"host: failed to write task envelope: {e}",
                "task_id": task_id, "usage": task_usage}

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {"error": "host: protocol deadline exceeded",
                    "task_id": task_id, "usage": task_usage}
        line = recv_line(remaining)
        if line is None:
            return {"error": "host: container closed stdout before result",
                    "task_id": task_id, "usage": task_usage}
        try:
            env = json.loads(line)
        except json.JSONDecodeError:
            if log_event:
                log_event("protocol_garbage", line=line[:200])
            continue
        kind = env.get("_kind") if isinstance(env, dict) else None
        if kind == "llm_request":
            response_env = llm_handler(env)
            if response_env.get("ok"):
                u = response_env.get("usage", {})
                task_usage["prompt_tokens"] += int(u.get("prompt_tokens", 0))
                task_usage["completion_tokens"] += int(u.get("completion_tokens", 0))
                task_usage["n_calls"] += 1
            try:
                send_line(json.dumps(response_env))
            except (BrokenPipeError, OSError) as e:
                return {"error": f"host: failed to write llm_response: {e}",
                        "task_id": task_id, "usage": task_usage}
        elif kind == "result":
            result = env.get("result", {}) or {}
            if "task_id" not in result:
                result["task_id"] = task_id
            # Orchestrator-tracked usage is ground truth.
            result["usage"] = task_usage
            return result
        else:
            if log_event:
                log_event("protocol_unknown_kind", kind=kind)


def run_one_task(
    gen_dir: Path,
    task: dict,
    config: dict,
    llm_handler: Optional[Callable[[dict], dict]] = None,
    work_dir: Optional[Path] = None,
    log_event: Optional[Callable[..., None]] = None,
) -> dict:
    """Run one task in the sandbox, proxying LLM calls back to the host."""
    work_owner = False
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="stem_work_"))
        work_owner = True

    if llm_handler is None:
        try:
            from openai import OpenAI
        except ImportError as e:
            return {"error": f"host: openai not installed: {e}", "task_id": task.get("id")}
        llm_handler = make_llm_handler(OpenAI(), config["model"]["name"])

    container_name = f"stem-{uuid.uuid4().hex[:12]}"
    cmd = _docker_run_cmd(gen_dir, work_dir, config, container_name)
    timeout_s = config["sandbox"]["wall_clock_timeout_s"]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
    except FileNotFoundError as e:
        if work_owner:
            shutil.rmtree(work_dir, ignore_errors=True)
        return {"error": f"host: docker not available: {e}", "task_id": task.get("id")}

    stderr_buf: list = []

    def _drain_stderr():
        try:
            for line in proc.stderr:
                stderr_buf.append(line.rstrip("\n"))
        except Exception:
            pass

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    def send_line(s: str) -> None:
        proc.stdin.write(s + "\n")
        proc.stdin.flush()

    def recv_line(remaining: Optional[float]) -> Optional[str]:
        rlist, _, _ = select.select([proc.stdout], [], [], remaining)
        if not rlist:
            return None
        line = proc.stdout.readline()
        if not line:
            return None
        return line.rstrip("\n")

    try:
        result = _run_task_protocol(
            send_line, recv_line, task, llm_handler,
            timeout_s=timeout_s + 10,  # host-side hard cutoff
            log_event=log_event,
        )
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", container_name], capture_output=True)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        stderr_thread.join(timeout=2)
        if work_owner:
            shutil.rmtree(work_dir, ignore_errors=True)

    markers = _parse_runner_markers(stderr_buf)
    if log_event:
        if "start" in markers:
            log_event(
                "task_run",
                task_id=task.get("id"),
                container=container_name,
                runner_host=markers["start"].get("host"),
                runner_py=markers["start"].get("py"),
                runner_start_ts=markers["start"].get("ts"),
                runner_end_rc=markers.get("end", {}).get("rc"),
                n_llm_calls=result.get("usage", {}).get("n_calls", 0),
            )
        else:
            log_event(
                "task_run_warning",
                task_id=task.get("id"),
                reason="no RUNNER:start marker on stderr",
                stderr_tail=stderr_buf[-10:],
            )

    return result


# --- benchmarking & scoring ------------------------------------------------

def _f1_for_class(per_task: list, cls: str) -> float:
    tp = sum(1 for t in per_task if t["expected"] == cls and t["predicted"] == cls)
    fp = sum(1 for t in per_task if t["expected"] != cls and t["predicted"] == cls)
    fn = sum(1 for t in per_task if t["expected"] == cls and t["predicted"] != cls)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _macro_f1(per_task: list) -> float:
    return (_f1_for_class(per_task, "vulnerable") + _f1_for_class(per_task, "safe")) / 2


def _accuracy(per_task: list) -> float:
    if not per_task:
        return 0.0
    return sum(1 for t in per_task if t["predicted"] == t["expected"]) / len(per_task)


def _estimate_cost_usd(usage: dict, config: dict) -> float:
    model = config["model"]["name"]
    pricing = config.get("pricing", {}).get(model)
    if not pricing:
        return 0.0
    return (
        usage.get("prompt_tokens", 0) * pricing["input"] / 1_000_000
        + usage.get("completion_tokens", 0) * pricing["output"] / 1_000_000
    )


def _normalize_label(value):
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v or None


def aggregate(results: list, config: dict) -> dict:
    per_task = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0}
    cost_total = 0.0
    errors = []
    for r in results:
        task_id = r.get("task_id")
        expected = _normalize_label(r.get("expected"))
        u = r.get("usage") or {}
        usage_total["prompt_tokens"] += int(u.get("prompt_tokens", 0))
        usage_total["completion_tokens"] += int(u.get("completion_tokens", 0))
        usage_total["n_calls"] += int(u.get("n_calls", 0))
        cost_total += _estimate_cost_usd(u, config)
        if "error" in r:
            errors.append({"task_id": task_id, "error": r["error"]})
            per_task.append({
                "task_id": task_id,
                "expected": expected,
                "predicted": None,
                "ok": False,
                "errored": True,
                "raw": None,
                "error": r.get("error"),
            })
            continue
        predicted = _normalize_label(r.get("label"))
        per_task.append({
            "task_id": task_id,
            "expected": expected,
            "predicted": predicted,
            "ok": predicted is not None and predicted == expected,
            "errored": False,
            "raw": r.get("raw"),
            "error": None,
        })
    return {
        "macro_f1": _macro_f1(per_task),
        "accuracy": _accuracy(per_task),
        "per_task": per_task,
        "usage": usage_total,
        "cost_usd": cost_total,
        "errors": errors,
        "n_tasks": len(per_task),
        "n_errors": len(errors),
    }


def run_candidate(
    gen_dir: Path,
    split: list,
    config: dict,
    llm_handler: Optional[Callable[[dict], dict]] = None,
    log_event: Optional[Callable[..., None]] = None,
) -> dict:
    """Run a candidate over a list of tasks and return aggregated scores.

    Strips ground-truth labels before sending to the runner, then
    re-attaches them on the host side for scoring.
    """
    results = []
    for task in split:
        task_for_agent = {k: v for k, v in task.items() if k != "label"}
        r = run_one_task(gen_dir, task_for_agent, config,
                         llm_handler=llm_handler, log_event=log_event)
        r["expected"] = task.get("label")
        results.append(r)
        if log_event:
            if "error" in r:
                log_event("error_task", task_id=task.get("id"), error=r["error"])
            elif _normalize_label(r.get("label")) is None:
                log_event("parse_failed", task_id=task.get("id"),
                          raw=(r.get("raw") or "")[:500])
    return aggregate(results, config)


# --- gates ------------------------------------------------------------------

def smoke_test(
    gen_dir: Path,
    config: dict,
    task: Optional[dict] = None,
    project_root: Optional[Path] = None,
    llm_handler: Optional[Callable[[dict], dict]] = None,
    log_event: Optional[Callable[..., None]] = None,
) -> bool:
    """Run gen on one task; pass iff the result has no `error` field."""
    if task is None:
        if project_root is None:
            raise ValueError("smoke_test needs either a task or project_root")
        train = _load_split(config, project_root, "train")
        if not train:
            if log_event:
                log_event("smoke_skip", reason="empty train split")
            return False
        task = train[0]
    task_for_agent = {k: v for k, v in task.items() if k != "label"}
    result = run_one_task(gen_dir, task_for_agent, config,
                          llm_handler=llm_handler, log_event=log_event)
    if "error" in result:
        if log_event:
            log_event("smoke_fail", task_id=result.get("task_id"), error=result["error"])
        return False
    if log_event:
        log_event("smoke_pass", task_id=result.get("task_id"))
    return True


def validation_gate(parent_macro_f1: float, candidate_macro_f1: float) -> bool:
    """Strict improvement: candidate must beat parent. Ties don't pass."""
    return candidate_macro_f1 > parent_macro_f1


# --- run logging ------------------------------------------------------------

class RunLogger:
    """Append-only JSONL logger keyed to a single run dir."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.log_path = run_dir / "log.jsonl"
        self._gen: Optional[int] = None

    def set_gen(self, gen: Optional[int]) -> None:
        self._gen = gen

    def __call__(self, event: str, **fields) -> None:
        rec = {
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "event": event,
            "gen": fields.pop("gen", self._gen),
        }
        rec.update(fields)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")


# --- LLM call logger --------------------------------------------------------

class LLMCallLogger:
    """Per-run, thread-safe JSONL logger for every LLM call through the gateway."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / "llm_calls.jsonl"
        self._lock = threading.Lock()
        self._gen: Optional[int] = None

    def set_gen(self, gen: Optional[int]) -> None:
        self._gen = gen

    def log(
        self, *,
        call_id: str,
        timestamp_iso: str,
        purpose: Optional[str],
        task_id: Optional[str],
        step_in_task: int,
        request: dict,
        response: Optional[dict],
        usage: Optional[dict],
        duration_ms: int,
        error: Optional[str],
    ) -> None:
        rec = {
            "call_id": call_id,
            "timestamp_iso": timestamp_iso,
            "purpose": purpose,
            "generation": self._gen,
            "task_id": task_id,
            "step_in_task": step_in_task,
            "request": request,
            "response": response,
            "usage": usage,
            "duration_ms": duration_ms,
            "error": error,
        }
        with self._lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")


# --- trajectory selection & proposal helpers --------------------------------

_MUTABLE_DIRS = ("agent", "tools")


def _select_trajectories(
    per_task: list,
    task_code_map: dict,
    max_failures: int = 3,
    max_successes: int = 2,
) -> list:
    """Pick up to N failures (errors first, then wrong predictions) and M successes.

    Each returned entry is the per_task dict augmented with the input `code`
    snippet from the dataset. The orchestrator owns this logic so reflect.py
    just consumes what it's given.
    """
    failures = []
    successes = []
    for t in per_task:
        entry = dict(t)
        entry["code"] = task_code_map.get(t.get("task_id"), "")
        if t.get("errored") or not t.get("ok"):
            failures.append(entry)
        else:
            successes.append(entry)

    failures.sort(key=lambda e: (0 if e.get("errored") else 1, str(e.get("task_id"))))
    successes.sort(key=lambda e: str(e.get("task_id")))
    return failures[:max_failures] + successes[:max_successes]


def _read_mutable_files(gen_dir: Path) -> dict:
    """Read bytes for all files the evolving agent may change.

    Multi-edit proposals can create/delete arbitrary files under agent/ and
    tools/, so no-op detection has to compare the whole mutable tree rather
    than only agent.py, prompt.txt, and tools/base.py.
    """
    out = {}
    for dirname in _MUTABLE_DIRS:
        root = gen_dir / dirname
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file():
                out[str(p.relative_to(gen_dir))] = p.read_bytes()
    return out


def _proposal_changes(proposal: dict) -> list:
    if not isinstance(proposal, dict):
        return []
    if "changes" in proposal:
        changes = proposal.get("changes")
        return changes if isinstance(changes, list) else []
    if "kind" in proposal:
        return [proposal]
    return []


def _proposal_kinds(proposal: dict) -> list:
    kinds = []
    for change in _proposal_changes(proposal):
        if isinstance(change, dict) and isinstance(change.get("kind"), str):
            kinds.append(change["kind"])
    return kinds


def _proposal_kind_label(proposal: dict) -> str:
    kinds = _proposal_kinds(proposal)
    if not kinds:
        return "none"
    if len(kinds) == 1:
        return kinds[0]
    return "bundle"


def _proposal_intent(proposal: dict) -> str:
    if isinstance(proposal, dict) and proposal.get("intent") == "halt":
        return "halt"
    return "iterate"


def _proposal_rationale(proposal: dict) -> str:
    if not isinstance(proposal, dict):
        return ""
    rationale = proposal.get("rationale")
    if isinstance(rationale, str):
        return rationale
    details = proposal.get("details") or {}
    fallback = details.get("rationale")
    return fallback if isinstance(fallback, str) else ""


def _rationale_cites_failure(proposal: dict, trajectories: list) -> bool:
    """True iff the proposal's rationale mentions any failure task_id shown."""
    rationale = _proposal_rationale(proposal)
    if not rationale:
        return False
    failure_ids = {
        str(t.get("task_id"))
        for t in trajectories
        if t.get("errored") or not t.get("ok")
    }
    return any(tid in rationale for tid in failure_ids if tid)


# --- main loop --------------------------------------------------------------

def _new_run_dir(config: dict, project_root: Path) -> tuple:
    runs_dir = project_root / config["paths"]["runs_dir"]
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = "run_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / run_id
    run_dir.mkdir()
    return run_id, run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Stem Agent orchestrator")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_config(config_path)

    # Load .env so the OpenAI client picks up OPENAI_API_KEY.
    try:
        from dotenv import load_dotenv
        load_dotenv(project_root / ".env")
    except ImportError:
        pass

    run_id, run_dir = _new_run_dir(config, project_root)
    shutil.copy(config_path, run_dir / "config.snapshot.yaml")

    log = RunLogger(run_dir)
    llm_log = LLMCallLogger(run_dir)
    log("start_run", run_id=run_id, config_path=str(config_path))

    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception as e:
        log("fatal_error", error=f"OpenAI client init failed: {e}")
        sys.exit(1)

    try:
        log.set_gen(0)
        llm_log.set_gen(0)
        handler = make_llm_handler(client, config["model"]["name"], llm_logger=llm_log)
        gen0 = bootstrap_gen0(config, project_root)
        log("bootstrap_gen0", gen_dir=str(gen0))

        val_split = _load_split(config, project_root, "val")
        log("eval_start", split="val", n_tasks=len(val_split))
        gen0_score = run_candidate(gen0, val_split, config, llm_handler=handler, log_event=log)
        log("eval_complete", split="val",
            macro_f1=gen0_score["macro_f1"],
            accuracy=gen0_score["accuracy"],
            n_errors=gen0_score["n_errors"],
            cost_usd=gen0_score["cost_usd"],
            tokens=gen0_score["usage"])

        parent_dir = gen0
        parent_score = gen0_score
        parent_macro_f1 = gen0_score["macro_f1"]
        best_macro_f1 = parent_macro_f1
        plateau = 0
        total_cost = gen0_score["cost_usd"]
        total_tokens = dict(gen0_score["usage"])
        task_code_map = {t.get("id"): t.get("code", "") for t in val_split}
        plateau_limit = config["stopping"]["plateau_generations"]

        from growth.reflect import reflect
        from growth.apply import apply_proposal

        def _bump_plateau_or_stop(stage_reason: str) -> bool:
            """Increment plateau, return True iff caller should break out of loop."""
            nonlocal plateau, stop_reason
            plateau += 1
            if plateau >= plateau_limit:
                stop_reason = "plateau"
                log("stop", reason=stop_reason, plateau=plateau, after=stage_reason)
                return True
            return False

        stop_reason = "max_generations"
        for gen_idx in range(1, config["budget"]["max_generations"] + 1):
            log.set_gen(gen_idx)
            llm_log.set_gen(gen_idx)
            log("start_generation",
                parent_macro_f1=parent_macro_f1,
                best_macro_f1=best_macro_f1,
                plateau=plateau,
                total_cost=total_cost)

            candidate_dir = project_root / config["paths"]["artifacts_dir"] / f"gen_{gen_idx}"
            copy_snapshot(parent_dir, candidate_dir)
            log("snapshot_copied", src=str(parent_dir), dst=str(candidate_dir))

            trajectories = _select_trajectories(
                parent_score.get("per_task", []), task_code_map,
            )
            log("trajectories_selected",
                n_failures=sum(1 for t in trajectories
                               if t.get("errored") or not t.get("ok")),
                n_successes=sum(1 for t in trajectories
                                if not t.get("errored") and t.get("ok")),
                task_ids=[t.get("task_id") for t in trajectories])

            # Reflection runs at the generation boundary; mark its log entry
            # with generation=null so it's not counted under any one gen.
            llm_log.set_gen(None)
            proposal = None
            try:
                proposal = reflect(
                    trajectories, str(parent_dir),
                    llm_handler=handler, config=config,
                    score=parent_score, gen_idx=gen_idx - 1,
                )
            except ValueError as e:
                log("reflect_parse_error", error=str(e))
            except Exception as e:
                log("reflect_error", error=f"{e.__class__.__name__}: {e}")
                shutil.rmtree(candidate_dir)
                stop_reason = "reflect_error"
                llm_log.set_gen(gen_idx)
                break
            finally:
                llm_log.set_gen(gen_idx)

            if proposal is None:
                log("reflect_no_proposal")
                shutil.rmtree(candidate_dir)
                if _bump_plateau_or_stop("reflect_no_proposal"):
                    break
                continue

            proposal_kind = _proposal_kind_label(proposal)
            proposal_kinds = _proposal_kinds(proposal)
            proposal_intent = _proposal_intent(proposal)
            proposal_rationale = _proposal_rationale(proposal)

            log("proposal",
                kind=proposal_kind,
                kinds=proposal_kinds,
                intent=proposal_intent,
                rationale=proposal_rationale[:300])

            if not _rationale_cites_failure(proposal, trajectories):
                log("reflect_rationale_unjustified",
                    kind=proposal_kind,
                    kinds=proposal_kinds,
                    intent=proposal_intent,
                    rationale=proposal_rationale[:300])

            before_files = _read_mutable_files(candidate_dir)
            try:
                apply_proposal(proposal, str(candidate_dir))
            except ValueError as e:
                log("proposal_invalid",
                    kind=proposal_kind,
                    kinds=proposal_kinds,
                    intent=proposal_intent,
                    error=str(e))
                shutil.rmtree(candidate_dir)
                if _bump_plateau_or_stop("proposal_invalid"):
                    break
                continue
            except Exception as e:
                log("apply_error", error=f"{e.__class__.__name__}: {e}")
                shutil.rmtree(candidate_dir)
                if _bump_plateau_or_stop("apply_error"):
                    break
                continue

            halt_requested = proposal_intent == "halt"
            after_files = _read_mutable_files(candidate_dir)
            if before_files == after_files and not halt_requested:
                log("proposal_noop", kind=proposal_kind, kinds=proposal_kinds)
                shutil.rmtree(candidate_dir)
                if _bump_plateau_or_stop("proposal_noop"):
                    break
                continue

            if halt_requested:
                log("agent_halt",
                    stage="requested",
                    rationale=proposal_rationale[:300],
                    kinds=proposal_kinds)

            if not smoke_test(candidate_dir, config, task=val_split[0],
                              llm_handler=handler, log_event=log):
                log("gate_reject", stage="smoke")
                shutil.rmtree(candidate_dir)
                if halt_requested:
                    stop_reason = "agent_halt"
                    log("stop", reason=stop_reason, after="smoke_fail")
                    break
                if _bump_plateau_or_stop("smoke_fail"):
                    break
                continue

            cand_score = run_candidate(candidate_dir, val_split, config,
                                       llm_handler=handler, log_event=log)
            total_cost += cand_score["cost_usd"]
            total_tokens["prompt_tokens"] += cand_score["usage"]["prompt_tokens"]
            total_tokens["completion_tokens"] += cand_score["usage"]["completion_tokens"]
            total_tokens["n_calls"] += cand_score["usage"]["n_calls"]
            log("eval_complete", split="val",
                macro_f1=cand_score["macro_f1"],
                accuracy=cand_score["accuracy"],
                n_errors=cand_score["n_errors"],
                cost_usd=cand_score["cost_usd"],
                tokens=cand_score["usage"])

            accepted = False
            if validation_gate(parent_macro_f1, cand_score["macro_f1"]):
                log("gate_accept",
                    parent_f1=parent_macro_f1,
                    candidate_f1=cand_score["macro_f1"])
                accepted = True
                parent_dir = candidate_dir
                parent_score = cand_score
                parent_macro_f1 = cand_score["macro_f1"]
                if cand_score["macro_f1"] > best_macro_f1:
                    best_macro_f1 = cand_score["macro_f1"]
                    plateau = 0
                else:
                    plateau += 1
            else:
                log("gate_reject", stage="val",
                    parent_f1=parent_macro_f1,
                    candidate_f1=cand_score["macro_f1"])
                plateau += 1

            if halt_requested:
                stop_reason = "agent_halt"
                log("stop",
                    reason=stop_reason,
                    accepted=accepted,
                    parent_f1=parent_macro_f1,
                    candidate_f1=cand_score["macro_f1"])
                break

            if plateau >= plateau_limit:
                stop_reason = "plateau"
                log("stop", reason=stop_reason, plateau=plateau)
                break

        log("end_run",
            stop_reason=stop_reason,
            parent_macro_f1=parent_macro_f1,
            best_macro_f1=best_macro_f1,
            total_cost_usd=total_cost,
            total_tokens=total_tokens)

    except Exception as e:
        log("fatal_error",
            error=f"{e.__class__.__name__}: {e}",
            traceback=traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
