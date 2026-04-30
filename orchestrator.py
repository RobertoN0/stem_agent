"""Orchestrator — runs the evolution loop.

This module is the rollback guardian and is IMMUTABLE infrastructure:
the evolving agent must never edit it. Responsibilities:

- load configuration from config.yaml
- snapshot each run's generations into artifacts/runs/<run_id>/generations/gen_N/
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
import shlex
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

def _snapshot_ignore(_dir: str, names: list[str]) -> set[str]:
    """Skip interpreter/cache artifacts when freezing agent/tool snapshots."""
    return {
        name for name in names
        if name == "__pycache__" or name.endswith((".pyc", ".pyo"))
    }


def bootstrap_gen0(
    config: dict,
    project_root: Path,
    artifacts_root: Optional[Path] = None,
) -> Path:
    """Create a gen_0 snapshot from the live agent/ + tools/ trees if missing.

    Refuses to bootstrap from an empty/missing source tree to prevent silent
    gen_0 generation from nothing.
    """
    if artifacts_root is None:
        artifacts_root = project_root / config["paths"]["artifacts_dir"]
    gen0 = artifacts_root / "gen_0"
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
    shutil.copytree(agent_src, gen0 / "agent", ignore=_snapshot_ignore)
    shutil.copytree(tools_src, gen0 / "tools", ignore=_snapshot_ignore)
    return gen0


def copy_snapshot(src_dir: Path, dst_dir: Path) -> None:
    if dst_dir.exists():
        raise FileExistsError(f"snapshot already exists: {dst_dir}")
    shutil.copytree(src_dir, dst_dir, ignore=_snapshot_ignore)


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


def _tool_call_names(tool_calls: Optional[list]) -> list[str]:
    if not tool_calls:
        return []
    names = []
    for tc in tool_calls:
        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
        name = fn.get("name")
        names.append(str(name) if name else "?")
    return names


def _summarize_static_scan_result(content) -> str:
    try:
        data = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError:
        return "static_scan:unparseable"
    if not isinstance(data, dict):
        return "static_scan:unparseable"
    findings = data.get("heuristic_findings") or []
    external = data.get("external") or {}
    return (
        f"static_scan:{data.get('language', '?')}:"
        f"findings={len(findings)}:ext={external.get('status', '?')}"
    )


def _summarize_tool_results(messages: list) -> list[str]:
    """Return short, payload-safe summaries of tool outputs in a chat history."""
    call_names: dict[str, str] = {}
    summaries = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                call_id = tc.get("id")
                name = (tc.get("function") or {}).get("name")
                if call_id and name:
                    call_names[call_id] = name
        elif msg.get("role") == "tool":
            tool_call_id = msg.get("tool_call_id")
            name = call_names.get(tool_call_id, "tool")
            content = msg.get("content")
            if name == "static_scan":
                summaries.append(_summarize_static_scan_result(content))
            else:
                summaries.append(f"{name}:chars={len(str(content or ''))}")
    return summaries[-3:]


def make_llm_handler(
    client,
    default_model: str,
    timeout_s: int = 60,
    llm_logger: Optional["LLMCallLogger"] = None,
    console_logger: Optional[Callable[..., None]] = None,
    usage_tracker: Optional["LLMUsageTracker"] = None,
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
        tool_result_summaries = _summarize_tool_results(env.get("messages", []))
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

        duration_ms = int((time.monotonic() - t0) * 1000)

        if llm_logger is not None:
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

        if usage_tracker is not None and usage is not None:
            usage_tracker.add(usage, model=model)

        if console_logger is not None:
            console_usage = usage or {}
            console_logger(
                "llm_call",
                purpose=env.get("purpose"),
                task_id=env.get("task_id"),
                step_in_task=env.get("step_in_task", 0),
                model=model,
                duration_ms=duration_ms,
                prompt_tokens=console_usage.get("prompt_tokens", 0),
                completion_tokens=console_usage.get("completion_tokens", 0),
                response_tools=_tool_call_names((response_dict or {}).get("tool_calls")),
                tool_results=tool_result_summaries,
                response_kind=(
                    "tools" if (response_dict or {}).get("tool_calls")
                    else ("text" if (response_dict or {}).get("content") else "empty")
                ),
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


def _pricing_for_model(model: Optional[str], config: dict) -> Optional[dict]:
    """Return pricing for exact or dated model names, if configured."""
    if not model:
        model = config.get("model", {}).get("name")
    pricing_by_model = config.get("pricing", {})
    if model in pricing_by_model:
        return pricing_by_model[model]
    for configured_model, pricing in sorted(
        pricing_by_model.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if model and model.startswith(f"{configured_model}-"):
            return pricing
    return None


def _estimate_cost_usd(usage: dict, config: dict, model: Optional[str] = None) -> float:
    model = model or usage.get("model") or config.get("model", {}).get("name")
    pricing = _pricing_for_model(model, config)
    if not pricing:
        return 0.0
    return (
        usage.get("prompt_tokens", 0) * pricing["input"] / 1_000_000
        + usage.get("completion_tokens", 0) * pricing["output"] / 1_000_000
    )


class LLMUsageTracker:
    """Accumulate run-wide billable usage across solve, smoke, and reflection calls."""

    def __init__(self, config: dict):
        self.config = config
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.n_calls = 0
        self.cost_usd = 0.0
        self.by_model: dict[str, dict] = {}

    def add(self, usage: dict, model: Optional[str] = None) -> None:
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        usage_model = usage.get("model") or model or self.config.get("model", {}).get("name")
        cost_usd = _estimate_cost_usd(usage, self.config, model=usage_model)
        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.n_calls += 1
            self.cost_usd += cost_usd
            model_bucket = self.by_model.setdefault(
                usage_model,
                {"prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0, "cost_usd": 0.0},
            )
            model_bucket["prompt_tokens"] += prompt_tokens
            model_bucket["completion_tokens"] += completion_tokens
            model_bucket["n_calls"] += 1
            model_bucket["cost_usd"] += cost_usd

    def tokens(self) -> dict:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "n_calls": self.n_calls,
            }

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "n_calls": self.n_calls,
                "cost_usd": self.cost_usd,
                "by_model": {
                    model: dict(values)
                    for model, values in sorted(self.by_model.items())
                },
            }


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
    n_tasks = len(split)
    for idx, task in enumerate(split, start=1):
        task_for_agent = {k: v for k, v in task.items() if k != "label"}
        if log_event:
            log_event("task_start", task_idx=idx, n_tasks=n_tasks, task_id=task.get("id"))
        r = run_one_task(gen_dir, task_for_agent, config,
                         llm_handler=llm_handler, log_event=log_event)
        r["expected"] = task.get("label")
        results.append(r)
        if log_event:
            predicted = _normalize_label(r.get("label"))
            expected = _normalize_label(r.get("expected"))
            log_event(
                "task_result",
                task_idx=idx,
                n_tasks=n_tasks,
                task_id=task.get("id"),
                predicted=predicted,
                expected=expected,
                ok=("error" not in r and predicted is not None and predicted == expected),
                errored="error" in r,
                error=r.get("error"),
                n_llm_calls=(r.get("usage") or {}).get("n_calls", 0),
            )
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


def _score_macro_f1(score_or_f1) -> float:
    if isinstance(score_or_f1, dict):
        return float(score_or_f1.get("macro_f1", 0.0))
    return float(score_or_f1)


def _score_errors(score_or_f1) -> int:
    if isinstance(score_or_f1, dict):
        return int(score_or_f1.get("n_errors", 0))
    return 0


def validation_gate(parent_score_or_f1, candidate_score_or_f1) -> bool:
    """Strict improvement with runnable candidates only; ties don't pass."""
    if _score_errors(candidate_score_or_f1) > 0:
        return False
    return _score_macro_f1(candidate_score_or_f1) > _score_macro_f1(parent_score_or_f1)


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


class EventFanout:
    """Send structured events to multiple logger-like callables."""

    def __init__(self, *loggers):
        self.loggers = [logger for logger in loggers if logger is not None]

    def set_gen(self, gen: Optional[int]) -> None:
        for logger in self.loggers:
            set_gen = getattr(logger, "set_gen", None)
            if set_gen is not None:
                set_gen(gen)

    def __call__(self, event: str, **fields) -> None:
        for logger in self.loggers:
            logger(event, **fields)

    def close(self) -> None:
        for logger in self.loggers:
            close = getattr(logger, "close", None)
            if close is not None:
                close()


class ConsoleLogger:
    """Human-readable stderr progress logger for long experiment runs."""

    def __init__(self, stream=None, *, enabled: bool = True, verbose: bool = False):
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = enabled
        self.verbose = verbose
        self._gen: Optional[int] = None

    def set_gen(self, gen: Optional[int]) -> None:
        self._gen = gen

    def __call__(self, event: str, **fields) -> None:
        if not self.enabled:
            return
        line = self._format(event, fields)
        if not line:
            return
        ts = _dt.datetime.now().strftime("%H:%M:%S")
        print(f"[stem {ts}] {line}", file=self.stream, flush=True)

    def _format(self, event: str, fields: dict) -> Optional[str]:
        gen = fields.get("gen", self._gen)
        prefix = f"gen={gen} " if gen is not None else ""

        if event == "start_run":
            return f"run start id={fields.get('run_id')} config={fields.get('config_path')}"
        if event == "transcript_start":
            return (
                f"terminal transcript path={fields.get('path')} "
                f"command={fields.get('command')}"
            )
        if event == "bootstrap_gen0":
            return f"{prefix}bootstrap gen_0 dir={fields.get('gen_dir')}"
        if event == "benchmark_start":
            limit = fields.get("limit")
            limit_str = f" limit={limit}" if limit is not None else ""
            return f"benchmark start gen={fields.get('gen')} split={fields.get('split')}{limit_str}"
        if event == "eval_start":
            limit = fields.get("limit")
            limit_str = f" limit={limit}" if limit is not None else ""
            return f"{prefix}eval start split={fields.get('split')} tasks={fields.get('n_tasks', '?')}{limit_str}"
        if event == "start_generation":
            return (
                f"{prefix}generation start parent_f1={_fmt_float(fields.get('parent_macro_f1'))} "
                f"best_f1={_fmt_float(fields.get('best_macro_f1'))} "
                f"plateau={fields.get('plateau')} cost={_fmt_usd(fields.get('total_cost'))}"
            )
        if event == "trajectories_selected":
            return (
                f"{prefix}reflection context failures={fields.get('n_failures')} "
                f"successes={fields.get('n_successes')} task_ids={fields.get('task_ids')}"
            )
        if event == "reflect_start":
            return f"{prefix}reflection start parent={fields.get('parent_dir')}"
        if event == "proposal":
            return (
                f"{prefix}proposal kind={fields.get('kind')} intent={fields.get('intent')} "
                f"kinds={fields.get('kinds')} rationale={_short(fields.get('rationale'))}"
            )
        if event == "reflect_rationale_unjustified":
            return (
                f"{prefix}proposal rationale warning kind={fields.get('kind')} "
                f"rationale={_short(fields.get('rationale'))}"
            )
        if event == "apply_success":
            return f"{prefix}apply success changed_files={fields.get('n_changed_files')}"
        if event == "proposal_invalid":
            return f"{prefix}proposal invalid kind={fields.get('kind')} error={_short(fields.get('error'))}"
        if event == "proposal_noop":
            return f"{prefix}proposal noop kind={fields.get('kind')}"
        if event == "apply_error":
            return f"{prefix}apply error {_short(fields.get('error'))}"
        if event == "task_start":
            return (
                f"{prefix}task {fields.get('task_idx')}/{fields.get('n_tasks')} "
                f"start id={fields.get('task_id')}"
            )
        if event == "task_result":
            status = "error" if fields.get("errored") else ("ok" if fields.get("ok") else "wrong")
            return (
                f"{prefix}task {fields.get('task_idx')}/{fields.get('n_tasks')} done "
                f"id={fields.get('task_id')} status={status} pred={fields.get('predicted')} "
                f"expected={fields.get('expected')} calls={fields.get('n_llm_calls')}"
            )
        if event == "task_run" and self.verbose:
            return (
                f"{prefix}runner task={fields.get('task_id')} container={fields.get('container')} "
                f"rc={fields.get('runner_end_rc')} calls={fields.get('n_llm_calls')}"
            )
        if event == "task_run_warning":
            return f"{prefix}runner warning task={fields.get('task_id')} reason={fields.get('reason')}"
        if event == "llm_call":
            tool_suffix = _format_llm_tool_suffix(fields)
            if fields.get("error"):
                return (
                    f"llm error purpose={fields.get('purpose')} task={fields.get('task_id')} "
                    f"step={fields.get('step_in_task')} model={fields.get('model')} "
                    f"duration={_fmt_ms(fields.get('duration_ms'))}{tool_suffix} "
                    f"error={_short(fields.get('error'))}"
                )
            return (
                f"llm purpose={fields.get('purpose')} task={fields.get('task_id')} "
                f"step={fields.get('step_in_task')} model={fields.get('model')} "
                f"duration={_fmt_ms(fields.get('duration_ms'))} "
                f"tokens={fields.get('prompt_tokens', 0)}+{fields.get('completion_tokens', 0)}"
                f"{tool_suffix}"
            )
        if event == "smoke_pass":
            return f"{prefix}smoke pass task={fields.get('task_id')}"
        if event == "smoke_fail":
            return f"{prefix}smoke fail task={fields.get('task_id')} error={_short(fields.get('error'))}"
        if event == "eval_complete" or event == "benchmark_complete":
            return (
                f"{prefix}eval done split={fields.get('split')} "
                f"macro_f1={_fmt_float(fields.get('macro_f1'))} "
                f"accuracy={_fmt_float(fields.get('accuracy'))} "
                f"errors={fields.get('n_errors')} cost={_fmt_usd(fields.get('cost_usd'))} "
                f"calls={(fields.get('tokens') or {}).get('n_calls', '?')}"
            )
        if event == "gate_accept":
            return (
                f"{prefix}gate accept parent_f1={_fmt_float(fields.get('parent_f1'))} "
                f"candidate_f1={_fmt_float(fields.get('candidate_f1'))}"
            )
        if event == "gate_reject":
            return (
                f"{prefix}gate reject stage={fields.get('stage')} "
                f"parent_f1={_fmt_float(fields.get('parent_f1'))} "
                f"candidate_f1={_fmt_float(fields.get('candidate_f1'))}"
            )
        if event == "agent_halt":
            return f"{prefix}agent halt stage={fields.get('stage')} rationale={_short(fields.get('rationale'))}"
        if event == "stop":
            return f"{prefix}stop reason={fields.get('reason')} after={fields.get('after')}"
        if event == "end_run":
            return (
                f"{prefix}run end reason={fields.get('stop_reason')} "
                f"parent_f1={_fmt_float(fields.get('parent_macro_f1'))} "
                f"best_f1={_fmt_float(fields.get('best_macro_f1'))} "
                f"cost={_fmt_usd(fields.get('total_cost_usd'))} "
                f"calls={(fields.get('total_tokens') or {}).get('n_calls', '?')}"
            )
        if event in ("fatal_error", "reflect_error", "reflect_parse_error", "reflect_no_proposal"):
            return f"{prefix}{event} {_short(fields.get('error') or fields)}"
        if self.verbose:
            return f"{prefix}{event} {fields}"
        return None


class TranscriptLogger(ConsoleLogger):
    """Human-readable progress log written to terminal_output.out in a run dir."""

    def __init__(self, run_dir: Path, *, verbose: bool = False):
        self.path = run_dir / "terminal_output.out"
        self._file = open(self.path, "a", buffering=1)
        super().__init__(stream=self._file, enabled=True, verbose=verbose)

    def close(self) -> None:
        self._file.close()


def _format_llm_tool_suffix(fields: dict) -> str:
    parts = []
    tool_results = fields.get("tool_results") or []
    if tool_results:
        parts.append("tool_results=" + ",".join(str(item) for item in tool_results))
    response_tools = fields.get("response_tools") or []
    if response_tools:
        parts.append("response_tools=" + ",".join(str(name) for name in response_tools))
    elif fields.get("response_kind") in {"text", "empty"}:
        parts.append(f"response={fields.get('response_kind')}")
    return (" " + " ".join(parts)) if parts else ""


def _fmt_float(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "?"


def _fmt_usd(value) -> str:
    return f"${value:.4f}" if isinstance(value, (int, float)) else "$?"


def _fmt_ms(value) -> str:
    if not isinstance(value, (int, float)):
        return "?"
    if value >= 1000:
        return f"{value / 1000:.1f}s"
    return f"{int(value)}ms"


def _short(value, limit: int = 180) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit - 3] + "..."


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
    parser.add_argument("--quiet", action="store_true",
                        help="disable human-readable terminal progress logs")
    parser.add_argument("--verbose", action="store_true",
                        help="include lower-level runner/event details in terminal logs")
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
    gen_root = run_dir / "generations"
    shutil.copy(config_path, run_dir / "config.snapshot.yaml")

    run_log = RunLogger(run_dir)
    console_log = ConsoleLogger(enabled=not args.quiet, verbose=args.verbose)
    transcript_log = TranscriptLogger(run_dir, verbose=args.verbose)
    progress_log = EventFanout(console_log, transcript_log)
    log = EventFanout(run_log, progress_log)
    llm_log = LLMCallLogger(run_dir)
    usage_tracker = LLMUsageTracker(config)
    log("transcript_start",
        path=str(transcript_log.path),
        command=shlex.join([sys.executable, *sys.argv]))
    log("start_run", run_id=run_id, config_path=str(config_path))

    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception as e:
        log("fatal_error", error=f"OpenAI client init failed: {e}")
        transcript_log.close()
        sys.exit(1)

    try:
        log.set_gen(0)
        llm_log.set_gen(0)
        handler = make_llm_handler(
            client,
            config["model"]["name"],
            llm_logger=llm_log,
            console_logger=progress_log,
            usage_tracker=usage_tracker,
        )
        gen0 = bootstrap_gen0(config, project_root, artifacts_root=gen_root)
        log("bootstrap_gen0", gen_dir=str(gen0))

        train_split = _load_split(config, project_root, "train")
        val_split = _load_split(config, project_root, "val")
        log("eval_start", split="train", n_tasks=len(train_split))
        gen0_train_score = run_candidate(
            gen0, train_split, config, llm_handler=handler, log_event=log
        )
        log("eval_complete", split="train",
            macro_f1=gen0_train_score["macro_f1"],
            accuracy=gen0_train_score["accuracy"],
            n_errors=gen0_train_score["n_errors"],
            cost_usd=gen0_train_score["cost_usd"],
            tokens=gen0_train_score["usage"])

        log("eval_start", split="val", n_tasks=len(val_split))
        gen0_val_score = run_candidate(
            gen0, val_split, config, llm_handler=handler, log_event=log
        )
        log("eval_complete", split="val",
            macro_f1=gen0_val_score["macro_f1"],
            accuracy=gen0_val_score["accuracy"],
            n_errors=gen0_val_score["n_errors"],
            cost_usd=gen0_val_score["cost_usd"],
            tokens=gen0_val_score["usage"])

        parent_dir = gen0
        parent_train_score = gen0_train_score
        parent_val_score = gen0_val_score
        parent_train_macro_f1 = gen0_train_score["macro_f1"]
        parent_val_macro_f1 = gen0_val_score["macro_f1"]
        best_macro_f1 = parent_val_macro_f1
        plateau = 0
        train_task_code_map = {t.get("id"): t.get("code", "") for t in train_split}
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
                parent_macro_f1=parent_val_macro_f1,
                parent_train_macro_f1=parent_train_macro_f1,
                best_macro_f1=best_macro_f1,
                plateau=plateau,
                total_cost=usage_tracker.cost_usd)

            candidate_dir = gen_root / f"gen_{gen_idx}"
            copy_snapshot(parent_dir, candidate_dir)
            log("snapshot_copied", src=str(parent_dir), dst=str(candidate_dir))

            trajectories = _select_trajectories(
                parent_train_score.get("per_task", []), train_task_code_map,
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
                log("reflect_start", parent_dir=str(parent_dir), n_trajectories=len(trajectories))
                proposal = reflect(
                    trajectories, str(parent_dir),
                    llm_handler=handler, config=config,
                    score=parent_train_score, gen_idx=gen_idx - 1,
                    score_split="train",
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
            changed_files = sorted(
                set(before_files.keys()) ^ set(after_files.keys())
                | {k for k in before_files.keys() & after_files.keys()
                   if before_files[k] != after_files[k]}
            )
            log("apply_success",
                kind=proposal_kind,
                kinds=proposal_kinds,
                intent=proposal_intent,
                n_changed_files=len(changed_files),
                changed_files=changed_files[:20])
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

            if not smoke_test(candidate_dir, config, task=train_split[0],
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

            cand_val_score = run_candidate(candidate_dir, val_split, config,
                                           llm_handler=handler, log_event=log)
            log("eval_complete", split="val",
                macro_f1=cand_val_score["macro_f1"],
                accuracy=cand_val_score["accuracy"],
                n_errors=cand_val_score["n_errors"],
                cost_usd=cand_val_score["cost_usd"],
                tokens=cand_val_score["usage"])

            accepted = False
            if validation_gate(parent_val_score, cand_val_score):
                log("gate_accept",
                    parent_f1=parent_val_macro_f1,
                    candidate_f1=cand_val_score["macro_f1"])
                accepted = True
                cand_train_score = run_candidate(candidate_dir, train_split, config,
                                                 llm_handler=handler, log_event=log)
                log("eval_complete", split="train",
                    macro_f1=cand_train_score["macro_f1"],
                    accuracy=cand_train_score["accuracy"],
                    n_errors=cand_train_score["n_errors"],
                    cost_usd=cand_train_score["cost_usd"],
                    tokens=cand_train_score["usage"])
                parent_dir = candidate_dir
                parent_train_score = cand_train_score
                parent_val_score = cand_val_score
                parent_train_macro_f1 = cand_train_score["macro_f1"]
                parent_val_macro_f1 = cand_val_score["macro_f1"]
                if cand_val_score["macro_f1"] > best_macro_f1:
                    best_macro_f1 = cand_val_score["macro_f1"]
                    plateau = 0
                else:
                    plateau += 1
            else:
                reject_stage = "val_errors" if cand_val_score["n_errors"] > 0 else "val"
                log("gate_reject", stage=reject_stage,
                    parent_f1=parent_val_macro_f1,
                    candidate_f1=cand_val_score["macro_f1"],
                    candidate_errors=cand_val_score["n_errors"])
                plateau += 1

            if halt_requested:
                stop_reason = "agent_halt"
                log("stop",
                    reason=stop_reason,
                    accepted=accepted,
                    parent_f1=parent_val_macro_f1,
                    candidate_f1=cand_val_score["macro_f1"])
                break

            if plateau >= plateau_limit:
                stop_reason = "plateau"
                log("stop", reason=stop_reason, plateau=plateau, after="val_gate")
                break

        log("end_run",
            stop_reason=stop_reason,
            parent_macro_f1=parent_val_macro_f1,
            parent_train_macro_f1=parent_train_macro_f1,
            best_macro_f1=best_macro_f1,
            total_cost_usd=usage_tracker.cost_usd,
            total_tokens=usage_tracker.tokens(),
            llm_usage=usage_tracker.snapshot())

    except Exception as e:
        log("fatal_error",
            error=f"{e.__class__.__name__}: {e}",
            traceback=traceback.format_exc())
        sys.exit(1)
    finally:
        transcript_log.close()


if __name__ == "__main__":
    main()
