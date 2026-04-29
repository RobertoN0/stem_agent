"""Generic tools available to the agent inside the sandbox.

Filesystem operations are confined to /work (read-write scratch) and
/agent (read-only snapshot). The container has no network — all LLM
access must go through `llm_call`, which proxies the request back to
the orchestrator on the host.

The RPC channels used by `llm_call` are injected by `sandbox/runner.py`
at startup via `_set_rpc_channels`. Outside that runner (e.g., during
host-side tests) `llm_call` raises immediately.
"""

import datetime as _dt
import json as _json
import os as _os
import subprocess as _subprocess
import uuid as _uuid
from pathlib import Path


# Module-level prefixes so tests can monkey-patch.
_READABLE_PREFIXES = ("/work", "/agent")
_WRITABLE_PREFIXES = ("/work",)

# Set by the runner before solve_task runs.
_RPC_SEND_LINE = None
_RPC_RECV_LINE = None

# Task context injected by the runner so every llm_call envelope carries
# provenance fields (task_id, step_in_task, purpose) for the LLM call log.
_TASK_ID = None
_STEP = 0
_DEFAULT_PURPOSE = "solve_task"


def _set_rpc_channels(send_line, recv_line) -> None:
    """Wire the RPC line-IO callables. Called by the sandbox runner only."""
    global _RPC_SEND_LINE, _RPC_RECV_LINE
    _RPC_SEND_LINE = send_line
    _RPC_RECV_LINE = recv_line


def _set_task_context(task_id, default_purpose: str = "solve_task") -> None:
    """Set task provenance for llm_call logging. Called by the runner after
    parsing the task envelope, before solve_task is invoked."""
    global _TASK_ID, _STEP, _DEFAULT_PURPOSE
    _TASK_ID = task_id
    _STEP = 0
    _DEFAULT_PURPOSE = default_purpose


def _safe_path(path, *, write: bool) -> Path:
    p = Path(path).resolve()
    prefixes = _WRITABLE_PREFIXES if write else _READABLE_PREFIXES
    if not any(str(p) == pre or str(p).startswith(pre + "/") for pre in prefixes):
        kind = "writable" if write else "readable"
        raise PermissionError(
            f"path {p!s} is outside the {kind} sandbox dirs ({', '.join(prefixes)})"
        )
    return p


def read_file(path: str) -> str:
    return _safe_path(path, write=False).read_text()


def write_file(path: str, content: str) -> None:
    p = _safe_path(path, write=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def list_dir(path: str) -> list:
    p = _safe_path(path, write=False)
    return sorted(_os.listdir(p))


def run_bash(cmd: str, timeout: int = 30) -> dict:
    """Run a shell command in /work. Filesystem-level restrictions are
    enforced by the Docker mount (/agent is RO at the kernel level), so we
    don't try to validate the command itself."""
    try:
        proc = _subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd="/work",
        )
    except _subprocess.TimeoutExpired as e:
        return {
            "stdout": e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or ""),
            "stderr": f"timeout after {timeout}s",
            "returncode": -1,
        }
    return {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}


def note(text: str) -> None:
    """Append a timestamped note to /work/notes.txt for cross-step scratch."""
    ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
    with open("/work/notes.txt", "a") as f:
        f.write(f"[{ts}] {text}\n")


def llm_call(
    messages: list,
    model: str | None = None,
    response_format: dict | None = None,
    tools: list | None = None,
    tool_choice: str | dict | None = None,
    purpose: str | None = None,
) -> dict:
    """Make an LLM call via the host orchestrator.

    Returns {"content": str | None, "tool_calls": list | None,
             "usage": {prompt_tokens, completion_tokens, model}}.

    `content` is None when the model emitted only tool calls. `tool_calls` is
    None when the model emitted text only. Tool-call entries are dicts shaped
    {"id", "type", "function": {"name", "arguments"}} where arguments is a
    JSON string (matching OpenAI's wire format).

    All LLM access in this project must go through this function — direct
    `import openai` in agent code is forbidden (see agent/AGENT.md). The
    orchestrator owns ground-truth token tracking by virtue of being the only
    process that talks to the OpenAI API.
    """
    global _STEP
    if _RPC_SEND_LINE is None or _RPC_RECV_LINE is None:
        raise RuntimeError(
            "tools.base.llm_call: RPC channels not initialized. "
            "This function only works inside the sandbox runner."
        )
    _STEP += 1
    request_id = str(_uuid.uuid4())
    envelope = {
        "_kind": "llm_request",
        "id": request_id,
        "messages": messages,
        "model": model,
        "response_format": response_format,
        "tools": tools,
        "tool_choice": tool_choice,
        "task_id": _TASK_ID,
        "purpose": purpose if purpose is not None else _DEFAULT_PURPOSE,
        "step_in_task": _STEP,
    }
    _RPC_SEND_LINE(_json.dumps(envelope))

    while True:
        line = _RPC_RECV_LINE(None)  # block until something arrives
        if line is None:
            raise RuntimeError("llm_call: host closed stdin before response")
        try:
            response = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        if response.get("_kind") != "llm_response":
            continue
        if response.get("id") != request_id:
            raise RuntimeError(
                f"llm_call: id mismatch (expected {request_id}, got {response.get('id')})"
            )
        if not response.get("ok"):
            raise RuntimeError(f"llm_call upstream error: {response.get('error')}")
        return {
            "content": response.get("content"),
            "tool_calls": response.get("tool_calls"),
            "usage": response.get("usage", {}),
        }
