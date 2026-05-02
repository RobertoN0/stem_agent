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
import re as _re
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
_RUNTIME_HELPER_CALLS = {}
_RUNTIME_HELPER_ERRORS = {}


def _reset_runtime_helper_telemetry() -> None:
    """Reset per-task direct helper-call telemetry."""
    _RUNTIME_HELPER_CALLS.clear()
    _RUNTIME_HELPER_ERRORS.clear()


def _bump_runtime_helper_count(bucket: dict, name: str) -> None:
    bucket[name] = int(bucket.get(name, 0)) + 1


def _record_runtime_helper_call(name: str) -> None:
    _bump_runtime_helper_count(_RUNTIME_HELPER_CALLS, name)


def _record_runtime_helper_error(name: str) -> None:
    _bump_runtime_helper_count(_RUNTIME_HELPER_ERRORS, name)


def _runtime_helper_telemetry_snapshot() -> dict:
    """Return cumulative per-task helper calls made inside the sandbox."""
    return {
        "helper_calls_by_name": dict(sorted(_RUNTIME_HELPER_CALLS.items())),
        "helper_errors_by_name": dict(sorted(_RUNTIME_HELPER_ERRORS.items())),
    }


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
    _reset_runtime_helper_telemetry()


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
    _record_runtime_helper_call("read_file")
    try:
        return _safe_path(path, write=False).read_text()
    except Exception:
        _record_runtime_helper_error("read_file")
        raise


def write_file(path: str, content: str) -> None:
    _record_runtime_helper_call("write_file")
    try:
        p = _safe_path(path, write=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    except Exception:
        _record_runtime_helper_error("write_file")
        raise


def list_dir(path: str) -> list:
    _record_runtime_helper_call("list_dir")
    try:
        p = _safe_path(path, write=False)
        return sorted(_os.listdir(p))
    except Exception:
        _record_runtime_helper_error("list_dir")
        raise


def run_bash(cmd: str, timeout: int = 30) -> dict:
    """Run a shell command in /work. Filesystem-level restrictions are
    enforced by the Docker mount (/agent is RO at the kernel level), so we
    don't try to validate the command itself."""
    _record_runtime_helper_call("run_bash")
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


def _normalize_language(code: str, language: str | None) -> str:
    if isinstance(language, str):
        lang = language.strip().lower().replace("-", "_")
        if lang in {"py", "python"}:
            return "python"
        if lang in {"c", "cpp", "c++", "cc", "c_cpp"}:
            return "c_cpp"

    sample = code[:2000]
    if _re.search(r"^\s*(def|import|from)\s+", sample, _re.MULTILINE):
        return "python"
    if any(token in sample for token in ("#include", "::", "->", "malloc(", "free(", ";")):
        return "c_cpp"
    return "unknown"


def _add_finding(findings: list, rule_id: str, severity: str, message: str, evidence: str) -> None:
    findings.append({
        "rule_id": rule_id,
        "severity": severity,
        "message": message,
        "evidence": evidence.strip()[:240],
    })


def _heuristic_scan(code: str, language: str) -> list:
    findings = []
    checks = []
    if language == "python":
        checks = [
            (r"\beval\s*\(", "PY-EVAL", "high", "eval() on attacker-influenced input can execute code."),
            (r"\bexec\s*\(", "PY-EXEC", "high", "exec() on attacker-influenced input can execute code."),
            (r"\bpickle\.loads?\s*\(", "PY-PICKLE", "high", "pickle deserialization can execute code."),
            (r"\byaml\.load\s*\(", "PY-YAML-LOAD", "medium", "yaml.load() can deserialize unsafe objects."),
            (r"\bos\.system\s*\(", "PY-OS-SYSTEM", "high", "os.system() can allow command injection."),
            (r"subprocess\.[A-Za-z_]+\s*\([^)]*shell\s*=\s*True", "PY-SHELL-TRUE", "high", "subprocess with shell=True is injection-prone."),
        ]
    elif language == "c_cpp":
        checks = [
            (r"\bgets\s*\(", "CWE-242", "high", "gets() cannot bound input and is inherently unsafe."),
            (r"\bstrcpy\s*\(", "CWE-120", "high", "strcpy() copies without a destination size."),
            (r"\bstrcat\s*\(", "CWE-120", "high", "strcat() appends without a destination size."),
            (r"\bsprintf\s*\(", "CWE-120", "high", "sprintf() writes without an output bound."),
            (r"\bscanf\s*\(\s*\"[^\"]*%s", "CWE-120", "medium", "scanf %s without a width limit can overflow."),
            (r"\bmemcpy\s*\([^;]+,\s*[^;]+,\s*[^;]+\)", "CWE-119", "medium", "memcpy() needs independent size validation."),
            (r"\b(?:malloc|realloc|new|reserve|resize|String)\s*\([^;\n]*(?:\*|<<|\+)[^;\n]*\)", "CWE-190", "medium", "allocation or reserve size uses arithmetic that may need overflow checks."),
            (r"\bfree\s*\([^)]*\)\s*;[^{}]*(?:return|goto)?[^{}]*\b\w+\s*=", "CWE-416", "low", "state after free should be checked for dangling use or double-free risk."),
        ]
    for pattern, rule_id, severity, message in checks:
        match = _re.search(pattern, code, _re.IGNORECASE | _re.DOTALL)
        if match:
            _add_finding(findings, rule_id, severity, message, match.group(0))
    return findings


def _external_scan(code: str, language: str, timeout: int, work_dir: str) -> dict:
    if language == "python":
        ext = ".py"
        cmd = ["bandit", "-q", "-f", "json"]
        tool = "bandit"
    elif language == "c_cpp":
        ext = ".cpp"
        cmd = ["semgrep", "--quiet", "--json", "--config=p/c"]
        tool = "semgrep"
    else:
        return {"status": "skipped", "reason": f"unsupported language: {language}"}

    root = _safe_path(work_dir, write=True)
    root.mkdir(parents=True, exist_ok=True)
    snippet = root / f"snippet{ext}"
    snippet.write_text(code)
    try:
        proc = _subprocess.run(
            cmd + [str(snippet)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(root),
        )
    except FileNotFoundError:
        return {"status": "unavailable", "tool": tool}
    except _subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        return {
            "status": "timeout",
            "tool": tool,
            "timeout_s": timeout,
            "stdout_tail": stdout[-1000:],
            "stderr_tail": stderr[-1000:],
        }

    return {
        "status": "ok",
        "tool": tool,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def static_scan(
    code: str,
    language: str | None = None,
    run_external: bool = False,
    timeout: int = 5,
    work_dir: str = "/work/static_scan",
) -> dict:
    """Run a bounded, structured static scan over a code snippet."""
    _record_runtime_helper_call("static_scan")
    try:
        code = code or ""
        try:
            timeout_s = max(1, min(int(timeout), 8))
        except (TypeError, ValueError):
            timeout_s = 5
        scan_code = code[:50000]
        lang = _normalize_language(scan_code, language)
        result = {
            "language": lang,
            "truncated": len(code) > len(scan_code),
            "heuristic_findings": _heuristic_scan(scan_code, lang),
            "external": {"status": "skipped", "reason": "run_external is false"},
        }
        if run_external:
            result["external"] = _external_scan(scan_code, lang, timeout_s, work_dir)
        return result
    except Exception:
        _record_runtime_helper_error("static_scan")
        raise


def note(text: str) -> None:
    """Append a timestamped note to /work/notes.txt for cross-step scratch."""
    _record_runtime_helper_call("note")
    try:
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        with open("/work/notes.txt", "a") as f:
            f.write(f"[{ts}] {text}\n")
    except Exception:
        _record_runtime_helper_error("note")
        raise


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
        "runtime_helper_telemetry": _runtime_helper_telemetry_snapshot(),
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
