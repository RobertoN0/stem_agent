"""Container entry point — runs solve_task() inside the Docker sandbox.

Implements a line-delimited JSON RPC protocol over stdio:
- First stdin line: {"_kind": "task", "task": {...}}
- During solve_task, llm_call writes {"_kind": "llm_request", ...} to stdout
  and reads matching {"_kind": "llm_response", ...} from stdin.
- Final stdout line: {"_kind": "result", "result": {...}}

solve_task's own stdout is captured to a buffer so it doesn't pollute the
RPC channel. Stderr is reserved for runner markers (RUNNER:start /
RUNNER:end), which the orchestrator parses for per-task Docker proof.
"""

import datetime
import io
import json
import signal
import socket
import sys
import traceback


SNAPSHOT_ROOT = "/agent"
DEFAULT_TIMEOUT_S = 120


class _TaskTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _TaskTimeout("solve_task exceeded wall-clock timeout")


def _emit_result(channel, result: dict) -> None:
    channel.write(json.dumps({"_kind": "result", "result": result}) + "\n")
    channel.flush()


def _stderr_marker(name: str, **fields) -> None:
    parts = [f"{k}={v}" for k, v in fields.items()]
    sys.stderr.write(f"RUNNER:{name} {' '.join(parts)}\n")
    sys.stderr.flush()


def _make_send_line(channel):
    def send(line: str) -> None:
        channel.write(line + "\n")
        channel.flush()
    return send


def _make_recv_line(channel):
    def recv(_timeout):  # timeout ignored on the container side
        line = channel.readline()
        if not line:
            return None
        return line.rstrip("\n")
    return recv


def main() -> None:
    real_stdout = sys.stdout
    real_stdin = sys.stdin

    _stderr_marker(
        "start",
        host=socket.gethostname(),
        py=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )

    raw = real_stdin.readline()
    try:
        env = json.loads(raw)
    except json.JSONDecodeError as e:
        _emit_result(real_stdout, {
            "error": f"runner: bad task envelope JSON: {e}",
            "task_id": None,
        })
        _stderr_marker("end", task_id="?", rc=1)
        return

    if not isinstance(env, dict) or env.get("_kind") != "task" or "task" not in env:
        _emit_result(real_stdout, {
            "error": f"runner: expected task envelope, got {env.get('_kind') if isinstance(env, dict) else type(env).__name__!r}",
            "task_id": None,
        })
        _stderr_marker("end", task_id="?", rc=1)
        return

    task = env["task"]
    task_id = task.get("id") if isinstance(task, dict) else None

    if SNAPSHOT_ROOT not in sys.path:
        sys.path.insert(0, SNAPSHOT_ROOT)

    try:
        from tools import base as tools_base
        tools_base._set_rpc_channels(_make_send_line(real_stdout), _make_recv_line(real_stdin))
        tools_base._set_task_context(task_id)
    except Exception as e:
        _emit_result(real_stdout, {
            "error": f"runner: tools.base import failed: {e.__class__.__name__}: {e}",
            "traceback": traceback.format_exc(),
            "task_id": task_id,
        })
        _stderr_marker("end", task_id=task_id or "?", rc=1)
        return

    try:
        from agent.agent import solve_task
    except Exception as e:
        _emit_result(real_stdout, {
            "error": f"runner: import failed: {e.__class__.__name__}: {e}",
            "traceback": traceback.format_exc(),
            "task_id": task_id,
        })
        _stderr_marker("end", task_id=task_id or "?", rc=1)
        return

    captured = io.StringIO()
    sys.stdout = captured

    timeout_s = int(task.get("_timeout_s", DEFAULT_TIMEOUT_S))
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_s)

    rc = 0
    try:
        out = solve_task(task)
        if not isinstance(out, dict):
            result = {
                "error": f"runner: solve_task returned non-dict ({type(out).__name__})",
                "task_id": task_id,
            }
            rc = 1
        else:
            result = out
            if "task_id" not in result:
                result["task_id"] = task_id
    except _TaskTimeout as e:
        result = {"error": f"runner: timeout: {e}", "task_id": task_id}
        rc = 1
    except Exception as e:
        result = {
            "error": f"runner: solve_task raised: {e.__class__.__name__}: {e}",
            "traceback": traceback.format_exc(),
            "task_id": task_id,
        }
        rc = 1
    finally:
        signal.alarm(0)
        sys.stdout = real_stdout

    captured_text = captured.getvalue()
    if captured_text:
        result["_stdout_captured"] = captured_text[:2000]

    _emit_result(real_stdout, result)
    _stderr_marker("end", task_id=task_id or "?", rc=rc)


if __name__ == "__main__":
    main()
