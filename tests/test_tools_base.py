"""Tests for sandbox tool helpers."""

import json
import subprocess

from tools import base


def test_static_scan_detects_obvious_c_buffer_risk(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "_WRITABLE_PREFIXES", (str(tmp_path),))
    base._reset_runtime_helper_telemetry()

    result = base.static_scan(
        "void f(char *s) { char buf[8]; gets(buf); }",
        language="c_cpp",
        work_dir=str(tmp_path),
    )

    assert result["language"] == "c_cpp"
    assert result["external"]["status"] == "skipped"
    assert any(finding["rule_id"] == "CWE-242" for finding in result["heuristic_findings"])
    telemetry = base._runtime_helper_telemetry_snapshot()
    assert telemetry["helper_calls_by_name"]["static_scan"] == 1


def test_static_scan_detects_obvious_python_command_risk(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "_WRITABLE_PREFIXES", (str(tmp_path),))
    base._reset_runtime_helper_telemetry()

    result = base.static_scan(
        "import os\nos.system(user_input)\n",
        language="python",
        work_dir=str(tmp_path),
    )

    assert result["language"] == "python"
    assert any(finding["rule_id"] == "PY-OS-SYSTEM" for finding in result["heuristic_findings"])


def test_static_scan_external_timeout_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "_WRITABLE_PREFIXES", (str(tmp_path),))
    base._reset_runtime_helper_telemetry()

    def fake_run(*_args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd="bandit",
            timeout=kwargs["timeout"],
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

    monkeypatch.setattr(base._subprocess, "run", fake_run)

    result = base.static_scan(
        "print('hello')",
        language="python",
        run_external=True,
        timeout=99,
        work_dir=str(tmp_path),
    )

    assert result["external"]["status"] == "timeout"
    assert result["external"]["timeout_s"] == 8
    assert "partial stderr" in result["external"]["stderr_tail"]


def test_runtime_helper_telemetry_tracks_direct_calls_and_resets(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "_READABLE_PREFIXES", (str(tmp_path),))
    monkeypatch.setattr(base, "_WRITABLE_PREFIXES", (str(tmp_path),))
    base._set_task_context("t1")

    (tmp_path / "x.txt").write_text("hello")
    assert base.read_file(str(tmp_path / "x.txt")) == "hello"
    base.write_file(str(tmp_path / "y.txt"), "world")
    base.static_scan("strcpy(dst, src);", language="c_cpp", work_dir=str(tmp_path))

    telemetry = base._runtime_helper_telemetry_snapshot()
    assert telemetry["helper_calls_by_name"] == {
        "read_file": 1,
        "static_scan": 1,
        "write_file": 1,
    }
    assert telemetry["helper_errors_by_name"] == {}

    base._set_task_context("t2")
    assert base._runtime_helper_telemetry_snapshot()["helper_calls_by_name"] == {}


def test_runtime_helper_telemetry_tracks_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "_READABLE_PREFIXES", (str(tmp_path),))
    base._set_task_context("t1")

    try:
        base.read_file(str(tmp_path / "missing.txt"))
    except FileNotFoundError:
        pass

    telemetry = base._runtime_helper_telemetry_snapshot()
    assert telemetry["helper_calls_by_name"] == {"read_file": 1}
    assert telemetry["helper_errors_by_name"] == {"read_file": 1}


def test_llm_call_sends_runtime_helper_telemetry(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "_WRITABLE_PREFIXES", (str(tmp_path),))
    sent = []

    def send_line(line):
        sent.append(json.loads(line))

    def recv_line(_timeout):
        return json.dumps({
            "_kind": "llm_response",
            "id": sent[0]["id"],
            "ok": True,
            "content": "ok",
            "tool_calls": None,
            "usage": {},
        })

    base._set_task_context("t1")
    base._set_rpc_channels(send_line, recv_line)
    base.static_scan("strcpy(dst, src);", language="c_cpp", work_dir=str(tmp_path))

    result = base.llm_call([{"role": "user", "content": "hi"}])

    assert result["content"] == "ok"
    assert sent[0]["runtime_helper_telemetry"]["helper_calls_by_name"] == {
        "static_scan": 1,
    }
