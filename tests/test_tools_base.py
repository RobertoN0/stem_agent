"""Tests for sandbox tool helpers."""

import subprocess

from tools import base


def test_static_scan_detects_obvious_c_buffer_risk(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "_WRITABLE_PREFIXES", (str(tmp_path),))

    result = base.static_scan(
        "void f(char *s) { char buf[8]; gets(buf); }",
        language="c_cpp",
        work_dir=str(tmp_path),
    )

    assert result["language"] == "c_cpp"
    assert result["external"]["status"] == "skipped"
    assert any(finding["rule_id"] == "CWE-242" for finding in result["heuristic_findings"])


def test_static_scan_detects_obvious_python_command_risk(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "_WRITABLE_PREFIXES", (str(tmp_path),))

    result = base.static_scan(
        "import os\nos.system(user_input)\n",
        language="python",
        work_dir=str(tmp_path),
    )

    assert result["language"] == "python"
    assert any(finding["rule_id"] == "PY-OS-SYSTEM" for finding in result["heuristic_findings"])


def test_static_scan_external_timeout_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "_WRITABLE_PREFIXES", (str(tmp_path),))

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
