"""Tests for the LLM call logging system.

Drives make_llm_handler with a fake OpenAI client and verifies that
llm_calls.jsonl is written with the correct schema. Also covers the
inspect_llm_calls.iter_calls filter API.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import orchestrator
from orchestrator import LLMCallLogger, make_llm_handler
from eval.inspect_llm_calls import iter_calls, summarize


# --- fake OpenAI client helpers --------------------------------------------

def _fake_usage(prompt=10, completion=5):
    u = SimpleNamespace()
    u.prompt_tokens = prompt
    u.completion_tokens = completion
    u.total_tokens = prompt + completion
    return u


def _fake_response(content="ok", model="gpt-5.4-mini", prompt=10, completion=5,
                   tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg)
    resp = SimpleNamespace(
        choices=[choice],
        usage=_fake_usage(prompt, completion),
        model=model,
    )
    return resp


def _fake_tool_call(name, arguments_json, call_id="call_001"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments_json),
    )


def _fake_client(content="ok", model="gpt-5.4-mini", prompt=10, completion=5,
                 raises=None, tool_calls=None):
    client = MagicMock()
    if raises:
        client.chat.completions.create.side_effect = raises
    else:
        client.chat.completions.create.return_value = _fake_response(
            content, model, prompt, completion, tool_calls=tool_calls,
        )
    return client


# --- schema tests -----------------------------------------------------------

def test_successful_call_writes_correct_schema(tmp_path):
    """A successful call produces a log record with all required fields."""
    llm_log = LLMCallLogger(tmp_path)
    llm_log.set_gen(0)

    handler = make_llm_handler(_fake_client(prompt=12, completion=3),
                               "gpt-5.4-mini", llm_logger=llm_log)
    env = {
        "_kind": "llm_request",
        "id": "r1",
        "messages": [{"role": "user", "content": "hi"}],
        "model": "gpt-5.4-mini",
        "response_format": None,
        "task_id": "primevul_1",
        "purpose": "solve_task",
        "step_in_task": 1,
    }
    result = handler(env)

    assert result["ok"] is True
    log_path = tmp_path / "llm_calls.jsonl"
    assert log_path.exists()

    lines = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    rec = lines[0]

    # Required fields present
    assert "call_id" in rec
    assert "timestamp_iso" in rec
    assert rec["purpose"] == "solve_task"
    assert rec["generation"] == 0
    assert rec["task_id"] == "primevul_1"
    assert rec["step_in_task"] == 1
    assert rec["request"]["model"] == "gpt-5.4-mini"
    assert rec["request"]["messages"] == [{"role": "user", "content": "hi"}]
    assert rec["response"]["content"] == "ok"
    assert rec["usage"]["prompt_tokens"] == 12
    assert rec["usage"]["completion_tokens"] == 3
    assert rec["usage"]["total_tokens"] == 15
    assert rec["usage"]["model"] == "gpt-5.4-mini"
    assert isinstance(rec["duration_ms"], int) and rec["duration_ms"] >= 0
    assert rec["error"] is None
    # No cost_usd field
    assert "cost_usd" not in rec


def test_failed_call_logs_error_and_null_response(tmp_path):
    """A failed call writes error string, null response/usage, ok=False."""
    llm_log = LLMCallLogger(tmp_path)
    llm_log.set_gen(1)

    handler = make_llm_handler(
        _fake_client(raises=RuntimeError("API down")),
        "gpt-5.4-mini",
        llm_logger=llm_log,
    )
    env = {
        "_kind": "llm_request",
        "id": "r2",
        "messages": [],
        "model": None,
        "response_format": None,
        "task_id": "t9",
        "purpose": "solve_task",
        "step_in_task": 1,
    }
    result = handler(env)

    assert result["ok"] is False
    assert "error" in result

    log_path = tmp_path / "llm_calls.jsonl"
    lines = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    rec = lines[0]

    assert rec["error"] is not None and "API down" in rec["error"]
    assert rec["response"] is None
    assert rec["usage"] is None
    assert rec["generation"] == 1


def test_per_run_isolation(tmp_path):
    """Each LLMCallLogger writes to its own file; separate runs don't mix."""
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()

    log_a = LLMCallLogger(run_a)
    log_b = LLMCallLogger(run_b)

    env = {
        "_kind": "llm_request", "id": "x", "messages": [], "model": None,
        "response_format": None, "task_id": "t1", "purpose": "solve_task",
        "step_in_task": 1,
    }
    make_llm_handler(_fake_client(), "gpt-5.4-mini", llm_logger=log_a)(env)
    make_llm_handler(_fake_client(), "gpt-5.4-mini", llm_logger=log_b)(env)
    make_llm_handler(_fake_client(), "gpt-5.4-mini", llm_logger=log_b)(env)

    lines_a = (run_a / "llm_calls.jsonl").read_text().splitlines()
    lines_b = (run_b / "llm_calls.jsonl").read_text().splitlines()
    assert len(lines_a) == 1
    assert len(lines_b) == 2


def test_generation_field_tracks_set_gen(tmp_path):
    """Records reflect the generation set on the logger at call time."""
    llm_log = LLMCallLogger(tmp_path)
    env = {
        "_kind": "llm_request", "id": "x", "messages": [], "model": None,
        "response_format": None, "task_id": "t1", "purpose": "solve_task",
        "step_in_task": 1,
    }
    handler = make_llm_handler(_fake_client(), "gpt-5.4-mini", llm_logger=llm_log)

    llm_log.set_gen(0)
    handler(env)
    llm_log.set_gen(2)
    handler(env)

    lines = [json.loads(l) for l in (tmp_path / "llm_calls.jsonl").read_text().splitlines()]
    assert lines[0]["generation"] == 0
    assert lines[1]["generation"] == 2


# --- inspect_llm_calls filter tests ----------------------------------------

def _write_calls(path: Path, records: list) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_iter_calls_no_filter(tmp_path):
    log = tmp_path / "llm_calls.jsonl"
    _write_calls(log, [
        {"call_id": "a", "task_id": "t1", "purpose": "solve_task",
         "generation": 0, "error": None},
        {"call_id": "b", "task_id": "t2", "purpose": "solve_task",
         "generation": 1, "error": None},
    ])
    result = list(iter_calls(log))
    assert len(result) == 2


def test_iter_calls_filter_task_id(tmp_path):
    log = tmp_path / "llm_calls.jsonl"
    _write_calls(log, [
        {"call_id": "a", "task_id": "t1", "purpose": "solve_task",
         "generation": 0, "error": None},
        {"call_id": "b", "task_id": "t2", "purpose": "solve_task",
         "generation": 0, "error": None},
    ])
    result = list(iter_calls(log, task_id="t1"))
    assert len(result) == 1
    assert result[0]["call_id"] == "a"


def test_iter_calls_filter_generation(tmp_path):
    log = tmp_path / "llm_calls.jsonl"
    _write_calls(log, [
        {"call_id": "a", "generation": 0, "task_id": "t1",
         "purpose": "solve_task", "error": None},
        {"call_id": "b", "generation": 1, "task_id": "t2",
         "purpose": "solve_task", "error": None},
        {"call_id": "c", "generation": 1, "task_id": "t3",
         "purpose": "solve_task", "error": None},
    ])
    result = list(iter_calls(log, generation=1))
    assert len(result) == 2
    assert all(r["generation"] == 1 for r in result)


def test_iter_calls_filter_error_only(tmp_path):
    log = tmp_path / "llm_calls.jsonl"
    _write_calls(log, [
        {"call_id": "a", "error": None, "task_id": "t1",
         "generation": 0, "purpose": "solve_task"},
        {"call_id": "b", "error": "timeout", "task_id": "t2",
         "generation": 0, "purpose": "solve_task"},
    ])
    result = list(iter_calls(log, error_only=True))
    assert len(result) == 1
    assert result[0]["error"] == "timeout"


# --- tool-calling round trip ----------------------------------------------

_TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Run a shell command in /work.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    },
]


def test_handler_forwards_tools_to_openai_client():
    """The `tools` and `tool_choice` fields in the envelope reach the API call."""
    client = _fake_client()
    handler = make_llm_handler(client, "gpt-5-mini")
    handler({
        "_kind": "llm_request",
        "id": "r1",
        "messages": [{"role": "user", "content": "hi"}],
        "model": "gpt-5-mini",
        "tools": _TOOL_SPEC,
        "tool_choice": "auto",
    })
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["tools"] == _TOOL_SPEC
    assert kwargs["tool_choice"] == "auto"


def test_handler_returns_tool_calls_when_model_emits_them():
    """When the OpenAI message has tool_calls, the response envelope mirrors them."""
    tc = _fake_tool_call("run_bash", '{"cmd": "ls"}', call_id="call_xyz")
    client = _fake_client(content=None, tool_calls=[tc])
    handler = make_llm_handler(client, "gpt-5-mini")

    result = handler({
        "_kind": "llm_request", "id": "r1",
        "messages": [], "model": "gpt-5-mini",
        "tools": _TOOL_SPEC,
    })

    assert result["ok"] is True
    assert result["content"] is None
    assert result["tool_calls"] == [{
        "id": "call_xyz",
        "type": "function",
        "function": {"name": "run_bash", "arguments": '{"cmd": "ls"}'},
    }]


def test_handler_returns_none_tool_calls_for_text_only_response():
    """Text-only responses leave tool_calls as None, not [], for clean JSON serialization."""
    client = _fake_client(content="just text")
    handler = make_llm_handler(client, "gpt-5-mini")
    result = handler({"_kind": "llm_request", "id": "r1", "messages": []})
    assert result["content"] == "just text"
    assert result["tool_calls"] is None


def test_llm_logger_records_tools_and_tool_calls(tmp_path):
    """Both the request's tools list and the response's tool_calls land in the ledger."""
    llm_log = LLMCallLogger(tmp_path)
    llm_log.set_gen(0)

    tc = _fake_tool_call("run_bash", '{"cmd": "echo hi"}')
    handler = make_llm_handler(
        _fake_client(content=None, tool_calls=[tc]),
        "gpt-5-mini",
        llm_logger=llm_log,
    )
    handler({
        "_kind": "llm_request", "id": "r1",
        "messages": [{"role": "user", "content": "do it"}],
        "model": "gpt-5-mini",
        "tools": _TOOL_SPEC,
        "tool_choice": "auto",
        "task_id": "t1",
        "purpose": "solve_task",
        "step_in_task": 1,
    })

    rec = json.loads((tmp_path / "llm_calls.jsonl").read_text().strip())
    assert rec["request"]["tools"] == _TOOL_SPEC
    assert rec["request"]["tool_choice"] == "auto"
    assert rec["response"]["tool_calls"][0]["function"]["name"] == "run_bash"


def test_summarize(tmp_path):
    log = tmp_path / "llm_calls.jsonl"
    _write_calls(log, [
        {"call_id": "a", "error": None, "generation": 0, "purpose": "solve_task",
         "task_id": "t1", "duration_ms": 200,
         "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        {"call_id": "b", "error": "fail", "generation": 0, "purpose": "solve_task",
         "task_id": "t2", "duration_ms": 100,
         "usage": None},
        {"call_id": "c", "error": None, "generation": 1, "purpose": "solve_task",
         "task_id": "t3", "duration_ms": 300,
         "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}},
    ])
    s = summarize(log)
    assert s["n_calls"] == 3
    assert s["n_errors"] == 1
    assert s["prompt_tokens"] == 30
    assert s["completion_tokens"] == 15
    assert s["total_tokens"] == 45
    assert s["by_generation"][0] == 2
    assert s["by_generation"][1] == 1
