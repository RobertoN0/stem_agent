"""Tests for the orchestrator's gates, protocol, and bootstrap.

Docker is not invoked — tests either monkeypatch run_one_task or drive
_run_task_protocol directly with fake send/recv callables.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import orchestrator


# --- fixtures --------------------------------------------------------------

@pytest.fixture
def basic_config() -> dict:
    return {
        "paths": {
            "artifacts_dir": "artifacts",
            "runs_dir": "artifacts/runs",
            "data_dir": "data/primevul",
            "agent_dir": "agent",
            "tools_dir": "tools",
        },
        "model": {"name": "gpt-5.4-mini"},
        "budget": {"max_generations": 3},
        "splits": {"train": 5, "val": 3, "test": 5},
        "stopping": {"plateau_generations": 2},
        "sandbox": {
            "image": "stem-agent-sandbox",
            "memory_mb": 1024,
            "cpu_cores": 1,
            "wall_clock_timeout_s": 5,
            "network": False,
        },
        "pricing": {"gpt-5.4-mini": {"input": 0.75, "output": 4.50}},
    }


@pytest.fixture
def seeded_project(tmp_path: Path) -> Path:
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "agent.py").write_text(
        "def solve_task(task):\n"
        "    return {'task_id': task.get('id'), 'label': 'safe'}\n"
    )
    (tmp_path / "agent" / "prompt.txt").write_text("dummy")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "base.py").write_text("# stub\n")
    return tmp_path


# --- the three required tests ----------------------------------------------

def test_smoke_test_rejects_import_error(seeded_project, basic_config, monkeypatch):
    """smoke_test returns False when run_one_task surfaces an error."""
    gen_dir = orchestrator.bootstrap_gen0(basic_config, seeded_project)
    monkeypatch.setattr(orchestrator, "run_one_task",
                        lambda *a, **kw: {
                            "error": "runner: import failed: ImportError: bad",
                            "task_id": "t1",
                        })
    assert orchestrator.smoke_test(gen_dir, basic_config,
                                    task={"id": "t1", "code": "int x;"}) is False


def test_validation_gate_rejects_worse_candidate():
    assert orchestrator.validation_gate(0.7, 0.5) is False
    assert orchestrator.validation_gate(0.5, 0.7) is True
    # ties don't pass
    assert orchestrator.validation_gate(0.6, 0.6) is False


def test_bootstrap_refuses_when_agent_missing(tmp_path, basic_config):
    with pytest.raises(FileNotFoundError, match="missing source dirs"):
        orchestrator.bootstrap_gen0(basic_config, tmp_path)


# --- bonus tests for infrastructure paths ----------------------------------

def test_bootstrap_refuses_when_only_tools_missing(tmp_path, basic_config):
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "agent.py").write_text("def solve_task(t): pass\n")
    with pytest.raises(FileNotFoundError, match="tools"):
        orchestrator.bootstrap_gen0(basic_config, tmp_path)


def test_aggregate_counts_errors_as_wrong():
    config = {"model": {"name": "gpt-5.4-mini"},
              "pricing": {"gpt-5.4-mini": {"input": 0.75, "output": 4.50}}}
    results = [
        {"task_id": "a", "label": "vulnerable", "expected": "vulnerable",
         "usage": {"prompt_tokens": 100, "completion_tokens": 10, "n_calls": 1}},
        {"task_id": "b", "error": "host: timeout", "expected": "vulnerable"},
        {"task_id": "c", "label": "safe", "expected": "safe",
         "usage": {"prompt_tokens": 100, "completion_tokens": 10, "n_calls": 1}},
    ]
    out = orchestrator.aggregate(results, config)
    assert out["n_tasks"] == 3
    assert out["n_errors"] == 1
    assert out["accuracy"] == pytest.approx(2 / 3)
    assert out["usage"]["prompt_tokens"] == 200
    assert out["usage"]["completion_tokens"] == 20
    assert out["cost_usd"] == pytest.approx((200 * 0.75 + 20 * 4.50) / 1_000_000)


def test_aggregate_treats_null_label_as_wrong():
    """A None label (parse failure) is wrong, not an error."""
    config = {"model": {"name": "gpt-5.4-mini"}, "pricing": {}}
    results = [
        {"task_id": "a", "label": None, "expected": "vulnerable", "raw": "garbage"},
        {"task_id": "b", "label": "safe", "expected": "safe"},
    ]
    out = orchestrator.aggregate(results, config)
    assert out["n_errors"] == 0  # None label isn't an "error"
    assert out["accuracy"] == 0.5  # 1 of 2 correct
    assert out["per_task"][0]["predicted"] is None
    assert out["per_task"][0]["ok"] is False


# --- protocol tests --------------------------------------------------------

def _drive_protocol(container_lines, task, llm_handler, timeout_s=5.0):
    """Helper: simulate a container by replaying canned stdout lines."""
    sent = []
    pending = list(container_lines)

    def send_line(s):
        sent.append(s)

    def recv_line(_remaining):
        if not pending:
            return None
        return pending.pop(0)

    result = orchestrator._run_task_protocol(
        send_line, recv_line, task, llm_handler, timeout_s=timeout_s,
    )
    return result, sent


def test_protocol_proxies_llm_request_and_returns_result():
    """An llm_request envelope is handled, then a result envelope ends the loop."""
    container_lines = [
        json.dumps({"_kind": "llm_request", "id": "r1",
                    "messages": [{"role": "user", "content": "hi"}]}),
        json.dumps({"_kind": "result", "result": {"task_id": "t1", "label": "safe"}}),
    ]

    handler_calls = []

    def handler(env):
        handler_calls.append(env)
        return {"_kind": "llm_response", "id": env["id"], "ok": True,
                "content": "ok",
                "usage": {"prompt_tokens": 12, "completion_tokens": 3,
                          "model": "gpt-5.4-mini"}}

    result, sent = _drive_protocol(container_lines, {"id": "t1"}, handler)

    assert result["task_id"] == "t1"
    assert result["label"] == "safe"
    # Orchestrator-tracked usage is the source of truth on the result.
    assert result["usage"] == {"prompt_tokens": 12, "completion_tokens": 3, "n_calls": 1}
    assert len(handler_calls) == 1
    # Sent lines: task envelope, then llm_response.
    assert json.loads(sent[0])["_kind"] == "task"
    assert json.loads(sent[1])["_kind"] == "llm_response"


def test_protocol_returns_error_when_container_closes_early():
    """If the container closes stdout before sending a result, surface an error."""
    container_lines = []  # empty: recv_line returns None immediately

    def handler(env):
        pytest.fail("handler should not be called")

    result, _ = _drive_protocol(container_lines, {"id": "t9"}, handler)
    assert "error" in result
    assert result["task_id"] == "t9"
    assert result["usage"]["n_calls"] == 0


def test_protocol_passes_model_through_to_handler():
    """The agent's `model` field reaches the handler; otherwise default is used."""
    container_lines = [
        json.dumps({"_kind": "llm_request", "id": "r1", "model": "gpt-5.4-mini",
                    "messages": []}),
        json.dumps({"_kind": "result", "result": {"task_id": "t1", "label": "safe"}}),
    ]

    seen = []

    def handler(env):
        seen.append(env.get("model"))
        return {"_kind": "llm_response", "id": env["id"], "ok": True,
                "content": "ok", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    _drive_protocol(container_lines, {"id": "t1"}, handler)
    assert seen == ["gpt-5.4-mini"]


# --- runner-marker parser --------------------------------------------------

# --- trajectory selection --------------------------------------------------

def test_select_trajectories_failures_first_then_successes():
    per_task = [
        {"task_id": "t1", "ok": True, "errored": False, "predicted": "safe", "expected": "safe"},
        {"task_id": "t2", "ok": False, "errored": False, "predicted": "safe", "expected": "vulnerable"},
        {"task_id": "t3", "ok": False, "errored": True, "predicted": None, "expected": "vulnerable"},
        {"task_id": "t4", "ok": True, "errored": False, "predicted": "vulnerable", "expected": "vulnerable"},
        {"task_id": "t5", "ok": False, "errored": False, "predicted": "vulnerable", "expected": "safe"},
    ]
    code_map = {f"t{i}": f"code{i}" for i in range(1, 6)}
    sel = orchestrator._select_trajectories(per_task, code_map,
                                             max_failures=3, max_successes=2)
    # t3 (errored) comes before non-error failures, then t2, t5 alphabetically.
    # Then up to 2 successes (t1, t4 alphabetically).
    ids = [t["task_id"] for t in sel]
    assert ids[0] == "t3"  # errored first
    assert set(ids[1:3]) == {"t2", "t5"}  # other failures
    assert set(ids[3:]) == {"t1", "t4"}  # successes
    # Each entry has the input code attached.
    assert all("code" in t for t in sel)


def test_select_trajectories_caps_failures_and_successes():
    per_task = [{"task_id": f"f{i}", "ok": False, "errored": False,
                 "predicted": "safe", "expected": "vulnerable"} for i in range(5)]
    per_task += [{"task_id": f"s{i}", "ok": True, "errored": False,
                  "predicted": "safe", "expected": "safe"} for i in range(5)]
    sel = orchestrator._select_trajectories(per_task, {},
                                             max_failures=3, max_successes=2)
    assert sum(1 for t in sel if not t["ok"]) == 3
    assert sum(1 for t in sel if t["ok"]) == 2


def test_rationale_cites_failure_true_and_false():
    trajectories = [
        {"task_id": "t1", "ok": False, "errored": False},
        {"task_id": "t2", "ok": True, "errored": False},
    ]
    yes = orchestrator._rationale_cites_failure(
        {"details": {"rationale": "task t1 was wrongly classified"}},
        trajectories,
    )
    no = orchestrator._rationale_cites_failure(
        {"details": {"rationale": "no specific task referenced"}},
        trajectories,
    )
    cites_only_success = orchestrator._rationale_cites_failure(
        {"details": {"rationale": "t2 worked, so..."}},
        trajectories,
    )
    assert yes is True
    assert no is False
    assert cites_only_success is False


def test_aggregate_per_task_carries_raw_and_error():
    config = {"model": {"name": "gpt-5.4-mini"}, "pricing": {}}
    results = [
        {"task_id": "a", "label": "vulnerable", "expected": "vulnerable",
         "raw": '{"label":"vulnerable"}'},
        {"task_id": "b", "error": "host: timeout", "expected": "safe"},
    ]
    out = orchestrator.aggregate(results, config)
    assert out["per_task"][0]["raw"] == '{"label":"vulnerable"}'
    assert out["per_task"][0]["error"] is None
    assert out["per_task"][1]["raw"] is None
    assert out["per_task"][1]["error"] == "host: timeout"


def test_runner_marker_parser_extracts_start_and_end():
    stderr_lines = [
        "RUNNER:start host=ab12 py=3.11.9 ts=2026-04-26T10:00:00",
        "some unrelated line",
        "RUNNER:end task_id=t42 rc=0",
    ]
    markers = orchestrator._parse_runner_markers(stderr_lines)
    assert markers["start"]["host"] == "ab12"
    assert markers["start"]["py"] == "3.11.9"
    assert markers["end"]["rc"] == "0"
    assert markers["end"]["task_id"] == "t42"
