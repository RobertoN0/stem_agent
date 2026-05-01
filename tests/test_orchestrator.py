"""Tests for the orchestrator's gates, protocol, and bootstrap.

Docker is not invoked — tests either monkeypatch run_one_task or drive
_run_task_protocol directly with fake send/recv callables.
"""

import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import orchestrator
from growth.manifest import load_mutation_manifest


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


def test_bootstrap_can_target_run_scoped_generation_root(seeded_project, basic_config):
    run_gen_root = seeded_project / "artifacts" / "runs" / "run_x" / "generations"

    gen0 = orchestrator.bootstrap_gen0(
        basic_config,
        seeded_project,
        artifacts_root=run_gen_root,
    )

    assert gen0 == run_gen_root / "gen_0"
    assert (gen0 / "agent" / "agent.py").exists()
    assert (gen0 / "tools" / "base.py").exists()
    assert not (seeded_project / "artifacts" / "gen_0").exists()


def test_bootstrap_uses_manifest_snapshot_roots(seeded_project, basic_config):
    (seeded_project / "knowledge").mkdir()
    (seeded_project / "knowledge" / "strategy.md").write_text("seed\n")
    manifest = {
        "snapshot_roots": ["agent", "tools", "knowledge"],
        "mutable_paths": ["agent/**", "tools/**", "knowledge/**"],
        "protected_files": ["agent/agent.py", "agent/prompt.txt", "tools/base.py"],
        "protected_symbols": {},
    }

    gen0 = orchestrator.bootstrap_gen0(
        basic_config,
        seeded_project,
        artifacts_root=seeded_project / "runs" / "generations",
        manifest=manifest,
    )

    assert (gen0 / "knowledge" / "strategy.md").read_text() == "seed\n"


def test_snapshot_mutation_manifest_writes_run_copy(seeded_project):
    run_dir = seeded_project / "artifacts" / "runs" / "run_x"
    run_dir.mkdir(parents=True)
    (seeded_project / "mutation_manifest.yaml").write_text(
        "version: 1\nsnapshot_roots: [agent, tools, knowledge]\n"
    )
    manifest = load_mutation_manifest(seeded_project)

    snapshot = orchestrator._snapshot_mutation_manifest(seeded_project, run_dir, manifest)

    assert snapshot == run_dir / "mutation_manifest.snapshot.yaml"
    assert "knowledge" in snapshot.read_text()


def test_snapshot_ignores_python_cache_artifacts(seeded_project, basic_config):
    (seeded_project / "agent" / "__pycache__").mkdir()
    (seeded_project / "agent" / "__pycache__" / "agent.cpython-312.pyc").write_bytes(b"x")
    (seeded_project / "tools" / "__pycache__").mkdir()
    (seeded_project / "tools" / "__pycache__" / "base.cpython-312.pyc").write_bytes(b"x")

    gen0 = orchestrator.bootstrap_gen0(basic_config, seeded_project)
    gen1 = seeded_project / "artifacts" / "gen_1"
    orchestrator.copy_snapshot(gen0, gen1)

    assert not (gen0 / "agent" / "__pycache__").exists()
    assert not (gen0 / "tools" / "__pycache__").exists()
    assert not (gen1 / "agent" / "__pycache__").exists()
    assert not (gen1 / "tools" / "__pycache__").exists()


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
    assert result["telemetry"]["llm_calls"] == 1
    assert result["telemetry"]["text_response_turns"] == 1
    assert result["telemetry"]["tool_calls_by_name"] == {}
    assert len(handler_calls) == 1
    # Sent lines: task envelope, then llm_response.
    assert json.loads(sent[0])["_kind"] == "task"
    assert json.loads(sent[1])["_kind"] == "llm_response"


def test_protocol_tracks_solve_tool_call_telemetry():
    container_lines = [
        json.dumps({"_kind": "llm_request", "id": "r1",
                    "step_in_task": 1, "messages": []}),
        json.dumps({"_kind": "result", "result": {"task_id": "t1", "label": "safe"}}),
    ]

    def handler(env):
        return {
            "_kind": "llm_response",
            "id": env["id"],
            "ok": True,
            "content": None,
            "tool_calls": [
                {"id": "scan_1", "type": "function",
                 "function": {"name": "static_scan", "arguments": "{}"}},
                {"id": "final_1", "type": "function",
                 "function": {"name": "finalize", "arguments": "{}"}},
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }

    result, _ = _drive_protocol(container_lines, {"id": "t1"}, handler)

    assert result["telemetry"]["llm_calls"] == 1
    assert result["telemetry"]["tool_calls_by_name"] == {
        "finalize": 1,
        "static_scan": 1,
    }
    assert result["telemetry"]["inspection_tool_calls"] == 1
    assert result["telemetry"]["used_static_scan"] is True
    assert result["telemetry"]["immediate_finalize"] is True


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

def test_select_trajectories_balances_failure_and_success_buckets():
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
    ids = [t["task_id"] for t in sel]
    assert ids == ["t3", "t2", "t5", "t4", "t1"]
    assert [t["reflection_bucket"] for t in sel] == [
        "error",
        "false_negative",
        "false_positive",
        "vulnerable_success",
        "safe_success",
    ]
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


def test_select_trajectories_round_robins_false_negative_and_false_positive():
    per_task = [
        {"task_id": "fn1", "ok": False, "errored": False,
         "predicted": "safe", "expected": "vulnerable"},
        {"task_id": "fn2", "ok": False, "errored": False,
         "predicted": "safe", "expected": "vulnerable"},
        {"task_id": "fp1", "ok": False, "errored": False,
         "predicted": "vulnerable", "expected": "safe"},
        {"task_id": "fp2", "ok": False, "errored": False,
         "predicted": "vulnerable", "expected": "safe"},
    ]

    sel = orchestrator._select_trajectories(per_task, {},
                                             max_failures=4, max_successes=0)

    assert [t["task_id"] for t in sel] == ["fn1", "fp1", "fn2", "fp2"]


def test_select_trajectories_rotates_with_generation_offset():
    per_task = [
        {"task_id": "fn1", "ok": False, "errored": False,
         "predicted": "safe", "expected": "vulnerable"},
        {"task_id": "fn2", "ok": False, "errored": False,
         "predicted": "safe", "expected": "vulnerable"},
        {"task_id": "fp1", "ok": False, "errored": False,
         "predicted": "vulnerable", "expected": "safe"},
        {"task_id": "fp2", "ok": False, "errored": False,
         "predicted": "vulnerable", "expected": "safe"},
    ]

    sel = orchestrator._select_trajectories(
        per_task, {}, max_failures=4, max_successes=0, offset=1,
    )

    assert [t["task_id"] for t in sel] == ["fn2", "fp2", "fn1", "fp1"]


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


def test_rationale_cites_failure_reads_top_level_bundle_rationale():
    trajectories = [
        {"task_id": "t1", "ok": False, "errored": False},
        {"task_id": "t2", "ok": True, "errored": False},
    ]
    proposal = {
        "rationale": "task t1 exposed a missed bounds check",
        "intent": "iterate",
        "changes": [
            {"kind": "edit_prompt", "details": {"content": "new prompt"}},
        ],
    }
    assert orchestrator._rationale_cites_failure(proposal, trajectories) is True


def test_proposal_summary_helpers_support_bundles_and_halt():
    bundle = {
        "rationale": "task t1",
        "intent": "halt",
        "changes": [
            {"kind": "create_file", "details": {"path": "agent/x.py", "content": ""}},
            {"kind": "edit_solve_loop", "details": {"content": "def solve_task(t): pass"}},
        ],
    }
    legacy = {"kind": "edit_prompt", "details": {"rationale": "task t1"}}

    assert orchestrator._proposal_kinds(bundle) == ["create_file", "edit_solve_loop"]
    assert orchestrator._proposal_kind_label(bundle) == "bundle"
    assert orchestrator._proposal_intent(bundle) == "halt"
    assert orchestrator._proposal_rationale(bundle) == "task t1"

    assert orchestrator._proposal_kinds(legacy) == ["edit_prompt"]
    assert orchestrator._proposal_kind_label(legacy) == "edit_prompt"
    assert orchestrator._proposal_intent(legacy) == "iterate"
    assert orchestrator._proposal_rationale(legacy) == "task t1"


def test_read_mutable_files_covers_created_and_deleted_files(tmp_path):
    gen_dir = tmp_path
    (gen_dir / "agent").mkdir()
    (gen_dir / "agent" / "agent.py").write_text("def solve_task(t): pass\n")
    (gen_dir / "agent" / "extra.py").write_text("EXTRA = 1\n")
    (gen_dir / "tools").mkdir()
    (gen_dir / "tools" / "base.py").write_text("# tools\n")

    before = orchestrator._read_mutable_files(gen_dir)
    assert set(before) == {"agent/agent.py", "agent/extra.py", "tools/base.py"}

    (gen_dir / "agent" / "extra.py").unlink()
    (gen_dir / "tools" / "new_tool.py").write_text("NEW = 1\n")
    after = orchestrator._read_mutable_files(gen_dir)

    assert "agent/extra.py" not in after
    assert after["tools/new_tool.py"] == b"NEW = 1\n"
    assert before != after


def test_read_mutable_files_uses_manifest_roots(tmp_path):
    gen_dir = tmp_path
    (gen_dir / "agent").mkdir()
    (gen_dir / "agent" / "agent.py").write_text("def solve_task(t): pass\n")
    (gen_dir / "tools").mkdir()
    (gen_dir / "tools" / "base.py").write_text("# tools\n")
    (gen_dir / "knowledge").mkdir()
    (gen_dir / "knowledge" / "strategy.md").write_text("seed\n")
    manifest = {
        "snapshot_roots": ["agent", "tools", "knowledge"],
        "mutable_paths": ["agent/**", "tools/**", "knowledge/**"],
    }

    files = orchestrator._read_mutable_files(gen_dir, manifest=manifest)

    assert files["knowledge/strategy.md"] == b"seed\n"


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


def test_aggregate_summarizes_task_telemetry():
    config = {"model": {"name": "gpt-5.4-mini"}, "pricing": {}}
    results = [
        {
            "task_id": "a",
            "label": "safe",
            "expected": "safe",
            "telemetry": {
                "llm_calls": 1,
                "tool_calls_by_name": {"finalize": 1},
                "inspection_tool_calls": 0,
                "first_tool": "finalize",
                "used_finalize_tool": True,
                "immediate_finalize": True,
            },
        },
        {
            "task_id": "b",
            "label": "vulnerable",
            "expected": "vulnerable",
            "telemetry": {
                "llm_calls": 2,
                "tool_calls_by_name": {"static_scan": 1, "finalize": 1},
                "inspection_tool_calls": 1,
                "first_tool": "static_scan",
                "used_inspection_tool": True,
                "used_static_scan": True,
                "used_finalize_tool": True,
                "immediate_finalize": False,
            },
        },
    ]

    out = orchestrator.aggregate(results, config)

    assert out["telemetry"]["n_tasks"] == 2
    assert out["telemetry"]["llm_calls_total"] == 3
    assert out["telemetry"]["tool_calls_by_name"] == {
        "finalize": 2,
        "static_scan": 1,
    }
    assert out["telemetry"]["tasks_with_static_scan"] == 1
    assert out["telemetry"]["tasks_finalized_on_step_1"] == 1
    assert out["telemetry"]["inspection_tool_use_rate"] == pytest.approx(0.5)


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


def test_console_logger_formats_progress_without_prompt_payloads():
    stream = io.StringIO()
    logger = orchestrator.ConsoleLogger(stream=stream)
    logger.set_gen(2)

    logger("task_start", task_idx=1, n_tasks=3, task_id="t1")
    logger("llm_call",
           purpose="solve_task",
           task_id="t1",
           step_in_task=1,
           model="gpt-5.4-mini",
           duration_ms=1234,
           prompt_tokens=100,
           completion_tokens=25,
           messages=[{"role": "user", "content": "secret prompt payload"}])
    logger("task_result",
           task_idx=1,
           n_tasks=3,
           task_id="t1",
           predicted="safe",
           expected="vulnerable",
           ok=False,
           errored=False,
           n_llm_calls=1)

    out = stream.getvalue()
    assert "gen=2 task 1/3 start id=t1" in out
    assert "llm purpose=solve_task task=t1 step=1 model=gpt-5.4-mini" in out
    assert "tokens=100+25" in out
    assert "status=wrong pred=safe expected=vulnerable calls=1" in out
    assert "secret prompt payload" not in out


def test_console_logger_summarizes_tool_activity():
    stream = io.StringIO()
    logger = orchestrator.ConsoleLogger(stream=stream)

    logger("llm_call",
           purpose="solve_task",
           task_id="t1",
           step_in_task=2,
           model="gpt-5.4",
           duration_ms=250,
           prompt_tokens=200,
           completion_tokens=30,
           tool_results=["static_scan:c_cpp:findings=1:ext=skipped"],
           response_tools=["finalize"],
           response_kind="tools")

    out = stream.getvalue()
    assert "tool_results=static_scan:c_cpp:findings=1:ext=skipped" in out
    assert "response_tools=finalize" in out


def test_transcript_logger_writes_terminal_output_file(tmp_path):
    logger = orchestrator.TranscriptLogger(tmp_path)
    logger.set_gen(3)
    logger("task_start", task_idx=1, n_tasks=2, task_id="t1")
    logger.close()

    out_path = tmp_path / "terminal_output.out"
    assert out_path.exists()
    assert "gen=3 task 1/2 start id=t1" in out_path.read_text()


def test_event_fanout_writes_jsonl_and_console(tmp_path):
    run_logger = orchestrator.RunLogger(tmp_path)
    stream = io.StringIO()
    console = orchestrator.ConsoleLogger(stream=stream)
    logger = orchestrator.EventFanout(run_logger, console)

    logger.set_gen(4)
    logger("proposal",
           kind="edit_prompt",
           kinds=["edit_prompt"],
           intent="iterate",
           rationale="task t1")

    rec = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[0])
    assert rec["gen"] == 4
    assert rec["event"] == "proposal"
    assert "gen=4 proposal kind=edit_prompt intent=iterate" in stream.getvalue()


def test_validation_gate_rejects_candidate_with_execution_errors():
    parent = {"macro_f1": 0.70, "n_errors": 0}
    better_but_errorful = {"macro_f1": 0.90, "n_errors": 1}
    clean_improvement = {"macro_f1": 0.71, "n_errors": 0}

    assert orchestrator.validation_gate(parent, better_but_errorful) is False
    assert orchestrator.validation_gate(parent, clean_improvement) is True


def test_cost_estimate_uses_usage_model_not_only_default_model():
    config = {
        "model": {"name": "gpt-5.4-mini"},
        "pricing": {
            "gpt-5.4-mini": {"input": 1.0, "output": 2.0},
            "gpt-5.4": {"input": 10.0, "output": 20.0},
        },
    }
    usage = {
        "prompt_tokens": 1_000_000,
        "completion_tokens": 1_000_000,
        "model": "gpt-5.4-2026-03-17",
    }

    assert orchestrator._estimate_cost_usd(usage, config) == pytest.approx(30.0)


def test_pricing_model_match_prefers_longest_prefix():
    config = {
        "model": {"name": "gpt-5.4-mini"},
        "pricing": {
            "gpt-5.4": {"input": 10.0, "output": 20.0},
            "gpt-5.4-mini": {"input": 1.0, "output": 2.0},
        },
    }
    usage = {
        "prompt_tokens": 1_000_000,
        "completion_tokens": 1_000_000,
        "model": "gpt-5.4-mini-2026-03-17",
    }

    assert orchestrator._estimate_cost_usd(usage, config) == pytest.approx(3.0)


def test_llm_usage_tracker_counts_reflection_model_costs():
    config = {
        "model": {"name": "gpt-5.4-mini"},
        "pricing": {
            "gpt-5.4-mini": {"input": 1.0, "output": 2.0},
            "gpt-5.4": {"input": 10.0, "output": 20.0},
        },
    }
    tracker = orchestrator.LLMUsageTracker(config)

    tracker.add({
        "prompt_tokens": 1_000,
        "completion_tokens": 100,
        "model": "gpt-5.4-mini-2026-03-17",
    })
    tracker.add({
        "prompt_tokens": 2_000,
        "completion_tokens": 200,
        "model": "gpt-5.4-2026-03-17",
    })

    snapshot = tracker.snapshot()
    assert snapshot["n_calls"] == 2
    assert snapshot["prompt_tokens"] == 3_000
    assert snapshot["completion_tokens"] == 300
    assert snapshot["cost_usd"] == pytest.approx(
        ((1_000 * 1.0) + (100 * 2.0) + (2_000 * 10.0) + (200 * 20.0)) / 1_000_000
    )
