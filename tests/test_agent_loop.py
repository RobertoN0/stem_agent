"""Tests for the seed agent's solve_task loop behavior."""

import importlib


def test_agent_exposes_static_scan_instead_of_open_ended_bash():
    agent_module = importlib.import_module("agent.agent")
    tool_names = {
        spec["function"]["name"]
        for spec in agent_module._TOOL_SPECS
    }

    assert "static_scan" in tool_names
    assert "list_dir" in tool_names
    assert "run_bash" not in tool_names


def test_dispatch_static_scan_uses_current_task_code(monkeypatch):
    agent_module = importlib.import_module("agent.agent")
    calls = []

    def fake_static_scan(code, **kwargs):
        calls.append((code, kwargs))
        return {"language": "c_cpp", "heuristic_findings": [], "external": {"status": "skipped"}}

    monkeypatch.setattr(agent_module, "static_scan", fake_static_scan)

    result = agent_module._dispatch(
        "static_scan",
        {"language": "c_cpp", "run_external": True, "timeout": 3},
        task_code="int main(void) { return 0; }",
    )

    assert result["language"] == "c_cpp"
    assert calls == [
        (
            "int main(void) { return 0; }",
            {"language": "c_cpp", "run_external": True, "timeout": 3},
        )
    ]


def test_dispatch_list_dir_uses_bounded_tool(monkeypatch):
    agent_module = importlib.import_module("agent.agent")
    calls = []

    def fake_list_dir(path):
        calls.append(path)
        return ["agent.py", "prompt.txt"]

    monkeypatch.setattr(agent_module, "list_dir", fake_list_dir)

    result = agent_module._dispatch("list_dir", {"path": "/agent/agent"})

    assert result == ["agent.py", "prompt.txt"]
    assert calls == ["/agent/agent"]


def test_solve_task_accepts_text_only_safe_verdict(monkeypatch):
    agent_module = importlib.import_module("agent.agent")

    def fake_llm_call(*_args, **_kwargs):
        return {"content": "safe\nNo concrete vulnerability is visible.", "tool_calls": None}

    monkeypatch.setattr(agent_module, "llm_call", fake_llm_call)

    result = agent_module.solve_task({"id": "t_safe", "code": "int x = 0;"})

    assert result["task_id"] == "t_safe"
    assert result["label"] == "safe"
    assert "No concrete vulnerability" in result["reasoning"]


def test_solve_task_accepts_text_only_vulnerable_verdict_with_cwe(monkeypatch):
    agent_module = importlib.import_module("agent.agent")

    def fake_llm_call(*_args, **_kwargs):
        return {
            "content": "vulnerable CWE-120 The copy can overflow the destination.",
            "tool_calls": None,
        }

    monkeypatch.setattr(agent_module, "llm_call", fake_llm_call)

    result = agent_module.solve_task({"id": "t_vuln", "code": "strcpy(dst, src);"})

    assert result["label"] == "vulnerable"
    assert result["cwe"] == "CWE-120"


def test_solve_task_repairs_unstructured_text_by_requesting_finalize(monkeypatch):
    agent_module = importlib.import_module("agent.agent")
    calls = []

    def fake_llm_call(messages, **kwargs):
        calls.append(([dict(message) for message in messages], kwargs))
        if len(calls) == 1:
            return {"content": "I should inspect this manually.", "tool_calls": None}
        return {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "finalize",
                        "arguments": '{"label": "safe", "reasoning": "bounded wrapper"}',
                    },
                }
            ],
        }

    monkeypatch.setattr(agent_module, "llm_call", fake_llm_call)

    result = agent_module.solve_task({"id": "t_retry", "code": "return;"})

    assert result["label"] == "safe"
    assert len(calls) == 2
    assert "Use the finalize tool now" in calls[1][0][-1]["content"]
