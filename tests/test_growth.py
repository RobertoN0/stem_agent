"""Tests for growth/apply.py and growth/reflect.py.

apply.py is pure file IO + ast validation — tested directly. reflect.py is
driven with a fake llm_handler so we don't need an OpenAI key.
"""

import json
from pathlib import Path

import pytest

from growth.apply import apply_proposal
from growth.reflect import (
    reflect,
    _compute_confusion,
    _extract_tools_api,
    _build_messages,
    _parse_proposal,
)


# --- fixtures --------------------------------------------------------------

@pytest.fixture
def gen_dir(tmp_path: Path) -> Path:
    """A minimal gen snapshot (agent/agent.py, prompt.txt, tools/base.py)."""
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "agent.py").write_text(
        "from tools.base import llm_call\n\n"
        "def solve_task(task: dict) -> dict:\n"
        "    return {'task_id': task.get('id'), 'label': 'safe'}\n"
    )
    (tmp_path / "agent" / "prompt.txt").write_text("be a security reviewer.")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "base.py").write_text(
        '"""Tools."""\n\n'
        "def read_file(path: str) -> str:\n"
        '    """Read a file."""\n'
        "    return open(path).read()\n\n\n"
        "def llm_call(messages: list, model: str | None = None) -> dict:\n"
        '    """Make an LLM call via the host."""\n'
        "    return {'content': '', 'usage': {}}\n"
    )
    return tmp_path


# --- apply.py: edit_prompt --------------------------------------------------

def test_edit_prompt_writes_new_content(gen_dir):
    apply_proposal(
        {"kind": "edit_prompt",
         "details": {"content": "new prompt body", "rationale": "task t1 was wrong"}},
        str(gen_dir),
    )
    assert (gen_dir / "agent" / "prompt.txt").read_text() == "new prompt body"


def test_edit_prompt_rejects_empty_content(gen_dir):
    with pytest.raises(ValueError, match="non-empty"):
        apply_proposal(
            {"kind": "edit_prompt", "details": {"content": "   "}},
            str(gen_dir),
        )


# --- apply.py: edit_solve_loop ---------------------------------------------

def test_edit_solve_loop_replaces_agent_py(gen_dir):
    new_code = (
        "from tools.base import llm_call\n\n"
        "def solve_task(task: dict) -> dict:\n"
        "    return {'task_id': task.get('id'), 'label': 'vulnerable'}\n"
    )
    apply_proposal(
        {"kind": "edit_solve_loop",
         "details": {"content": new_code, "rationale": "t1"}},
        str(gen_dir),
    )
    assert "label': 'vulnerable'" in (gen_dir / "agent" / "agent.py").read_text()


def test_edit_solve_loop_rejects_missing_solve_task(gen_dir):
    with pytest.raises(ValueError, match="solve_task"):
        apply_proposal(
            {"kind": "edit_solve_loop",
             "details": {"content": "x = 1\n"}},
            str(gen_dir),
        )


def test_edit_solve_loop_rejects_syntax_error(gen_dir):
    with pytest.raises(ValueError, match="SyntaxError"):
        apply_proposal(
            {"kind": "edit_solve_loop",
             "details": {"content": "def solve_task(t): return {\n"}},
            str(gen_dir),
        )


# --- apply.py: add_tool ----------------------------------------------------

def test_add_tool_appends_function(gen_dir):
    code = (
        "def count_branches(src: str) -> int:\n"
        '    """Count branch keywords in src."""\n'
        "    import re\n"
        "    return len(re.findall(r'\\b(if|while|for|switch)\\b', src))\n"
    )
    apply_proposal(
        {"kind": "add_tool",
         "details": {"name": "count_branches", "code": code, "rationale": "t1"}},
        str(gen_dir),
    )
    new_src = (gen_dir / "tools" / "base.py").read_text()
    assert "def count_branches" in new_src
    # Existing functions still present
    assert "def read_file" in new_src
    assert "def llm_call" in new_src


def test_add_tool_rejects_shadowing_existing_name(gen_dir):
    code = "def read_file(p):\n    return ''\n"
    with pytest.raises(ValueError, match="already defined"):
        apply_proposal(
            {"kind": "add_tool",
             "details": {"name": "read_file", "code": code}},
            str(gen_dir),
        )


def test_add_tool_rejects_disallowed_import(gen_dir):
    code = (
        "def use_requests(url: str) -> str:\n"
        "    import requests\n"
        "    return requests.get(url).text\n"
    )
    with pytest.raises(ValueError, match="not in the allowlist"):
        apply_proposal(
            {"kind": "add_tool",
             "details": {"name": "use_requests", "code": code}},
            str(gen_dir),
        )


def test_add_tool_rejects_multiple_functions(gen_dir):
    code = "def a():\n    pass\n\ndef b():\n    pass\n"
    with pytest.raises(ValueError, match="exactly one function"):
        apply_proposal(
            {"kind": "add_tool", "details": {"name": "a", "code": code}},
            str(gen_dir),
        )


def test_add_tool_rejects_name_mismatch(gen_dir):
    code = "def actual_name():\n    pass\n"
    with pytest.raises(ValueError, match="expected 'declared_name'"):
        apply_proposal(
            {"kind": "add_tool",
             "details": {"name": "declared_name", "code": code}},
            str(gen_dir),
        )


def test_add_tool_rejects_invalid_identifier(gen_dir):
    with pytest.raises(ValueError, match="valid identifier"):
        apply_proposal(
            {"kind": "add_tool",
             "details": {"name": "1bad-name", "code": "def x(): pass\n"}},
            str(gen_dir),
        )


def test_add_tool_allows_stdlib_from_allowlist(gen_dir):
    code = (
        "def hash_str(s: str) -> str:\n"
        '    """Hash a string."""\n'
        "    import hashlib\n"
        "    return hashlib.sha256(s.encode()).hexdigest()\n"
    )
    apply_proposal(
        {"kind": "add_tool",
         "details": {"name": "hash_str", "code": code}},
        str(gen_dir),
    )
    assert "hash_str" in (gen_dir / "tools" / "base.py").read_text()


# --- apply.py: multi-edit bundles -----------------------------------------

def test_multi_edit_bundle_applies_all_changes(gen_dir):
    tool_code = (
        "def count_chars(src: str) -> int:\n"
        '    """Count source characters."""\n'
        "    return len(src)\n"
    )
    apply_proposal(
        {
            "rationale": "task t1 showed weak prompt and missing helper",
            "intent": "iterate",
            "changes": [
                {"kind": "edit_prompt", "details": {"content": "bundle prompt"}},
                {"kind": "create_file",
                 "details": {"path": "agent/heuristics.py", "content": "RULES = []\n"}},
                {"kind": "add_tool",
                 "details": {"name": "count_chars", "code": tool_code}},
            ],
        },
        str(gen_dir),
    )

    assert (gen_dir / "agent" / "prompt.txt").read_text() == "bundle prompt"
    assert (gen_dir / "agent" / "heuristics.py").read_text() == "RULES = []\n"
    assert "def count_chars" in (gen_dir / "tools" / "base.py").read_text()


def test_multi_edit_bundle_rolls_back_when_later_change_invalid(gen_dir):
    original_prompt = (gen_dir / "agent" / "prompt.txt").read_text()
    original_tools = (gen_dir / "tools" / "base.py").read_text()

    with pytest.raises(ValueError, match="not in the allowlist"):
        apply_proposal(
            {
                "rationale": "task t1",
                "intent": "iterate",
                "changes": [
                    {"kind": "edit_prompt", "details": {"content": "should rollback"}},
                    {"kind": "add_tool",
                     "details": {
                         "name": "fetch_url",
                         "code": (
                             "def fetch_url(url: str) -> str:\n"
                             "    import requests\n"
                             "    return requests.get(url).text\n"
                         ),
                     }},
                ],
            },
            str(gen_dir),
        )

    assert (gen_dir / "agent" / "prompt.txt").read_text() == original_prompt
    assert (gen_dir / "tools" / "base.py").read_text() == original_tools


# --- apply.py: edit_tool / delete_tool ------------------------------------

def test_edit_tool_replaces_existing_function(gen_dir):
    code = (
        "def read_file(path: str) -> str:\n"
        '    """Read a file with a marker."""\n'
        "    return 'patched:' + path\n"
    )
    apply_proposal(
        {"kind": "edit_tool", "details": {"name": "read_file", "code": code}},
        str(gen_dir),
    )
    new_src = (gen_dir / "tools" / "base.py").read_text()
    assert "patched:" in new_src
    assert "open(path).read()" not in new_src


def test_edit_tool_rejects_missing_function(gen_dir):
    code = "def missing_tool() -> int:\n    return 1\n"
    with pytest.raises(ValueError, match="function not found"):
        apply_proposal(
            {"kind": "edit_tool",
             "details": {"name": "missing_tool", "code": code}},
            str(gen_dir),
        )


def test_delete_tool_removes_existing_function(gen_dir):
    tools_path = gen_dir / "tools" / "base.py"
    tools_path.write_text(
        tools_path.read_text()
        + "\n\ndef helper_tool() -> int:\n"
        + "    return 1\n"
    )

    apply_proposal(
        {"kind": "delete_tool", "details": {"name": "helper_tool"}},
        str(gen_dir),
    )
    assert "def helper_tool" not in tools_path.read_text()


def test_delete_tool_rejects_protected_core_tool(gen_dir):
    with pytest.raises(ValueError, match="protected core tool"):
        apply_proposal(
            {"kind": "delete_tool", "details": {"name": "read_file"}},
            str(gen_dir),
        )


# --- apply.py: create_file / delete_file ----------------------------------

def test_create_file_creates_nested_file_under_agent(gen_dir):
    apply_proposal(
        {"kind": "create_file",
         "details": {"path": "agent/heuristics/rules.py", "content": "RULES = []\n"}},
        str(gen_dir),
    )
    assert (gen_dir / "agent" / "heuristics" / "rules.py").read_text() == "RULES = []\n"


def test_create_file_rejects_path_escape(gen_dir):
    with pytest.raises(ValueError, match="relative"):
        apply_proposal(
            {"kind": "create_file",
             "details": {"path": "../escape.py", "content": "x = 1\n"}},
            str(gen_dir),
        )


def test_delete_file_deletes_existing_file(gen_dir):
    target = gen_dir / "tools" / "obsolete.py"
    target.write_text("OLD = True\n")
    apply_proposal(
        {"kind": "delete_file", "details": {"path": "tools/obsolete.py"}},
        str(gen_dir),
    )
    assert not target.exists()


def test_delete_file_rejects_protected_file(gen_dir):
    with pytest.raises(ValueError, match="protected file"):
        apply_proposal(
            {"kind": "delete_file", "details": {"path": "agent/agent.py"}},
            str(gen_dir),
        )


# --- apply.py: top-level rejection -----------------------------------------

def test_unknown_kind_raises(gen_dir):
    with pytest.raises(ValueError, match="kind must be one of"):
        apply_proposal(
            {"kind": "rewrite_universe", "details": {}},
            str(gen_dir),
        )


def test_proposal_must_be_dict(gen_dir):
    with pytest.raises(ValueError, match="proposal must be a dict"):
        apply_proposal("not a dict", str(gen_dir))


# --- reflect.py: helpers ---------------------------------------------------

def test_compute_confusion_basic():
    per_task = [
        {"expected": "vulnerable", "predicted": "vulnerable", "ok": True, "errored": False},
        {"expected": "vulnerable", "predicted": "safe", "ok": False, "errored": False},
        {"expected": "vulnerable", "predicted": None, "ok": False, "errored": False},
        {"expected": "vulnerable", "predicted": None, "ok": False, "errored": True},
        {"expected": "safe", "predicted": "safe", "ok": True, "errored": False},
        {"expected": "safe", "predicted": "vulnerable", "ok": False, "errored": False},
    ]
    cm = _compute_confusion(per_task)
    assert cm["tp_v"] == 1
    assert cm["fn_v"] == 2  # one wrong-label + one None, errors counted separately
    assert cm["err_v"] == 1
    assert cm["fp_s"] == 1
    assert cm["tn_s"] == 1
    # recall_v denom = TP + FN + err = 1 + 2 + 1 = 4
    assert cm["rec_v"] == pytest.approx(1 / 4)
    # precision_v denom = TP + FP = 1 + 1 = 2
    assert cm["prec_v"] == pytest.approx(1 / 2)


def test_extract_tools_api_pulls_signatures(gen_dir):
    api = _extract_tools_api(gen_dir)
    assert "def read_file(path: str) -> str:" in api
    assert "Read a file." in api
    assert "def llm_call" in api


def test_build_messages_includes_failures_and_code(gen_dir):
    score = {
        "per_task": [
            {"task_id": "t1", "expected": "vulnerable",
             "predicted": "safe", "ok": False, "errored": False},
        ],
        "macro_f1": 0.0, "accuracy": 0.0, "n_tasks": 1, "n_errors": 0,
    }
    trajectories = [
        {"task_id": "t1", "expected": "vulnerable", "predicted": "safe",
         "ok": False, "errored": False, "raw": "{\"label\":\"safe\"}",
         "code": "char buf[10]; gets(buf);"},
    ]
    messages = _build_messages(
        gen_idx=0, score=score, trajectories=trajectories,
        agent_py="def solve_task(t): pass",
        prompt_txt="be a reviewer",
        tools_api="def llm_call(...): ...",
    )
    user = messages[1]["content"]
    assert "task_id=t1" in user
    assert "gets(buf)" in user
    assert "FAILURE 1" in user
    assert "edit_prompt" in user
    assert "edit_solve_loop" in user
    assert "add_tool" in user
    assert "edit_tool" in user
    assert "create_file" in user
    assert "\"intent\": \"iterate\"" in user
    assert "\"changes\"" in user


# --- reflect.py: handler interaction ---------------------------------------

def _fake_handler_returning(content: str, calls: list | None = None):
    """Build a fake llm_handler that returns the given content as ok response."""
    def handler(env):
        if calls is not None:
            calls.append(env)
        return {"_kind": "llm_response", "id": env["id"],
                "ok": True, "content": content,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    return handler


def _fake_handler_failing():
    def handler(env):
        return {"_kind": "llm_response", "id": env["id"],
                "ok": False, "error": "API down"}
    return handler


def test_reflect_returns_parsed_proposal(gen_dir):
    proposal_json = json.dumps({
        "rationale": "task t1 showed prompt confusion",
        "intent": "iterate",
        "changes": [
            {"kind": "edit_prompt", "details": {"content": "new prompt"}},
        ],
    })
    calls = []
    handler = _fake_handler_returning(proposal_json, calls=calls)

    out = reflect(
        trajectories=[],
        current_gen_dir=str(gen_dir),
        llm_handler=handler,
        config={"model": {"name": "gpt-5.4-mini"}},
        score={"per_task": [], "macro_f1": 0.0, "accuracy": 0.0, "n_tasks": 0},
        gen_idx=0,
    )
    assert out["rationale"] == "task t1 showed prompt confusion"
    assert out["intent"] == "iterate"
    assert out["changes"][0]["kind"] == "edit_prompt"
    assert out["changes"][0]["details"]["content"] == "new prompt"

    # The envelope must carry purpose=reflect, task_id=None, response_format=json_object
    assert calls[0]["purpose"] == "reflect"
    assert calls[0]["task_id"] is None
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_reflect_returns_none_when_handler_fails(gen_dir):
    out = reflect(
        trajectories=[],
        current_gen_dir=str(gen_dir),
        llm_handler=_fake_handler_failing(),
        config={"model": {"name": "gpt-5.4-mini"}},
        score={"per_task": []},
    )
    assert out is None


def test_reflect_retries_once_on_bad_json(gen_dir):
    """First response is garbage; second must be valid JSON. Should succeed."""
    valid = json.dumps({
        "rationale": "task t1",
        "intent": "iterate",
        "changes": [{"kind": "edit_prompt", "details": {"content": "x"}}],
    })
    responses = ["this is not json", valid]
    calls = []

    def handler(env):
        calls.append(env)
        return {"_kind": "llm_response", "id": env["id"],
                "ok": True, "content": responses[len(calls) - 1],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    out = reflect(
        trajectories=[],
        current_gen_dir=str(gen_dir),
        llm_handler=handler,
        config={"model": {"name": "gpt-5.4-mini"}},
        score={"per_task": []},
    )
    assert out["changes"][0]["kind"] == "edit_prompt"
    assert len(calls) == 2  # original + one retry


def test_reflect_wraps_legacy_single_edit_proposal(gen_dir):
    legacy = json.dumps({
        "kind": "edit_prompt",
        "details": {"content": "legacy prompt", "rationale": "task t1"},
    })

    out = reflect(
        trajectories=[],
        current_gen_dir=str(gen_dir),
        llm_handler=_fake_handler_returning(legacy),
        config={"model": {"name": "gpt-5.4-mini"}},
        score={"per_task": []},
    )

    assert out == {
        "rationale": "task t1",
        "intent": "iterate",
        "changes": [
            {
                "kind": "edit_prompt",
                "details": {"content": "legacy prompt", "rationale": "task t1"},
            },
        ],
    }


def test_parse_proposal_accepts_halt_with_empty_changes():
    out = _parse_proposal(json.dumps({
        "rationale": "validation is saturated on shown tasks",
        "intent": "halt",
        "changes": [],
    }))
    assert out["intent"] == "halt"
    assert out["changes"] == []


def test_parse_proposal_rejects_bad_change_details():
    with pytest.raises(ValueError, match="changes\\[0\\]\\.details"):
        _parse_proposal(json.dumps({
            "rationale": "task t1",
            "intent": "iterate",
            "changes": [{"kind": "edit_prompt", "details": "not an object"}],
        }))


def test_reflect_raises_when_retry_also_fails(gen_dir):
    def handler(env):
        return {"_kind": "llm_response", "id": env["id"],
                "ok": True, "content": "still not json",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    with pytest.raises(ValueError):
        reflect(
            trajectories=[],
            current_gen_dir=str(gen_dir),
            llm_handler=handler,
            config={"model": {"name": "gpt-5.4-mini"}},
            score={"per_task": []},
        )


def test_reflect_raises_when_kind_invalid(gen_dir):
    bad = json.dumps({"kind": "delete_universe", "details": {"x": 1}})

    def handler(env):
        return {"_kind": "llm_response", "id": env["id"],
                "ok": True, "content": bad,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    with pytest.raises(ValueError):
        reflect(
            trajectories=[],
            current_gen_dir=str(gen_dir),
            llm_handler=handler,
            config={"model": {"name": "gpt-5.4-mini"}},
            score={"per_task": []},
        )
