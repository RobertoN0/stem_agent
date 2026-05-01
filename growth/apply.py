"""Apply a JSON proposal to a snapshot directory, producing the next generation.

Translates structured proposal changes into concrete edits inside a candidate
generation directory. Edits are confined by mutation_manifest.yaml.

Two proposal formats are accepted:
- legacy single-edit: {"kind": "...", "details": {...}}
- bundle: {"rationale": "...", "intent": "iterate"|"halt", "changes": [...]}

Validation is strict and atomic: every requested change is validated into an
in-memory file plan before any write happens. If a write/delete fails, touched
files are restored to their pre-apply bytes.
"""

import ast
from pathlib import Path

from growth.manifest import (
    is_mutable_path,
    is_protected_file,
    load_mutation_manifest,
    normalize_rel_path,
    protected_symbols_for,
)


_VALID_KINDS = frozenset([
    "edit_prompt",
    "edit_solve_loop",
    "add_tool",
    "edit_tool",
    "delete_tool",
    "create_file",
    "delete_file",
    "replace_file",
    "add_function",
    "replace_function",
])

_PROTECTED_TOOLS = frozenset([
    "llm_call",
    "read_file",
    "write_file",
    "list_dir",
    "run_bash",
    "note",
    "_set_rpc_channels",
    "_set_task_context",
    "_safe_path",
])

_PROTECTED_FILES = frozenset([
    "agent/agent.py",
    "agent/prompt.txt",
    "tools/base.py",
])

# Modules an add_tool/edit_tool function may import. Keep tight — wider
# surface means wider risk for the agent to depend on host-only libraries.
_ALLOWED_IMPORTS = frozenset([
    "re", "ast", "json", "os", "os.path", "pathlib", "subprocess",
    "math", "collections", "itertools", "functools", "typing",
    "datetime", "hashlib", "base64", "textwrap", "uuid",
    "__future__", "dataclasses", "tools", "tools.base", "agent",
])

_FORBIDDEN_IMPORT_ROOTS = frozenset([
    "openai",
    "requests",
    "httpx",
    "urllib",
    "socket",
    "datasets",
])


def _ensure_dict(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dict, got {type(value).__name__}")
    return value


def _ensure_change(change, name: str) -> tuple[str, dict]:
    change = _ensure_dict(change, name)
    kind = change.get("kind")
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"{name}.kind must be one of {sorted(_VALID_KINDS)}, got {kind!r}"
        )
    details = _ensure_dict(change.get("details"), f"{name}.details")
    return kind, details


def _normalize_proposal(proposal: dict) -> list[tuple[str, dict]]:
    proposal = _ensure_dict(proposal, "proposal")

    # Backwards compatibility for the old single-edit schema.
    if "kind" in proposal:
        return [_ensure_change(proposal, "proposal")]

    rationale = proposal.get("rationale")
    if not isinstance(rationale, str):
        raise ValueError("proposal.rationale must be a string")

    intent = proposal.get("intent")
    if intent not in ("iterate", "halt"):
        raise ValueError("proposal.intent must be 'iterate' or 'halt'")

    changes = proposal.get("changes")
    if not isinstance(changes, list):
        raise ValueError("proposal.changes must be a list")

    return [
        _ensure_change(change, f"proposal.changes[{idx}]")
        for idx, change in enumerate(changes)
    ]


def _check_import_allowlist(tree: ast.Module, kind: str, func_name: str) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _FORBIDDEN_IMPORT_ROOTS:
                    raise ValueError(
                        f"{kind} '{func_name}' imports forbidden module '{alias.name}'"
                    )
                if alias.name not in _ALLOWED_IMPORTS and root not in _ALLOWED_IMPORTS:
                    raise ValueError(
                        f"{kind} '{func_name}' imports '{alias.name}' which is "
                        f"not in the allowlist"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            if root in _FORBIDDEN_IMPORT_ROOTS:
                raise ValueError(
                    f"{kind} '{func_name}' imports from forbidden module '{module}'"
                )
            if module not in _ALLOWED_IMPORTS and root not in _ALLOWED_IMPORTS:
                raise ValueError(
                    f"{kind} '{func_name}' imports from '{module}' which is "
                    f"not in the allowlist"
                )


def _validate_tool_code(kind: str, name: str, code: str) -> None:
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"{kind}: details.name must be a valid identifier, got {name!r}")
    if not isinstance(code, str) or not code.strip():
        raise ValueError(f"{kind}: details.code must be a non-empty string")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"{kind} '{name}': SyntaxError: {e}")

    func_defs = [
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(func_defs) != 1:
        raise ValueError(
            f"{kind} '{name}': code must define exactly one function, "
            f"got {len(func_defs)}"
        )
    if func_defs[0].name != name:
        raise ValueError(
            f"{kind} '{name}': function name in code is "
            f"'{func_defs[0].name}', expected '{name}'"
        )

    _check_import_allowlist(tree, kind, name)


def _existing_top_level_names(source: str) -> set:
    """Return all top-level function/class/assignment names in tools/base.py."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    names.add(tgt.id)
    return names


def _find_top_level_function(source: str, name: str, kind: str):
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"{kind} '{name}': tools/base.py has SyntaxError: {e}")

    matches = [
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    ]
    if not matches:
        raise ValueError(f"{kind} '{name}': function not found in tools/base.py")
    if len(matches) > 1:
        raise ValueError(f"{kind} '{name}': function defined multiple times")
    node = matches[0]
    if getattr(node, "end_lineno", None) is None:
        raise ValueError(f"{kind} '{name}': could not determine function span")
    return node


def _function_span(node) -> tuple[int, int]:
    start = node.lineno
    if getattr(node, "decorator_list", None):
        start = min([start] + [d.lineno for d in node.decorator_list])
    return start - 1, node.end_lineno


def _replace_top_level_function(source: str, name: str, code: str, kind: str) -> str:
    node = _find_top_level_function(source, name, kind)
    start, end = _function_span(node)
    lines = source.splitlines()
    new_lines = lines[:start] + code.strip().splitlines() + lines[end:]
    return "\n".join(new_lines).rstrip() + "\n"


def _delete_top_level_function(source: str, name: str, kind: str) -> str:
    node = _find_top_level_function(source, name, kind)
    start, end = _function_span(node)
    lines = source.splitlines()
    new_lines = lines[:start] + lines[end:]
    return "\n".join(new_lines).rstrip() + "\n"


def _rel_key(rel: Path) -> str:
    return rel.as_posix()


def _read_virtual(gen_dir: Path, planned: dict[str, str | None], rel: Path, kind: str):
    key = _rel_key(rel)
    if key in planned:
        return planned[key]

    target = gen_dir / rel
    if not target.exists():
        return None
    if not target.is_file():
        raise ValueError(f"{kind}: target is not a file: {key}")
    return target.read_text()


def _require_virtual_file(
    gen_dir: Path,
    planned: dict[str, str | None],
    rel: Path,
    kind: str,
) -> str:
    content = _read_virtual(gen_dir, planned, rel, kind)
    if content is None:
        raise ValueError(f"{kind}: target file missing: {_rel_key(rel)}")
    return content


def _set_virtual(planned: dict[str, str | None], rel: Path, content: str | None) -> None:
    planned[_rel_key(rel)] = content


def _safe_rel_path(path, kind: str, manifest: dict) -> Path:
    try:
        key = normalize_rel_path(path)
    except ValueError as e:
        raise ValueError(f"{kind}: details.path {e}") from e
    if not is_mutable_path(key, manifest):
        raise ValueError(f"{kind}: path is not mutable by manifest: {key}")
    return Path(key)


def _validate_python_source(kind: str, rel: Path, content: str) -> None:
    if rel.suffix != ".py":
        return
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        raise ValueError(f"{kind}: {rel.as_posix()} has SyntaxError: {e}")
    _check_import_allowlist(tree, kind, rel.as_posix())


def _validate_replacement_file(
    kind: str,
    rel: Path,
    content: str,
    existing: str | None,
    manifest: dict,
) -> None:
    if rel.as_posix() == "agent/agent.py" and "def solve_task" not in content:
        raise ValueError(f"{kind}: agent/agent.py must define solve_task")
    if rel.as_posix() == "tools/base.py" and protected_symbols_for(rel.as_posix(), manifest):
        raise ValueError(
            f"{kind}: tools/base.py contains protected symbols; use "
            "add_function or replace_function for non-protected functions"
        )
    _validate_python_source(kind, rel, content)


def _reject_protected_symbol(kind: str, rel: Path, name: str, manifest: dict) -> None:
    protected = protected_symbols_for(rel.as_posix(), manifest)
    if name in protected:
        raise ValueError(
            f"{kind} '{name}': protected symbol cannot be changed in {rel.as_posix()}"
        )


def _plan_edit_prompt(
    details: dict,
    gen_dir: Path,
    planned: dict[str, str | None],
    manifest: dict,
) -> None:
    content = details.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("edit_prompt: details.content must be a non-empty string")
    rel = Path("agent/prompt.txt")
    if not (gen_dir / rel.parent).exists():
        raise ValueError(f"edit_prompt: target dir missing: {gen_dir / rel.parent}")
    _set_virtual(planned, rel, content)


def _plan_edit_solve_loop(
    details: dict,
    gen_dir: Path,
    planned: dict[str, str | None],
    manifest: dict,
) -> None:
    content = details.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("edit_solve_loop: details.content must be a non-empty string")
    if "def solve_task" not in content:
        raise ValueError("edit_solve_loop: content must define solve_task")
    try:
        ast.parse(content)
    except SyntaxError as e:
        raise ValueError(f"edit_solve_loop: content has SyntaxError: {e}")
    _validate_python_source("edit_solve_loop", Path("agent/agent.py"), content)
    rel = Path("agent/agent.py")
    if not (gen_dir / rel.parent).exists():
        raise ValueError(f"edit_solve_loop: target dir missing: {gen_dir / rel.parent}")
    _set_virtual(planned, rel, content)


def _plan_add_tool(
    details: dict,
    gen_dir: Path,
    planned: dict[str, str | None],
    manifest: dict,
) -> None:
    name = details.get("name")
    code = details.get("code")
    _validate_tool_code("add_tool", name, code)

    rel = Path("tools/base.py")
    _reject_protected_symbol("add_tool", rel, name, manifest)
    existing = _require_virtual_file(gen_dir, planned, rel, "add_tool")
    if name in _existing_top_level_names(existing):
        raise ValueError(f"add_tool '{name}': name already defined in tools/base.py")

    _set_virtual(planned, rel, existing.rstrip() + "\n\n\n" + code.strip() + "\n")


def _plan_edit_tool(
    details: dict,
    gen_dir: Path,
    planned: dict[str, str | None],
    manifest: dict,
) -> None:
    name = details.get("name")
    code = details.get("code")
    _validate_tool_code("edit_tool", name, code)

    rel = Path("tools/base.py")
    _reject_protected_symbol("edit_tool", rel, name, manifest)
    existing = _require_virtual_file(gen_dir, planned, rel, "edit_tool")
    _set_virtual(planned, rel, _replace_top_level_function(existing, name, code, "edit_tool"))


def _plan_delete_tool(
    details: dict,
    gen_dir: Path,
    planned: dict[str, str | None],
    manifest: dict,
) -> None:
    name = details.get("name")
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"delete_tool: details.name must be a valid identifier, got {name!r}")
    rel = Path("tools/base.py")
    if name in _PROTECTED_TOOLS or name in protected_symbols_for(rel.as_posix(), manifest):
        raise ValueError(f"delete_tool '{name}': protected core tool cannot be deleted")

    existing = _require_virtual_file(gen_dir, planned, rel, "delete_tool")
    _set_virtual(planned, rel, _delete_top_level_function(existing, name, "delete_tool"))


def _plan_create_file(
    details: dict,
    gen_dir: Path,
    planned: dict[str, str | None],
    manifest: dict,
) -> None:
    rel = _safe_rel_path(details.get("path"), "create_file", manifest)
    content = details.get("content")
    if not isinstance(content, str):
        raise ValueError("create_file: details.content must be a string")
    if _read_virtual(gen_dir, planned, rel, "create_file") is not None:
        raise ValueError(f"create_file: target already exists: {_rel_key(rel)}")
    _validate_python_source("create_file", rel, content)
    _set_virtual(planned, rel, content)


def _plan_delete_file(
    details: dict,
    gen_dir: Path,
    planned: dict[str, str | None],
    manifest: dict,
) -> None:
    rel = _safe_rel_path(details.get("path"), "delete_file", manifest)
    key = _rel_key(rel)
    if key in _PROTECTED_FILES or is_protected_file(key, manifest):
        raise ValueError(f"delete_file: protected file cannot be deleted: {key}")
    if _read_virtual(gen_dir, planned, rel, "delete_file") is None:
        raise ValueError(f"delete_file: target file missing: {key}")
    _set_virtual(planned, rel, None)


def _plan_replace_file(
    details: dict,
    gen_dir: Path,
    planned: dict[str, str | None],
    manifest: dict,
) -> None:
    rel = _safe_rel_path(details.get("path"), "replace_file", manifest)
    content = details.get("content")
    if not isinstance(content, str):
        raise ValueError("replace_file: details.content must be a string")
    existing = _read_virtual(gen_dir, planned, rel, "replace_file")
    if existing is None:
        raise ValueError(f"replace_file: target file missing: {_rel_key(rel)}")
    _validate_replacement_file("replace_file", rel, content, existing, manifest)
    _set_virtual(planned, rel, content)


def _function_details(details: dict, kind: str) -> tuple[str, str, str]:
    rel = details.get("path")
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError(f"{kind}: details.path must be a non-empty string")
    name = details.get("name")
    code = details.get("code")
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"{kind}: details.name must be a valid identifier, got {name!r}")
    if not isinstance(code, str) or not code.strip():
        raise ValueError(f"{kind}: details.code must be a non-empty string")
    return rel, name, code


def _plan_add_function(
    details: dict,
    gen_dir: Path,
    planned: dict[str, str | None],
    manifest: dict,
) -> None:
    raw_path, name, code = _function_details(details, "add_function")
    rel = _safe_rel_path(raw_path, "add_function", manifest)
    if rel.suffix != ".py":
        raise ValueError(f"add_function: target must be a Python file: {_rel_key(rel)}")
    _reject_protected_symbol("add_function", rel, name, manifest)
    _validate_tool_code("add_function", name, code)

    existing = _require_virtual_file(gen_dir, planned, rel, "add_function")
    if name in _existing_top_level_names(existing):
        raise ValueError(f"add_function '{name}': name already defined in {_rel_key(rel)}")
    _set_virtual(planned, rel, existing.rstrip() + "\n\n\n" + code.strip() + "\n")


def _plan_replace_function(
    details: dict,
    gen_dir: Path,
    planned: dict[str, str | None],
    manifest: dict,
) -> None:
    raw_path, name, code = _function_details(details, "replace_function")
    rel = _safe_rel_path(raw_path, "replace_function", manifest)
    if rel.suffix != ".py":
        raise ValueError(f"replace_function: target must be a Python file: {_rel_key(rel)}")
    _reject_protected_symbol("replace_function", rel, name, manifest)
    _validate_tool_code("replace_function", name, code)

    existing = _require_virtual_file(gen_dir, planned, rel, "replace_function")
    _set_virtual(
        planned,
        rel,
        _replace_top_level_function(existing, name, code, "replace_function"),
    )


_PLANNERS = {
    "edit_prompt": _plan_edit_prompt,
    "edit_solve_loop": _plan_edit_solve_loop,
    "add_tool": _plan_add_tool,
    "edit_tool": _plan_edit_tool,
    "delete_tool": _plan_delete_tool,
    "create_file": _plan_create_file,
    "delete_file": _plan_delete_file,
    "replace_file": _plan_replace_file,
    "add_function": _plan_add_function,
    "replace_function": _plan_replace_function,
}


def _apply_planned(gen_dir: Path, planned: dict[str, str | None]) -> None:
    backups: dict[str, bytes | None] = {}

    try:
        for key in planned:
            target = gen_dir / key
            if target.exists():
                if not target.is_file():
                    raise ValueError(f"apply: target is not a file: {key}")
                backups[key] = target.read_bytes()
            else:
                backups[key] = None

        for key, content in planned.items():
            target = gen_dir / key
            if content is None:
                if target.exists():
                    target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
    except Exception:
        for key, original in backups.items():
            target = gen_dir / key
            try:
                if original is None:
                    if target.exists() and target.is_file():
                        target.unlink()
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(original)
            except Exception:
                pass
        raise


def apply_proposal(
    proposal: dict,
    target_gen_dir: str,
    manifest: dict | None = None,
) -> None:
    """Apply a proposal to a candidate snapshot directory, mutating files in place."""
    changes = _normalize_proposal(proposal)

    gen_dir = Path(target_gen_dir)
    if not gen_dir.exists():
        raise ValueError(f"target gen dir missing: {gen_dir}")
    if manifest is None:
        manifest = load_mutation_manifest(gen_dir)

    planned: dict[str, str | None] = {}
    for kind, details in changes:
        _PLANNERS[kind](details, gen_dir, planned, manifest)

    _apply_planned(gen_dir, planned)
