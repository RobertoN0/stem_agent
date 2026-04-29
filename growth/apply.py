"""Apply a JSON proposal to a snapshot directory, producing the next generation.

Translates the structured proposal kinds (`add_tool`, `edit_prompt`,
`edit_solve_loop`) into concrete file edits inside the candidate gen
directory. Edits are confined to agent/ and tools/.

Validation is strict: a malformed proposal raises ValueError so the
orchestrator can log `proposal_invalid` and move on without a half-applied
snapshot. The orchestrator handles the no-op case (file byte-identical to
parent) by comparing before/after — apply just writes what it's told.
"""

import ast
from pathlib import Path


_VALID_KINDS = frozenset(["edit_prompt", "edit_solve_loop", "add_tool"])

# Modules an `add_tool` function may import. Keep tight — wider surface =
# wider risk for the agent to depend on host-only libraries.
_ALLOWED_IMPORTS = frozenset([
    "re", "ast", "json", "os", "os.path", "pathlib", "subprocess",
    "math", "collections", "itertools", "functools", "typing",
    "datetime", "hashlib", "base64", "textwrap",
])


def _ensure_dict(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dict, got {type(value).__name__}")
    return value


def _check_import_allowlist(tree: ast.Module, func_name: str) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if alias.name not in _ALLOWED_IMPORTS and root not in _ALLOWED_IMPORTS:
                    raise ValueError(
                        f"add_tool '{func_name}' imports '{alias.name}' which is "
                        f"not in the allowlist"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            if module not in _ALLOWED_IMPORTS and root not in _ALLOWED_IMPORTS:
                raise ValueError(
                    f"add_tool '{func_name}' imports from '{module}' which is "
                    f"not in the allowlist"
                )


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


def _apply_edit_prompt(details: dict, gen_dir: Path) -> None:
    content = details.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("edit_prompt: details.content must be a non-empty string")
    target = gen_dir / "agent" / "prompt.txt"
    if not target.parent.exists():
        raise ValueError(f"edit_prompt: target dir missing: {target.parent}")
    target.write_text(content)


def _apply_edit_solve_loop(details: dict, gen_dir: Path) -> None:
    content = details.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("edit_solve_loop: details.content must be a non-empty string")
    if "def solve_task" not in content:
        raise ValueError("edit_solve_loop: content must define solve_task")
    try:
        ast.parse(content)
    except SyntaxError as e:
        raise ValueError(f"edit_solve_loop: content has SyntaxError: {e}")
    target = gen_dir / "agent" / "agent.py"
    if not target.parent.exists():
        raise ValueError(f"edit_solve_loop: target dir missing: {target.parent}")
    target.write_text(content)


def _apply_add_tool(details: dict, gen_dir: Path) -> None:
    name = details.get("name")
    code = details.get("code")
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"add_tool: details.name must be a valid identifier, got {name!r}")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("add_tool: details.code must be a non-empty string")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"add_tool '{name}': SyntaxError: {e}")

    func_defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(func_defs) != 1:
        raise ValueError(
            f"add_tool '{name}': code must define exactly one function, "
            f"got {len(func_defs)}"
        )
    if func_defs[0].name != name:
        raise ValueError(
            f"add_tool '{name}': function name in code is "
            f"'{func_defs[0].name}', expected '{name}'"
        )

    _check_import_allowlist(tree, name)

    tools_path = gen_dir / "tools" / "base.py"
    if not tools_path.exists():
        raise ValueError(f"add_tool '{name}': tools/base.py missing in {gen_dir}")
    existing = tools_path.read_text()
    if name in _existing_top_level_names(existing):
        raise ValueError(f"add_tool '{name}': name already defined in tools/base.py")

    new_content = existing.rstrip() + "\n\n\n" + code.strip() + "\n"
    tools_path.write_text(new_content)


_DISPATCH = {
    "edit_prompt": _apply_edit_prompt,
    "edit_solve_loop": _apply_edit_solve_loop,
    "add_tool": _apply_add_tool,
}


def apply_proposal(proposal: dict, target_gen_dir: str) -> None:
    """Apply a proposal to a candidate snapshot directory, mutating files in place."""
    proposal = _ensure_dict(proposal, "proposal")
    kind = proposal.get("kind")
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"proposal.kind must be one of {sorted(_VALID_KINDS)}, got {kind!r}"
        )
    details = _ensure_dict(proposal.get("details"), "proposal.details")
    gen_dir = Path(target_gen_dir)
    if not gen_dir.exists():
        raise ValueError(f"target gen dir missing: {gen_dir}")
    _DISPATCH[kind](details, gen_dir)
