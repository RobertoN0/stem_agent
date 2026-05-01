"""Mutation manifest helpers for the self-modification boundary.

The root manifest is immutable experiment infrastructure. It tells the host
which directories become candidate snapshots, which paths proposals may touch,
and which files/symbols remain protected even inside mutable roots.
"""

from __future__ import annotations

import copy
import fnmatch
from pathlib import Path
from typing import Any

import yaml


_LEGACY_MANIFEST: dict[str, Any] = {
    "version": 1,
    "snapshot_roots": ["agent", "tools"],
    "mutable_paths": ["agent/**", "tools/**"],
    "protected_files": ["agent/agent.py", "agent/prompt.txt", "tools/base.py"],
    "protected_paths": [],
    "protected_symbols": {
        "tools/base.py": [
            "llm_call",
            "read_file",
            "write_file",
            "list_dir",
            "run_bash",
            "note",
            "_set_rpc_channels",
            "_set_task_context",
            "_safe_path",
        ],
    },
    "reflection_context": {
        "max_tree_files": 120,
        "max_file_chars": 4000,
        "max_snapshot_file_chars": 12000,
        "deny_paths": [
            "data/**",
            "artifacts/**",
            ".env",
            ".codex",
            ".codex/**",
            ".git",
            ".git/**",
            ".venv/**",
        ],
        "files": [],
    },
}


def _as_list(value) -> list:
    return value if isinstance(value, list) else []


def normalize_rel_path(path) -> str:
    """Return a safe POSIX relative path or raise ValueError."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    rel = Path(path)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"path must be relative and stay in the snapshot: {path!r}")
    cleaned = rel.as_posix()
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if not cleaned or cleaned == ".":
        raise ValueError("path must name a file")
    return cleaned


def _clean_pattern(pattern) -> str:
    cleaned = str(pattern).strip()
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


def _normalize_manifest(raw: dict[str, Any]) -> dict[str, Any]:
    manifest = copy.deepcopy(_LEGACY_MANIFEST)
    if isinstance(raw, dict):
        manifest.update(raw)

    manifest["snapshot_roots"] = [
        normalize_rel_path(root).rstrip("/")
        for root in _as_list(manifest.get("snapshot_roots"))
    ]
    manifest["mutable_paths"] = [
        _clean_pattern(pattern)
        for pattern in _as_list(manifest.get("mutable_paths"))
    ]
    manifest["protected_files"] = [
        normalize_rel_path(path)
        for path in _as_list(manifest.get("protected_files"))
    ]
    manifest["protected_paths"] = [
        _clean_pattern(pattern)
        for pattern in _as_list(manifest.get("protected_paths"))
    ]

    protected_symbols = manifest.get("protected_symbols") or {}
    if not isinstance(protected_symbols, dict):
        protected_symbols = {}
    manifest["protected_symbols"] = {
        normalize_rel_path(path): [
            str(name) for name in _as_list(names) if isinstance(name, str) and name
        ]
        for path, names in protected_symbols.items()
    }

    ctx = manifest.get("reflection_context") or {}
    if not isinstance(ctx, dict):
        ctx = {}
    ctx.setdefault("max_tree_files", 120)
    ctx.setdefault("max_file_chars", 4000)
    ctx.setdefault("max_snapshot_file_chars", 12000)
    ctx["deny_paths"] = [
        _clean_pattern(pattern)
        for pattern in _as_list(ctx.get("deny_paths"))
    ]
    ctx["files"] = [
        item for item in _as_list(ctx.get("files")) if isinstance(item, dict)
    ]
    manifest["reflection_context"] = ctx
    return manifest


def load_mutation_manifest(
    project_root: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Load the root mutation manifest, falling back to legacy agent/tools rules."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    if manifest_path is None:
        manifest_path = project_root / "mutation_manifest.yaml"

    if manifest_path.exists():
        with open(manifest_path) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = copy.deepcopy(_LEGACY_MANIFEST)
    return _normalize_manifest(raw)


def matches_any(rel_path: str, patterns: list[str]) -> bool:
    return any(
        fnmatch.fnmatch(rel_path, pattern)
        or (pattern.endswith("/**") and rel_path == pattern[:-3].rstrip("/"))
        for pattern in patterns
    )


def snapshot_roots(manifest: dict[str, Any]) -> list[str]:
    return list(manifest.get("snapshot_roots") or [])


def is_mutable_path(rel_path, manifest: dict[str, Any]) -> bool:
    rel = normalize_rel_path(str(rel_path))
    return matches_any(rel, manifest.get("mutable_paths") or [])


def is_protected_file(rel_path, manifest: dict[str, Any]) -> bool:
    rel = normalize_rel_path(str(rel_path))
    return rel in set(manifest.get("protected_files") or [])


def protected_symbols_for(rel_path, manifest: dict[str, Any]) -> set[str]:
    rel = normalize_rel_path(str(rel_path))
    return set((manifest.get("protected_symbols") or {}).get(rel) or [])


def reflection_context_config(manifest: dict[str, Any]) -> dict[str, Any]:
    return dict(manifest.get("reflection_context") or {})
