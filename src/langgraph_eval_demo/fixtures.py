"""Frozen tool-result fixtures + arg digests (record/replay for agent CI).

OSS/learning only — not LangSmith datasets or HTTP cassette libraries.
Interview angle: golden the *environment* (tool outputs), not just planner names.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_FIXTURES_DIR = Path(__file__).resolve().parents[2] / "evals" / "tool_fixtures"
_CATALOG_NAME = "catalog.json"


def tool_arg_digest(args: dict[str, Any]) -> str:
    """Stable SHA-256 digest of canonicalized tool args (sorted JSON)."""
    canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fixtures_dir() -> Path:
    return _FIXTURES_DIR


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    path = path or (_FIXTURES_DIR / _CATALOG_NAME)
    if not path.exists():
        return {"tools": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def lookup_fixture(
    tool_name: str,
    args: dict[str, Any],
    catalog: dict[str, Any] | None = None,
) -> str | None:
    """Return frozen output for (tool, arg_digest) if present."""
    catalog = catalog if catalog is not None else load_catalog()
    digest = tool_arg_digest(args)
    entries = (catalog.get("tools") or {}).get(tool_name) or []
    for entry in entries:
        if entry.get("args_digest") == digest:
            return entry.get("output")
    return None


def verify_against_fixture(
    tool_name: str,
    args: dict[str, Any],
    actual_output: str,
    catalog: dict[str, Any] | None = None,
) -> tuple[bool, str | None]:
    """If a fixture exists for this call, require exact output match."""
    expected = lookup_fixture(tool_name, args, catalog=catalog)
    if expected is None:
        return True, None  # no fixture → skip
    if actual_output == expected:
        return True, None
    return (
        False,
        f"fixture mismatch for {tool_name}@{tool_arg_digest(args)[:12]}…: "
        f"expected={expected!r} actual={actual_output!r}",
    )
