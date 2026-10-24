"""
opendraco_runs/tools.py

What a public-OpenDraco tool call is, from its name alone.

OpenDraco ships many tool sets (``opendraco/tools`` and ``opendraco/tools/repo/*``,
modelled on SWE-agent, OpenHands, Claude Coder, Trae and others), each naming
its read/write/search/shell tools differently, and users add their own. The
event log records only the name, so tools are sorted into the kinds the phase
mapping understands by an explicit table first and name patterns after.

The log also records arguments only as ``args_preview``: ``k=v, k=v`` with every
value cut to 40 characters (long paths shortened to ``…/dir/file``), which
``parse_args_preview`` turns back into a dict of those truncated strings.
"""

import re
from typing import Optional

# Exact names from opendraco/tools/*.py, where a pattern would guess wrong.
EXACT = {
    "read_file": "read",
    "list_files": "search",
    "search_code": "search",
    "detect_bug_class": "search",
    "apply_patch": "write",
    "write_patch": "write",
    "agent_write_patch": "write",
    "write_file": "write",
    "run_tests": "test",
    "run_flake8": "test",
    "generate_diff": "general",
    "normalize_patch": "general",
    "reset_repo": "general",
    "derive_description_fix": "general",
}

# Checked in order; the first match wins.
_PATTERNS = [
    ("shell", r"bash|shell|cmd_?run|execute_?command|run_?command|exec_?command|terminal"),
    ("test", r"test|lint|flake8|pylint|mypy"),
    ("write", r"write|edit|patch|replace|create|insert|save|append|modify|delete"),
    ("read", r"read|view|open|cat_?file|show"),
    ("search", r"search|grep|glob|find|list|explore|locate|^loc_|(^|_)ast(_|$)|symbol|manifest|ckg|tree"),
    ("localization", r"web|browser|url|fetch|screenshot"),
    ("general", r"think|finish|complete|done|exit|ask|spawn|condens|plan|todo"),
]
_COMPILED = [(kind, re.compile(rx)) for kind, rx in _PATTERNS]

_PATH_KEYS = ("path", "file_path", "file", "filename", "target", "dest")
_COMMAND_KEYS = ("command", "cmd", "script", "code")


def snake(name: str) -> str:
    """``readFileTool`` / ``CmdRunTool`` → ``read_file_tool`` / ``cmd_run_tool``."""
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name or "")
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def tool_kind(name: str) -> Optional[str]:
    """``shell`` | ``read`` | ``write`` | ``search`` | ``test`` | a phase | None."""
    if name in EXACT:
        return EXACT[name]
    flat = snake(name)
    if flat in EXACT:
        return EXACT[flat]
    for kind, rx in _COMPILED:
        if rx.search(flat):
            return kind
    return None


def parse_args_preview(preview) -> dict:
    """``"path=…/src/app.py, line=12"`` → ``{"path": "…/src/app.py", "line": "12"}``."""
    if isinstance(preview, dict):
        return dict(preview)
    if not isinstance(preview, str) or not preview.strip():
        return {}
    out = {}
    for part in re.split(r", (?=[A-Za-z_][A-Za-z0-9_]*=)", preview):
        key, sep, value = part.partition("=")
        if sep:
            out[key.strip()] = value
        elif out:  # a value that itself contained ", " without a key after it
            last = next(reversed(out))
            out[last] += ", " + part
        else:
            out["_raw"] = part
    return out


def path_of(args: dict) -> Optional[str]:
    for key in _PATH_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def command_of(args: dict) -> str:
    for key in _COMMAND_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def result_is_error(preview) -> Optional[bool]:
    """``result_preview`` is ≤80 characters, ``ok=…``, or ``N items``."""
    text = str(preview or "").strip()
    if text.startswith(("Error", "error:", "Traceback")) or text.startswith("ok=False"):
        return True
    if text.startswith("ok=True"):
        return False
    return None
