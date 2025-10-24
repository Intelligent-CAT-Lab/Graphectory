#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase classifier for agent actions (robust to dict/sequence `command`, heredocs, and shell None tool).

Phases:
  - "localization" : gathering info, searching, reading, or generating/trying tests *before* any patch
  - "patch"        : creating/editing/deleting non-test assets
  - "validation"   : (re-)running tests or test-like commands *after* a patch; viewing/creating/editing test assets *after* a patch
  - "general"      : everything else

Key rule (test generation & execution):
  • If test generation/execution happens with NO prior "patch" in the phase history → "localization".
  • If it happens AFTER a "patch" → "validation".

Other bash commands:
  • grep/find/cat/nl WITHOUT redirection (>, >>) → "localization" or ("validation" if test-related after patch).
  • Piped read-only operations (e.g., nl file.py | sed -n '10,20p') → "localization" (or "validation" if test-related after patch).
  • If those commands CREATE/EDIT files (via redirection/heredoc/tee/in-place), treat as edits:
      - if target is **non-test** → "patch"
      - if target is **test** → apply key rule (loc before first patch; validation after)

Function:
  get_phase(tool, subcommand, command, args, prev_phases=None)
"""

from __future__ import annotations
import re
from typing import Iterable, List, Tuple, Any, Optional

# --------------------------- Configurable Heuristics ---------------------------

# Tokens/paths hinting that something is test-related.
TEST_HINTS: Tuple[str, ...] = (
    "test_", "reproduc", "debug", "_test", "/tests/", "/test/",
)

# Commands that typically *read/search* only; with redirection they can become edits.
READONLY_CMDS: Tuple[str, ...] = ("grep", "find", "cat", "echo", "ls", "head", "tail", "awk", "nl")

# Commands that are clearly *editing* or *creating* content.
EDIT_CMDS: Tuple[str, ...] = ("sed", "touch")

# str_replace_editor subcommands that indicate edits vs reads.
SRE_EDIT_SUBCMDS: Tuple[str, ...] = ("create", "str_replace", "insert", "undo_edit")
SRE_READONLY_SUBCMDS: Tuple[str, ...] = ("view",)

# Python commands that usually execute code/tests.
PY_CMDS: Tuple[str, ...] = ("python", "python3", "python2", "pytest", "pylint")

# --------------------------- Utilities ---------------------------

def _flatten_args(args: Any) -> List[str]:
    """Normalize args into a flat list of lowercase string tokens."""
    tokens: List[str] = []
    if isinstance(args, dict):
        for v in args.values():
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                tokens.extend(str(x) for x in v)
            else:
                tokens.append(str(v))
    elif isinstance(args, (list, tuple)):
        tokens = [str(x) for x in args]
    elif isinstance(args, str):
        tokens = [args]
    return [t.lower() for t in tokens]

_PATHISH = re.compile(r"(^[/~.]|/|\.py$)")

def _extract_paths(args: Any) -> List[str]:
    """Extract path-like strings from args."""
    tokens = _flatten_args(args)
    return [t for t in tokens if _PATHISH.search(t)]

def _has_prior_patch(prev_phases: Optional[Iterable[str]]) -> bool:
    return any(p == "patch" for p in (prev_phases or []))

def _contains_redirection(tokens: List[str]) -> bool:
    """
    Detect shell redirection/heredoc/tee implying writes/edits.
    Handles both separated tokens (">", ">>", "<<") and embedded heredocs like "cat <<'EOF' > file".
    """
    if not tokens:
        return False
    # Exact tokens / prefixed tokens
    redir_ops = {">", ">>", "1>", "2>", ">|", "<<<", "<<", "<>", ">&", "2>&1"}
    if any(t in redir_ops or t.startswith((">", ">>", "1>", "2>")) for t in tokens):
        return True
    # Embedded operators (e.g., "cat << 'EOF' > file", or script blobs)
    embedded_ops = (" <<", "<<", " >>", ">>", " 1>", " 2>", " >"," >|","<>", ">&", "2>&1")
    if any(any(op in t for op in embedded_ops) for t in tokens):
        return True
    # 'tee' writes to files via pipe
    return any("tee" == t or " tee " in t for t in tokens)

def _is_piped_readonly_operation(cmd: str, tokens: List[str]) -> bool:
    """
    Detect if this is a piped read-only operation (e.g., nl file.py | sed -n '10,20p').
    Returns True if:
      - The command is a read-only command (nl, cat, grep, etc.)
      - There's a pipe (|) in the tokens
      - There's no output redirection (>, >>, tee)
    This indicates the command is for viewing/filtering only, not editing.
    """
    if cmd not in READONLY_CMDS:
        return False
    has_pipe = "|" in tokens or any("|" in t for t in tokens)
    has_output_redir = _contains_redirection(tokens)
    return has_pipe and not has_output_redir

def _is_test_related(tokens: List[str], paths: List[str]) -> bool:
    """Test-related if any path contains a hint from TEST_HINTS."""
    return any(any(h in s for h in TEST_HINTS) for s in paths)

def _sre_phase(subcommand: Optional[str]) -> str:
    sub = (subcommand or "").lower()
    if sub in SRE_EDIT_SUBCMDS:
        return "patch"
    if sub in SRE_READONLY_SUBCMDS:
        return "localization"
    return "general"

def _normalize_command_and_merge_args(command: Any, args: Any) -> Tuple[str, List[str], List[str]]:
    """
    Normalize `command` into a lowercase command string (may be empty if not a simple str)
    and merge any command-embedded arguments into the args token/path sets.

    Returns: (cmd_str, merged_tokens, merged_paths)
    """
    # Determine command string if possible
    if isinstance(command, str) or command is None:
        cmd_str = (command or "").lower().strip()
        cmd_tokens = []
    else:
        # If command is dict/list/tuple, treat its contents as additional tokens/paths.
        cmd_str = ""
        cmd_tokens = _flatten_args(command)

    arg_tokens = _flatten_args(args)
    merged_tokens = arg_tokens + cmd_tokens
    merged_paths  = _extract_paths(args) + _extract_paths(command)
    return cmd_str, merged_tokens, merged_paths

# --------------------------- Core classification ---------------------------

def get_phase(
    tool: Optional[str],
    subcommand: Optional[str],
    command: Optional[str | dict | list | tuple],
    args: Any,
    prev_phases: Optional[Iterable[str]] = None,
) -> str:
    """
    Map a (tool, subcommand, command, args, prev_phases) to a phase:
        "localization" | "patch" | "validation" | "general"
    """
    cmd, tokens, paths = _normalize_command_and_merge_args(command, args)
    has_patch = _has_prior_patch(prev_phases)

    # 1) str_replace_editor decisions (tool-specific)
    if (tool or "").lower() == "str_replace_editor":
        phase = _sre_phase(subcommand)
        if phase == "patch":
            # If SRE edit targets tests, apply key rule (loc vs val by prior patch)
            if _is_test_related(tokens, paths):
                return "validation" if has_patch else "localization"
            return "patch"

        # 'view' (read-only) remains localization unless it's test-related AFTER a patch → validation
        if phase == "localization" and (subcommand or "").lower() in SRE_READONLY_SUBCMDS:
            if _is_test_related(tokens, paths) and has_patch:
                return "validation"

        return phase  # "localization" or "general"

    # 2) Python / pytest / pylint
    #    - Execution: apply key rule regardless of file hints.
    #    - If command line includes redirection (creating/editing files), treat as edit-like and use heuristics.
    if cmd in PY_CMDS:
        if _contains_redirection(tokens):
            # Edit-like via redirection (e.g., python -c '...' > tests/test_x.py)
            return ("validation" if has_patch else "localization") if _is_test_related(tokens, paths) else "patch"
        # Default: test/code execution → key rule
        return "validation" if has_patch else "localization"

    # 3) Read-only commands (grep/find/cat/ls/head/tail/awk/echo/nl)
    if cmd in READONLY_CMDS:
        # Piped operations without output redirection (e.g., nl file.py | sed -n '10,20p') are read-only
        if _is_piped_readonly_operation(cmd, tokens):
            # Viewing content: test-related AFTER patch → validation; otherwise → localization
            if _is_test_related(tokens, paths) and has_patch:
                return "validation"
            return "localization"

        if _contains_redirection(tokens):
            # These become edits when redirecting to files or using tee/heredoc
            return ("validation" if has_patch else "localization") if _is_test_related(tokens, paths) else "patch"

        # read-only, test-related AFTER a prior patch counts as validation; otherwise localization
        if _is_test_related(tokens, paths) and has_patch:
            return "validation"
        return "localization"

    # 4) Edit/creation commands (sed/touch)
    if cmd in EDIT_CMDS or (cmd == "sed"):
        # sed with/without -i still considered edit by config; treat targets accordingly
        return ("validation" if has_patch else "localization") if _is_test_related(tokens, paths) else "patch"

    # 5) Fallbacks:
    #    If any redirection is present (even embedded), treat as edit-like.
    if _contains_redirection(tokens):
        return ("validation" if has_patch else "localization") if _is_test_related(tokens, paths) else "patch"

    #    Otherwise, unknown → general.
    return "general"


# --------------------------- Self-checks ---------------------------
if __name__ == "__main__":
    # Simple tests
    test_cases = [
        # (tool, subcommand, command, args, prev_phases, expected_phase)
        (None, None, "grep", ["def foo():", "file.py"], None, "localization"),
        (None, None, "grep", ["def foo():", "test_file.py"], ["patch"], "validation"),
        (None, None, "grep", ["def foo():", "file.py", ">", "out.txt"], None, "patch"),
        (None, None, "grep", ["def test_foo():", "file.py", ">", "tests/test_file.py"], None, "localization"),
        (None, None, "grep", ["def test_foo():", "file.py", ">", "tests/test_file.py"], ["patch"], "validation"),
        (None, None, "sed", ["-i", "s/foo/bar/g", "file.py"], None, "patch"),
        (None, None, "sed", ["s/foo/bar/g", "file.py"], None, "patch"),
        (None, None, "python", ["script.py"], None, "localization"),
        (None, None, "python", ["script.py"], ["patch"], "validation"),
        (None, None, "python", ["-c", "'print(42)'", ">", "out.txt"], None, "patch"),
        (None, None, "python", ["-c", "'print(42)'", ">", "tests/test_out.py"], None, "localization"),
        (None, None, "python", ["-c", "'print(42)'", ">", "tests/test_out.py"], ["patch"], "validation"),
        # str_replace_editor where `command` may be a dict (observed in traces)
        ("str_replace_editor", "create", {"path": "file.py"}, None, None, "patch"),
        ("str_replace_editor", "create", {"path": "tests/test_file.py"}, None, None, "localization"),
        ("str_replace_editor", "create", {"path": "tests/test_file.py"}, None, ["patch"], "validation"),
        ("str_replace_editor", "view", {"path": "test_file.py"}, None, ["patch"], "validation"),
        ("str_replace_editor", "str_replace", {"path": "/testbed/django/db/backends/postgresql/client.py", "new_str": "        temp_pgpass = None\n        sigint_handler = signal.getsignal(signal.SIGINT)\n        try:\n            print(f\"DEBUG: passwd = '{passwd}'\")  # DEBUG\n            if passwd:\n                print(\"DEBUG: Creating temporary .pgpass file\")  # DEBUG\n                # Create temporary .pgpass file.\n                temp_pgpass = NamedTemporaryFile(mode='w+')}"}, None, None, "patch"),
        # Heredoc embedded in a single token (should be detected as redirection → edit-like).
        # Target is test-related and no prior patch → localization (test generation).
        (None, None, "complex_command",
         ["cat << 'EOF' > /workspace/test_hstack_fix.py\nprint('hi')\nEOF"], None, "localization"),

        # nl piped commands (read-only viewing operations)
        # nl file.py | sed -n '10,20p' - viewing regular file before patch
        (None, None, "nl", ["filename.py", "|", "sed"], None, "localization"),
        # nl test_file.py | sed -n '10,20p' - viewing test file before patch
        (None, None, "nl", ["test_file.py", "|", "sed"], None, "localization"),
        # nl test_file.py | sed -n '10,20p' - viewing test file AFTER patch
        (None, None, "nl", ["test_file.py", "|", "sed"], ["patch"], "validation"),
        # nl file.py | sed -n '10,20p' - viewing regular file AFTER patch
        (None, None, "nl", ["filename.py", "|", "sed"], ["patch"], "localization"),

        # nl with output redirection (becomes an edit operation)
        # nl file.py > output.txt - creating/editing non-test file
        (None, None, "nl", ["file.py", ">", "output.txt"], None, "patch"),
        # nl file.py > test_output.py - creating/editing test file before patch
        (None, None, "nl", ["file.py", ">", "test_output.py"], None, "localization"),
        # nl file.py > test_output.py - creating/editing test file AFTER patch
        (None, None, "nl", ["file.py", ">", "test_output.py"], ["patch"], "validation"),
    ]

    for i, (tool, subcmd, cmd, args, prev, expected) in enumerate(test_cases, 1):
        result = get_phase(tool, subcmd, cmd, args, prev)
        assert result == expected, f"Test case {i} failed: got {result}, expected {expected}"
    print("All test cases passed.")
