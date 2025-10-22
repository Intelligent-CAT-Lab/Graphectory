#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action-role classifier for agent steps (robust to dict/sequence `command`, heredocs, and shell None tool).

Roles:
  - "L_reproduce"            : generating/viewing/executing tests *before* any patch
  - "L_navigate"             : other localization *before* any patch (read/search/browse)
  - "patch"                  : creating/editing/deleting *non-test* assets (or generic edits)
  - "V_newly_generated_test" : test-related actions *after* a patch that target tests created in this run
  - "V_regression_test"      : test-related actions *after* a patch that target existing tests (or suite w/o paths)
  - "general"                : everything else

State:
  - `created_tests` (set[str]) is updated in-place when a command creates a test file.
"""

from __future__ import annotations
import re
from typing import Iterable, List, Tuple, Any, Optional, Set

# --------------------------- Configurable Heuristics ---------------------------

TEST_HINTS: Tuple[str, ...] = (
    "test_", "reproduc", "debug", "_test", "/tests/", "/test/",
)

READONLY_CMDS: Tuple[str, ...] = ("grep", "find", "cat", "echo", "ls", "head", "tail", "awk")
EDIT_CMDS: Tuple[str, ...] = ("sed", "touch")
SRE_EDIT_SUBCMDS: Tuple[str, ...] = ("create", "str_replace", "insert", "undo_edit")
SRE_READONLY_SUBCMDS: Tuple[str, ...] = ("view",)
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

def _has_prior_patch(prev_roles: Optional[Iterable[str]]) -> bool:
    """Whether a 'patch' role has occurred earlier."""
    return any(r == "patch" for r in (prev_roles or []))

def _contains_redirection(tokens: List[str]) -> bool:
    """
    Detect shell redirection/heredoc/tee implying writes/edits.
    Handles both separated tokens (">", ">>", "<<") and embedded heredocs like "cat <<'EOF' > file".
    """
    if not tokens:
        return False
    redir_ops = {">", ">>", "1>", "2>", ">|", "<<<", "<<", "<>", ">&", "2>&1"}
    if any(t in redir_ops or t.startswith((">", ">>", "1>", "2>")) for t in tokens):
        return True
    embedded_ops = (" <<", "<<", " >>", ">>", " 1>", " 2>", " >", " >|", "<>", ">&", "2>&1")
    if any(any(op in t for op in embedded_ops) for t in tokens):
        return True
    return any("tee" == t or " tee " in t for t in tokens)

def _is_test_path(s: str) -> bool:
    return any(h in s for h in TEST_HINTS)

def _is_test_related(tokens: List[str], paths: List[str]) -> bool:
    return any(_is_test_path(p) for p in paths)

def _paths_after_redirection(tokens: List[str]) -> List[str]:
    """
    Best-effort: collect candidate target paths immediately following redirection tokens.
    """
    targets: List[str] = []
    redir_starts = {">", ">>", "1>", "2>", ">|"}
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if t in redir_starts or t.startswith((">", ">>", "1>", "2>")) or (" >" in t):
            if i + 1 < n:
                nxt = tokens[i + 1]
                if _PATHISH.search(nxt):
                    targets.append(nxt)
        i += 1
    return targets

def _sre_role(subcommand: Optional[str]) -> str:
    """Map str_replace_editor subcommand to a role family (will be refined later)."""
    sub = (subcommand or "").lower()
    if sub in SRE_EDIT_SUBCMDS:
        return "patch"
    if sub in SRE_READONLY_SUBCMDS:
        return "L_navigate"  # read-only by default; refined by context
    return "general"

def _normalize_command_and_merge_args(command: Any, args: Any) -> Tuple[str, List[str], List[str]]:
    """
    Normalize `command` into a lowercase command string (may be empty if not a simple str)
    and merge any command-embedded arguments into the args token/path sets.

    Returns: (cmd_str, merged_tokens, merged_paths)
    """
    if isinstance(command, str) or command is None:
        cmd_str = (command or "").lower().strip()
        cmd_tokens = []
    else:
        cmd_str = ""
        cmd_tokens = _flatten_args(command)

    arg_tokens = _flatten_args(args)
    merged_tokens = arg_tokens + cmd_tokens
    merged_paths  = _extract_paths(args) + _extract_paths(command)
    return cmd_str, merged_tokens, merged_paths

def _postpatch_validation_kind(
    targets: List[str], created_tests: Optional[Set[str]]
) -> str:
    """
    Decide V_newly_generated_test vs V_regression_test given explicit targets.
    If no explicit targets, default to regression tests.
    """
    if not targets:
        return "V_regression_test"
    if not created_tests:
        return "V_regression_test"
    for p in targets:
        if p in created_tests:
            return "V_newly_generated_test"
    return "V_regression_test"

def _record_created_tests(
    targets: List[str], created_tests: Optional[Set[str]]
) -> None:
    """Add test targets to `created_tests` if present."""
    if created_tests is None:
        return
    for p in targets:
        if _is_test_path(p):
            created_tests.add(p)

# --------------------------- Core classification ---------------------------

def get_action_role(
    tool: Optional[str],
    subcommand: Optional[str],
    command: Optional[str | dict | list | tuple],
    args: Any,
    prev_roles: Optional[Iterable[str]] = None,
    *,
    created_tests: Optional[Set[str]] = None,
) -> str:
    """
    Classify an action into a refined role:
        "L_reproduce" | "L_navigate" | "patch" |
        "V_newly_generated_test" | "V_regression_test" | "general"

    Side-effect: updates `created_tests` when detecting creation of test files.
    """
    cmd, tokens, paths = _normalize_command_and_merge_args(command, args)
    has_patch = _has_prior_patch(prev_roles)
    test_related = _is_test_related(tokens, paths)

    # 1) str_replace_editor decisions (tool-specific)
    if (tool or "").lower() == "str_replace_editor":
        role = _sre_role(subcommand)

        # SRE edit-like subcommands
        if role == "patch":
            targets = [p for p in paths if _is_test_path(p)]
            if (subcommand or "").lower() == "create":
                _record_created_tests(targets, created_tests)

            if test_related:
                if has_patch:
                    return _postpatch_validation_kind(targets, created_tests)
                else:
                    return "L_reproduce"
            return "patch"

        # SRE 'view' (read-only)
        if role == "L_navigate":
            if test_related:
                if has_patch:
                    targets = [p for p in paths if _is_test_path(p)]
                    return _postpatch_validation_kind(targets, created_tests)
                else:
                    return "L_reproduce"
            return "L_navigate"

        return role  # general

    # 2) Python / pytest / pylint
    if cmd in PY_CMDS:
        if _contains_redirection(tokens):
            # Generates files via redirection (e.g., python -c ... > tests/test_x.py)
            redir_targets = _paths_after_redirection(tokens)
            _record_created_tests(redir_targets, created_tests)
            if any(_is_test_path(t) for t in redir_targets):
                return _postpatch_validation_kind(redir_targets, created_tests) if has_patch else "L_reproduce"
            return "patch"
        # No redirection: treat as execution; classify by pre/post-patch + whether tests are implicated
        if has_patch:
            explicit_test_targets = [p for p in paths if _is_test_path(p)]
            return _postpatch_validation_kind(explicit_test_targets, created_tests)
        else:
            return "L_reproduce" if test_related or cmd == "pytest" else "L_navigate"

    # 3) Read-only commands
    if cmd in READONLY_CMDS:
        if _contains_redirection(tokens):
            redir_targets = _paths_after_redirection(tokens)
            _record_created_tests(redir_targets, created_tests)
            if any(_is_test_path(t) for t in redir_targets):
                return _postpatch_validation_kind(redir_targets, created_tests) if has_patch else "L_reproduce"
            return "patch"

        if test_related:
            return _postpatch_validation_kind([p for p in paths if _is_test_path(p)], created_tests) if has_patch else "L_reproduce"
        return "L_navigate"

    # 4) Edit/creation commands (sed/touch)
    if cmd in EDIT_CMDS or (cmd == "sed"):
        targets = [p for p in paths if _is_test_path(p)]
        if targets:
            # In-place edits on tests don't mark as "created"
            return _postpatch_validation_kind(targets, created_tests) if has_patch else "L_reproduce"
        return "patch"

    # 5) Generic shell with redirection/heredoc/tee → write-like
    if _contains_redirection(tokens):
        redir_targets = _paths_after_redirection(tokens)
        _record_created_tests(redir_targets, created_tests)
        if any(_is_test_path(t) for t in redir_targets):
            return _postpatch_validation_kind(redir_targets, created_tests) if has_patch else "L_reproduce"
        return "patch"

    # 6) Fallbacks
    if test_related:
        return "V_regression_test" if has_patch else "L_reproduce"
    return "L_navigate" if not has_patch else "general"


# --------------------------- Self-checks ---------------------------
if __name__ == "__main__":
    created = set()

    # Pre-patch: grep writing a new test -> L_reproduce + record
    assert get_action_role(None, None, "grep",
                           ["def test_foo():", "file.py", ">", "tests/test_file.py"],
                           None, created_tests=created) == "L_reproduce"
    assert "tests/test_file.py" in created

    # Post-patch: viewing created test -> V_newly_generated_test
    assert get_action_role("str_replace_editor", "view", {"path": "tests/test_file.py"},
                           None, ["patch"], created_tests=created) == "V_newly_generated_test"

    # Post-patch: pytest with no explicit paths -> V_regression_test
    assert get_action_role(None, None, "pytest", [], ["patch"], created_tests=created) == "V_regression_test"

    # Post-patch: python -c ... > tests/new_test.py -> V_newly_generated_test
    assert get_action_role(None, None, "python",
                           ["-c", "'print()'", ">", "tests/new_test.py"],
                           ["patch"], created_tests=created) == "V_newly_generated_test"
    assert "tests/new_test.py" in created

    # Non-test edit remains patch
    assert get_action_role(None, None, "sed", ["-i", "s/x/y/g", "src/file.py"],
                           None, created_tests=created) == "patch"

    # Pre-patch read-only non-test -> L_navigate
    assert get_action_role(None, None, "grep", ["pattern", "src/file.py"],
                           None, created_tests=created) == "L_navigate"

    print("All action-role tests passed.")
