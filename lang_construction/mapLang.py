#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action-role classifier for agent steps.

Phases before first patch ("localization / reproduction"):
  - L_reproduce : generating / viewing / executing tests (reproducing / understanding bug)
  - L_navigate  : non-test browsing/searching/reading
  - P           : creating/editing/deleting non-test assets (or generic edits)

Phases after first patch ("validation"):
  - V_newly_generated_test :
        Any interaction (create / view / edit / run) with tests that did NOT
        originally exist in the repo, including:
          - new concrete test files created in this run (tracked in created_tests)
          - inline/ephemeral validation code like `python -c "assert ..."`
            or `python -m adhoc_runner` that is not from disk and not pytest,
            tracked in created_dynamic_suites
        Repeats of those same new tests are still newly_generated.
  - V_regression_test :
        Any interaction (create / view / edit / run) with tests that DID
        originally exist in the repo, including re-running pytest or editing
        existing tests.
  
general : Everything else.

We persist across steps:
  - created_tests: set[str]
        Paths for test files first created this run.
        After patch, any interaction with those paths is V_newly_generated_test.
  - created_dynamic_suites: set[str]
        Stable keys for inline / ephemeral validation (python -c/-m with no paths).
        After patch, reusing those is still V_newly_generated_test.
"""

from __future__ import annotations
import re
from typing import Iterable, List, Tuple, Any, Optional, Set, Dict, Union

# --------------------------- Configurable Heuristics ---------------------------

TEST_HINTS: Tuple[str, ...] = (
    "test_", "reproduc", "debug", "_test", "/tests/", "/test/",
)

READONLY_CMDS: Tuple[str, ...] = (
    "grep", "find", "cat", "echo", "ls", "head", "tail", "awk", "nl"
)
EDIT_CMDS: Tuple[str, ...] = ("sed", "touch")
SRE_EDIT_SUBCMDS: Tuple[str, ...] = ("create", "str_replace", "insert", "undo_edit")
SRE_READONLY_SUBCMDS: Tuple[str, ...] = ("view",)
PY_CMDS: Tuple[str, ...] = ("python", "python3", "python2", "pytest", "pylint")

_PATHISH = re.compile(r"(^[/~.]|/|\.py$)")

# --------------------------- Flatten / token helpers ---------------------------

def _flatten_any(val: Any) -> List[str]:
    """
    Lowercased tokens from arbitrary val:
      - str -> [val]
      - list/tuple -> [each]
      - dict -> include BOTH keys and values
        (important for python flags like {"c": "assert ..."} which encode -c)
    """
    toks: List[str] = []
    if isinstance(val, dict):
        for k, v in val.items():
            if k is not None:
                toks.append(str(k))
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                toks.extend(str(x) for x in v)
            else:
                toks.append(str(v))
    elif isinstance(val, (list, tuple)):
        toks = [str(x) for x in val]
    elif isinstance(val, str):
        toks = [val]
    return [t.lower() for t in toks]

def _extract_paths_generic(*vals: Any) -> List[str]:
    """
    Heuristic path extraction for non-SRE commands:
      - starts with '/', '~', or '.'
      - OR contains '/'
      - OR endswith '.py'
    We apply this to command/args/flags in general shell/python use.
    """
    all_toks: List[str] = []
    for v in vals:
        all_toks.extend(_flatten_any(v))
    return [t for t in all_toks if _PATHISH.search(t)]

def _extract_sre_paths(args: Any) -> List[str]:
    """
    STRICT path extraction for str_replace_editor:
    We *only* trust the declared "path" / "paths" fields from the args dict.
    We do NOT scan other keys like "old_str", "new_str", etc.
    This prevents us from accidentally treating arbitrary substrings as file paths.
    """
    out: List[str] = []
    if isinstance(args, dict):
        p = args.get("path")
        if isinstance(p, str):
            out.append(p.lower())
        ps = args.get("paths")
        if isinstance(ps, (list, tuple)):
            for x in ps:
                if isinstance(x, str):
                    out.append(x.lower())
    return out

def _gather_command_context(
    command: Any,
    args: Any,
    flags: Any,
    *,
    for_sre: bool,
) -> Tuple[str, List[str], List[str]]:
    """
    Build:
      cmd_str  : base command (lowercased) if `command` is a plain str, else ""
      tokens   : merged lowered tokens from args + command(+subfields) + flags
      paths    : merged list of path-like items
                 - if for_sre=True: use ONLY _extract_sre_paths(args)
                 - else           : use heuristic _extract_paths_generic(...)
    """
    if isinstance(command, str) or command is None:
        cmd_str = (command or "").lower().strip()
        cmd_tokens: List[str] = []
    else:
        cmd_str = ""
        cmd_tokens = _flatten_any(command)

    arg_tokens  = _flatten_any(args)
    flag_tokens = _flatten_any(flags)

    merged_tokens = arg_tokens + cmd_tokens + flag_tokens

    if for_sre:
        merged_paths = _extract_sre_paths(args)
    else:
        merged_paths = _extract_paths_generic(args, command, flags)

    return cmd_str, merged_tokens, merged_paths

# --------------------------- Context / intent helpers ---------------------------

def _has_prior_patch(prev_roles: Optional[Iterable[str]]) -> bool:
    """True iff we've already seen a 'P' (a code patch) earlier in the run."""
    return any(r == "P" for r in (prev_roles or []))

def _is_test_path(s: str) -> bool:
    """Heuristic: does this look like a test/repro harness path?"""
    return any(h in s for h in TEST_HINTS)

def _is_test_related(paths: List[str]) -> bool:
    """Test-related if ANY collected path-like token looks like a test."""
    return any(_is_test_path(p) for p in paths)

# --------------------------- Shell helpers ---------------------------

def _contains_redirection(tokens: List[str]) -> bool:
    """
    Detect shell output redirection / heredocs / tee (== writing).
    """
    if not tokens:
        return False
    redir_ops = {">", ">>", "1>", "2>", ">|", "<<<", "<<", "<>", ">&", "2>&1"}
    if any(t in redir_ops or t.startswith((">", ">>", "1>", "2>")) for t in tokens):
        return True
    embedded_ops = (
        " <<", "<<",
        " >>", ">>",
        " 1>", " 2>", " >", " >|",
        "<>", ">&", "2>&1"
    )
    if any(any(op in t for op in embedded_ops) for t in tokens):
        return True
    return any("tee" == t or " tee " in t for t in tokens)

def _is_piped_readonly_operation(cmd: str, tokens: List[str]) -> bool:
    """
    Detect "view-only via pipe", e.g. `nl file.py | sed -n '10,20p'`.
    """
    if cmd not in READONLY_CMDS:
        return False
    has_pipe = "|" in tokens or any("|" in t for t in tokens)
    return has_pipe and not _contains_redirection(tokens)

def _paths_after_redirection(tokens: List[str]) -> List[str]:
    """
    Guess file(s) being written: tokens that follow >, >>, etc.
    """
    targets: List[str] = []
    redir_starts = {">", ">>", "1>", "2>", ">|"}
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if (
            t in redir_starts
            or t.startswith((">", ">>", "1>", "2>"))
            or (" >" in t)
        ):
            if i + 1 < n:
                nxt = tokens[i + 1]
                if _PATHISH.search(nxt):
                    targets.append(nxt)
        i += 1
    return targets

# --------------------------- str_replace_editor helpers ---------------------------

def _sre_role(subcommand: Optional[str]) -> str:
    """
    Rough mapping from str_replace_editor subcommand to a role family.
    """
    sub = (subcommand or "").lower()
    if sub in SRE_EDIT_SUBCMDS:
        return "P"
    if sub in SRE_READONLY_SUBCMDS:
        return "L_navigate"
    return "general"

# --------------------------- Test provenance tracking ---------------------------

def _record_created_tests(
    targets: List[str],
    created_tests: Optional[Set[str]]
) -> None:
    """
    If we write to something that looks like a test file,
    remember that path as "newly created this run".
    """
    if created_tests is None:
        return
    for p in targets:
        if _is_test_path(p):
            created_tests.add(p)

def _dynamic_key_for_inline_test(
    cmd: str,
    tokens: List[str],
    paths: List[str],
) -> Optional[str]:
    """
    Detect inline / ephemeral validation (like python -c/-m assertions) that is NOT
    executing a file path from disk.

    Dynamic if ALL:
      - cmd is python / python2 / python3
      - we see an inline-exec style flag ("-c", "-m", "c", "m", or "-cFOO"/"-mFOO")
      - there are NO path-like args in `paths`
      - it's not just delegating to pytest / py.test
    """
    if cmd not in ("python", "python2", "python3"):
        return None

    def _is_inline_flag(tok: str) -> bool:
        return (
            tok in ("-c", "-m", "c", "m")
            or tok.startswith("-c")
            or tok.startswith("-m")
        )

    # If it's just invoking pytest, that's regression, not "new".
    for i, tok in enumerate(tokens):
        if tok.startswith("pytest") or tok.startswith("py.test"):
            return None
        if tok in ("-m", "m") and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt.startswith("pytest") or nxt.startswith("py.test"):
                return None
        if tok.startswith("-m") and tok not in ("-m",):
            maybe_mod = tok[2:]
            if maybe_mod.startswith("pytest") or maybe_mod.startswith("py.test"):
                return None

    # must have inline exec flag
    if not any(_is_inline_flag(t) for t in tokens):
        return None

    # if we reference on-disk paths, it's not ephemeral inline
    if paths:
        return None

    # build stable key for reuse:
    # module after -m / m / "-mFOO"
    for i, tok in enumerate(tokens):
        if tok in ("-m", "m") and i + 1 < len(tokens):
            mod = tokens[i + 1]
            return f"module:{mod[:200]}"
        if tok.startswith("-m") and tok not in ("-m",):
            mod = tok[2:]
            if mod:
                return f"module:{mod[:200]}"

    # inline code after -c / c / "-cCODE"
    for i, tok in enumerate(tokens):
        if tok in ("-c", "c") and i + 1 < len(tokens):
            code_snip = tokens[i + 1]
            return f"inline:{code_snip[:200]}"
        if tok.startswith("-c") and tok not in ("-c",):
            code_snip = tok[2:]
            if code_snip:
                return f"inline:{code_snip[:200]}"

    return "inline:<unknown>"

# --------------------------- Post-patch validation decision ---------------------------

def _postpatch_validation_kind(
    targets: List[str],
    *,
    created_tests: Optional[Set[str]],
    dynamic_key: Optional[str],
    created_dynamic_suites: Optional[Set[str]],
) -> str:
    """
    After patch: choose V_newly_generated_test vs V_regression_test.
    """
    if targets:
        if created_tests:
            for p in targets:
                if p in created_tests:
                    return "V_newly_generated_test"
        return "V_regression_test"

    if dynamic_key:
        if created_dynamic_suites is not None:
            created_dynamic_suites.add(dynamic_key)
        return "V_newly_generated_test"

    return "V_regression_test"

# --------------------------- Core classification ---------------------------

def get_action_role(
    tool: Optional[str],
    subcommand: Optional[str],
    command: Optional[Union[str, dict, list, tuple]],
    args: Any,
    flags: Optional[Dict[str, Any]] = None,
    prev_roles: Optional[Iterable[str]] = None,
    *,
    created_tests: Optional[Set[str]] = None,
    created_dynamic_suites: Optional[Set[str]] = None,
) -> str:
    """
    Classify a step into:
      "L_reproduce", "L_navigate", "P",
      "V_newly_generated_test", "V_regression_test",
      "general"

    flags:
        e.g. {"c": "assert ..."} for python -c inline code
    """
    flags = flags or {}
    is_sre = (tool or "").lower() == "str_replace_editor"

    cmd, tokens, paths = _gather_command_context(
        command,
        args,
        flags,
        for_sre=is_sre,
    )

    has_patch = _has_prior_patch(prev_roles)
    test_related = _is_test_related(paths)

    # 1) str_replace_editor
    if is_sre:
        role_family = _sre_role(subcommand)

        # NOTE: for SRE we already restricted `paths` using ONLY args["path"/"paths"],
        # so `paths` here is clean and does NOT accidentally pull from old/new text.
        targets = paths  # already filtered

        if role_family == "P":
            if (subcommand or "").lower() == "create":
                _record_created_tests(targets, created_tests)

            if test_related:
                if has_patch:
                    return _postpatch_validation_kind(
                        targets,
                        created_tests=created_tests,
                        dynamic_key=None,
                        created_dynamic_suites=created_dynamic_suites,
                    )
                return "L_reproduce"

            return "P"

        if role_family == "L_navigate":
            if test_related:
                if has_patch:
                    return _postpatch_validation_kind(
                        targets,
                        created_tests=created_tests,
                        dynamic_key=None,
                        created_dynamic_suites=created_dynamic_suites,
                    )
                return "L_reproduce"
            return "L_navigate"

        return role_family  # "general"

    # 2) Python / pytest / pylint / etc.
    if cmd in PY_CMDS:
        if _contains_redirection(tokens):
            redir_targets = _paths_after_redirection(tokens)
            _record_created_tests(redir_targets, created_tests)
            return (
                _postpatch_validation_kind(
                    [p for p in redir_targets if _is_test_path(p)],
                    created_tests=created_tests,
                    dynamic_key=None,
                    created_dynamic_suites=created_dynamic_suites,
                )
                if has_patch
                else "L_reproduce"
            )

        if has_patch:
            explicit_test_targets = [p for p in paths if _is_test_path(p)]
            dyn_key = _dynamic_key_for_inline_test(cmd, tokens, paths)

            if dyn_key and created_dynamic_suites is not None and dyn_key in created_dynamic_suites:
                return "V_newly_generated_test"

            return _postpatch_validation_kind(
                explicit_test_targets,
                created_tests=created_tests,
                dynamic_key=dyn_key,
                created_dynamic_suites=created_dynamic_suites,
            )
        else:
            dyn_key = _dynamic_key_for_inline_test(cmd, tokens, paths)
            return "L_reproduce" if (test_related or cmd == "pytest" or dyn_key) else "general"

    # 3) Read-only commands
    if cmd in READONLY_CMDS:
        if _is_piped_readonly_operation(cmd, tokens):
            test_targets = [p for p in paths if _is_test_path(p)]
            if test_targets:
                return (
                    _postpatch_validation_kind(
                        test_targets,
                        created_tests=created_tests,
                        dynamic_key=None,
                        created_dynamic_suites=created_dynamic_suites,
                    )
                    if has_patch else
                    "L_reproduce"
                )
            return "L_navigate"

        if _contains_redirection(tokens):
            redir_targets = _paths_after_redirection(tokens)
            _record_created_tests(redir_targets, created_tests)
            if any(_is_test_path(t) for t in redir_targets):
                return (
                    _postpatch_validation_kind(
                        [p for p in redir_targets if _is_test_path(p)],
                        created_tests=created_tests,
                        dynamic_key=None,
                        created_dynamic_suites=created_dynamic_suites,
                    )
                    if has_patch else
                    "L_reproduce"
                )
            return "P"

        test_targets = [p for p in paths if _is_test_path(p)]
        if test_targets:
            return (
                _postpatch_validation_kind(
                    test_targets,
                    created_tests=created_tests,
                    dynamic_key=None,
                    created_dynamic_suites=created_dynamic_suites,
                )
                if has_patch else
                "L_reproduce"
            )
        return "L_navigate"

    # 4) Edit/creation commands like sed/touch
    if cmd in EDIT_CMDS or cmd == "sed":
        edit_targets = [p for p in paths if _is_test_path(p)]
        if edit_targets:
            return (
                _postpatch_validation_kind(
                    edit_targets,
                    created_tests=created_tests,
                    dynamic_key=None,
                    created_dynamic_suites=created_dynamic_suites,
                )
                if has_patch else
                "L_reproduce"
            )
        return "P"

    # 5) Generic shell redirection (>, >>, tee, etc.)
    if _contains_redirection(tokens):
        redir_targets = _paths_after_redirection(tokens)
        _record_created_tests(redir_targets, created_tests)
        test_targets = [p for p in redir_targets if _is_test_path(p)]
        if test_targets:
            return (
                _postpatch_validation_kind(
                    test_targets,
                    created_tests=created_tests,
                    dynamic_key=None,
                    created_dynamic_suites=created_dynamic_suites,
                )
                if has_patch else
                "L_reproduce"
            )
        return "P"

    # 6) Fallback
    return "general"
