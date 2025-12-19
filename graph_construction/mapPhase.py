#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase classifier for agent actions.

Phases:
  - "localization" : gathering info, searching, reading, or generating/trying tests *before* any patch
  - "patch"        : creating/editing/deleting non-test assets
  - "validation"   : (re-)running tests or test-like commands *after* a patch; viewing/creating/editing test assets *after* a patch
  - "general"      : everything else

Key rule (test generation & execution):
  • If test generation/execution happens with NO prior "patch" in the phase history → "localization".
  • If it happens AFTER a "patch" → "validation".

Function:
  get_phase(tool, subcommand, command, args, prev_phases=None)
"""

from __future__ import annotations
import ast
import re
from typing import Iterable, List, Tuple, Any, Optional, Set, Dict

# --------------------------- Configurable Heuristics ---------------------------

# Regex patterns for test file detection
TEST_FILE_PATTERNS = (
    r'/tests?/',           # Directory: /test/ or /tests/
    r'\btest_[^/]*\.py$',  # File: test_*.py
    r'\b[^/]*_tests?\.py$', # File: *_test.py or *_tests.py
    r'.*reproduc.*\.py$',  # File containing 'reproduc'
    r'.*debug.*\.py$',     # File containing 'debug'
)

READONLY_CMDS: Tuple[str, ...] = (
    "grep", "find", "cat", "ls", "head", "tail", "awk", "nl"
)
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

def _has_prior_patch(prev_phases: Optional[Iterable[str]]) -> bool:
    """True if we've already seen a 'patch' earlier in the run."""
    return any(p == "patch" for p in (prev_phases or []))

def _is_test_path(s: str) -> bool:
    """Heuristic: does this look like a test path?"""
    return any(re.search(pattern, s) for pattern in TEST_FILE_PATTERNS)

def _is_test_related(paths: List[str]) -> bool:
    """Test-related if ANY collected path-like token looks like a test."""
    return any(_is_test_path(p) for p in paths)

def _contains_redirection(tokens: List[str]) -> bool:
    """Detect shell output redirection / heredocs / tee (== writing)."""
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
    """Detect "view-only via pipe", e.g. `nl file.py | sed -n '10,20p'`."""
    if cmd not in READONLY_CMDS:
        return False
    has_pipe = "|" in tokens or any("|" in t for t in tokens)
    return has_pipe and not _contains_redirection(tokens)

def _paths_after_redirection(tokens: List[str]) -> List[str]:
    """Guess file(s) being written: tokens that follow >, >>, etc."""
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

def _extract_edited_files_from_python_code(code: str) -> List[str]:
    """
    Analyze Python code via AST to extract file paths being edited/created.
    Looks for patterns like:
    - Path('file.py').write_text(...)
    - open('file.py', 'w').write(...)
    - with open('file.py', 'w') as f: ...
    Returns list of file paths found.
    """
    if not code or not isinstance(code, str):
        return []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    # First pass: collect all variable assignments
    path_vars: Dict[str, str] = {}
    string_vars: Dict[str, str] = {}

    class VariableCollector(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign):
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name) and node.value.func.id == 'Path':
                    if node.value.args and isinstance(node.value.args[0], ast.Constant):
                        filepath = node.value.args[0].value
                        if isinstance(filepath, str):
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    path_vars[target.id] = filepath
            elif isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        string_vars[target.id] = node.value.value
            self.generic_visit(node)

    var_collector = VariableCollector()
    var_collector.visit(tree)

    # Second pass: detect file edits
    edited_files: List[str] = []
    with_files: Set[str] = set()

    class FileEditVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            # Pattern 1: Path('file.py').write_text(...)
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in ('write_text', 'write_bytes'):
                    if isinstance(node.func.value, ast.Call):
                        if isinstance(node.func.value.func, ast.Name) and node.func.value.func.id == 'Path':
                            if node.func.value.args and isinstance(node.func.value.args[0], ast.Constant):
                                filepath = node.func.value.args[0].value
                                if isinstance(filepath, str):
                                    edited_files.append(filepath)
                    elif isinstance(node.func.value, ast.Name):
                        var_name = node.func.value.id
                        if var_name in path_vars:
                            edited_files.append(path_vars[var_name])

            # Pattern 2: open('file.py', 'w')
            if isinstance(node.func, ast.Name) and node.func.id == 'open':
                if len(node.args) >= 2:
                    filename = None
                    if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        filename = node.args[0].value
                    elif isinstance(node.args[0], ast.Name):
                        var_name = node.args[0].id
                        if var_name in string_vars:
                            filename = string_vars[var_name]

                    if filename:
                        if filename in with_files:
                            self.generic_visit(node)
                            return
                        if isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                            mode = node.args[1].value
                            if any(m in mode for m in ['w', 'a', 'x']):
                                edited_files.append(filename)

            self.generic_visit(node)

        def visit_With(self, node: ast.With):
            # Pattern 3: with open('file.py', 'w') as f: ...
            for item in node.items:
                if isinstance(item.context_expr, ast.Call):
                    call = item.context_expr
                    if isinstance(call.func, ast.Name) and call.func.id == 'open':
                        if len(call.args) >= 2:
                            filename = None
                            if isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
                                filename = call.args[0].value
                            elif isinstance(call.args[0], ast.Name):
                                var_name = call.args[0].id
                                if var_name in string_vars:
                                    filename = string_vars[var_name]

                            if filename and isinstance(call.args[1], ast.Constant) and isinstance(call.args[1].value, str):
                                mode = call.args[1].value
                                if any(m in mode for m in ['w', 'a', 'x']):
                                    edited_files.append(filename)
                                    with_files.add(filename)
            self.generic_visit(node)

    visitor = FileEditVisitor()
    visitor.visit(tree)

    return edited_files

def _sre_phase(subcommand: Optional[str]) -> str:
    """Rough mapping from str_replace_editor subcommand to a phase family."""
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
    test_related = _is_test_related(paths)

    # 1) str_replace_editor decisions (tool-specific)
    if (tool or "").lower() == "str_replace_editor":
        phase = _sre_phase(subcommand)

        if phase == "patch":
            # If SRE edit targets tests, apply key rule
            if test_related:
                return "validation" if has_patch else "localization"
            return "patch"

        # 'view' (read-only)
        if phase == "localization":
            if test_related and has_patch:
                return "validation"
            if test_related:
                return "localization"
            return "localization"

        return phase  # "general"

    # 2) Python / pytest / pylint
    if cmd in PY_CMDS:
        # Check for output redirection (python ... > file)
        if _contains_redirection(tokens):
            redir_targets = _paths_after_redirection(tokens)
            test_targets = [p for p in redir_targets if _is_test_path(p)]
            if test_targets:
                return "validation" if has_patch else "localization"
            return "patch"

        # Check for inline code execution that edits files
        # Look for Python code in args (heredoc, -c flag content)
        code_content = None
        if args:
            args_list = args if isinstance(args, (list, tuple)) else [args]
            for item in args_list:
                if isinstance(item, str):
                    # Check if this looks like Python code
                    is_code = (
                        len(item) > 20 or
                        '\n' in item or
                        'Path(' in item or
                        'open(' in item or
                        'write' in item
                    )
                    if is_code and item not in ['-', '>']:
                        code_content = item
                        break

        # If inline code is editing files, classify based on what files are being edited
        if code_content:
            edited_files_from_code = _extract_edited_files_from_python_code(code_content)
            if edited_files_from_code:
                test_files_edited = [f for f in edited_files_from_code if _is_test_path(f.lower())]
                if test_files_edited:
                    # Editing/creating test files
                    return "validation" if has_patch else "localization"
                else:
                    # Editing non-test files → patching
                    return "patch"

        # Default: Python execution (running tests/code)
        # After patch: always validation
        # Before patch: localization if test-related or pytest; otherwise localization
        if has_patch:
            return "validation"
        return "localization"

    # 3) Read-only commands (grep/find/cat/ls/head/tail/awk/nl)
    if cmd in READONLY_CMDS:
        # Piped operations without output redirection are read-only
        if _is_piped_readonly_operation(cmd, tokens):
            test_targets = [p for p in paths if _is_test_path(p)]
            if test_targets:
                return "validation" if has_patch else "localization"
            return "localization"

        if _contains_redirection(tokens):
            # These become edits when redirecting to files
            redir_targets = _paths_after_redirection(tokens)
            test_targets = [p for p in redir_targets if _is_test_path(p)]
            if test_targets:
                return "validation" if has_patch else "localization"
            return "patch"

        # Read-only, test-related AFTER patch → validation; otherwise localization
        test_targets = [p for p in paths if _is_test_path(p)]
        if test_targets:
            return "validation" if has_patch else "localization"
        return "localization"

    # 4) Edit/creation commands (sed/touch)
    if cmd in EDIT_CMDS or cmd == "sed":
        test_targets = [p for p in paths if _is_test_path(p)]
        if test_targets:
            return "validation" if has_patch else "localization"
        return "patch"

    # 5) Fallbacks: If any redirection is present, treat as edit-like
    if _contains_redirection(tokens):
        redir_targets = _paths_after_redirection(tokens)
        test_targets = [p for p in redir_targets if _is_test_path(p)]
        if test_targets:
            return "validation" if has_patch else "localization"
        return "patch"

    # Otherwise, unknown → general
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
        # str_replace_editor
        ("str_replace_editor", "create", {"path": "file.py"}, None, None, "patch"),
        ("str_replace_editor", "create", {"path": "tests/test_file.py"}, None, None, "localization"),
        ("str_replace_editor", "create", {"path": "tests/test_file.py"}, None, ["patch"], "validation"),
        ("str_replace_editor", "view", {"path": "test_file.py"}, None, ["patch"], "validation"),
        ("str_replace_editor", "str_replace", {"path": "/workspace/pytest-dev__pytest__6.0/src/_pytest/logging.py"}, None, None, "patch"),
        ("str_replace_editor", "create", {"path": "/workspace/test_example.py"}, None, None, "localization"),
        ("str_replace_editor", "create", {"path": "/workspace/example_test.py"}, None, None, "localization"),
        ("str_replace_editor", "create", {"path": "/workspace/tests/example.py"}, None, None, "localization"),
        # Heredoc embedded (redirection → edit-like, test target → localization before patch)
        (None, None, "cat", ["<<", "'EOF'", ">", "/workspace/test_file.py"], None, "localization"),
        # nl piped commands (read-only viewing operations)
        (None, None, "nl", ["filename.py", "|", "sed"], None, "localization"),
        (None, None, "nl", ["test_file.py", "|", "sed"], None, "localization"),
        (None, None, "nl", ["test_file.py", "|", "sed"], ["patch"], "validation"),
        (None, None, "nl", ["filename.py", "|", "sed"], ["patch"], "localization"),
        # nl with output redirection (becomes an edit operation)
        (None, None, "nl", ["file.py", ">", "output.txt"], None, "patch"),
        (None, None, "nl", ["file.py", ">", "test_output.py"], None, "localization"),
        (None, None, "nl", ["file.py", ">", "test_output.py"], ["patch"], "validation"),
    ]

    for i, (tool, subcmd, cmd, args, prev, expected) in enumerate(test_cases, 1):
        result = get_phase(tool, subcmd, cmd, args, prev)
        assert result == expected, f"Test case {i} failed: got {result}, expected {expected}"
    print("All test cases passed.")
