import yaml
import shlex
import re
import bashlex
from typing import Dict, List, Optional, Any, Tuple


# --- ToolDefinition class for parsing specific tool commands (for SWE-agent) ----------------
class ToolDefinition:
    def __init__(self, tool_name: str, subcommands: List[str], arg_specs: Dict[str, dict]):
        self.tool_name = tool_name
        self.subcommands = subcommands
        self.arg_specs = arg_specs

    def parse(self, tokens: List[str]) -> Optional[Dict[str, Any]]:
        if not tokens or tokens[0] != self.tool_name:
            return None

        has_subcommand = bool(self.subcommands)
        subcommand = tokens[1] if len(tokens) > 1 and has_subcommand else None
        if has_subcommand and subcommand not in self.subcommands:
            return None

        parsed = {
            "tool": self.tool_name.strip(),
            "subcommand": subcommand.strip() if subcommand else None,
            "args": {}
        }

        args = tokens[2:] if has_subcommand else tokens[1:]
        positional = [
            spec for spec in self.arg_specs.values()
            if "argument_format" not in spec and spec.get("name") != "command"
        ]

        i = 0
        pos_idx = 0
        while i < len(args):
            token = args[i]
            if token.startswith("--"):
                key = token[2:]
                spec = self.arg_specs.get(key)
                if not spec:
                    i += 1
                    continue

                value = True
                arg_type = spec.get("type")

                if arg_type == "array":
                    i += 1
                    value = []
                    while i < len(args) and not args[i].startswith("--"):
                        try:
                            value.append(int(args[i]))
                        except ValueError:
                            break
                        i += 1
                    parsed["args"][key] = value
                    continue
                elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                    value = args[i + 1]
                    if arg_type == "integer":
                        try:
                            value = int(value)
                        except ValueError:
                            pass
                    parsed["args"][key] = value
                    i += 2
                    continue
                else:
                    parsed["args"][key] = value
            else:
                if pos_idx < len(positional):
                    name = positional[pos_idx]["name"]
                    value = token
                    if positional[pos_idx].get("type") == "integer":
                        try:
                            value = int(value)
                        except ValueError:
                            pass
                    parsed["args"][name] = value
                    pos_idx += 1
            i += 1

        return parsed


class CommandParser:
    """
    Bash command parser that properly handles:
    - Heredocs (<<, <<-)
    - Compound commands with proper separator detection (; && ||)
    - Nested structures (command substitution, subshells)
    - Environment variables
    - Pipelines and redirections

    Uses bashlex AST for accurate parsing - respects quotes and nesting.
    """

    def __init__(self):
        self.tool_map: Dict[str, ToolDefinition] = {}

    def _clean(self, s: str) -> str:
        # Remove NULs (can trip bashlex) and trim
        return (s or "").replace("\x00", "").strip()

    def _safe_bashlex_parse(self, s: str) -> Optional[List[Any]]:
        """Return bashlex AST parts or None on ANY failure."""
        try:
            return bashlex.parse(s)
        except Exception:
            return None

    def load_tool_yaml_files(self, yaml_paths: List[str]):
        for path in yaml_paths:
            try:
                with open(path, "r") as f:
                    content = yaml.safe_load(f)
                    for tool_name, spec in content.get("tools", {}).items():
                        if not spec:
                            continue
                        subcommands = next(
                            (arg.get("enum", []) for arg in spec.get("arguments", []) if arg.get("name") == "command"),
                            []
                        )
                        arg_specs = {arg["name"]: arg for arg in spec.get("arguments", [])}
                        self.tool_map[tool_name] = ToolDefinition(tool_name, subcommands, arg_specs)
            except Exception as e:
                print(f"Failed to load YAML from {path}: {e}")

    def is_complex(self, cmd_str: str) -> bool:
        """
        Determine if a command is too complex for detailed parsing.
        Complex commands include: for loops, while loops, if statements, case statements, functions.
        Heredocs and simple compound commands (; && ||) are NOT considered complex.
        This is checked BEFORE splitting by operators.
        """
        s = self._clean(cmd_str)
        if not s:
            return False

        parts = self._safe_bashlex_parse(s)
        if parts is None:
            # If bashlex can't parse it, might be complex
            # But check if it's just a heredoc first
            if self._is_simple_heredoc(s):
                return False
            return True

        for part in parts:
            if part is None:
                return True
            # Only check for actual control structures at the top level
            # Don't check for compound/pipeline here - we'll check those after splitting
            if self._has_top_level_control_structure(part):
                return True
        return False

    def _has_top_level_control_structure(self, node) -> bool:
        """Check ONLY for control structures (for, while, if, case, function) at top level."""
        if node is None:
            return False
        if hasattr(node, 'kind') and node.kind in {
            'if', 'for', 'while', 'until', 'case', 'function'
        }:
            return True
        # Don't recurse - only check top level
        return False

    def _is_individual_command_complex(self, cmd_str: str) -> bool:
        """
        Check if an individual command (after splitting by &&, ||, ;) is complex.
        Complex individual commands include:
        - Subshells: (...)
        - Command groups: {...}
        - Pipelines: cmd1 | cmd2
        - Command substitution at the start: var=$(...)
        """
        s = cmd_str.strip()
        if not s:
            return False

        # Parse this individual command
        parts = self._safe_bashlex_parse(s)
        if parts is None:
            return False  # If can't parse, let it through for shlex to handle

        for part in parts:
            if part is None:
                continue
            # Check if this command contains compound or pipeline structures
            if self._has_control_structure(part):
                return True

        return False

    def _has_control_structure(self, node) -> bool:
        """
        Check for control structures and complex constructs that should not be parsed.
        Complex includes:
        - Control structures: for, while, if, case, function
        - Compound commands: subshells (...), command groups {...}
        - Pipelines are NOT considered complex (they're common and parseable)
        """
        if node is None:
            return False

        # Check for control structures
        if hasattr(node, 'kind') and node.kind in {
            'if', 'for', 'while', 'until', 'case', 'function'
        }:
            return True

        # Check for compound nodes (subshells, command groups)
        if hasattr(node, 'kind') and node.kind == 'compound':
            # Compound nodes represent (...) subshells or {...} groups
            # These should be treated as complex
            return True

        # Note: Pipelines are NOT checked here - they're parseable
        # We'll handle pipelines by keeping the | in the args

        # Recursively check parts (but skip pipeline nodes)
        if hasattr(node, 'parts') and node.parts:
            for part in node.parts:
                # Skip pipeline nodes themselves, but check their contents
                if hasattr(part, 'kind') and part.kind == 'pipeline':
                    # Don't mark as complex just because it's a pipeline
                    continue
                if self._has_control_structure(part):
                    return True

        # Recursively check list
        if hasattr(node, 'list') and node.list:
            for sub in node.list:
                if self._has_control_structure(sub):
                    return True

        return False

    def _is_simple_heredoc(self, cmd_str: str) -> bool:
        """
        Check if this is a simple heredoc command.
        Examples: cat << EOF, python3 - <<'PY', cat <<'EOF' > file.txt

        A command is a simple heredoc if:
        1. It contains the heredoc operator (<<-?) followed by a delimiter
        2. The part BEFORE << doesn't contain bash control keywords
        3. The command doesn't contain compound operators (&&, ||, ;) that would
           split it into multiple commands (this includes compound commands with
           multiple heredocs like: cmd1 <<EOF ... EOF && cmd2 <<EOF ... EOF)

        The content after the delimiter can be anything - it's data, not bash commands.
        """
        s = cmd_str.strip()

        # Check if heredoc operator exists with a valid delimiter
        # Delimiter can be quoted or unquoted
        if not re.search(r'<<-?\s*([\'"]?)\w+\1', s):
            return False

        # Extract the part before the first << operator
        before_heredoc = s.split('<<')[0]

        # Check if the part before << contains bash control keywords
        # These indicate control structures, not simple heredocs
        control_pattern = r'\b(if|then|fi|for|while|do|done|case|esac|until|function)\b'
        if re.search(control_pattern, before_heredoc):
            return False

        # Check if the part before << contains compound operators
        # If so, this should be split into multiple commands first
        compound_operators = ['&&', '||', ';']
        if any(op in before_heredoc for op in compound_operators):
            return False

        # Check if there are multiple heredocs (indicates compound command)
        # Pattern: <<DELIM followed by content and delimiter on its own line, then another <<
        # This catches cases like: python3 <<'PY' ... PY\n&& python3 <<'PY' ... PY
        heredoc_count = len(re.findall(r'<<-?\s*([\'"]?)\w+\1', s))
        if heredoc_count > 1:
            return False

        return True

    def _parse_heredoc(self, cmd_str: str) -> Optional[Dict[str, Any]]:
        """
        Parse a heredoc command.
        Formats: cat <<EOF, python3 - <<'PY', cat <<'EOF' > file.txt

        Returns: {command: str, args: [arg1, arg2, ..., heredoc_content], flags: {}}
        Example: python3 - <<'PY' ... PY → {command: 'python3', args: ['-', content], flags: {}}
        """
        s = cmd_str.strip()

        # Find the heredoc operator and extract delimiter
        heredoc_match = re.search(r'<<-?\s*([\'"]?)(\w+)\1', s)
        if not heredoc_match:
            return None

        delimiter = heredoc_match.group(2)
        heredoc_start = heredoc_match.start()

        # Extract everything before << (command and args)
        before_heredoc = s[:heredoc_start].strip()

        # Extract everything after the heredoc operator
        after_operator = s[heredoc_match.end():].strip()

        # Parse command and args before <<
        try:
            tokens = shlex.split(before_heredoc)
        except ValueError:
            tokens = before_heredoc.split()

        if not tokens:
            return None

        command = tokens[0]
        args = list(tokens[1:])  # Args before the heredoc

        # Handle output redirection (e.g., cat <<'EOF' > file.txt)
        redirect_match = re.match(r'>\s*([^\n]+)', after_operator)
        output_file = None
        heredoc_content = after_operator

        if redirect_match:
            output_file = redirect_match.group(1).strip()
            heredoc_content = after_operator[redirect_match.end():].strip()

        # Remove closing delimiter if present
        if f"\n{delimiter}" in heredoc_content:
            heredoc_content = heredoc_content.split(f"\n{delimiter}")[0]

        # Add heredoc content to args
        if heredoc_content:
            args.append(heredoc_content)

        # Add redirection if present
        if output_file:
            args.append(">")
            args.append(output_file)

        return {
            "command": command,
            "args": args,
            "flags": {"__heredoc__": True}  # Mark as heredoc for mapLang
        }

    def _extract_command_text(self, cmd_str: str, node) -> str:
        """Extract the text for a specific AST node from the original command string."""
        if hasattr(node, 'pos'):
            start = node.pos[0]
            end = node.pos[1]
            return cmd_str[start:end]
        return ""

    def _split_by_operators(self, cmd_str: str) -> List[str]:
        """
        Split command by top-level operators (; && ||) using bashlex AST.
        Returns list of individual command strings.
        This properly handles quotes and nesting - operators inside quotes or $(...) are NOT split.
        """
        s = self._clean(cmd_str)
        if not s:
            return []

        parts = self._safe_bashlex_parse(s)
        if parts is None:
            # If bashlex fails, return the whole command as-is
            return [s]

        commands = []

        for part in parts:
            if part is None:
                continue

            # Process this part for operators
            extracted = self._extract_commands_from_node(s, part)
            commands.extend(extracted)

        return commands if commands else [s]

    def _extract_commands_from_node(self, cmd_str: str, node) -> List[str]:
        """
        Recursively extract individual commands from a bashlex node.
        Handles operator nodes (; && ||) by splitting on them.

        Bashlex creates a tree structure where nodes with kind='list' contain a 'parts' list
        that alternates between command nodes and operator nodes:
        [command, operator, command, operator, command, ...]
        """
        commands = []

        if node is None:
            return commands

        # Check if this node represents a list with operators
        if hasattr(node, 'kind') and node.kind == 'list':
            if hasattr(node, 'parts') and node.parts:
                # Parts alternate: command, operator, command, operator, command...
                # Extract only the non-operator nodes
                for part in node.parts:
                    # Skip operator nodes
                    if hasattr(part, 'kind') and part.kind == 'operator':
                        continue

                    # Recursively extract commands (in case of nested lists)
                    sub_cmds = self._extract_commands_from_node(cmd_str, part)
                    commands.extend(sub_cmds)
            else:
                # List without parts - extract as single command
                cmd_text = self._extract_command_text(cmd_str, node)
                if cmd_text:
                    commands.append(cmd_text)
        else:
            # Not a list node - extract the command text
            cmd_text = self._extract_command_text(cmd_str, node)
            if cmd_text:
                commands.append(cmd_text)

        return commands

    def _extract_env_prefix(self, cmd_str: str) -> Tuple[Optional[str], str]:
        """
        Extract environment variable prefix from command.
        Returns (env_part, remaining_command)
        """
        env_pattern = re.compile(r"^((?:\w+=[^ \t\n\r\f\v]+[ \t]*)+)(.+)?")
        match = env_pattern.match(cmd_str.strip())

        if match:
            env_part = match.group(1).strip()
            rest = (match.group(2) or "").strip()
            return (env_part, rest)
        return (None, cmd_str)

    def parse(self, cmd_str: str) -> List[Dict[str, Any]]:
        """
        Parse command string into structured format.
        Handles compound commands, heredocs, and environment variables.
        """
        s = cmd_str.strip()
        if not s:
            return []

        # Check if this is a simple heredoc command (before complexity check)
        if self._is_simple_heredoc(s):
            parsed = self._parse_heredoc(s)
            if parsed:
                return [parsed]
            # If heredoc parsing failed, treat as complex
            return [{"command": "complex_command", "args": [s]}]

        # Check for complexity
        if self.is_complex(s):
            return [{"command": "complex_command", "args": [s]}]

        # Split by top-level operators using bashlex
        command_parts = self._split_by_operators(s)

        results = []

        for cmd_text in command_parts:
            cmd_text = cmd_text.strip()
            if not cmd_text:
                continue

            # Check if this individual command contains complex structures
            # (subshells, pipelines, etc.) AFTER splitting
            if self._is_individual_command_complex(cmd_text):
                results.append({"command": "complex_command", "args": [cmd_text]})
                continue

            # Check for environment variable assignment
            env_part, remaining = self._extract_env_prefix(cmd_text)

            if env_part:
                # Add environment variable assignment
                results.append({"command": "set_env", "args": [env_part]})
                cmd_text = remaining

            if not cmd_text:
                continue

            # Check if pure env assignment (no command after)
            if re.fullmatch(r"\w+=.+", cmd_text):
                results.append({"command": "set_env", "args": [cmd_text]})
                continue

            # Try to tokenize with shlex
            try:
                tokens = shlex.split(cmd_text)
            except ValueError:
                # If shlex fails, treat as complex
                results.append({"command": "complex_command", "args": [cmd_text]})
                continue

            if not tokens:
                continue

            tool = tokens[0]

            # Check if this is a known tool
            if tool in self.tool_map:
                result = self.tool_map[tool].parse(tokens)
                if result:
                    results.append(result)
            else:
                # Parse as bash command
                result = self.parse_bash_command(tokens)
                if result:
                    results.append(result)

        return results

    def parse_bash_command(self, tokens: List[str]) -> Optional[Dict[str, Any]]:
        """Parse a bash command from tokens."""
        if not tokens:
            return None

        command = tokens[0]
        args = []
        flags = {}
        i = 1

        # Interpreters that embed inline code via -c/-e (and bash -lc)
        interpreters_with_inline = {
            "python", "python3", "bash", "sh", "zsh",
            "node", "ruby", "perl", "psql", "mysql", "sqlite3"
        }

        while i < len(tokens):
            token = tokens[i]

            if token.startswith('--'):
                # Long flag
                if '=' in token:
                    key, value = token[2:].split('=', 1)
                    flags[key] = value
                else:
                    key = token[2:]
                    value = True
                    if i + 1 < len(tokens) and not tokens[i + 1].startswith('-'):
                        value = tokens[i + 1]
                        i += 1
                    flags[key] = value

            elif token.startswith('-') and len(token) > 1:
                # Short flag(s)
                if len(token) > 2:
                    # Bundled short flags (e.g., -xzvf, -lc)
                    if command in interpreters_with_inline and 'c' in token[1:]:
                        # Set other bundled flags True, capture next token as code for -c
                        for ch in token[1:]:
                            if ch != 'c':
                                flags[ch] = True
                        code_val = True
                        if i + 1 < len(tokens):
                            code_val = tokens[i + 1]
                            i += 1
                        flags['c'] = code_val
                        i += 1
                        # Remaining tokens are positional args
                        while i < len(tokens):
                            args.append(tokens[i])
                            i += 1
                        break
                    else:
                        # All flags are boolean
                        for ch in token[1:]:
                            flags[ch] = True
                else:
                    # Single short flag
                    key = token[1:]
                    # Special handling for inline code
                    if command in interpreters_with_inline and key in {'c', 'e'}:
                        code_val = True
                        if i + 1 < len(tokens):
                            code_val = tokens[i + 1]
                            i += 1
                        flags[key] = code_val
                        i += 1
                        # Remaining tokens are positional args
                        while i < len(tokens):
                            args.append(tokens[i])
                            i += 1
                        break
                    # Default short-flag behavior
                    value = True
                    if i + 1 < len(tokens) and not tokens[i + 1].startswith('-'):
                        value = tokens[i + 1]
                        i += 1
                    flags[key] = value

            else:
                # Positional argument
                args.append(token)

            i += 1

        return {
            "command": command.strip(),
            "args": args,
            "flags": flags
        }


if __name__ == "__main__":
    parser = CommandParser()

    commands = [
        "cd /home/user",
        "ls -la",
        "grep --color=auto 'pattern' file.txt",
        "rm -rf /tmp/*",
        "echo 'Hello, World!' > /testbed/reproduce_error.py",
        "python3 script.py --input=data.txt --verbose",
        "cd /project && python3 run.py",
        "PYTHONPATH=/testbed",
        "PYTHONPATH=/testbed python3 main.py",
        "nl -ba filename.py | sed -n '10,20p'",
        "cd /workspace/django__django__3.0 && find . -name \"*.py\" | grep -i test | head -5",
        "cd /workspace/django__django__3.0 && grep -r \"class Avg\" --include=\"*.py\" .",
        "PYTHONPATH=/project python3 script.py --input file.txt && echo \"done\" ; rm temp.log;",
        "cd /workspace/sympy__sympy__1.9 && python -m pytest sympy/polys/tests/test_monomials.py -v",
        "cd /project || python -c 'print(\"Failed to change directory\")'",
        "\ncd /workspace/django__django__4.0\necho \"SECRET_KEY = 'dummy'\" > test_settings.py\necho \"DATABASES = {'default': {'ENGINE': 'django.db.backends.dummy'}}\" >> test_settings.py\necho \"INSTALLED_APPS = []\" >> test_settings.py\n",
        "PYTHONPATH=/workspace/scikit-learn__scikit-learn__0.22",
        "\ncd /workspace/astropy__astropy__4.3 && \nfind . -name \"*.py\" -exec grep -l \"class TimeSeries\" {} \;\n\n",
        "\ncd /workspace/psf__requests__2.0 && \n(grep -ri \"test\" README* || grep -ri \"test\" .github/workflows/* || grep -ri \"pytest\" setup.* || true) && \nfind . -name \"*test*.py\" | head -5",
        'cat << \'EOF\' > /workspace/test_hstack_fix.py\nimport sympy as sy\n\n# Test case 1: Zero-height matrices\nM1 = sy.Matrix.zeros(0, 0)\nM2 = sy.Matrix.zeros(0, 1)\nM3 = sy.Matrix.zeros(0, 2)\nM4 = sy.Matrix.zeros(0, 3)\nresult = sy.Matrix.hstack(M1, M2, M3, M4).shape\nprint(f"Zero-height hstack result: {result} (should be (0, 6))")\n\n# Test case 2: Non-zero height matrices\nM1 = sy.Matrix.zeros(1, 0)\nM2 = sy.Matrix.zeros(1, 1)\nM3 = sy.Matrix.zeros(1, 2)\nM4 = sy.Matrix.zeros(1, 3)\nresult = sy.Matrix.hstack(M1, M2, M3, M4).shape\nprint(f"Non-zero height hstack result: {result} (should be (1, 6))")\nEOF',
        "python3 - <<'PY'\ndef f(self):\n    return 1\nprop = property(f)\nprint(\"before:\", prop.__doc__)\nprop.__doc__ = \"assigned\"\nprint(\"after assign:\", prop.__doc__)\nclass C:\n    @classmethod\n    def cm(cls): \"cmm\"; return 1\nprint(\"classmethod doc before:\", C.cm.__doc__)\n# Try creating a standalone classmethod object and setting __doc__\ndef g(cls): \"gdoc\"; return 2\ncm_obj = classmethod(g)\nprint(\"cm_obj.__doc__ before:\", cm_obj.__doc__)\ncm_obj.__func__.__doc__ = \"changed_gdoc\"\nprint(\"cm_obj.__doc__ after func doc change:\", cm_obj.__doc__)\ncm_obj.__doc__ = \"assigned_cm\"\nprint(\"cm_obj.__doc__ after assign:\", cm_obj.__doc__)\nPY",
        ]

    for cmd in commands:
        result = parser.parse(cmd)
        print(f"\n>>> {cmd}")
        for r in result:
            print(r)

    complex_bash = """
for file in $(git status --porcelain | grep -E "^(M| M|\\?\\?|A| A)" | cut -c4-); do
    if [ -f "$file" ] && (file "$file" | grep -q "executable" || git check-attr binary "$file" | grep -q "binary: set"); then
        git rm -f "$file" 2>/dev/null || rm -f "$file"
        echo "Removed: $file"
    fi
done
    """

    result = parser.parse(complex_bash)
    print(f"\n>>> Complex Bash:\n{complex_bash}")
    for r in result:
        print(r)