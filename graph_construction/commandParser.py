import yaml
import shlex
import re
import bashlex
from typing import Dict, List, Optional, Any

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
        subcommand = tokens[1] if has_subcommand else None
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
            # bashlex can throw AttributeError/IndexError internally; treat as failure
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
        s = self._clean(cmd_str)
        if not s:
            return False  # empty is not complex

        parts = self._safe_bashlex_parse(s)
        if parts is None:
            return True  # fail-safe on ANY parse error

        for part in parts:
            if part is None:
                # Defensive: rare bashlex behavior; treat conservatively
                return True
            if self._has_control_structure(part):
                return True
        return False

    def _has_control_structure(self, node) -> bool:
        if node is None:
            return False
        if hasattr(node, 'kind') and node.kind in {
            'if', 'for', 'while', 'until', 'case', 'function'
        }:
            return True
        if hasattr(node, 'parts') and node.parts:
            for part in node.parts:
                if self._has_control_structure(part):
                    return True
        if hasattr(node, 'list') and node.list:
            for sub in node.list:
                if self._has_control_structure(sub):
                    return True
        return False

    # --- quote-aware splitter for ;, &&, || outside quotes ----------------
    def _split_outside_quotes(self, s: str) -> List[str]:
        """
        Split on ;, &&, || ONLY when outside single/double quotes.
        Whitespace around separators is ignored.
        """
        parts: List[str] = []
        buf: List[str] = []
        i, n = 0, len(s)
        in_single = False
        in_double = False
        escape = False

        while i < n:
            ch = s[i]

            if escape:
                # keep the escaped char verbatim
                buf.append(ch)
                escape = False
                i += 1
                continue

            if ch == '\\':
                buf.append(ch)
                escape = True
                i += 1
                continue

            if ch == "'" and not in_double:
                in_single = not in_single
                buf.append(ch)
                i += 1
                continue

            if ch == '"' and not in_single:
                in_double = not in_double
                buf.append(ch)
                i += 1
                continue

            if not in_single and not in_double:
                # two-char separators first: &&, ||
                if ch == '&' and i + 1 < n and s[i + 1] == '&':
                    part = ''.join(buf).strip()
                    if part:
                        parts.append(part)
                    buf = []
                    i += 2
                    while i < n and s[i].isspace():
                        i += 1
                    continue
                if ch == '|' and i + 1 < n and s[i + 1] == '|':
                    part = ''.join(buf).strip()
                    if part:
                        parts.append(part)
                    buf = []
                    i += 2
                    while i < n and s[i].isspace():
                        i += 1
                    continue
                # one-char separator: ;
                if ch == ';':
                    part = ''.join(buf).strip()
                    if part:
                        parts.append(part)
                    buf = []
                    i += 1
                    while i < n and s[i].isspace():
                        i += 1
                    continue

            # default: accumulate
            buf.append(ch)
            i += 1

        tail = ''.join(buf).strip()
        if tail:
            parts.append(tail)
        return parts

    def split_env_and_command(self, cmd_str: str) -> List[str]:
        env_pattern = re.compile(r"^((?:\w+=[^ \t\n\r\f\v]+[ \t]*)+)(.+)?")
        match = env_pattern.match(cmd_str.strip())

        if match:
            env_part = match.group(1).strip()
            rest = (match.group(2) or "").strip()
            return [env_part] + (self._split_outside_quotes(rest) if rest else [])
        else:
            return self._split_outside_quotes(cmd_str.strip())

    def _split_env_preserving_rest_no_split(self, s: str) -> List[str]:
        env_pattern = re.compile(r"^((?:\w+=[^ \t\n\r\f\v]+[ \t]*)+)(.+)?")
        m = env_pattern.match(s.strip())
        if m:
            env_part = m.group(1).strip()
            rest = (m.group(2) or "").strip()
            return [env_part] + ([rest] if rest else [])
        return [s]

    def parse(self, cmd_str: str) -> List[Dict[str, Any]]:
        if self.is_complex(cmd_str):
            return [{"command": "complex_command", "args": [cmd_str.strip()]}]

        s = cmd_str.strip()
        # Always do quote-aware splitting; separators inside quotes are ignored,
        # but separators outside quotes still split.
        parts = self.split_env_and_command(s)

        results = []

        for subcmd in parts:
            if re.fullmatch(r"\w+=.+", subcmd.strip()) or all("=" in token for token in subcmd.strip().split()):
                results.append({"command": "set_env", "args": [subcmd.strip()]})
                continue

            try:
                tokens = shlex.split(subcmd.strip())
            except ValueError:
                continue

            if not tokens:
                continue

            tool = tokens[0]
            if tool in self.tool_map:
                result = self.tool_map[tool].parse(tokens)
                if result:
                    results.append(result)
            else:
                result = self.parse_bash_command(tokens)
                if result:
                    results.append(result)
        return results

    def parse_bash_command(self, tokens: List[str]) -> Optional[Dict[str, Any]]:
        if not tokens:
            return None

        command = tokens[0]
        args = []
        flags = {}
        i = 1

        # Interpreters that embed inline code via -c/-e (and bash -lc)
        interpreters_with_inline = {"python", "python3", "bash", "sh", "zsh", "node", "ruby", "perl", "psql", "mysql", "sqlite3"}

        while i < len(tokens):
            token = tokens[i]
            if token.startswith('--'):
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
                # bundled short flags (e.g., -xzvf, -lc)
                if len(token) > 2:
                    if command in interpreters_with_inline and 'c' in token[1:]:
                        # set other bundled flags True, capture next token as code for -c
                        for ch in token[1:]:
                            if ch != 'c':
                                flags[ch] = True
                        code_val = True
                        if i + 1 < len(tokens):
                            code_val = tokens[i + 1]
                            i += 1
                        flags['c'] = code_val
                        i += 1
                        # remaining tokens are positional args; stop parsing flags
                        while i < len(tokens):
                            args.append(tokens[i])
                            i += 1
                        break
                    else:
                        for ch in token[1:]:
                            flags[ch] = True

                else:
                    key = token[1:]
                    # capture inline code for interpreters with -c/-e
                    if command in interpreters_with_inline and key in {'c', 'e'}:
                        code_val = True
                        if i + 1 < len(tokens):
                            code_val = tokens[i + 1]
                            i += 1
                        flags[key] = code_val
                        i += 1
                        # remaining tokens are positional args; stop parsing flags
                        while i < len(tokens):
                            args.append(tokens[i])
                            i += 1
                        break
                    # default short-flag behavior
                    value = True
                    if i + 1 < len(tokens) and not tokens[i + 1].startswith('-'):
                        value = tokens[i + 1]
                        i += 1
                    flags[key] = value

            else:
                args.append(token)
            i += 1

        return {
            "command": command.strip(),
            "args": args,
            "flags": flags
        }