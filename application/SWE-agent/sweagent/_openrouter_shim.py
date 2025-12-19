# sweagent/_openrouter_shim.py
from __future__ import annotations
from typing import Any, Dict, List

def _is_openrouter_target(model_name: str | None, api_base: str | None) -> bool:
    name = (model_name or "").lower()
    base = (api_base or "").lower()
    return ("openrouter.ai" in base) or name.startswith("openrouter/")

def _flatten_content_parts_to_text(content: Any) -> str:
    if isinstance(content, list):
        out: List[str] = []
        for part in content:
            if isinstance(part, dict):
                # remove Anthropic-only field safely
                if "cache_control" in part:
                    part = {k: v for k, v in part.items() if k != "cache_control"}
                txt = part.get("text") or part.get("content")
                if isinstance(txt, str):
                    out.append(txt)
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(out)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    return "" if content is None else str(content)

def sanitize_messages_for_openrouter(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert to OpenAI chat-completions shape with STRING content,
    but PRESERVE function-calling fields (tool_calls, tool_call_id, name).
    """
    sanitized: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")

        # Normalize accidental {"user":{"content":...}} shape
        if role is None:
            for cand in ("user", "assistant", "system", "tool"):
                if cand in m and isinstance(m[cand], dict):
                    role = cand
                    content = m[cand].get("content", content)
                    break
        if role is None:
            role = "user"

        # Tool result messages must include tool_call_id and name; keep fields
        if role == "tool":
            out = {"role": "tool"}
            # preserve required keys if present
            if "tool_call_id" in m:
                out["tool_call_id"] = m["tool_call_id"]
            if "name" in m:
                out["name"] = m["name"]
            # content must be a string
            out["content"] = _flatten_content_parts_to_text(content)
            sanitized.append(out)
            continue

        # Assistant messages that trigger tools: preserve tool_calls
        tool_calls = m.get("tool_calls")
        if isinstance(tool_calls, list) and len(tool_calls) > 0:
            # Make a shallow copy to avoid mutating upstream
            tc_copy: List[Dict[str, Any]] = []
            for i, tc in enumerate(tool_calls):
                if not isinstance(tc, dict):
                    continue
                tc_new = dict(tc)
                # Ensure an id exists (Mistral requires it)
                if "id" not in tc_new or not tc_new["id"]:
                    tc_new["id"] = f"call_{i+1}"
                # Ensure type and function fields are present
                if "type" not in tc_new:
                    tc_new["type"] = "function"
                func = tc_new.get("function") or {}
                # keep function name/arguments as-is; do not stringify
                tc_new["function"] = func
                tc_copy.append(tc_new)
            out = {
                "role": "assistant",
                # assistant content can be empty string; keep it stringy
                "content": _flatten_content_parts_to_text(content),
                "tool_calls": tc_copy,
            }
            sanitized.append(out)
            continue

        # Plain system/user/assistant text messages → just flatten to a string
        sanitized.append({
            "role": role,
            "content": _flatten_content_parts_to_text(content),
        })

    return sanitized