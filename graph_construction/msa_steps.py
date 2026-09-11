"""Read mini-swe-agent v2 chat messages without executing their commands.

The software version and trajectory_format are different version numbers:
v2 uses ``mini-swe-agent-1.1``, also used by older Responses-style samples.
Keep format detection structural and leave those older samples to their parser.
"""

import re


def _extra(message):
    value = message.get("extra")
    return value if isinstance(value, dict) else {}


def _text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            item["text"] for item in value
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""


def is_msa_v2(traj_data):
    """Identify role-based v2 files, including interrupted/empty v2 runs."""
    if traj_data.get("trajectory_format") == "mini-swe-agent-1":
        return False
    messages = traj_data.get("messages", [])
    if any(isinstance(m, dict) and isinstance(m.get("output"), list) for m in messages):
        return False
    info = traj_data.get("info") or {}
    version = str(info.get("mini_version", "")) if isinstance(info, dict) else ""
    return version.startswith("2.") or any(
        isinstance(m, dict) and m.get("role") == "assistant"
        and isinstance(_extra(m).get("actions"), list)
        for m in messages
    )


def _result(message, message_index):
    extra = _extra(message)
    content = _text(message.get("content"))
    raw = extra.get("raw_output")
    if isinstance(raw, str):
        observation = raw
    else:
        match = re.search(r"<output>(.*?)</output>", content, re.DOTALL)
        observation = match.group(1).strip() if match else content
    returncode = extra.get("returncode")
    if returncode is None:
        match = re.search(r"<returncode>\s*(-?\d+)\s*</returncode>", content)
        if match:
            returncode = int(match.group(1))
    return {
        "observation": observation,
        "result_message_index": message_index,
        "result_missing": False,
        "returncode": returncode,
        "exception_info": extra.get("exception_info"),
    }


def iter_msa_v2_steps(traj_data):
    """Yield one step per assistant turn, with results attached per action.

    extra.actions is the agent's parsed action list; raw tool_calls may also
    contain rejected calls, so they are deliberately not used as a fallback.
    Result matching is bounded by the next assistant turn. Unkeyed regex-model
    observations are matched in order only when their counts agree exactly.
    """
    messages = traj_data.get("messages", [])
    assistants = [
        index for index, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    for step_index, message_index in enumerate(assistants):
        message = messages[message_index]
        stop = assistants[step_index + 1] if step_index + 1 < len(assistants) else len(messages)
        parts = []
        for value in (message.get("reasoning_content"), message.get("content")):
            text = _text(value)
            if text and text not in parts:
                parts.append(text)

        raw_actions = _extra(message).get("actions") or []
        actions = [a for a in raw_actions if isinstance(a, dict)
                   and isinstance(a.get("command"), str) and a["command"].strip()]
        keyed_results = {}
        unkeyed_results = []
        feedback = []
        exit_status = None
        for index in range(message_index + 1, stop):
            result = messages[index]
            if not isinstance(result, dict):
                continue
            call_id = result.get("tool_call_id")
            if _extra(result).get("exit_status"):
                exit_status = _extra(result)["exit_status"]
            if result.get("role") == "tool" and call_id:
                keyed_results.setdefault(call_id, []).append((index, result))
            elif result.get("role") == "user" and not call_id:
                extra = _extra(result)
                if "returncode" in extra or "raw_output" in extra or "<output>" in _text(result.get("content")):
                    unkeyed_results.append((index, result))
                elif extra.get("interrupt_type"):
                    feedback.append(_text(result.get("content")))

        unkeyed_actions = [a for a in actions if not a.get("tool_call_id")]
        positional = iter(unkeyed_results) if len(unkeyed_actions) == len(unkeyed_results) else iter(())
        normalized = []
        for action_index, action in enumerate(actions):
            call_id = action.get("tool_call_id")
            candidates = keyed_results.get(call_id, []) if call_id else []
            pair = (candidates[0] if len(candidates) == 1 else None) if call_id else next(positional, None)
            result = _result(pair[1], pair[0]) if pair else {
                "observation": "", "result_message_index": None,
                "result_missing": True, "returncode": None, "exception_info": None,
            }
            normalized.append({
                "command": action["command"], "tool_call_id": call_id,
                "action_index": action_index, **result,
            })
        yield {
            "step_index": step_index, "message_index": message_index,
            "thought": "\n\n".join(parts), "actions": normalized,
            "feedback": "\n\n".join(feedback), "exit_status": exit_status,
        }
