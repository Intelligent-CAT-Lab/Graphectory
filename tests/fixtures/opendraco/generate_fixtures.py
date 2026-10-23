"""Generate the OpenDraco `run.jsonl` fixtures in this directory.

Every log here is written by OpenDraco's own executor (`run_topology`) against its
scripted `FakeProvider`, so the records are genuine rather than hand-laid. No Ollama,
no Docker: the one scenario that uses `bash` gets a scripted in-process sandbox.

Regenerate (from the OpenDraco backend checkout, so its environment is used):

    cd <OpenDraco>/backend
    UV_INDEX_URL=https://pypi.org/simple uv run python \\
        <Graphectory>/tests/fixtures/opendraco/generate_fixtures.py . [WORK_DIR]

The first argument is the backend directory (or set OPENDRACO_BACKEND). WORK_DIR, where
the throw-away projects are built, defaults to a fresh temporary directory. Each
scenario becomes `<name>.run.jsonl` next to this script. Timestamps, uuids and git
shas differ on every regeneration; the record structure does not.

The graph/script helpers come from OpenDraco's `tests/export/test_archive.py`; only the
run driver is replicated here, because that module's `_run` pins one run id.
"""

import asyncio
import os
from datetime import UTC, datetime, timedelta
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _backend() -> Path:
    raw = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OPENDRACO_BACKEND")
    if not raw:
        sys.exit("usage: generate_fixtures.py <opendraco-backend-dir> [work-dir]")
    return Path(raw).resolve()


BACKEND = _backend()
sys.path.insert(0, str(BACKEND / "tests" / "export"))
sys.path.insert(0, str(BACKEND / "src"))

import test_archive as ta  # noqa: E402

from opendraco.execution.continuation import fork_point, resumption  # noqa: E402
from opendraco.execution.runner import run_topology  # noqa: E402
from opendraco.execution.state import RunHalted  # noqa: E402
from opendraco.models.agent import AgentConfig  # noqa: E402
from opendraco.models.topology import NodeMode  # noqa: E402
from opendraco.providers.base import (  # noqa: E402
    CallMetrics,
    CallRole,
    ModelProvider,
    ModelResponse,
    ToolCall,
    Usage,
)
from opendraco.providers.fake import FakeProvider  # noqa: E402
from opendraco.runlog.records import RECORD_ADAPTER, ActivationStartedRecord, RunStatus  # noqa: E402
from opendraco.sandbox.base import ExecResult, SandboxBackend  # noqa: E402

LOOP_EDGE = "e5"  # reviewer -> writer, the edge that closes the ping-pong loop


# --- run driver ----------------------------------------------------------------------

# FakeProvider reports no per-HTTP-call metrics, and the executor writes one `model_call`
# record per entry of `ModelResponse.calls`, so a bare scripted response produces none.
# Every scripted response is given one primary call's metrics here (what OllamaProvider
# would have attached); a response built with `_with_fallback` carries two.


def _metrics(role=CallRole.PRIMARY, tokens=(120, 30)):
    ended = datetime.now(UTC)
    usage = Usage(input_tokens=tokens[0], output_tokens=tokens[1], total_tokens=sum(tokens))
    return CallMetrics(
        role=role,
        started_at=ended - timedelta(milliseconds=850),
        ended_at=ended,
        usage=usage,
        load_ms=12.5,
        prompt_eval_ms=210.0,
        eval_ms=600.0,
    )


def _metered(entry):
    if isinstance(entry, ModelResponse) and not entry.calls:
        call = _metrics()
        return entry.model_copy(update={"calls": [call], "usage": call.usage})
    return entry


def _with_fallback(response: ModelResponse) -> ModelResponse:
    """A routing answer that took a retry: a primary call, then a fallback one."""
    calls = [_metrics(CallRole.PRIMARY), _metrics(CallRole.FALLBACK)]
    return response.model_copy(update={"calls": calls, "usage": calls[-1].usage})



class _Held:
    """A FakeProvider that can hold its n-th call open, so a run can be paused inside it."""

    def __init__(self, script, *, held_call=None):
        self._fake = FakeProvider([_metered(e) for e in script])
        self._held_call = held_call
        self._asked = 0
        self.entered = asyncio.Event()
        self.released = asyncio.Event()

    async def _ask(self):
        self._asked += 1
        if self._asked == self._held_call:
            self.entered.set()
            await self.released.wait()

    async def generate(self, messages, *, tools=None, output_schema=None):
        await self._ask()
        return await self._fake.generate(messages, tools=tools, output_schema=output_schema)

    async def stream(self, messages, *, tools=None):
        await self._ask()
        async for item in self._fake.stream(messages, tools=tools):
            yield item


def _providers(scripts):
    by_agent = {
        a: s if isinstance(s, _Held) else FakeProvider([_metered(e) for e in s])
        for a, s in scripts.items()
    }

    def provider_for(agent: AgentConfig) -> ModelProvider:
        return by_agent[agent.id]

    return provider_for


def _no_sandbox(workspace: Path) -> SandboxBackend:
    raise AssertionError("this scenario declares no bash tool")


async def _run(project, run_id, scripts, *, sandbox_for=_no_sandbox, continuation=None):
    result = await run_topology(
        project,
        run_id,
        provider_for=_providers(scripts),
        sandbox_for=sandbox_for,
        continuation=continuation,
    )
    assert result.status is RunStatus.COMPLETED, (run_id, result.status, result.error)
    return result


def _log_lines(project, run_id):
    return project.layout.run_log_path(run_id).read_text(encoding="utf-8").splitlines()


def _activations(project, run_id, agent_id):
    records = [RECORD_ADAPTER.validate_json(line) for line in _log_lines(project, run_id)]
    return [r for r in records if isinstance(r, ActivationStartedRecord) and r.agent_id == agent_id]


# --- scenarios -----------------------------------------------------------------------


async def one_agent_write(root):
    project = ta._project(root, ta._one_agent(tools={ta._WRITER: [ta._WRITE_TOOL]}), name="one")
    run_id = "2026-10-23-a001"
    await _run(
        project,
        run_id,
        {ta._WRITER: [ta._writes(ta._ARTIFACT_PATH, ta._ARTIFACT), ta._says(ta._DRAFT)]},
    )
    return [(None, project, run_id)]


async def two_agents(root):
    project = ta._project(root, ta._two_agents(), name="two")
    run_id = "2026-10-23-a002"
    await _run(
        project,
        run_id,
        {ta._WRITER: [ta._says("a first draft")], ta._REVIEWER: [ta._says("looks good")]},
    )
    return [(None, project, run_id)]


async def conditional_route(root):
    project = ta._project(
        root,
        ta._conditional_writer(),
        name="cond",
        edge_prompts={ta._DRAFT_EDGE: ta._HANDOFF_DECISION, ta._FINAL_EDGE: ta._ACCEPT_DECISION},
    )
    run_id = "2026-10-23-a003"
    await _run(
        project,
        run_id,
        {
            ta._WRITER: [ta._says(ta._DRAFT), ta._routes_to(ta._REVIEWER)],
            ta._REVIEWER: [ta._says("reviewed, ship it")],
        },
    )
    return [(None, project, run_id)]


def _ping_pong_graph():
    """START -> writer; writer -?-> reviewer | END; reviewer -> writer."""
    graph = ta._graph(
        {ta._WRITER: NodeMode.CONDITIONAL, ta._REVIEWER: NodeMode.UNCONDITIONAL},
        [
            (ta._TASK_EDGE, ta._START_ID, ta._WRITER, ta._SLOT),
            (ta._DRAFT_EDGE, ta._WRITER, ta._REVIEWER, ta._SLOT),
            (ta._FINAL_EDGE, ta._WRITER, ta._END_ID, ta._SLOT),
            (LOOP_EDGE, ta._REVIEWER, ta._WRITER, ta._SLOT),
        ],
    )
    return graph, (ta._WRITER, ta._REVIEWER)


async def ping_pong_loop(root):
    project = ta._project(
        root,
        _ping_pong_graph(),
        name="loop",
        edge_prompts={ta._DRAFT_EDGE: ta._HANDOFF_DECISION, ta._FINAL_EDGE: ta._ACCEPT_DECISION},
    )
    run_id = "2026-10-23-a004"
    rounds = 3
    writer = []
    for n in range(1, rounds + 1):
        route = ta._routes_to(ta._REVIEWER)
        # The second hand-off needed a routing retry, so one route_fallback call appears.
        writer += [ta._says(f"draft v{n}"), _with_fallback(route) if n == 2 else route]
    writer += [ta._says(f"draft v{rounds + 1}, final"), ta._routes_to(ta._END_ID)]
    reviewer = [ta._says(f"review {n}: tighten it") for n in range(1, rounds + 1)]
    await _run(project, run_id, {ta._WRITER: writer, ta._REVIEWER: reviewer})
    return [(None, project, run_id)]


class _ScriptedSandbox:
    """A SandboxBackend answering each command from a fixed table."""

    ANSWERS = {
        "ls -la": ExecResult(exit_code=0, stdout="total 0\n-rw-r--r-- 1 u u 0 notes.md\n", stderr=""),
        "cat missing.txt": ExecResult(
            exit_code=1, stdout="", stderr="cat: missing.txt: No such file or directory\n"
        ),
    }

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    async def start(self) -> None:
        pass

    async def run(self, command: str, timeout_s: float | None = None) -> ExecResult:
        return self.ANSWERS[command]

    async def stop(self) -> None:
        pass


def _bash(command: str, call_id: str) -> ModelResponse:
    return ModelResponse(
        text="",
        model=ta._MODEL,
        tool_calls=[ToolCall(name="bash", args={"command": command}, id=call_id)],
    )


async def bash_tool(root):
    project = ta._project(root, ta._one_agent(tools={ta._WRITER: ["bash"]}), name="bash")
    run_id = "2026-10-23-a005"
    await _run(
        project,
        run_id,
        {
            ta._WRITER: [
                _bash("ls -la", "c1"),
                _bash("cat missing.txt", "c2"),
                ta._says("the workspace holds notes.md; missing.txt does not exist"),
            ]
        },
        sandbox_for=_ScriptedSandbox,
    )
    return [(None, project, run_id)]


async def paused_resumed(root):
    """writer -> reviewer -> END, paused while the reviewer's call is in flight, then
    resumed: the reviewer's activation appears twice and a `run_resumed` is written."""
    project = ta._project(root, ta._two_agents(), name="resume")
    run_id = "2026-10-23-a006"
    reviewer = _Held([ta._says("the review")], held_call=1)
    scripts = {ta._WRITER: [ta._says("the draft")], ta._REVIEWER: reviewer}

    halted = RunHalted()
    task = asyncio.create_task(
        run_topology(
            project, run_id, provider_for=_providers(scripts), sandbox_for=_no_sandbox, halted=halted
        )
    )
    await reviewer.entered.wait()
    halted.pause()
    task.cancel()
    paused = await task
    assert paused.status is RunStatus.PAUSED, paused

    # Release the held call so the resumed activation is answered from the script.
    reviewer._held_call = None
    reviewer._fake = FakeProvider([_metered(ta._says("the review"))])
    continued = await resumption(project.layout, run_id)
    await _run(project, run_id, scripts, continuation=continued)
    return [(None, project, run_id)]


async def forked(root):
    """A completed writer -> reviewer run, then a fork from the reviewer's activation."""
    project = ta._project(root, ta._two_agents(), name="fork")
    source_id = "2026-10-23-a007"
    fork_id = "2026-10-23-f007"
    await _run(
        project,
        source_id,
        {ta._WRITER: [ta._says("the draft")], ta._REVIEWER: [ta._says("the review")]},
    )
    (activation,) = _activations(project, source_id, ta._REVIEWER)
    point = await fork_point(project.layout, source_id, activation.uuid)
    await _run(
        project,
        fork_id,
        {ta._WRITER: [], ta._REVIEWER: [ta._says("a stricter review")]},
        continuation=point,
    )
    return [("fork_source", project, source_id), ("forked", project, fork_id)]


SCENARIOS = {
    "one_agent_write": one_agent_write,
    "two_agents": two_agents,
    "conditional_route": conditional_route,
    "ping_pong_loop": ping_pong_loop,
    "bash_tool": bash_tool,
    "paused_resumed": paused_resumed,
    "forked": forked,
}


async def main() -> None:
    work = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path(tempfile.mkdtemp())
    for name, scenario in SCENARIOS.items():
        root = work / name
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        for out_name, project, run_id in await scenario(root):
            out = HERE / f"{out_name or name}.run.jsonl"
            shutil.copyfile(project.layout.run_log_path(run_id), out)
            types = Counter(
                RECORD_ADAPTER.validate_json(line).type.value for line in _log_lines(project, run_id)
            )
            print(f"{out.name}: {sum(types.values())} lines {dict(types)}")


if __name__ == "__main__":
    asyncio.run(main())
