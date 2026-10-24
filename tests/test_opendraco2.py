"""
Tests for OpenDraco2 run logs (graph_construction/opendraco2_runs) and the
``agent=opendraco2`` path of POST /api/analyze.

Two kinds of input:

* real logs in ``tests/fixtures/opendraco2/*.run.jsonl``, written by OpenDraco's own
  executor with a scripted model (``tests/fixtures/opendraco2/generate_fixtures.py``);
* small hand-written logs built with ``LogBuilder`` below, for the edge cases a
  scripted run does not produce on demand.

Golden JSON lives in ``tests/golden/opendraco2/``. Regenerate after an intended
output change with ``GRAPHECTORY_UPDATE_GOLDEN=1``.

Run from the repository root:

    python -m unittest discover -s tests
"""

import base64
import http.client
import json
import os
import re
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "graph_construction"))
os.chdir(ROOT)  # tool configs and assets resolve relative to the repo root

import live_graph_server  # noqa: E402
from opendraco2_runs.analyze import build, to_json  # noqa: E402
from opendraco2_runs.log import looks_like_opendraco2, parse_run_log  # noqa: E402
from server.handler import GraphHandler  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "opendraco2"
GOLDEN = ROOT / "tests" / "golden" / "opendraco2"
UPDATE_GOLDEN = os.environ.get("GRAPHECTORY_UPDATE_GOLDEN") == "1"
PARSER = live_graph_server._setup_cmd_parser()

HOSTILE = '<script>alert("pwned")</script><img src=x onerror=alert(1)>&amp;'


def fixture(name: str) -> bytes:
    return (FIXTURES / f"{name}.run.jsonl").read_bytes()


# ── Hand-written logs ────────────────────────────────────────────────────────

class LogBuilder:
    """Records with OpenDraco's envelope, in write (global_seq) order."""

    def __init__(self, run_id: str, agents, edges, *, conditional=(), forked_from=None):
        self.run_id, self.seq, self.lines, self.uuids = run_id, 0, [], 0
        nodes = [{"id": "start", "kind": "start"}, {"id": "end", "kind": "end"}]
        nodes += [{"id": a, "kind": "agent", "mode": "conditional" if a in conditional else "unconditional"}
                  for a in agents]
        self.started = self.rec("run_started", None, topology={
            "nodes": nodes,
            "edges": [{"id": e, "source": s, "target": t, "slot": "task"} for e, s, t in edges],
        }, agents={a: {"id": a, "model": "fake-model"} for a in agents},
            topology_hash="h", framework_version="0.test", forked_from=forked_from)

    def _uuid(self, kind: str) -> str:
        self.uuids += 1
        return f"{kind}-{self.uuids:03d}"

    def rec(self, rtype, parent, *, agent=None, turn=None, uuid=None, **fields) -> str:
        self.seq += 1
        record = {"type": rtype, "uuid": uuid or self._uuid(rtype.split("_")[0]), "parent_uuid": parent,
                  "run_id": self.run_id, "global_seq": self.seq, "agent_id": agent, "local_turn": turn,
                  "timestamp": f"2026-10-23T10:00:{self.seq:02d}+00:00", **fields}
        self.lines.append(json.dumps(record))
        return record["uuid"]

    def deliver(self, frm, to, edge, value, parent=None) -> str:
        return self.rec("delivery", parent or self.started, from_agent=frm, to_agent=to, edge_id=edge,
                        slot="task", policy="append", value=value)

    def activate(self, agent, turn, superstep, caused_by) -> "Turn":
        uuid = self.rec("activation_started", None, agent=agent, turn=turn, caused_by=caused_by,
                        superstep=superstep)
        return Turn(self, uuid, agent, turn)

    def body(self, *, finished="completed", tail: str = "") -> bytes:
        lines = list(self.lines)
        if finished:
            self.seq += 1
            lines.append(json.dumps({"type": "run_finished", "uuid": "run-end", "parent_uuid": self.started,
                                     "run_id": self.run_id, "global_seq": self.seq, "agent_id": None,
                                     "local_turn": None, "timestamp": "2026-10-23T10:59:00+00:00",
                                     "status": finished, "error_uuid": None}))
        return ("\n".join(lines) + "\n" + tail).encode()


class Turn:
    def __init__(self, log: LogBuilder, uuid, agent, turn):
        self.log, self.uuid, self.agent, self.turn = log, uuid, agent, turn

    def rec(self, rtype, **fields):
        return self.log.rec(rtype, self.uuid, agent=self.agent, turn=self.turn, **fields)

    def say(self, text, kind="message"):
        for i in range(0, len(text), 4):  # OpenDraco streams 4-character chunks
            self.rec(kind, content=text[i:i + 4])

    def model(self, purpose="work", tokens=(100, 20)):
        self.rec("model_call", purpose=purpose, started_at="2026-10-23T10:00:00+00:00",
                 ended_at="2026-10-23T10:00:01+00:00",
                 usage={"input_tokens": tokens[0], "output_tokens": tokens[1], "total_tokens": sum(tokens)})

    def call(self, name, args, call_id="call_0"):
        self.rec("tool_call", call={"name": name, "args": args, "id": call_id})

    def result(self, name, text, call_id="call_0"):
        self.rec("tool_result", call_id=call_id, name=name, result=text)

    def commit(self, *paths):
        self.rec("workspace_commit", activation_uuid=self.uuid, sha="abc123", changed_paths=list(paths))

    def finish(self, tokens=120, duration=1.5):
        self.rec("activation_finished", usage={"input_tokens": tokens - 20, "output_tokens": 20,
                                                "total_tokens": tokens}, duration_s=duration, stop_reason="stop")

    def hand_off(self, to, edge, value):
        return self.log.deliver(self.agent, to, edge, value, parent=self.uuid)


def interleaved_parallel() -> bytes:
    """START feeds a and b in one superstep; their records interleave, both use call_0."""
    log = LogBuilder("hand-parallel", ["a", "b"], [("e1", "start", "a"), ("e2", "start", "b"),
                                                   ("e3", "a", "end"), ("e4", "b", "end")])
    da, db = log.deliver("start", "a", "e1", "task for a"), log.deliver("start", "b", "e2", "task for b")
    a, b = log.activate("a", 1, 1, [da]), log.activate("b", 1, 1, [db])
    a.model(); b.model()
    a.call("bash", {"command": "ls src"}); b.call("read", {"path": "README.md"})
    b.result("read", "B-README-CONTENT"); a.result("bash", "A-LS-OUTPUT")
    a.say("a done"); b.say("b done")
    b.model("tool_turn"); a.model("tool_turn")
    a.finish(); b.finish()
    a.hand_off("end", "e3", "a done"); b.hand_off("end", "e4", "b done")
    return log.body()


def repeated_call_ids() -> bytes:
    """One agent, two turns, every tool call named call_0 by the model."""
    log = LogBuilder("hand-callids", ["w"], [("e1", "start", "w"), ("e2", "w", "w"), ("e3", "w", "end")])
    d1 = log.deliver("start", "w", "e1", "go")
    t1 = log.activate("w", 1, 1, [d1])
    t1.model(); t1.call("bash", {"command": "ls"}); t1.result("bash", "TURN1-LS")
    t1.model("tool_turn"); t1.call("read", {"path": "a.py"}); t1.result("read", "TURN1-READ")
    t1.say("again"); t1.model("tool_turn"); t1.finish()
    d2 = t1.hand_off("w", "e2", "again")
    t2 = log.activate("w", 2, 2, [d1, d2])  # append policy: caused_by keeps old deliveries
    t2.model(); t2.call("read", {"path": "b.py"}); t2.result("read", "TURN2-READ")
    t2.say("done"); t2.model("tool_turn"); t2.finish()
    t2.hand_off("end", "e3", "done")
    return log.body()


def writer_tester(web: bool = False) -> bytes:
    """writer edits src/app.py (superstep 1); tester runs the tests (superstep 2)."""
    log = LogBuilder("hand-tester", ["writer", "tester"],
                     [("e1", "start", "writer"), ("e2", "writer", "tester"), ("e3", "tester", "end")])
    d1 = log.deliver("start", "writer", "e1", "fix the bug")
    w = log.activate("writer", 1, 1, [d1])
    if web:
        w.model(); w.call("web_search", {"query": "python off by one"}); w.result("web_search", "results")
    w.model("work" if not web else "tool_turn")
    w.call("write", {"path": "src/app.py", "content": "print(1)"}, "call_1")
    w.result("write", "wrote 8 bytes", "call_1")
    w.say("fixed"); w.model("tool_turn"); w.finish(); w.commit("src/app.py")
    d2 = w.hand_off("tester", "e2", "fixed")
    t = log.activate("tester", 1, 2, [d2])
    t.model(); t.call("bash", {"command": "python -m pytest tests"}); t.result("bash", "1 passed")
    t.say("tests pass"); t.model("tool_turn"); t.finish()
    t.hand_off("end", "e3", "tests pass")
    return log.body()


def thinking_only() -> bytes:
    log = LogBuilder("hand-think", ["solo"], [("e1", "start", "solo"), ("e2", "solo", "end")])
    d = log.deliver("start", "solo", "e1", "question")
    s = log.activate("solo", 1, 1, [d])
    s.say("let me think about it", kind="thinking"); s.say("the answer is 42"); s.model(); s.finish()
    s.hand_off("end", "e2", "the answer is 42")
    return log.body()


def hostile() -> bytes:
    """Model-controlled text everywhere, including an agent id."""
    agent = "ev<i>l"
    log = LogBuilder("hand-hostile", [agent], [("e1", "start", agent), ("e2", agent, "end")])
    d = log.deliver("start", agent, "e1", "delivered " + HOSTILE)
    t = log.activate(agent, 1, 1, [d])
    t.rec("prompt_rendered", system="sys " + HOSTILE, specialize=HOSTILE, input=HOSTILE, overflow=None)
    t.say("thinking " + HOSTILE, kind="thinking")
    t.model(); t.call("bash", {"command": "echo '" + HOSTILE + "'"})
    t.result("bash", "output " + HOSTILE)
    t.say("answer " + HOSTILE); t.model("tool_turn"); t.finish()
    t.hand_off("end", "e2", "answer " + HOSTILE)
    return log.body()


def in_progress_half_line() -> bytes:
    """A real two-agent log with run_finished gone and the last line cut mid-write."""
    lines = fixture("two_agents").decode().splitlines()
    lines = [l for l in lines if '"run_finished"' not in l]
    return ("\n".join(lines[:-1]) + "\n" + lines[-1][: len(lines[-1]) // 2]).encode()


HAND = {
    "interleaved_parallel": interleaved_parallel,
    "repeated_call_ids": repeated_call_ids,
    "writer_tester": writer_tester,
    "thinking_only": thinking_only,
    "in_progress_half_line": in_progress_half_line,
}


def observations(data: dict, agent: str) -> dict:
    """{node label + args summary: observation text} for one agent's graph."""
    out = {}
    for node in data["agents"][agent]["graph"]["nodes"]:
        for visit in node.get("step_data") or []:
            text = visit.get("observation")
            if text:
                # The builder records a step's observation as "<tool>: <output>".
                out.setdefault(node["label"], []).append(re.sub(r"\A[\w.-]+: ", "", text))
    return out


def phases_of(data: dict, agent: str) -> dict:
    return {n["label"]: n["phases"] for n in data["agents"][agent]["graph"]["nodes"]}


def analyze_json(body: bytes, **kw) -> dict:
    run, graphs, rg = build(body, PARSER, **kw)
    return json.loads(json.dumps(to_json(run, graphs, rg), default=str))


# ── Unit-level tests ─────────────────────────────────────────────────────────

class RealLogTest(unittest.TestCase):

    def test_every_generated_fixture_parses(self):
        names = sorted(p.name.split(".")[0] for p in FIXTURES.glob("*.run.jsonl"))
        self.assertGreaterEqual(len(names), 8)
        for name in names:
            with self.subTest(name):
                self.assertTrue(looks_like_opendraco2(fixture(name)))
                data = analyze_json(fixture(name))
                self.assertEqual(data["status"], "completed")
                self.assertEqual(data["agent_type"], "opendraco2")

    def test_message_chunks_are_joined(self):
        data = analyze_json(fixture("two_agents"))
        answers = [n["args"]["text"] for n in data["agents"]["writer"]["graph"]["nodes"]
                   if n["label"] == "final_answer"]
        self.assertEqual(answers, ["a first draft"])

    def test_delivery_is_context_for_next_step(self):
        data = analyze_json(fixture("two_agents"))
        first = data["agents"]["reviewer"]["graph"]["nodes"][0]["step_data"][0]["thought"]
        self.assertTrue(first.startswith("from writer: a first draft"), first)

    def test_bash_exit_code_drives_outcome(self):
        data = analyze_json(fixture("bash_tool"))
        nodes = {n["label"]: n for n in data["agents"]["writer"]["graph"]["nodes"]}
        self.assertEqual(nodes["cat"].get("command_outcome"), "failure")
        self.assertNotEqual(nodes["ls"].get("command_outcome"), "failure")
        self.assertEqual(nodes["ls"]["phases"], ["localization"])

    def test_write_is_patch_and_commit_marks_workspace(self):
        data = analyze_json(fixture("one_agent_write"))
        self.assertIn("patch", phases_of(data, "writer")["write"])
        writer = next(n for n in data["run_graph"]["nodes"] if n["id"] == "writer")
        self.assertTrue(writer["changed_workspace"])
        self.assertEqual(writer["commits"], 1)

    def test_resumed_run_shows_rerun_and_marks_abandoned(self):
        data = analyze_json(fixture("paused_resumed"))
        blocks = data["run_graph"]["graph"]["timeline"]["activations"]
        reviewer = [b for b in blocks if b["agent_id"] == "reviewer"]
        self.assertEqual([b["status"] for b in reviewer], ["abandoned", "finished"])
        self.assertEqual(reviewer[1]["rerun_of"], reviewer[0]["uuid"])
        writer = next(b for b in blocks if b["agent_id"] == "writer")
        self.assertEqual(writer["status"], "finished")
        node = next(n for n in data["run_graph"]["nodes"] if n["id"] == "reviewer")
        self.assertEqual((node["activations"], node["abandoned_activations"]), (1, 1))

    def test_fork_causes_come_from_parent_run(self):
        data = analyze_json(fixture("forked"))
        thought = data["agents"]["reviewer"]["graph"]["nodes"][0]["step_data"][0]["thought"]
        self.assertTrue(thought.startswith("from the parent run"), thought)
        self.assertIsNotNone(data["run_graph"]["graph"]["forked_from"])
        block = data["run_graph"]["graph"]["timeline"]["activations"][0]
        self.assertGreater(block["from_parent_run"], 0)

    def test_ping_pong_loop_patterns(self):
        data = analyze_json(fixture("ping_pong_loop"))
        kinds = {p["kind"] for p in data["patterns"]}
        self.assertTrue({"ping_pong", "router_churn"} <= kinds, kinds)
        pp = next(p for p in data["patterns"] if p["kind"] == "ping_pong")
        self.assertEqual(pp["agents"], ["reviewer", "writer"])
        route = [e for e in data["run_graph"]["edges"] if e["type"] == "route" and e["choice"] == "reviewer"]
        self.assertEqual((route[0]["count"], route[0]["of"]), (3, 4))

    def test_two_agent_run_detects_nothing(self):
        self.assertEqual(analyze_json(fixture("two_agents"))["patterns"], [])


class HandLogTest(unittest.TestCase):

    def test_interleaved_parallel_agents_keep_their_own_records(self):
        data = analyze_json(interleaved_parallel())
        self.assertEqual(observations(data, "a"), {"ls": ["A-LS-OUTPUT"]})
        self.assertEqual(observations(data, "b"), {"read": ["B-README-CONTENT"]})

    def test_repeated_call_0_pairs_within_activation(self):
        obs = observations(analyze_json(repeated_call_ids()), "w")
        self.assertEqual(obs["ls"], ["TURN1-LS"])
        self.assertEqual(sorted(obs["read"]), ["TURN1-READ", "TURN2-READ"])

    def test_append_policy_shows_only_new_deliveries(self):
        data = analyze_json(repeated_call_ids())
        thoughts = [v["thought"] for n in data["agents"]["w"]["graph"]["nodes"] for v in n["step_data"]]
        self.assertEqual(sum(t.count("from start: go") for t in thoughts), 1)
        self.assertEqual(sum(t.count("from w: again") for t in thoughts), 1)

    def test_in_progress_with_half_written_line(self):
        body = in_progress_half_line()
        run = parse_run_log(body)
        self.assertTrue(run.truncated)
        data = analyze_json(body)
        self.assertEqual(data["status"], "in_progress")
        self.assertTrue(data["run_graph"]["graph"]["truncated"])
        self.assertIn("writer", data["agents"])

    def test_agent_that_never_calls_a_tool(self):
        data = analyze_json(thinking_only())
        nodes = data["agents"]["solo"]["graph"]["nodes"]
        self.assertEqual([n["label"] for n in nodes], ["final_answer"])
        self.assertEqual(nodes[0]["args"]["text"], "the answer is 42")
        self.assertIn("let me think about it", nodes[0]["step_data"][0]["thought"])

    def test_tester_validates_another_agents_patch(self):
        self.assertEqual(phases_of(analyze_json(writer_tester()), "tester")["python"], ["validation"])
        strict = analyze_json(writer_tester(), run_phases=False)
        self.assertEqual(strict["phase_scope"], "agent")
        self.assertEqual(phases_of(strict, "tester")["python"], ["localization"])

    def test_web_search_is_localization(self):
        self.assertEqual(phases_of(analyze_json(writer_tester(web=True)), "writer")["web_search"],
                         ["localization"])

    def test_agent_id_returns_one_agent_in_single_agent_shape(self):
        run, graphs, rg = build(writer_tester(), PARSER, agent_id="tester")
        data = to_json(run, graphs, rg)
        self.assertIsNone(rg)
        self.assertEqual((data["instance_id"], data["agent_type"]), ("tester", "opendraco2"))
        self.assertIn("graph", data)


class GoldenTest(unittest.TestCase):
    """The JSON output for fixed inputs, byte for byte."""

    CASES = {**{f"real_{n}": (lambda n=n: fixture(n)) for n in ("two_agents", "ping_pong_loop", "bash_tool",
                                                                 "paused_resumed", "forked")},
             **{f"hand_{n}": f for n, f in HAND.items()}}

    def test_golden_json(self):
        GOLDEN.mkdir(parents=True, exist_ok=True)
        for name, make in self.CASES.items():
            with self.subTest(name):
                got = json.dumps(analyze_json(make()), indent=1, sort_keys=True, ensure_ascii=False) + "\n"
                path = GOLDEN / f"{name}.json"
                if UPDATE_GOLDEN or not path.exists():
                    path.write_text(got, encoding="utf-8")
                    if not UPDATE_GOLDEN:
                        self.fail(f"golden file {path.name} was missing and has been written; re-run")
                self.assertEqual(got, path.read_text(encoding="utf-8"),
                                 f"{name} differs from {path}; GRAPHECTORY_UPDATE_GOLDEN=1 to accept")


# ── Through the HTTP server ──────────────────────────────────────────────────

class OpenDracoApiTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        GraphHandler.cmd_parser = PARSER
        GraphHandler.assets_dir = ROOT / "graph_construction"
        GraphHandler.log_message = lambda *_args: None
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), GraphHandler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def post(self, body, query="", content_type="application/x-ndjson"):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        conn.request("POST", f"/api/analyze{query}", body=body, headers={"Content-Type": content_type})
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, resp.getheader("Content-Type"), data

    def test_auto_detected_json_schema(self):
        status, ctype, data = self.post(fixture("two_agents"), "?format=json")
        self.assertEqual(status, 200, data[:300])
        self.assertTrue(ctype.startswith("application/json"))
        result = json.loads(data)
        self.assertTrue({"run_id", "agent_type", "status", "agents", "run_graph", "patterns"} <= set(result))
        self.assertEqual(set(result["agents"]), {"writer", "reviewer"})
        self.assertTrue({"nodes", "edges", "directed", "multigraph"} <= set(result["run_graph"]))

    def test_html_page_has_run_view_and_one_tab_per_agent(self):
        status, ctype, data = self.post(fixture("ping_pong_loop"), "?agent=opendraco2")
        self.assertEqual(status, 200)
        page = data.decode()
        self.assertIn('id="run-graph"', page)
        self.assertIn('id="timeline"', page)
        pages = json.loads(re.search(r'<script type="application/json" id="agent-pages">(.*?)</script>',
                                     page, re.S).group(1))
        self.assertEqual(set(pages), {"writer", "reviewer"})
        self.assertNotIn('src="/static/', page)  # standalone: dagre is inline
        self.assertIn("dagre", page)

    def test_agent_id_page_is_standalone(self):
        status, _, data = self.post(fixture("two_agents"), "?agent=opendraco2&agent_id=reviewer")
        self.assertEqual(status, 200)
        self.assertNotIn(b'src="/static/', data)

    def test_unknown_agent_id_is_400(self):
        status, _, data = self.post(fixture("two_agents"), "?agent_id=nobody")
        self.assertEqual(status, 400)
        self.assertIn(b"nobody", data)

    def test_not_a_run_log_is_400(self):
        status, _, _ = self.post(b'{"type": "something"}\n', "?agent=opendraco2")
        self.assertEqual(status, 400)

    def test_wrong_content_type_is_415(self):
        status, _, _ = self.post(fixture("two_agents"), content_type="text/plain")
        self.assertEqual(status, 415)

    def test_bad_phases_param_is_400(self):
        status, _, _ = self.post(fixture("two_agents"), "?phases=everything")
        self.assertEqual(status, 400)

    def test_html_contains_no_unescaped_run_text(self):
        body = hostile()
        status, _, data = self.post(body)
        self.assertEqual(status, 200)
        page = data.decode()
        texts = set()
        for line in body.decode().splitlines():
            _collect_strings(json.loads(line), texts)
        # Message/thinking records are 4-character stream chunks; a chunk such as
        # ")</s" can coincide with the page's own markup without being run text,
        # so check every string long enough to carry markup, plus the joined payload.
        risky = [t for t in texts if any(c in t for c in "<>&") and len(t) >= 10]
        self.assertTrue(risky)
        for text in risky + [HOSTILE, "answer " + HOSTILE, "thinking " + HOSTILE]:
            self.assertNotIn(text, page)
        self.assertNotIn("<script>alert", page)
        self.assertNotIn("<img src=x", page)
        # Inside each agent tab the upstream renderer must escape it too.
        pages = json.loads(re.search(r'<script type="application/json" id="agent-pages">(.*?)</script>',
                                     page, re.S).group(1))
        for b64 in pages.values():
            inner = base64.b64decode(b64).decode()
            self.assertNotIn("<script>alert", inner)
            self.assertNotIn("<img src=x", inner)


def _collect_strings(value, out: set):
    if isinstance(value, str):
        out.add(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            _collect_strings(k, out)
            _collect_strings(v, out)
    elif isinstance(value, list):
        for v in value:
            _collect_strings(v, out)


if __name__ == "__main__":
    unittest.main()
