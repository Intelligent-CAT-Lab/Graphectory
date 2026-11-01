"""
Tests for the public OpenDraco's inference event logs (graph_construction/
opendraco_runs) and the ``agent=opendraco`` path of POST /api/analyze.

Logs are built with ``EventLog`` below, mirroring what OpenDraco's
``api/routers/inference.py`` emits (github.com/OpenDraco/OpenDraco @ 7bc7f84):
the same event types, fields and order, with ``args_preview`` values cut to 40
characters and ``result_preview`` as ``ok=…`` / ``N items`` / ≤80 characters.
``tests/fixtures/opendraco/two_instances.inference.ndjson`` is ``two_instances()``
written out, for curl examples and the deploy check.

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
os.chdir(ROOT)

import live_graph_server  # noqa: E402
from opendraco2_runs.analyze import OpenDracoError  # noqa: E402
from opendraco_runs.analyze import analyze  # noqa: E402
from opendraco_runs.events import looks_like_opendraco, parse_event_log  # noqa: E402
from opendraco_runs.tools import parse_args_preview, tool_kind  # noqa: E402
from server.handler import GraphHandler  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "opendraco" / "two_instances.inference.ndjson"
PARSER = live_graph_server._setup_cmd_parser()
HOSTILE = '<script>alert("pwned")</script><img src=x onerror=alert(1)>&amp;'
TS = "2026-10-24T09:00:00"


class EventLog:
    """Events in the order inference.py's worker emits them."""

    def __init__(self, instance_ids, run_id="20261024-0900-swebench"):
        self.events = [{"type": "run_meta", "run_id": run_id, "config_name": "locator-coder-tester",
                        "config_sha": "abc1234", "instance_ids": list(instance_ids), "ts": TS}]

    def e(self, etype, **fields):
        self.events.append({"type": etype, **fields})
        return self

    def instance(self, iid, index, total):
        self.e("instance_start", instance_id=iid, index=index, total=total)
        return self.e("status", message=f"Cloning workspace for {iid}…")

    def think(self, agent, text):
        for i in range(0, len(text), 6):
            self.e("thinking_chunk", agent=agent, chunk=text[i:i + 6])
        return self

    def respond(self, agent, text):
        return self.e("response", agent=agent, content=text)

    def tool(self, agent, name, args_preview, result_preview):
        return self.e("tool_call", agent=agent, tool=name, args_preview=args_preview,
                      result_preview=result_preview)

    def finish(self, agent, inputs, delta, successors=()):
        self.e("agent_input", agent=agent, inputs=inputs)
        self.e("agent_event", agent=agent, delta=delta)
        for target in successors:
            self.e("handoff", **{"from": agent, "to": target, "keys": [agent], "summary": "1 slot",
                                 "preview": str(delta.get(agent, ""))[:120], "timestamp": TS})
        return self

    def body(self, tail=""):
        return ("\n".join(json.dumps(e) for e in self.events) + "\n" + tail).encode()


def first_instance(log: EventLog, *, hostile=False, finish=True) -> EventLog:
    iid = "django__django-10973"
    extra = (" " + HOSTILE) if hostile else ""
    log.instance(iid, 0, 2)
    log.think("locator", "The issue is about psql env passing." + extra)
    log.respond("locator", "")
    log.tool("locator", "search_code", "query=PGPASSWORD subprocess, directory=…/django/db", "5 items")
    log.respond("locator", "")
    log.tool("locator", "read_file", "path=…/postgresql/client.py", "import os\nimport signal" + extra)
    log.respond("locator", "<files>django/db/backends/postgresql/client.py</files>")
    log.finish("locator", {"issue": "Use subprocess.run and PGPASSWORD for client in postgres backend"},
               {"locator": "<files>django/db/backends/postgresql/client.py</files>"}, ["coder"])
    log.think("coder", "Replace check_call with run and env.")
    log.respond("coder", "I'll edit the client.")
    log.tool("coder", "write_file", "path=django/db/backends/postgresql/client.py, content=import os\nimport signal\nimport subproc",
             "ok=True")
    log.respond("coder", "")
    log.tool("coder", "execute_command", "command=python -m pytest tests/dbshell -q", "Error: 1 failed, 3 passed")
    log.respond("coder", "Patched client.py; one dbshell test still fails." + extra)
    log.finish("coder", {"locator": "<files>…</files>"}, {"coder": "diff --git a/django/db/…"}, ["tester"])
    log.respond("tester", "")
    log.tool("tester", "run_tests", "test_files=['tests/dbshell/test_postgresql.py']", "ok=True")
    log.respond("tester", "All selected tests pass.")
    log.finish("tester", {"coder": "diff --git …"}, {"tester": "All selected tests pass."}, ["__end__"])
    for agent, (i, o) in {"locator": (900, 120), "coder": (1500, 400), "tester": (600, 80)}.items():
        log.e("agent_tokens", agent=agent, input=i, output=o, total=i + o)
    if finish:
        log.e("instance_done", instance_id=iid, index=0, total=2, output_path="results/predictions/x.jsonl",
              run_id="20261024-0900-swebench",
              patch="diff --git a/django/db/backends/postgresql/client.py b/django/db/backends/postgresql/client.py\n")
    return log


def second_instance(log: EventLog) -> EventLog:
    """coder ↔ reviewer bounce with no writes: a ping-pong."""
    iid = "astropy__astropy-12907"
    log.instance(iid, 1, 2)
    log.respond("coder", "Draft 1")
    log.finish("coder", {"issue": "separability_matrix nested models"}, {"coder": "Draft 1"}, ["reviewer"])
    for n in range(1, 4):
        log.respond("reviewer", f"Needs work ({n})")
        log.finish("reviewer", {"coder": "Draft"}, {"reviewer": f"Needs work ({n})"}, ["coder"])
        log.respond("coder", f"Draft {n + 1}")
        log.finish("coder", {"issue": "…"}, {"coder": f"Draft {n + 1}"}, ["reviewer" if n < 3 else "__end__"])
    log.e("instance_done", instance_id=iid, index=1, total=2, output_path="results/predictions/x.jsonl",
          run_id="20261024-0900-swebench", patch="")
    return log


def two_instances() -> bytes:
    log = EventLog(["django__django-10973", "astropy__astropy-12907"])
    first_instance(log)
    second_instance(log)
    log.e("done", total=2)
    return log.body()


def run_json(body: bytes, **kw) -> dict:
    _, payload = analyze(body, PARSER, ROOT / "graph_construction", output="json", **kw)
    return json.loads(payload)


def phases(data: dict, agent: str) -> dict:
    return {n["label"]: n["phases"] for n in data["agents"][agent]["graph"]["nodes"]}


class ToolNameTest(unittest.TestCase):

    def test_kinds(self):
        cases = {"read_file": "read", "write_file": "write", "apply_patch": "write", "search_code": "search",
                 "run_tests": "test", "execute_command": "shell", "CmdRunTool": "shell", "readFileTool": "read",
                 "StrReplaceEditorTool": "write", "GrepTool": "search", "_ast_search": "search",
                 "webSearchTool": "search", "BrowserTool": "localization", "TaskDoneTool": "general",
                 "ThinkTool": "general", "my_custom_thing": None}
        for name, kind in cases.items():
            with self.subTest(name):
                self.assertEqual(tool_kind(name), kind)

    def test_args_preview(self):
        self.assertEqual(parse_args_preview("path=…/postgresql/client.py, line=12"),
                         {"path": "…/postgresql/client.py", "line": "12"})
        self.assertEqual(parse_args_preview("query=a, b and c, directory=…/db"),
                         {"query": "a, b and c", "directory": "…/db"})
        self.assertEqual(parse_args_preview(""), {})


class EventLogTest(unittest.TestCase):

    def test_detection(self):
        self.assertTrue(looks_like_opendraco(two_instances()))
        self.assertFalse(looks_like_opendraco((ROOT / "tests/fixtures/opendraco2/two_agents.run.jsonl").read_bytes()))

    def test_fixture_file_matches_builder(self):
        self.assertEqual(FIXTURE.read_bytes(), two_instances())

    def test_first_instance_by_default(self):
        data = run_json(two_instances())
        self.assertEqual((data["agent_type"], data["status"]), ("opendraco", "completed"))
        self.assertEqual(data["instance_id"], "django__django-10973")
        self.assertEqual(data["instances"], ["django__django-10973", "astropy__astropy-12907"])
        self.assertEqual(list(data["agents"]), ["locator", "coder", "tester"])
        self.assertEqual(data["run_graph"]["graph"]["final_patch_paths"],
                         ["django/db/backends/postgresql/client.py"])

    def test_phases_follow_tool_kinds_and_shared_history(self):
        data = run_json(two_instances())
        self.assertEqual(phases(data, "locator")["search_code"], ["localization"])
        self.assertEqual(phases(data, "locator")["read_file"], ["localization"])
        self.assertEqual(phases(data, "coder")["write_file"], ["patch"])
        # A shell tool becomes a parsed bash command; after the coder's write it validates.
        self.assertEqual(phases(data, "coder")["python"], ["validation"])
        # The tester's run comes after the coder's (inferred) commit in an earlier superstep.
        self.assertEqual(phases(data, "tester")["run_tests"], ["validation"])
        strict = run_json(two_instances(), run_phases=False)
        self.assertEqual(phases(strict, "tester")["run_tests"], ["localization"])

    def test_failures_from_result_previews(self):
        data = run_json(two_instances())
        node = next(n for n in data["agents"]["coder"]["graph"]["nodes"] if n["label"] == "python")
        self.assertEqual(node.get("command_outcome"), "failure")

    def test_inferred_commit_and_tokens(self):
        data = run_json(two_instances())
        coder = next(n for n in data["run_graph"]["nodes"] if n["id"] == "coder")
        self.assertTrue(coder["changed_workspace"])
        self.assertEqual(coder["changed_paths"], ["django/db/backends/postgresql/client.py"])
        self.assertEqual(coder["tokens"]["total_tokens"], 1900)
        self.assertEqual(data["run_graph"]["graph"]["derived"]["workspace_commits"],
                         "inferred from write-kind tool calls")

    def test_handoffs_drive_supersteps_and_context(self):
        data = run_json(two_instances())
        blocks = data["run_graph"]["graph"]["timeline"]["activations"]
        self.assertEqual([(b["agent_id"], b["superstep"]) for b in blocks],
                         [("locator", 1), ("coder", 2), ("tester", 3)])
        first = data["agents"]["coder"]["graph"]["nodes"][0]["step_data"][0]["thought"]
        self.assertTrue(first.startswith("from locator: <files>"), first)
        locator = data["agents"]["locator"]["graph"]["nodes"][0]["step_data"][0]["thought"]
        self.assertTrue(locator.startswith("from start: Use subprocess.run"), locator)

    def test_second_instance_ping_pong(self):
        data = run_json(two_instances(), label="astropy__astropy-12907")
        self.assertEqual(data["instance_id"], "astropy__astropy-12907")
        kinds = {p["kind"] for p in data["patterns"]}
        self.assertIn("ping_pong", kinds)
        coder = next(n for n in data["run_graph"]["nodes"] if n["id"] == "coder")
        self.assertEqual(coder["activations"], 4)

    def test_unknown_instance_is_an_error(self):
        with self.assertRaises(OpenDracoError):
            run_json(two_instances(), label="nope__nope-1")

    def test_in_progress_instance(self):
        log = EventLog(["django__django-10973"])
        body = first_instance(log, finish=False).body(tail='{"type": "tool_call", "agent": "co')
        run, _ = parse_event_log(body)
        self.assertEqual(run.status, "in_progress")
        self.assertEqual(run_json(body)["status"], "in_progress")

    def test_agent_id(self):
        data = run_json(two_instances(), agent_id="coder")
        self.assertEqual((data["instance_id"], data["agent_type"]), ("coder", "opendraco"))


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

    def post(self, body, query=""):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        conn.request("POST", f"/api/analyze{query}", body=body,
                     headers={"Content-Type": "application/x-ndjson"})
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, data

    def test_auto_detected(self):
        status, data = self.post(two_instances(), "?format=json")
        self.assertEqual(status, 200, data[:300])
        self.assertEqual(json.loads(data)["agent_type"], "opendraco")

    def test_opendraco2_still_auto_detected(self):
        body = (ROOT / "tests/fixtures/opendraco2/two_agents.run.jsonl").read_bytes()
        status, data = self.post(body, "?format=json")
        self.assertEqual((status, json.loads(data)["agent_type"]), (200, "opendraco2"))

    def test_forcing_the_wrong_format_is_400(self):
        status, _ = self.post(two_instances(), "?agent=opendraco2")
        self.assertEqual(status, 400)

    def test_unknown_instance_is_400(self):
        status, data = self.post(two_instances(), "?id=nope__nope-1")
        self.assertEqual(status, 400)
        self.assertIn(b"django__django-10973", data)

    def test_html_page_and_escaping(self):
        log = EventLog(["django__django-10973"])
        body = first_instance(log, hostile=True).body()
        status, data = self.post(body, "?agent=opendraco")
        self.assertEqual(status, 200)
        page = data.decode()
        self.assertIn("OpenDraco run", page)
        self.assertNotIn("<script>alert", page)
        self.assertNotIn("<img src=x", page)
        pages = json.loads(re.search(r'<script type="application/json" id="agent-pages">(.*?)</script>',
                                     page, re.S).group(1))
        self.assertEqual(set(pages), {"locator", "coder", "tester"})
        for b64 in pages.values():
            inner = base64.b64decode(b64).decode()
            self.assertNotIn("<script>alert", inner)


if __name__ == "__main__":
    unittest.main()
