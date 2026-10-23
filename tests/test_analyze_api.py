"""
End-to-end tests for POST /api/analyze and the POST/static guards.

Run from the repository root (tool configs are resolved relative to it):

    python -m unittest discover -s tests
"""

import http.client
import json
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "graph_construction"))

import live_graph_server  # noqa: E402
from server import handler as handler_module  # noqa: E402
from server.handler import GraphHandler  # noqa: E402

SAMPLES = ROOT / "data" / "samples"
SA_TRAJ = next((SAMPLES / "SWE-agent").rglob("django__django-10973.traj"))
MSA_TRAJ = next((SAMPLES / "mini-swe-agent-v2.1").rglob("astropy__astropy-12907.traj.json"))
OH_JSONL = next((SAMPLES / "OpenHands").rglob("*.jsonl"))

CLAUDE_JSONL = "\n".join(json.dumps(r) for r in [
    {"type": "user", "sessionId": "claude-demo", "message": {"role": "user", "content": "List files"}},
    {"type": "assistant", "sessionId": "claude-demo", "message": {
        "role": "assistant", "model": "claude-test",
        "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls -la"}}]}},
    {"type": "user", "sessionId": "claude-demo", "message": {
        "role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a.py"}]}},
]).encode()

CODEX_JSONL = "\n".join(json.dumps(r) for r in [
    {"type": "session_meta", "payload": {"id": "codex-demo", "cwd": "/repo"}},
    {"type": "turn_context", "payload": {"model": "gpt-test"}},
    {"type": "response_item", "payload": {
        "type": "function_call", "name": "shell", "call_id": "c1",
        "arguments": json.dumps({"command": ["bash", "-lc", "cat a.py"]})}},
    {"type": "response_item", "payload": {
        "type": "function_call_output", "call_id": "c1", "output": "print(1)"}},
]).encode()


class AnalyzeApiTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._cwd = Path.cwd()
        import os
        os.chdir(ROOT)
        GraphHandler.cmd_parser = live_graph_server._setup_cmd_parser()
        GraphHandler.assets_dir = ROOT / "graph_construction"
        GraphHandler.log_message = lambda *_args: None
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), GraphHandler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        import os
        os.chdir(cls._cwd)

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, data

    def analyze(self, body, query="", content_type="application/json", **headers):
        return self.request("POST", f"/api/analyze{query}", body,
                            {"Content-Type": content_type, **headers})

    # ── Each supported format builds a graph ─────────────────────────────────

    def assert_graph(self, body, agent, query="", content_type="application/json"):
        sep = "&" if query else "?"
        status, data = self.analyze(body, f"{query}{sep}format=json" if query else "?format=json",
                                    content_type)
        self.assertEqual(status, 200, data[:300])
        result = json.loads(data)
        self.assertEqual(result["agent_type"], agent)
        self.assertGreater(result["nodes"], 0)
        return result

    def test_swe_agent_traj(self):
        self.assert_graph(SA_TRAJ.read_bytes(), "sa", "?id=django__django-10973")

    def test_mini_swe_agent_traj(self):
        self.assert_graph(MSA_TRAJ.read_bytes(), "msa")

    def test_openhands_jsonl_picks_requested_entry(self):
        first = json.loads(OH_JSONL.read_text().splitlines()[-1])["instance_id"]
        result = self.assert_graph(OH_JSONL.read_bytes(), "oh", f"?id={first}",
                                   "application/x-ndjson")
        self.assertEqual(result["instance_id"], first)

    def test_claude_code_transcript(self):
        self.assert_graph(CLAUDE_JSONL, "claude", content_type="application/x-ndjson")

    def test_codex_rollout(self):
        self.assert_graph(CODEX_JSONL, "codex", content_type="application/x-ndjson")

    def test_html_output_is_a_graph_page(self):
        status, data = self.analyze(SA_TRAJ.read_bytes())
        self.assertEqual(status, 200)
        self.assertIn(b"<html", data[:2000].lower())

    # ── Rejections ───────────────────────────────────────────────────────────

    def test_rejects_simple_cross_site_content_type(self):
        status, _ = self.analyze(SA_TRAJ.read_bytes(), content_type="text/plain")
        self.assertEqual(status, 415)

    def test_rejects_foreign_origin(self):
        status, _ = self.analyze(SA_TRAJ.read_bytes(), Origin="https://evil.example")
        self.assertEqual(status, 403)

    def test_rejects_oversized_body_before_reading_it(self):
        original = handler_module._UPLOAD_MAX_BYTES
        handler_module._UPLOAD_MAX_BYTES = 10
        try:
            status, _ = self.analyze(b'{"trajectory": []}')
        finally:
            handler_module._UPLOAD_MAX_BYTES = original
        self.assertEqual(status, 413)

    def test_rejects_unknown_json_shape(self):
        status, data = self.analyze(b'{"hello": "world"}')
        self.assertEqual(status, 400)
        self.assertIn(b"Unrecognised", data)

    def test_rejects_traversal_in_id(self):
        status, _ = self.analyze(SA_TRAJ.read_bytes(), "?id=../../etc")
        self.assertEqual(status, 400)

    def test_unbuildable_trajectory_is_422_not_500(self):
        status, _ = self.analyze(b'{"trajectory": "not-a-list"}', "?agent=sa")
        self.assertIn(status, {400, 422})

    def test_script_payload_is_escaped_not_executed(self):
        traj = json.loads(SA_TRAJ.read_bytes())
        traj["trajectory"][0]["thought"] = "<script>alert(1)</script>"
        status, data = self.analyze(json.dumps(traj).encode())
        self.assertEqual(status, 200)
        self.assertNotIn(b"<script>alert(1)</script>", data)

    def test_static_traversal_is_blocked(self):
        for path in ("/static//etc/hostname", "/static/../handler.py"):
            status, _ = self.request("GET", path)
            self.assertEqual(status, 404, path)

    # ── /docs ────────────────────────────────────────────────────────────────

    def test_docs_page_is_served_standalone(self):
        for path in ("/docs", "/docs/", "/api/docs"):
            status, data = self.request("GET", path)
            self.assertEqual(status, 200, path)
            page = data.decode()
            self.assertIn("/api/analyze", page)
            # Nothing is fetched from elsewhere: scripts and styles are inline.
            self.assertNotRegex(page, r'<script[^>]+src=')
            self.assertNotRegex(page, r'<link[^>]+rel="stylesheet"')

    def test_docs_cover_every_query_parameter_the_handler_reads(self):
        import re as _re
        handler_src = (ROOT / "graph_construction" / "server" / "handler.py").read_text()
        # The two analyze handlers: _analyze_opendraco, then _api_analyze.
        start = handler_src.index("def _analyze_opendraco")
        body = handler_src[start:handler_src.index("def _api_get_config", start)]
        names = set(_re.findall(r'params\.get\("([a-z_]+)"', body))
        names |= set(_re.findall(r'_bool_param\(params, "([a-z_]+)"', body))
        names = {n for n in names if n != "prefetch"}
        self.assertTrue({"agent", "id", "format", "agent_id", "phases"} <= names, names)
        _, data = self.request("GET", "/docs")
        page = data.decode()
        for name in sorted(names):
            self.assertIn(f"<code>{name}</code>", page, name)


if __name__ == "__main__":
    unittest.main()
