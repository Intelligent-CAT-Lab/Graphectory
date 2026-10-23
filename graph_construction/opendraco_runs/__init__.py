"""
opendraco_runs

Read OpenDraco multi-agent run logs (``runs/<run_id>/run.jsonl``) into Graphectory.

An OpenDraco run is several agents handing work to each other, so one log yields
two kinds of output:

* one ordinary Graphectory graph per agent (``agents.py``), built by the upstream
  Claude Code builder from a per-agent step stream, so phases, outcomes and every
  existing metric behave exactly as they do for the other formats; and
* a run-level view of the agents working together (``run_graph.py``), with
  pattern detection over it (``patterns.py``) and a standalone HTML page
  (``page.py``).

``analyze.py`` is the entry point ``POST /api/analyze?agent=opendraco`` calls.

The package is named ``opendraco_runs`` rather than ``opendraco`` so it can never
shadow the real OpenDraco package where both are importable.
"""
