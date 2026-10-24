"""
opendraco_runs

Read the public OpenDraco's inference event logs
(github.com/OpenDraco/OpenDraco, ``opendraco/logs/inference_logs/<stem>.ndjson``)
into Graphectory.

``events.py`` rewrites one SWE-bench instance of the log as OpenDraco2 records,
so everything in ``opendraco2_runs`` (per-agent graphs, the run view, pattern
detection, the standalone page) is reused; ``tools.py`` sorts the framework's
many tool names into the kinds the phase mapping understands; ``analyze.py`` is
the entry point ``POST /api/analyze?agent=opendraco`` calls.

The package is named ``opendraco_runs`` rather than ``opendraco`` so it never
shadows the real OpenDraco package, which is itself called ``opendraco``.
"""
