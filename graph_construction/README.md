# Graph Construction

Interactive graph visualiser for SWE-agent and OpenHands trajectory files. Two entry points are provided: a **live server** that renders graphs on demand in the browser, and a **batch export script** that pre-generates graph JSON files to disk.

## Dependencies

```
networkx       # Graph data structure
bashlex        # Shell command parsing (used by commandParser)
datasets       # HuggingFace datasets — required by generatejson.py for difficulty lookup,
               # optional in the live server (omitting it just leaves difficulty as "unknown")
dagre (JS)     # Automatic graph layout — loaded from unpkg.com at render time
```

---

## Live Server

`live_graph_server.py` starts a local HTTP server and renders every trajectory as an interactive graph on demand. Nothing is written to disk. The agent type (SWE-agent or OpenHands) is detected automatically from the path you pass.

### Quick Start

**SWE-agent** — pass the directory that contains your `.traj` files:

```bash
python live_graph_server.py \
    --trajs path/to/trajectories \
    --eval_report path/to/report.json
```

**OpenHands** — pass the `output.jsonl` file directly:

```bash
python live_graph_server.py \
    --trajs path/to/output.jsonl \
    --eval_report path/to/report.json
```

Then open **http://localhost:8000** in your browser.

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--trajs` | ✓ | — | Directory of `.traj` files (SWE-agent) **or** path to an `output.jsonl` file (OpenHands). The agent type is inferred from which you pass. |
| `--eval_report` | ✓ | — | SWE-bench evaluation report JSON. Must contain `"resolved_ids"` and `"unresolved_ids"` arrays. Used to badge each instance as resolved, unresolved, or unsubmitted. |
| `--assets_dir` | | script directory | Directory containing `graph_template.html`, `styles.css`, and `graph_renderer.js`. Only needed if you have moved those files elsewhere. |
| `--port` | | `8000` | Port to serve on. |

### The Browser UI

The left sidebar lists every trajectory found in the provided path. Each entry shows the instance ID, a coloured status badge (resolved / unresolved / unsubmitted), and a step count. The search box filters the list in real time by instance ID substring.

Clicking an entry loads its graph into the main pane. The graph is rendered inside a sandboxed iframe; switching instances swaps the content without reloading the page.

#### View Toggles

Five toggles sit above the instance list. Changing any of them immediately re-requests the current graph with the new settings applied.

| Toggle | Default | Effect |
|---|---|---|
| **Verbose node labels** | On | Shows multi-line labels: action name, step number, and file path or view range. When off, nodes show only the action verb. |
| **Exclude quotes in thought length** | On | Strips content inside backticks and quote characters before measuring thought length, so arrowhead sizes reflect genuine reasoning rather than copied code. |
| **Filter cd (show ▲ hat)** | Off | Strips leading `cd` commands from multi-command steps and replaces them with a small orange triangle (▲) on the node. |
| **Show observation indicators** | Off | Draws a small coloured square on each edge at the 25% point, encoding the length and success/failure outcome of the previous step's tool response. |
| **Unique think nodes (by thought)** | Off | When on, each `think` step with distinct thought text becomes its own node rather than all think steps collapsing into one. Two steps with identical thought text still share a node. |

#### Reading the Graph

Nodes are coloured by **phase**:

| Colour | Phase | Meaning |
|---|---|---|
| Purple | Localization | Reading files, searching code, running tests before any patch |
| Orange | Patch | Creating or editing source files |
| Blue | Validation | Running tests or inspecting test files after a patch exists |
| Light blue | General | Everything else (think steps, navigation, etc.) |

A node can show two colours as a horizontal gradient when the same action was visited in multiple phases across repeated steps.

Edges are styled by **type**:

| Style | Meaning |
|---|---|
| Grey solid, scaled arrowhead | Normal execution. Arrowhead size encodes thought length — larger means more reasoning. |
| Red solid | Thought continuation — the model's thought for this step was identical to or a prefix of the previous step's, indicating cached reasoning. |
| Blue dashed | Intra-step — connects sub-actions within a single `&&`-chained step. |
| Green dashed | Hierarchy — drawn between `str_replace_editor view` nodes when one path is a subdirectory or line-range subset of another. |

Click any node to open a detail sidebar showing the full thought, action, and observation text for that node. If the node was visited multiple times, tab buttons let you page through each visit. The sidebar's left edge is draggable to resize it.

---

## Batch JSON Export (`generatejson.py`)

`generatejson.py` processes a batch of trajectories and writes one graph JSON file per instance. This is useful for archiving graphs, diffing runs offline, or loading graphs into other tools.

### Quick Start

**SWE-agent:**

```bash
python generatejson.py \
    --agent sa \
    --model dsk-v3 \
    --trajs path/to/trajectory/directory \
    --eval_report path/to/report.json \
    --output_dir output
```

**OpenHands:**

```bash
python generatejson.py \
    --agent oh \
    --model cld-4 \
    --trajs path/to/output.jsonl \
    --eval_report path/to/report.json \
    --output_dir output
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--agent` | ✓ | — | Agent type: `sa` (SWE-agent) or `oh` (OpenHands). |
| `--model` | ✓ | — | Model shorthand — controls the output subdirectory name. See table below. |
| `--trajs` | ✓ | — | Directory of `.traj` files (SA) or path to `output.jsonl` (OH). |
| `--eval_report` | ✓ | — | SWE-bench evaluation report JSON. |
| `--output_dir` | ✓ | — | Root output directory. Graphs are written under `{output_dir}/{Agent}/graphs/{model}/`. |
| `--workers` | | `8` | Number of parallel worker processes. |

#### Model Shorthands

| Flag | Full name written to disk |
|---|---|
| `dsk-v3` | `deepseek-v3` |
| `dsk-r1` | `deepseek-r1-0528` |
| `dev` | `devstral-small` |
| `cld-4` | `claude-sonnet-4` |

### Output Structure

```
output/
├── SWE-agent/
│   └── graphs/
│       └── deepseek-v3/
│           └── django__django-12345/
│               └── django__django-12345.json
└── OpenHands/
    └── graphs/
        └── claude-sonnet-4/
            └── django__django-12345/
                └── django__django-12345.json
```

Each JSON file is a NetworkX node-link graph. Nodes carry label, phase, step indices, thought lengths, tool/command metadata, and the full thought/action/observation text for every visit. Edges carry type (`exec`, `hier`), step index, and thought length.

---

## How Graphs Are Built

Both tools share the same graph construction pipeline.

**Parsing.** Each step's action string is parsed by `commandParser.py` into a list of structured records — one per distinct tool call or shell command. `&&`-chained commands produce multiple records per step.

**Node deduplication.** Each parsed action is hashed by its label, arguments, and flags. If the same action appears multiple times across the trajectory, all occurrences accumulate onto a single node (storing all step indices and thought texts) rather than creating duplicate nodes. This reveals loops and repetition clearly. Think steps are an exception: when the **Unique think nodes** toggle is on, they are keyed by their thought text so that meaningfully different reasoning steps remain distinct.

**Phase classification.** `mapPhase.py` classifies each action into one of four phases using rule-based heuristics that track what has happened so far in the trajectory. The key rule: test execution and test-file edits are **localization** before the first source patch, and **validation** afterward.

**Hierarchical edges.** After the main graph is built, a post-processing pass adds green hierarchy edges between `str_replace_editor view` nodes — connecting parent directories to child paths, and wider line ranges to narrower ones nested within them.

**Thought continuation.** If the current step's thought is identical to or a prefix of the previous step's thought, the connecting edge is flagged as a thought continuation and drawn in red, making it easy to spot steps where the model reused cached reasoning.