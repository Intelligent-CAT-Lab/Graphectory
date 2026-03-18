# Graphectory

Artifact repository for the paper [**Process-centric analysis of agentic software systems**](https://arxiv.org/abs/2512.02393), accepted to OOPSLA 2026.

Graphectory transforms agent execution traces into structured graphs that capture the problem-solving patterns of AI software engineering agents. By modeling agent actions as directed graphs with phase classification (localization, patching, validation), this tool enables systematic analysis of how agents approach and solve software engineering tasks.

New: Beyond the two agent frameworks studied in the paper (SWE-agent and OpenHands), the repository additionally supports **mini-swe-agent** (v2.0.0, `trajectory_format` version `mini-swe-agent-1.1`; `trajectory_format` version `mini-swe-agent-1`), a widely used scaffold in agentic research with over 3.3k GitHub stars.

---

## Dataset

**Pre-computed Graphs**: Full dataset (2 agents × 4 models) available under [data/{OpenHands|SWE-agent}/graphs](data/)

**Raw Trajectories**: Hosted on Zenodo due to file size: [https://zenodo.org/records/17364210](https://zenodo.org/records/17364210)

---

## Demo Video

[![Graphectory Walkthrough](https://img.shields.io/badge/▶_Watch_Demo-Video-blue?style=for-the-badge&logo=github)](https://github.com/Intelligent-CAT-Lab/Graphectory/blob/main/demo/video1050646930.mp4)

---

## Installation

```bash
git clone git@github.com:Intelligent-CAT-Lab/Graphectory.git
cd Graphectory
```

We recommend using conda or virtual environments (python>=3.12) to manage dependencies.

---

Note on PyGraphviz (Required for Live Visualization)
The live_graph_server.py tool requires pygraphviz. On Windows, a standard pip install often fails with a cgraph.h error because it cannot find the Graphviz C-libraries.

If you use Conda, we recommend installing the pre-compiled version from conda-forge to handle these dependencies automatically:

```bash
conda install -c conda-forge pygraphviz
python -m pip install -e .
```

If you are not using Conda, you must install the Graphviz system binaries manually and ensure they are added to your system PATH before running the pip install.

## Quick Start

Graphectory provides two tools for working with agent trajectories:

- **[generatejson.py](graph_construction/generatejson.py)**: Batch export graphs to JSON files
- **[live_graph_server.py](graph_construction/live_graph_server.py)**: Interactive browser-based graph visualization

For detailed usage and configuration options, see [graph_construction/README.md](graph_construction/README.md).

### Batch Export (generatejson.py)

Generate JSON graph files for offline analysis:

**SWE-agent with DeepSeek-V3:**
```bash
python graph_construction/generatejson.py \
  --agent sa --model dsk-v3 \
  --trajs data/samples/SWE-agent/trajectories/anthropic_filemap__deepseek--deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test \
  --eval_report data/SWE-agent/reports/deepseek-chat.json \
  --output_dir data/samples
```

**OpenHands with Claude-Sonnet-4:**
```bash
python graph_construction/generatejson.py \
  --agent oh --model cld-4 \
  --trajs data/samples/OpenHands/trajectories/deepseek-chat_maxiter_100_N_v0.40.0-no-hint-run_1/sample_output.jsonl \
  --eval_report data/samples/OpenHands/trajectories/deepseek-chat_maxiter_100_N_v0.40.0-no-hint-run_1/report.json \
  --output_dir data/samples
```

**mini-swe-agent with gpt-5-mini:**
```bash
python graph_construction/generatejson.py \
  --agent msa --model gpt-5-mini \
  --trajs data/samples/mini-swe-agent/trajectories/gpt-5-mini \
  --eval_report data/samples/mini-swe-agent/reports/gpt-5-mini.json \
  --output_dir data/samples
```

**Output**: `{output_dir}/{Agent}/graphs/{model}/{instance_id}/{instance_id}.json`

### Live Interactive Visualization (live_graph_server.py)

Launch a local server for exploring trajectories interactively in your browser:

```bash
python graph_construction/live_graph_server.py \
  --trajs <path_to_trajectories> \
  --eval_report <path_to_report.json>
```

Then open **http://localhost:8000** to browse and visualize graphs on demand. 

---

## Input Requirements

### generatejson.py

| Argument | Description | Format |
|----------|-------------|--------|
| `--agent` | Agent type | `sa` (SWE-agent), `oh` (OpenHands), `msa` (mini-swe-agent) |
| `--model` | Model identifier | `dsk-v3`, `dsk-r1`, `dev`, `cld-4` (extensible) |
| `--trajs` | Trajectory path | **SWE-agent**: directory with `.traj` files<br>**OpenHands**: `output.jsonl` file<br>**mini-swe-agent**: directory with `.traj.json` files|
| `--eval_report` | Evaluation report | JSON with `resolved_ids`/`unresolved_ids` keys |
| `--output_dir` | Base output directory | Organized as `{agent}/graphs/{model}/{instance_id}/` |
| `--workers` | Parallel workers (optional) | Default: 8 |

### live_graph_server.py

| Argument | Description | Format |
|----------|-------------|--------|
| `--trajs` | Trajectory path | **SWE-agent**: directory with `.traj` files<br>**OpenHands**: `output.jsonl` file<br>Agent type auto-detected |
| `--eval_report` | Evaluation report | JSON with `resolved_ids`/`unresolved_ids` keys |
| `--port` | Server port (optional) | Default: 8000 |
| `--assets_dir` | Assets directory (optional) | Default: script directory |

---

## Graph Construction Process

Both `generatejson.py` and `live_graph_server.py` share the same graph construction pipeline:

1. **Parsing**: Agent trajectories → atomic actions using [commandParser.py](graph_construction/commandParser.py)
2. **Node Deduplication**: Identical actions merged with occurrence tracking
3. **Phase Classification**: Actions categorized using heuristics ([mapPhase.py](graph_construction/mapPhase.py)):
   - **Localization**: Information gathering, searching, test generation before patching
   - **Patch**: Creating/editing non-test files
   - **Validation**: Running tests or editing test files after patching
   - **General**: Other actions (planning, environment setup)
4. **Edge Construction**: Execution edges (sequential flow) + hierarchical edges (structural relationship)
5. **Output**:
   - `generatejson.py`: JSON files (NetworkX node-link format)
   - `live_graph_server.py`: Interactive HTML visualization with phase-colored nodes

**Graph Metadata**: Each graph includes `resolution_status`, `instance_name`, and `debug_difficulty`

For detailed graph construction internals, see [buildGraph.py](graph_construction/buildGraph.py).

---

## Extending Graphectory

### Adding New Models

The four models (`dsk-v3`, `dsk-r1`, `dev`, `cld-4`) are pre-configured for paper reproducibility. To add new models, edit [generatejson.py:38](graph_construction/generatejson.py#L38):

```python
SUPPORTED_MODELS = {"dsk-v3", "dsk-r1", "dev", "cld-4", "my-model"}
```

Then run with your new model:
```bash
python graph_construction/generatejson.py \
  --agent sa --model my-model \
  --trajs <your_trajectories> \
  --eval_report <your_report> \
  --output_dir <output>
```

### Supporting New SWE-agent Tools

To parse custom SWE-agent tools, add their `config.yaml` files to [generatejson.py:558-562](graph_construction/generatejson.py#L558-L562):

```python
def setup_parser_for_agent(agent: str) -> CommandParser:
    parser = CommandParser()
    tool_configs = []
    if agent == "sa":
        tool_configs = [
            "data/SWE-agent/tools/edit_anthropic/config.yaml",
            "data/SWE-agent/tools/review_on_submit_m/config.yaml",
            "data/SWE-agent/tools/registry/config.yaml",
            "data/SWE-agent/tools/your_custom_tool/config.yaml",  # Add here
        ]
    if tool_configs:
        parser.load_tool_yaml_files(tool_configs)
    return parser
```

### Supporting New Agents

To add support for a new agent framework:

1. **Implement trajectory builder** in [buildGraph.py](graph_construction/buildGraph.py) (see existing functions at lines 274 & 365):
   ```python
    def build_graph_from_newagent_trajectory(traj_data, parser, instance_id, output_dir, eval_report_path):
        builder = GraphBuilder()
        # Parse agent-specific trajectory structure
        # Convert to builder.add_or_update_node() calls
        return builder.finalize_and_save(output_dir, instance_id, eval_report_path)
   ```

2. **Register the agent** in [generatejson.py:37-50](graph_construction/generatejson.py#L37-L50):
   - Update `SUPPORTED_AGENTS` and `AGENT_NAMES`

3. **Add trajectory loading logic** in [generatejson.py](graph_construction/generatejson.py):
   - Update `load_trajectories()` to handle NewAgent's file format
   - Add branch in `GraphProcessor.process_trajectory()` to call your builder function

**Key principle**: Different agents have different trajectory formats, but all generate the same unified graph structure (nodes with phases, execution/hierarchical edges, metadata).

---

## Graph Analysis

Pre-computed analysis results for the full dataset are available under [data/{OpenHands|SWE-agent}/analysis](data/), including Graphectory metrics.

### Analyze Pre-computed Graphs

```bash
python -m graph_analysis.batch_runner
```

### Analyze Custom Graphs

```bash
python -m graph_analysis.batch_runner --data-dir ./my_data --output-dir ./my_output
```

Results are saved to `trajectory_metrics.csv`.

---
