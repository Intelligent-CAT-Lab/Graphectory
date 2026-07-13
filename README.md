<p align="center">
  <img src="figures/logo.png" alt="Graphectory logo" width="260">
</p>
Artifact repository for the paper [Process-Centric Analysis of Agentic Software Systems!](https://dl.acm.org/doi/10.1145/3798271), accepted to the International Conference on Object-Oriented Programming Systems, Languages, and Applications (OOPSLA 2026).

## Demo Video and Online Demo (Try it Out! 🚀)

[![Graphectory Walkthrough](https://img.shields.io/badge/▶_Watch_Demo-Video-blue?style=for-the-badge&logo=github)](https://github.com/Intelligent-CAT-Lab/Graphectory/blob/main/demo/video1050646930.mp4)

[Try the Live Demo Here!](https://graphectory-viewer-demo.vercel.app/)

# Graphectory

Graphectory transforms agent execution traces into structured graphs that capture the problem-solving patterns of AI software engineering agents. By modeling agent actions as directed graphs with phase classification (localization, patching, validation), this tool enables systematic analysis of how agents solve software engineering tasks.

Graphectory is very easy to adopt (please see "Supporting New Agents" and "Supporting New SWE Agent Tools" in the ReadMe). If you have any question or need help, please post on the issue tracker with a sample of your trajectory and we would be happy to assist. 

New: Beyond the two agent frameworks studied in the paper (SWE-agent and OpenHands), the repository additionally supports **mini-swe-agent** (v2.0.0, `trajectory_format` version `mini-swe-agent-1.1`; and `trajectory_format` version `mini-swe-agent-1`), a widely used scaffold in agentic research with over 3.3k GitHub stars. 

---

## Dataset

**Pre-computed Graphs**: Full dataset (2 agents × 4 models) available under [data/{OpenHands|SWE-agent}/graphs](data/)

**Raw Trajectories**: Hosted on Zenodo due to file size: [https://zenodo.org/records/17364210](https://zenodo.org/records/17364210)

---

## Installation

### Docker (Recommended)
We provide a Dockerfile which includes the pre-computed graphs and installs all necessary dependencies to reproduce the results of Graphectory. Please download [Docker](https://www.docker.com/), and then execute the following to create a docker image and execute the container in interactive mode:
```
docker build --no-cache -t graphectory .
docker run -it graphectory bash
```

Note

If you are using MacOS with an Apple chip, please consider adding --platform=linux/amd64 in docker build.

Please refer to Reproduce Graphectory Results for instructions on how to reproduce the results of Graphectory. If you are interested in interactive graph construction and inspection, please refer to [graph_construction/README.md](graph_construction/README.md).

### Build Locally

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

---

## Quick Start

Graphectory provides two tools for working with agent trajectories:

- **[generatejson.py](graph_construction/generatejson.py)**: Batch export graphs to JSON files
- **[live_graph_server.py](graph_construction/live_graph_server.py)**: Interactive browser-based graph visualization

For detailed usage and configuration options, see [graph_construction/README.md](graph_construction/README.md).

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

## Graph Analysis

Pre-computed analysis results for the full dataset are available under [data/{OpenHands|SWE-agent}/analysis](data/), including Graphectory metrics.

### Analyze Pre-computed Graphs

```bash
python -m graph_analysis.batch_runner
```

### Analyze Custom Graphs

```bash
python -m graph_analysis.batch_runner --data-dir ./my_graphs --output-dir ./my_output
```

Results are saved to `trajectory_metrics.csv`.

---

## Reproduce Graphectory Results
