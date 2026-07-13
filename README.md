<p align="center">
  <img src="figures/logo.png" alt="Graphectory logo" width="260">
</p>
Artifact repository for the paper [Process-Centric Analysis of Agentic Software Systems](https://dl.acm.org/doi/10.1145/3798271), accepted to the International Conference on Object-Oriented Programming Systems, Languages, and Applications (OOPSLA 2026).

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
We provide a Dockerfile which includes the pre-computed graphs and installs all necessary dependencies to reproduce the results of Graphectory. Please download [Docker](https://www.docker.com/), and then build and run:
```bash
docker build -t graphectory .
docker run -it graphectory bash
```

**Multi-platform**: Works on Intel and Apple Silicon. For Apple Silicon, use:
```bash
docker build --platform linux/arm64 -t graphectory .
```

For Docker workflows, see [DOCKER.md](DOCKER.md). If interested in interactive graph construction, see [graph_construction/README.md](graph_construction/README.md).

### Build Locally

```bash
git clone git@github.com:Intelligent-CAT-Lab/Graphectory.git
cd Graphectory
```

Requires **Python ≥ 3.12**. We recommend using conda or virtual environments:

```bash
conda create -n graphectory python=3.12 && conda activate graphectory
python -m pip install -e .
```

**PyGraphviz Note** (Required for Live Visualization): On Windows, standard `pip install` often fails due to missing Graphviz C-libraries. Using conda is recommended:
```bash
conda install -c conda-forge pygraphviz
```
Otherwise, install Graphviz system binaries manually before `python -m pip install -e .`

---

## Quick Start

Graphectory provides two tools for working with agent trajectories:

- **[generatejson.py](graph_construction/generatejson.py)**: Batch export graphs to JSON files
- **[live_graph_server.py](graph_construction/live_graph_server.py)**: Interactive browser-based graph visualization

For detailed usage and configuration options, see [graph_construction/README.md](graph_construction/README.md).

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

Precomputed graphs are provided under `data/{OpenHands|SWE-agent}/graphs`. The reproduction pipeline has three optional stages:

### Reproduce Figures (Default)

Uses precomputed graphs and analysis. Inside Docker or local environment:

```bash
reproduce  # Docker: shortcut to docker/reproduce.sh
# or locally: python docker/reproduce.sh  (or run scripts individually)
```

Generates all paper figures:
- **RQ1**: `python plot/trajectory_heatmap_plot.py` → `figures/median_iqr_trajectory_heatmap.png` (Figure 3)
- **RQ2**: 
  - `python plot/sankey_phase_transition_plot.py` → `figures/sankey_grid.png` (Figure 7)
  - `python plot/end_phase_plot.py` → `figures/end_phase_donuts.png` (Figure 8)
  - `python plot/phase_transition_plot.py` → `figures/phase_transition_overview.png` (Figure 9)
- **RQ3**: `python plot/inefficiency_plot.py` → `figures/inefficiency_venn/*.pdf` (Figures 14-15)

### Optional: Graph Construction & Analysis

To generate graphs from raw trajectories (requires Zenodo data https://zenodo.org/records/17364210 or `data/samples/`):

**Docker:**
```bash
construct <trajectories_path> <eval_report.json>  # Stage 1: trajectories → graphs
analyze data/                                      # Stage 2: graphs → metrics
reproduce                                         # Stage 3: metrics → figures
```

**Local:**
See `docker/construct.sh`, `docker/analyze.sh`, and `docker/reproduce.sh` for detailed usage. For graph construction and interactive visualization details, see [graph_construction/README.md](graph_construction/README.md).