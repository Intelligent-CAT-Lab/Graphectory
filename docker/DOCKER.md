# Docker Setup for Graphectory

This document describes the Docker configuration for Graphectory, enabling reproducible graph analysis across platforms.

## Quick Start

### Build the Image
```bash
# Standard build (for your current platform)
docker build -t graphectory .

# Multi-platform build (both amd64 and arm64)
docker buildx build --platform linux/amd64,linux/arm64 -t graphectory:latest .
```

### Run the Container
```bash
# Interactive shell
docker run -it graphectory bash

# Generate figures (defaults to figures/ directory)
docker run -it graphectory reproduce

# Save output to host
docker run -it -v $(pwd)/output:/output graphectory reproduce
```

## Three-Stage Pipeline

Graphectory reproducibility workflow has three independent stages: **construct**, **analyze**, and **plot**. Only the final stage (plot) is required for generating publication figures; the first two are optional for custom workflows.

### Stage 1: Graph Construction (Optional)
Convert raw trajectories to graph JSON files.

```bash
docker run -it graphectory construct <trajectories_path> <eval_report.json> [output_dir] [model]

# Example: SWE-agent trajectories
docker run -it -v /path/to/data:/data graphectory construct \
    /data/trajectories \
    /data/report.json \
    /data/graphs
```

**Detects agent type automatically:**
- **SWE-agent**: Directory with `.traj` files
- **OpenHands**: Path to `output.jsonl`

**Model auto-inference** from directory names (or specify explicitly):
- `deepseek-chat` → dsk-v3
- `deepseek-r1` → dsk-r1
- `devstral-small` → dev
- `claude-sonnet-4` → cld-4
- `gpt-5-mini` → gpt-5-mini

### Stage 2: Graph Analysis (Optional)
Analyze graphs and compute trajectory metrics.

```bash
docker run -it graphectory analyze <data_dir> [output_dir] [--agent AGENT] [--model MODEL]

# Analyze all agents and models
docker run -it -v $(pwd)/data:/data graphectory analyze /data

# Filter by agent or model
docker run -it -v $(pwd)/data:/data graphectory analyze /data \
    ./output \
    --agent SWE-agent \
    --model deepseek

# Generate analysis metrics for figure generation
docker run -it graphectory analyze data/ ./analysis
```

**Expected structure:**
```
data/
  ├── SWE-agent/graphs/model1/instance/*.json
  └── OpenHands/graphs/model2/instance/*.json
```

### Stage 3: Figure Generation (Default)
Generate publication-ready plots from precomputed analysis results.

```bash
docker run -it graphectory reproduce

# Custom output directory
docker run -it -v $(pwd):/output graphectory reproduce -o /output/figs

# Host results directory
docker run -it -v $(pwd)/results:/output graphectory reproduce -o /output
```

**Generates (RQ1-RQ3):**
- `figures/median_iqr_trajectory_heatmap.png` — RQ1 metrics
- `figures/sankey_grid.png` — RQ2 phase transitions
- `figures/end_phase_donuts.png` — RQ2 end phases
- `figures/phase_transition_overview.png` — RQ2 phase flow
- `figures/inefficiency_venn/*.pdf` — RQ3 inefficiencies

## Image Details

**Base**: `python:3.12-slim-bookworm` (~200MB)
- Multi-architecture: `linux/amd64` and `linux/arm64`
- Broad library compatibility, minimal footprint

**Includes:**
- System: graphviz, libgraphviz-dev, build-essential, pkg-config
- Python: networkx, matplotlib, pygraphviz, pandas, seaborn, scipy, venn, pyyaml, datasets, bashlex, tqdm, sympy, requests
- Data: Precomputed graphs under `data/`
- Scripts: Three-stage pipeline (construct.sh, analyze.sh, reproduce.sh)

## Usage Patterns

### Full Reproducibility (All Stages)
From raw trajectories to figures:

```bash
docker run -it -v /path/to/trajectories:/traj graphectory bash
# Inside container:
construct /traj report.json data/
analyze data/
reproduce
```

### Analysis-Only (Stages 2-3)
If you have precomputed graphs:

```bash
docker run -it -v $(pwd)/data:/data graphectory bash
# Inside:
analyze /data ./results
reproduce -o ./results
```

### Figures-Only (Stage 3, Default)
With precomputed analysis:

```bash
docker run -it graphectory reproduce
```

## For Development

Mount the project directory to edit code:

```bash
docker run -it -v $(pwd):/opt/graphectory graphectory bash
# Changes persist on host; Python picks up modifications
```

Test individual modules:

```bash
docker run -it graphectory bash
python -m graph_analysis.batch_runner --help
python -m graph_construction.generatejson --help
python plot/trajectory_heatmap_plot.py
```

## Troubleshooting

### Out of Memory During Construction or Analysis
Increase Docker memory (Docker Desktop → Settings → Resources → Memory):
- Minimum: 4GB
- Recommended: 6-8GB

### Permission Issues with Output Files
Output files belong to container user `graphectory`. Fix on host:

```bash
sudo chown -R $USER:$USER ./output/
```

Or run with host user ID:

```bash
docker run --user $(id -u):$(id -g) -v $(pwd):/output graphectory reproduce
```

### pygraphviz Compilation Issues
The image handles this automatically via:
- `graphviz` system package (C libraries)
- `libgraphviz-dev` (headers)
- `pkg-config` (library discovery)

If building locally fails, use conda:
```bash
conda install -c conda-forge pygraphviz
pip install -e .
```

### Build for Apple Silicon
Docker Desktop detects ARM64 automatically:

```bash
docker build -t graphectory .
```

Or explicit:
```bash
docker build --platform linux/arm64 -t graphectory .
```

## References

- [Graphectory README](README.md)
- [Graph Construction Guide](graph_construction/README.md)
- [Docker Buildx](https://docs.docker.com/build/architecture/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
