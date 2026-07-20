# Docker Setup for Graphectory

Containerized reproducible environment for graph analysis of agentic software systems.

Tested platforms: Linux (Ubuntu 22.04, x86-64) and Windows (x64).
Docker build: ~136 seconds and ~13 GB of memory.

## Quick Start

### Build
```bash
docker build -t graphectory .
```

### Run
```bash
docker run -it graphectory bash
```

Enter the container and work interactively with scripts in `scripts/`:

```bash
# Generate figures from precomputed graphs
bash scripts/reproduce.sh

# Or build full pipeline from trajectories
bash scripts/construct.sh <trajectories> <eval_report.json>
bash scripts/analyze.sh data/
bash scripts/reproduce.sh
```

## Multi-Platform Support

```bash
docker build --platform linux/arm64 -t graphectory .
```

## Image Details

**Base**: `python:3.12-slim-bookworm` (~200MB)

**Includes**:
- System: graphviz, libgraphviz-dev, build-essential, pkg-config
- Python: networkx, matplotlib, pygraphviz, pandas, seaborn, scipy, venn, pyyaml, datasets, plus all dependencies in `pyproject.toml`
- Data: Precomputed graphs under `data/`
- Scripts: `scripts/construct.sh`, `scripts/analyze.sh`, `scripts/reproduce.sh`

**Non-root user**: Runs as `graphectory` (UID 10001) for security

## Usage Patterns

### Figures Only (Default)
With precomputed analysis:
```bash
docker run -it graphectory bash
bash scripts/reproduce.sh
```

### Full Pipeline
From trajectories to figures:
```bash
docker run -it graphectory bash
bash scripts/construct.sh <trajectories_path> <eval_report.json>
bash scripts/analyze.sh data/
bash scripts/reproduce.sh
```

### Volume Mounts
Mount host directory for input/output:
```bash
docker run -it -v $(pwd)/data:/opt/graphectory/data graphectory bash
bash scripts/construct.sh /opt/graphectory/data/trajs /opt/graphectory/data/report.json
```

### Development
Edit code and see changes live:
```bash
docker run -it -v $(pwd):/opt/graphectory graphectory bash
python -m graph_analysis.batch_runner --help
```

## Script Reference

All scripts provide help text:
```bash
bash scripts/construct.sh -h    # Graph construction
bash scripts/analyze.sh -h      # Graph analysis
bash scripts/reproduce.sh -h    # Figure generation
```

### Stage 1: Graph Construction
Converts trajectories to JSON graphs.
```bash
bash scripts/construct.sh <trajectories_path> <eval_report.json> [base_output_dir] [model]
```
- Auto-detects agent (SWE-agent or OpenHands)
- Auto-infers model from directory names

Example:
```bash
bash scripts/construct.sh data/samples/SWE-agent/trajectories/anthropic_filemap__deepseek--deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test data/samples/SWE-agent/reports/deepseek-chat.json data/samples
```

### Stage 2: Graph Analysis
Analyzes graphs and computes metrics.
```bash
bash scripts/analyze.sh <data_dir> [output_dir] [--agent AGENT] [--model MODEL]
```

### Stage 3: Figure Generation
Creates all RQ1-RQ3 publication plots.
```bash
bash scripts/reproduce.sh [-o output_dir]
```

## Troubleshooting

### Out of Memory
Increase Docker memory allocation:
- Docker Desktop: Settings → Resources → Memory (13-15GB recommended)

### Permission Issues with Output
Output files belong to container user. Fix on host:
```bash
sudo chown -R $USER:$USER output/
```

Or run with host user:
```bash
docker run --user $(id -u):$(id -g) -v $(pwd):/output graphectory bash
```

### PyGraphviz Compilation
Handled automatically in image via system dependencies. If building locally fails:
```bash
conda install -c conda-forge pygraphviz
pip install -e .
```

## Architecture

**Layer optimization**: Dependencies cached before source/data, enabling fast rebuilds when scripts change.

**Three independent stages**: Each can run separately. Stages 1-2 optional, stage 3 default.

```
Stage 1 (Optional)  → Stage 2 (Optional)  → Stage 3 (Default)
trajectories        → graphs              → analysis
↓                   ↓                      ↓
construct.sh        analyze.sh            reproduce.sh
↓                   ↓                      ↓
JSON graphs         CSV metrics           figures
```

## Local Install (No Docker)

```bash
git clone git@github.com:Intelligent-CAT-Lab/Graphectory.git
cd Graphectory
conda create -n graphectory python=3.12
conda activate graphectory
python -m pip install -e .
```

Run scripts the same way:
```bash
bash scripts/reproduce.sh
bash scripts/construct.sh <trajs> <report>
bash scripts/analyze.sh data/
```
