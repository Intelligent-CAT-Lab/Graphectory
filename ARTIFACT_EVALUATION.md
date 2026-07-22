# Graphectory Artifact Evaluation Guide

This document provides instructions for evaluating the artifact of the paper **“Process-Centric Analysis of Agentic Software Systems.”**

The artifact, **Graphectory**, transforms execution trajectories produced by autonomous software engineering agents into structured, phase-aware graphs and supports both individual trajectory inspection and aggregate process-centric analysis. The artifact includes:

* the Graphectory graph construction and analysis implementation;
* an interactive trajectory graph viewer;
* precomputed graphs for the trajectories studied in the paper;
* precomputed analysis results;
* scripts for reproducing the paper's main RQ1–RQ3 figures;
* sample raw trajectories for testing the complete trajectory-to-graph pipeline; and
* support for applying Graphectory to new trajectories.

The complete raw trajectory dataset used in the paper is hosted separately on Zenodo because of its size. **Downloading the full raw trajectory dataset is not required for either the Kick-the-Tires phase or reproducing the main paper figures.**

We recommend that reviewers first complete **Section 1: Kick-the-Tires**, which verifies installation and exercises the artifact's principal functionality. Reviewers can then follow **Section 2: Full Evaluation and Reproduction** to systematically evaluate the claims supported by the artifact.

---

# 1. Kick-the-Tires

**Purpose:** Verify that the artifact can be installed and that its principal graph construction, analysis, and reproduction functionality works correctly.

**Expected time:** approximately 15–30 minutes, excluding the initial Docker image build time.

## 1.1 Requirements

### Recommended environment: Docker

We recommend using Docker because it provides the complete software environment required by the artifact.

The Docker image is based on:

* Python 3.12 (`python:3.12-slim-bookworm`)
* the remaining Python dependencies specified in `pyproject.toml`

The container runs as a non-root user (`graphectory`, UID 10001).

The artifact has been tested on:

* Linux: Ubuntu 22.04, x86-64
* Windows x64 with Docker

We recommend allocating approximately **13–15 GB of memory** to Docker during the build.

---

## 1.2 Obtain the Artifact

Download the artifact:

```bash
cd Graphectory
```

All commands below assume that the current working directory is the repository root unless explicitly stated otherwise.

The repository already contains the following resources needed for the initial evaluation:

```text
Graphectory/
├── data/                   # Precomputed graphs, analysis results, and samples
├── graph_construction/     # Trajectory parsing and graph construction
├── graph_analysis/         # Process-centric graph analysis
├── lang_construction/      # Phase sequence-related analysis
├── plot/                   # Plotting scripts used for paper figures
├── scripts/                # End-to-end helper scripts
├── figures/                # Default location for reproduced figures
├── demo/                   # Viewer screenshots/demo materials
├── Dockerfile
├── DOCKER.md
├── pyproject.toml
└── README.md
```

The complete raw trajectory corpus used in the paper is hosted separately on Zenodo because of its size. It is required only for reviewers who wish to reconstruct the full collection of graphs from raw trajectories.

---

## 1.3 Build and Start the Docker Environment

From the repository root:

```bash
docker build -t graphectory .
```

> [!Note]
> Docker is also intended to support Apple Silicon through an ARM64 build.
>
> ```bash
> docker build --platform linux/arm64 -t graphectory .
> ```

Then start an interactive container:

```bash
docker run -it graphectory bash
```

All commands in the following Docker-based instructions should be executed **inside the container**.

---

## 1.4 Sanity Check A: Reproduce the Paper Figures

Inside the Docker container, run:

```bash
bash scripts/reproduce.sh
```

This command uses the precomputed analysis results included with the artifact and regenerates the principal figures associated with RQ1–RQ3.

Expected outputs include:

```text
figures/median_iqr_trajectory_heatmap.png
figures/sankey_grid.png
figures/end_phase_donuts.png
figures/phase_transition_overview.png
figures/inefficiency_venn/
```

These correspond to:

```text
RQ1
└── Figure 3:
    figures/median_iqr_trajectory_heatmap.png

RQ2
├── Figure 7:
│   figures/sankey_grid.png
├── Figure 8:
│   figures/end_phase_donuts.png
└── Figure 9:
    figures/phase_transition_overview.png

RQ3
└── Figures 14–15:
    figures/inefficiency_venn/*.pdf
```

### Expected result

The command should terminate successfully and produce the figures above.

The regenerated figures should show the same qualitative patterns and underlying results as the corresponding figures in the paper. Because this stage operates on the supplied deterministic, precomputed analysis results, substantial deviations from the paper are not expected. 

> [!NOTE]
> We made minor improvements to command parsing and phase assignment after paper submission. Consequently, some values may differ slightly (typically by one or two instances), but these differences do not affect the reported patterns, claims, or conclusions.

---

## 1.5 Sanity Check B: Construct Graphs from Sample Trajectories

The artifact includes sample trajectories so reviewers can exercise graph construction without downloading the full raw trajectory corpus.

For example, construct graphs for the supplied SWE-agent sample:

```bash
python graph_construction/generatejson.py \
  --agent sa \
  --model dsk-v3 \
  --trajs data/samples/SWE-agent/trajectories/anthropic_filemap__deepseek--deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test \
  --eval_report data/samples/SWE-agent/reports/deepseek-chat.json \
  --output_dir data/samples
```

The generated graph files follow the structure:

```text
data/samples/
└── SWE-agent/
    └── graphs/
        └── deepseek-v3/
            └── <instance_id>/
                └── <instance_id>.json
```

Each generated file is a NetworkX node-link graph.

### Expected result

JSON graphs should be generated successfully under the output directory.

This verifies the core transformation:

```text
raw agent trajectory -> trajectory parsing -> atomic actions -> node deduplication
-> phase labeling -> temporal + structural edge connection -> Graphectory graph
```

---

## 1.6 Sanity Check C: Inspect a Trajectory Interactively

Graphectory also provides an interactive browser-based graph viewer.

From the `graph_construction` directory, or by invoking the script from the repository root, run:

```bash
python graph_construction/live_graph_server.py \
  --trajs data/samples/SWE-agent/trajectories/anthropic_filemap__deepseek--deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test \
  --eval_report data/samples/SWE-agent/reports/deepseek-chat.json
```

Then open:

```text
http://localhost:8000
```

When Docker is used, expose the port when starting the container, for example:

```bash
docker run -it -p 8000:8000 graphectory bash
```

Then run the server inside the container.

### Expected result

The browser interface should display:

1. a list of trajectory instances;
2. resolution-status information for each instance;
3. an interactive graph for the selected trajectory;
4. phase-colored graph nodes;
5. temporal and structural relationships between actions; and
6. detailed thought/action/observation information when inspecting nodes.

The viewer supports interactive options for controlling graph presentation.

This step verifies the artifact's support for process-level inspection of individual agent executions.

---

## 1.7 Kick-the-Tires Completion Checklist

The Kick-the-Tires phase is successful if the reviewer can verify all of the following:

* [ ] The Docker image builds successfully.
* [ ] The container starts successfully.
* [ ] `bash scripts/reproduce.sh` generates the RQ1–RQ3 figures.

Reviewers do **not** need to download the complete Zenodo trajectory corpus to complete this phase.

---

# 2. Full Evaluation and Reproduction

This section evaluates the artifact more systematically.

The artifact pipeline consists of three stages:

```text
Stage 1                    Stage 2                 Stage 3
Raw trajectories     →     Graphectory graphs →   Analysis results
                                                   ↓
                                              Paper figures
```

The corresponding helper scripts are:

```text
scripts/construct.sh -> JSON graphs

scripts/analyze.sh -> CSV/process metrics

scripts/reproduce.sh -> PNG/PDF paper figures
```

The three stages are intentionally separable. For efficient artifact evaluation, reviewers can use the supplied precomputed graphs and analysis results rather than reconstructing the entire dataset from raw trajectories.

---

## 2.1 Evaluation Path A: Reproduce the Paper Results from Supplied Data

This is the **recommended evaluation path**.

The repository contains precomputed graphs under:

```text
data/SWE-agent/graphs/
data/OpenHands/graphs/
```

and precomputed analysis results under:

```text
data/SWE-agent/analysis/
data/OpenHands/analysis/
```

These data cover the agent/model configurations analyzed in the paper.

Generate the publication figures with:

```bash
bash scripts/reproduce.sh
```

This path is sufficient for evaluating the correspondence between the artifact's supplied experimental data and the main process-centric findings reported in RQ1–RQ3.

---

# 3. Claim-by-Claim Evaluation

This artifact primarily supports the empirical, process-centric claims associated with the paper.

The checklist below maps each supported claim to:

1. the corresponding part of the paper;
2. the relevant artifact data or implementation;
3. the command used for evaluation; and
4. the expected manifestation of the claim.

The purpose of this section is not to require reviewers to visually reproduce individual pixels of the published figures, but to make explicit how the supplied artifact evidence corresponds to each paper claim.

---

## Claim 1: Agent Execution Processes Can Be Represented and Quantified Using Graphectory

**Paper reference:** Graphectory methodology and RQ1.

**Claim.** Agent trajectories can be transformed into structured graphs that preserve process-level execution information and enable quantitative characterization beyond final resolved/unresolved outcomes.

**Artifact components:**

```text
graph_construction/
graph_analysis/
data/{SWE-agent|OpenHands}/graphs/
data/{SWE-agent|OpenHands}/analysis/
```

### Evaluation

First construct a sample graph:

```bash
python graph_construction/generatejson.py \
  --agent sa \
  --model dsk-v3 \
  --trajs data/samples/SWE-agent/trajectories/anthropic_filemap__deepseek--deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test \
  --eval_report data/samples/SWE-agent/reports/deepseek-chat.json \
  --output_dir data/samples
```

Then inspect the resulting JSON graph.

For aggregate analysis, the supplied graphs can be analyzed using:

```bash
bash scripts/analyze.sh data/
```

The resulting metrics are written to:

```text
trajectory_metrics.csv
```

### Expected evidence

The reviewer should observe that each trajectory is represented as a graph with process-related information rather than only a binary task outcome.

The aggregate analysis and the RQ1 figure expose measurable characteristics in execution structure across trajectories, agent/model configurations, and outcomes.

---

## Claim 2: Successful and Unsuccessful Agent Runs Exhibit Distinct Problem-Solving Patterns

**Paper reference:** RQ1.

**Claim.** Final task outcome alone does not capture the structure of an agent's execution. Process-centric metrics reveal systematic differences between successful and unsuccessful trajectories, while also showing substantial variation within each outcome category.

**Artifact components:**

```text
data/{SWE-agent|OpenHands}/graphs/
data/{SWE-agent|OpenHands}/analysis/
graph_analysis/
plot/
```

### Evaluation

Run:

```bash
bash scripts/reproduce.sh
```

Inspect:

```text
figures/median_iqr_trajectory_heatmap.png
```

Optionally inspect the underlying analysis output in:

```text
data/SWE-agent/analysis/
data/OpenHands/analysis/
```

### Expected evidence

The reproduced RQ1 analysis should show that trajectories differ in their process-level metrics.

The artifact should also demonstrate why these metrics provide information complementary to conventional outcome-only evaluation: two trajectories with the same task outcome may nevertheless exhibit substantially different execution structures.

The reviewer should compare the reproduced figure with **Figure 3** and the corresponding RQ1 discussion in the paper.

---

## Claim 3: Agent Problem-Solving Strategies Can Be Characterized Through Phase Transitions

**Paper reference:** RQ2.

**Claim.** Agent trajectories exhibit recognizable problem-solving strategies when actions are mapped into higher-level execution phases, including localization, patching, and validation.

**Artifact components:**

```text
graph_construction/mapPhase.py
lang_construction/
plot/
data/{SWE-agent|OpenHands}/
```

The graph construction pipeline assigns actions to execution phases including:

* **Localization**: information gathering, code search, code inspection, and related pre-patch activities;
* **Patch**: source-code creation or modification;
* **Validation**: post-patch testing and validation activities; and
* **General**: actions outside these principal categories.

### Evaluation

Run:

```bash
bash scripts/reproduce.sh
```

Inspect:

```text
figures/sankey_grid.png
figures/end_phase_donuts.png
figures/phase_transition_overview.png
```

These correspond to:

```text
Figure 7
Figure 8
Figure 9
```

respectively.

### Expected evidence

The Sankey and phase-transition visualizations should reveal structured transitions among problem-solving phases.

The reviewer should compare:

* dominant phase-transition flows;
* differences across agent/model configurations;
* differences between resolved and unresolved executions; and
* the phases in which trajectories terminate.

The reproduced results should exhibit the same qualitative transition patterns reported in RQ2.

---

## Claim 4: Successful Resolution Is Associated with More Coherent Problem-Solving Progressions

**Paper reference:** RQ2.

**Claim.** Successful trajectories more frequently exhibit coherent progressions through localization, patching, and validation, whereas unsuccessful executions more frequently display incomplete, irregular, or less productive progressions.

**Artifact components:**

```text
data/{SWE-agent|OpenHands}/analysis/
lang_construction/
plot/sankey_phase_transition_plot.py
plot/end_phase_plot.py
plot/phase_transition_plot.py
```

### Evaluation

Generate the RQ2 figures:

```bash
bash scripts/reproduce.sh
```

Inspect:

```text
figures/sankey_grid.png
figures/end_phase_donuts.png
figures/phase_transition_overview.png
```

### Expected evidence

The reviewer should observe differences between successful and unsuccessful trajectories in:

* phase ordering;
* phase-transition frequencies;
* terminal phases; and
* the presence or absence of complete problem-solving progressions.

These visual patterns should be consistent with the RQ2 observations reported in the paper.

---

## Claim 5: Execution Inefficiencies Occur Across Agent Trajectories, Including Successful Runs

**Paper reference:** RQ3.

**Claim.** Agent executions contain recurring process inefficiencies, and successful task completion does not imply an efficient execution process.

**Artifact components:**

```text
graph_analysis/
data/{SWE-agent|OpenHands}/analysis/
plot/
```

### Evaluation

Run:

```bash
bash scripts/reproduce.sh
```

Inspect:

```text
figures/inefficiency_venn/*.pdf
```

These outputs correspond to **Figures 14–15** in the paper.

### Expected evidence

The reproduced results should show that identified inefficiency patterns occur across the studied trajectories and are not restricted exclusively to unresolved instances.

The reviewer should compare the distribution and overlap of the reported inefficiency categories with the corresponding RQ3 figures and discussion in the paper.

The key qualitative result to verify is that **final task success and process efficiency are distinct dimensions of agent behavior**.

---

# 4. Optional End-to-End Reproduction from Raw Trajectories

The recommended evaluation above starts from the precomputed graphs and analysis files included in the repository.

Reviewers wishing to exercise the complete experimental pipeline can additionally reconstruct graphs from raw trajectories.

Because the full trajectory corpus is large, it is hosted separately on Zenodo:

```text
https://zenodo.org/records/17364210
```

After downloading and extracting the desired trajectories, the complete workflow is:

```bash
bash scripts/construct.sh \
  <trajectories_path> \
  <eval_report.json> \
  [output_dir] \
  [model]
```

then:

```bash
bash scripts/analyze.sh \
  <data_dir> \
  [output_dir] \
  [--agent AGENT] \
  [--model MODEL]
```

and finally:

```bash
bash scripts/reproduce.sh
```

Help for each script is available through:

```bash
bash scripts/construct.sh -h
bash scripts/analyze.sh -h
bash scripts/reproduce.sh -h
```

Reconstructing the complete graph corpus from every raw trajectory is **not required for standard artifact evaluation** because:

1. the full graphs are supplied with the artifact;
2. the corresponding analysis results are supplied;
3. the complete graph construction path can be exercised on the bundled sample trajectories; and
4. the figure-generation pipeline can be evaluated directly from the supplied analysis results.

This separation allows the central artifact claims to be evaluated within one hour without requiring reviewers to repeat the original large-scale agent executions.

---

# 5. Evaluating the Interactive Analysis Functionality

In addition to reproducing aggregate results, reviewers can inspect individual trajectories using the live graph viewer.

For SWE-agent trajectories:

```bash
python graph_construction/live_graph_server.py \
  --trajs path/to/trajectories \
  --eval_report path/to/report.json
```

For OpenHands trajectories:

```bash
python graph_construction/live_graph_server.py \
  --trajs path/to/output.jsonl \
  --eval_report path/to/report.json
```

Open:

```text
http://localhost:8000
```

The viewer automatically lists available trajectories and presents each execution as an interactive graph.

Graph nodes are color-coded by phases, and the interface allows reviewers to inspect the underlying execution information associated with each node.

---

# 6. Reusability

The core reusable component of the artifact is the pipeline that converts agent execution trajectories into a common process-centric graph representation.

The principal reusable components are:

```text
graph_construction/
├── generatejson.py
├── live_graph_server.py
├── buildGraph.py
├── commandParser.py
└── mapPhase.py

graph_analysis/
lang_construction/
```

The artifact is released under the license included in:

```text
LICENSE
```

---

## 6.1 Applying Graphectory to New Trajectories

For a supported agent framework, use:

```bash
python graph_construction/generatejson.py \
  --agent <agent> \
  --model <model> \
  --trajs <trajectory_path> \
  --eval_report <evaluation_report> \
  --output_dir <output_directory>
```

The evaluation report must provide:

```json
{
  "resolved_ids": [...],
  "unresolved_ids": [...]
}
```

Graphs are written under:

```text
<output_directory>/<Agent>/graphs/<model>/<instance_id>/<instance_id>.json
```

The resulting graph representation is shared across supported agent frameworks, enabling downstream analysis without requiring analysis code to depend directly on each framework's native trajectory format.

---

## 6.2 Supported Agent Formats

The artifact supports trajectory processing for the agent frameworks documented in the repository, including the two frameworks studied in the paper:

* SWE-agent
* OpenHands

The repository additionally supports **mini-swe-agent** trajectory formats for reuse beyond the original paper experiments.

Because different agent frameworks expose different trajectory structures, each framework requires an adapter that maps its native format into Graphectory's unified graph representation.

---

## 6.3 Adding a New Model

The four models (`dsk-v3`, `dsk-r1`, `dev`, `cld-4`) are pre-configured for paper reproducibility. To add new models, edit [graph_construction/generatejson.py](graph_construction/generatejson.py):

```python
SUPPORTED_MODELS = {"dsk-v3", "dsk-r1", "dev", "cld-4", "my-model"}
```

After registration, the normal graph generation command can be used with the new model identifier.

The graph representation itself is model-independent.

---

## 6.4 Supporting New SWE-agent Tools

SWE-agent trajectories can contain tool calls defined by tool-specific configuration files.

To support additional SWE-agent tools, add their `config.yaml` files to [graph_construction/generatejson.py](graph_construction/generatejson.py):

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

The command parser can then convert actions generated by the additional tools into Graphectory's atomic action representation.

---

## 6.5 Supporting a New Agent Framework

Supporting a new agent generally requires three steps.

### Step 1: Implement an agent-specific trajectory builder

Add trajectory parsing logic in:

```text
graph_construction/buildGraph.py
```

The adapter should convert agent-specific trajectory records into calls to Graphectory's common graph builder.

Conceptually:

```python
def build_graph_from_newagent_trajectory(
    traj_data,
    parser,
    instance_id,
    output_dir,
    eval_report_path,
):
    builder = GraphBuilder()

    # Parse the framework-specific trajectory.
    # Convert actions into Graphectory nodes and edges.

    return builder.finalize_and_save(
        output_dir,
        instance_id,
        eval_report_path,
    )
```

### Step 2: Register the agent

Update the supported-agent definitions in:

```text
graph_construction/generatejson.py
```

### Step 3: Add trajectory loading logic

Extend the trajectory-loading and processing paths in:

```text
graph_construction/generatejson.py
```

The central design principle is:

```text
Agent A trajectory ─┐
Agent B trajectory ─┼─→ unified Graphectory graph → shared analysis
Agent C trajectory ─┘
```

This unified representation is the primary mechanism through which Graphectory can be reused for additional software-agent frameworks.

---

# 7. Reproducibility Summary

The artifact supports reproduction at three different levels.

## Level 1: Figure Reproduction

**Required data:** Included with repository.

```bash
bash scripts/reproduce.sh
```

Reproduces the principal RQ1–RQ3 figures.

This is the fastest way to verify correspondence between the supplied experimental results and the published paper.

---

## Level 2: Graph Analysis Reproduction

**Required data:** Precomputed graphs included with repository.

```bash
bash scripts/analyze.sh data/
```

Recomputes process-centric analysis from Graphectory graphs.

---

## Level 3: End-to-End Graph Reconstruction

**Required data:** Raw trajectories, either the bundled samples or the complete Zenodo dataset.

```bash
bash scripts/construct.sh <trajectories> <eval_report>
bash scripts/analyze.sh data/
bash scripts/reproduce.sh
```

This evaluates the complete path from native agent execution traces to paper-level aggregate results.

For artifact evaluation, we recommend:

```text
Kick-the-Tires:
    bundled samples
        ↓
    graph construction + viewer
        +
    supplied analysis
        ↓
    figure reproduction

Full Evaluation:
    supplied graphs
        ↓
    process analysis
        ↓
    paper figures

Optional Deep Reproduction:
    Zenodo trajectories
        ↓
    complete pipeline
```

---

# 8. Expected Deviations and Limitations

## 8.1 Determinism

Graph construction and analysis operate on recorded agent trajectories. They do not rerun the underlying language models or software agents.

Therefore, reviewers are not expected to reproduce nondeterministic LLM executions.

Given the same trajectory inputs and artifact version, the generated graph structure and downstream analyses should be reproducible.

---

## 8.2 Full Dataset Size

The complete trajectory corpus is not stored directly in the Git repository because of its size.

It is hosted on Zenodo.

The precomputed graph representation and analysis results needed to evaluate the paper's central RQ1–RQ3 claims are included with the repository, while sample raw trajectories are provided to exercise graph construction.

---

## 8.3 Runtime

Figure reproduction from the supplied analysis results should complete relatively quickly.

Graph construction and analysis time depends on:

* the number of trajectories;
* trajectory size;
* available CPU resources; and
* the number of parallel workers.

Reconstructing the entire graph dataset from the full raw trajectory corpus is therefore expected to take substantially longer than running the sample-based Kick-the-Tires evaluation and is not necessary for verifying basic artifact functionality.

---

## 8.4 Platform Limitations

Docker is the recommended evaluation environment.

Local installation may require additional Graphviz/PyGraphviz configuration, particularly on Windows.

If Docker fails because of insufficient memory, increase the Docker Desktop memory allocation to approximately 13–15 GB.

---

# 9. Troubleshooting

## Docker Build Runs Out of Memory

Increase Docker's memory allocation.

For Docker Desktop:

```text
Settings → Resources → Memory
```

We recommend approximately 13–15 GB.

---

## PyGraphviz Installation Fails Locally

When using Conda:

```bash
conda install -c conda-forge pygraphviz
python -m pip install -e .
```

Alternatively, install the Graphviz system libraries before installing Graphectory.

Using the supplied Docker environment avoids this platform-specific setup.

---

## Output Permission Problems with Docker

If generated files are owned by the container user, ownership can be corrected on the host, for example:

```bash
sudo chown -R $USER:$USER output/
```

Alternatively, run the container using the host user:

```bash
docker run \
  --user $(id -u):$(id -g) \
  -v $(pwd):/output \
  graphectory bash
```

---

## Need Additional Script Information

All pipeline scripts provide built-in help:

```bash
bash scripts/construct.sh -h
bash scripts/analyze.sh -h
bash scripts/reproduce.sh -h
```

Detailed graph-construction and interactive-viewer documentation is available in:

```text
graph_construction/README.md
```

Docker-specific instructions are available in:

```text
DOCKER.md
```

---

# 10. Artifact Evaluation Checklist

For convenience, reviewers can use the following checklist during the evaluation.

## Phase 1: Kick-the-Tires

* [ ] Clone the repository.
* [ ] Build the Docker image.
* [ ] Start the Docker container.
* [ ] Run `bash scripts/reproduce.sh`.
* [ ] Confirm that RQ1–RQ3 figures are generated.
* [ ] Construct a graph from a bundled sample trajectory.
* [ ] Launch the interactive graph viewer.
* [ ] Inspect at least one trajectory graph and its node-level execution information.

## Phase 2: Evaluation

### RQ1

* [ ] Inspect/recompute Graphectory process metrics.
* [ ] Reproduce `median_iqr_trajectory_heatmap.png`.
* [ ] Compare the process-centric patterns with the RQ1 discussion.

### RQ2

* [ ] Reproduce `sankey_grid.png`.
* [ ] Reproduce `end_phase_donuts.png`.
* [ ] Reproduce `phase_transition_overview.png`.
* [ ] Compare phase-transition and termination patterns with the RQ2 discussion.

### RQ3

* [ ] Reproduce `inefficiency_venn/*.pdf`.
* [ ] Compare the resulting inefficiency distributions and overlaps with the RQ3 discussion.

### Reusability

* [ ] Verify graph construction on a bundled raw trajectory.
* [ ] Inspect the documented inputs and unified graph output format.
* [ ] Review the extension points for adding models, SWE-agent tools, or new agent frameworks.

---

# 11. Artifact–Claim Mapping at a Glance

| Paper claim                                                                                            | Artifact component                                | Evaluation procedure                       | Primary output                                     |
| ------------------------------------------------------------------------------------------------------ | ------------------------------------------------- | ------------------------------------------ | -------------------------------------------------- |
| Agent executions can be represented and quantitatively analyzed as process-centric graphs              | `graph_construction/`, `graph_analysis/`          | Construct sample graph; run graph analysis | JSON graphs, `trajectory_metrics.csv`              |
| Process metrics reveal behavioral differences not captured by final outcome alone                      | Precomputed graphs and RQ1 analysis               | `bash scripts/reproduce.sh`                | `median_iqr_trajectory_heatmap.png`                |
| Agent strategies can be characterized through execution-phase transitions                              | `mapPhase.py`, `lang_construction/`, RQ2 analysis | `bash scripts/reproduce.sh`                | `sankey_grid.png`, `phase_transition_overview.png` |
| Successful and unsuccessful executions exhibit different process progressions and termination behavior | RQ2 analysis and plots                            | `bash scripts/reproduce.sh`                | `sankey_grid.png`, `end_phase_donuts.png`          |
| Recurring inefficiencies occur in agent executions, including successful ones                          | `graph_analysis/`, RQ3 analysis                   | `bash scripts/reproduce.sh`                | `inefficiency_venn/*.pdf`                          |

---

# 12. Additional Artifact Resources

For additional information, see:

```text
README.md
```

for an overview of Graphectory and the reproduction pipeline;

```text
DOCKER.md
```

for container setup, platform notes, volume mounting, and troubleshooting; and

```text
graph_construction/README.md
```

for detailed documentation of graph construction, the interactive viewer, graph semantics, and extension to new agents and tools.

This organization is intended to support both **reproducibility of the paper's empirical findings** and **reuse of Graphectory as a general process-centric analysis framework for software-agent trajectories**.
