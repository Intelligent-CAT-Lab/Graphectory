# Graphectory Online Monitor

An online monitoring system that builds Graphectory and Langutory in real-time to monitor SWE agent execution, and intervene to break stagnation and plan violations.

## Overview

The monitor (implemented in `plan_monitor/`) is pluggable and can be integrated with any SWE agent framework. Currently supports integration with SWE-agent.

## Installation

### 1. Install Monitor Dependencies

```bash
cd application
python -m pip install -e .
```

### 2. Install SWE-agent Dependencies

```bash
cd SWE-agent
python -m pip install --upgrade pip && pip install --editable .
```

## Usage

### Run Comparison Analysis

Compare agent performance before and after monitor integration on problematic instances:

```bash
python application/compute_compare_overlap.py --config oscillation
```

This evaluates the monitor's impact on instances with identified inefficiency patterns (as reported in the revised paper).
