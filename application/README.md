# Graphectory Online Monitor

An online monitoring system that builds Graphectory and Langutory in real-time to monitor SWE agent execution, and intervene to break stagnation and plan violations.

## Run Comparison Analysis

Compare agent performance before and after monitor integration on problematic instances:

```bash
python application/compute_compare_overlap.py --config oscillation
```

This evaluates the monitor's impact on instances with identified inefficiency patterns (as reported in the revised paper).
