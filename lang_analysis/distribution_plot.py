"""

output_dir: str (defaults to figures/)
Two modes:
1. Multi-mode (default): Scans all agents/models, outputs one 2*4 grid of distribution plots
   Output: {output_dir}/lang_distribution/all_pairs_.pdf

2. Single-mode (custom data_dir): Requires --agent and --model
   Output: {output_dir}/lang_distribution/{agent}_{model}_.pdf