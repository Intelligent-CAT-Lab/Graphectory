# Scripts

## process_all_trajectories.py

Batch process all agent trajectories to generate graphs in parallel.

### Features

- **Auto-discovery**: Automatically finds all trajectory and report files
- **Parallel processing**: Processes multiple models concurrently
- **Flexible filtering**: Process specific agents or all
- **Dry-run mode**: Preview commands before execution
- **Progress tracking**: Real-time progress updates
- **Error handling**: Graceful handling of failures and timeouts

### Usage

#### Process all trajectories (both agents, all models)
```bash
python scripts/process_all_trajectories.py
```

#### Process only SWE-agent trajectories
```bash
python scripts/process_all_trajectories.py --agents sa
```

#### Process only OpenHands trajectories
```bash
python scripts/process_all_trajectories.py --agents oh
```

#### Process with more parallel workers
```bash
python scripts/process_all_trajectories.py --workers 8
```

#### Dry run (preview without executing)
```bash
python scripts/process_all_trajectories.py --dry-run
```

#### Custom data directory
```bash
python scripts/process_all_trajectories.py --data-dir /path/to/data
```

### Expected Data Structure

```
data/
├── SWE-agent/
│   ├── trajectories/
│   │   ├── anthropic_filemap__deepseek--deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test/
│   │   ├── anthropic_filemap__openrouter--anthropic--claude-sonnet-4__t-0.00__p-1.00__c-2.00___swe_bench_verified_test/
│   │   ├── anthropic_filemap__openrouter--deepseek--deepseek-r1-0528__t-0.00__p-1.00__c-2.00___swe_bench_verified_test/
│   │   └── anthropic_filemap__openrouter--mistralai--devstral-small__t-0.00__p-1.00__c-2.00___swe_bench_verified_test/
│   └── reports/
│       ├── deepseek-chat.json
│       ├── claude-sonnet-4.json
│       ├── deepseek-r1-0528.json
│       └── devstral-small.json
└── OpenHands/
    └── trajectories/
        ├── deepseek-chat_maxiter_100_N_v0.40.0-no-hint-run_1/
        │   ├── output.jsonl
        │   └── report.json
        ├── claude-sonnet-4_maxiter_100_N_v0.40.0-no-hint-run_1/
        │   ├── output.jsonl
        │   └── report.json
        ├── deepseek-r1-0528_maxiter_100_N_v0.40.0-no-hint-run_1/
        │   ├── output.jsonl
        │   └── report.json
        └── devstral-small_maxiter_100_N_v0.40.0-no-hint-run_1/
            ├── output.jsonl
            └── report.json
```

### Output

Graphs are generated in:
`data/{agent}/graphs/{model_name}/{instance_id}/{instance_id}.{json}`

### Supported Models

| Full Model Path                           | Short Name | Display Name      |
|-------------------------------------------|------------|-------------------|
| `deepseek/deepseek-chat`                  | `dsk-v3`   | deepseek-v3       |
| `openrouter/anthropic/claude-sonnet-4`    | `cld-4`    | claude-sonnet-4   |
| `openrouter/deepseek/deepseek-r1-0528`    | `dsk-r1`   | deepseek-r1-0528  |
| `openrouter/mistralai/devstral-small`     | `dev`      | devstral-small    |

### Notes

- The script uses relative paths from the project root
- Default parallelism is 4 workers (adjust with `--workers`)
- Each task has a 1-hour timeout
- Missing trajectory or report files are automatically skipped
