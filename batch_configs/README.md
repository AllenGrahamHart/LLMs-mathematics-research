# Batch Experiment Configurations

This directory contains configuration files for batch experiment runs.

## Example Config Format

See `batch_config_example.json` for a template.

```json
{
  "experiments": [
    {
      "name": "experiment_name",
      "papers": ["Paper1", "Paper2"],
      "max_iterations": 10
    }
  ]
}
```

## Fields

- `name`: Unique identifier for this experiment (required)
- `papers`: List of paper names (without .txt extension) from `problems/papers/` (required)
- `max_iterations`: Number of iterations for this experiment (optional, default: 10)

## Usage

Run experiments from this config:
```bash
python run_batch_experiments_grouped.py --config my_experiments.json
```

Or with full path:
```bash
python run_batch_experiments_grouped.py --config batch_configs/my_experiments.json
```

Progress will be automatically tracked in `batch_progress/<config_name>_progress.json`
