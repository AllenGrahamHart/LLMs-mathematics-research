# Datasets Directory

This directory contains datasets that can be provided to the AI researcher during experiments.

## Usage

Place your data files in this directory, then reference them when running experiments:

```bash
python run_experiment.py --papers MyPaper --data dataset1.csv dataset2.json
```

The specified data files will be copied into the session's output directory and made available to the agent.

## Organization

You can organize files in subdirectories or keep them flat. When referencing them, use the filename or relative path:

- Flat: `data/datasets/mydata.csv` → `--data mydata.csv`
- Nested: `data/datasets/timeseries/temp.csv` → `--data timeseries/temp.csv`

## Description Files (Optional)

For each dataset, you can create a corresponding `.txt` description file to provide context to the agent:

- `dataset1.csv` + `dataset1.txt` (contains human-readable description)

The agent will receive this description in the prompt, helping it understand the dataset's structure and purpose.

## Example

```
data/datasets/
  temperature_2020_2023.csv
  temperature_2020_2023.txt  # Description file
  social_network_graph.json
  social_network_graph.txt   # Description file
```
