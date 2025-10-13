# LLM Mathematics Research

Automated mathematics research using Large Language Models with a scaffolded generator-critic loop.

## Overview

This project implements an automated research system where an LLM iteratively:
1. Generates mathematical research (LaTeX papers + Python experiments)
2. Receives critique from a critic AI
3. Refines its work based on feedback

The system manages the full research workflow including code execution, LaTeX compilation, and comprehensive logging.

## Features

- **Scaffolded Research Loop**: Generator-critic architecture with configurable iteration budgets
- **Reference Papers**: Load ArXiv papers as context from `problems/papers/` directory
- **LaTeX Integration**: Automatic paper generation and PDF compilation
- **Code Execution**: Safe Python code execution with timeout protection
- **Comprehensive Logging**: Tracks all iterations, critiques, plans, and metrics
- **Cost Tracking**: Monitors API usage and costs
- **Configurable**: YAML-based configuration for timeouts, models, and limits

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-maths-research.git
cd llm-maths-research

# Install in development mode
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.10
- LaTeX distribution (for PDF compilation)
- Anthropic API key

## Setup

1. Create a `.env` file with your API key:
```bash
ANTHROPIC_API_KEY=your-api-key-here
```

2. Configure `config.yaml` to adjust:
   - Code execution timeout
   - LaTeX compilation settings
   - API parameters (model, tokens, costs)
   - Output preferences

## Usage

### Single Experiment

Use `run_experiment.py` to run a single experiment with reference papers:

```bash
# Basic usage - provide paper IDs
python run_experiment.py --papers Turrini2024 --max-iterations 10

# Multiple reference papers as context
python run_experiment.py --papers Turrini2024 Smith2025 --max-iterations 10

# Custom problem file (default: problems/open_research.txt)
python run_experiment.py --papers Turrini2024 --problem problems/my_problem.txt

# Custom session name
python run_experiment.py --papers Turrini2024 --session-name my_experiment

# Resume from a specific iteration
python run_experiment.py --papers Turrini2024 --start-iteration 5

# Resume at critic phase (generator already completed)
python run_experiment.py --papers Turrini2024 --resume-at-critic 3
```

### Batch Experiments (Sequential)

Use `run_batch_experiments.py` to run multiple papers sequentially (one paper per experiment):

```bash
# Run specific papers
python run_batch_experiments.py --papers Ashwin2025 Booth2025 Fernley2025 --max-iterations 10

# Run all papers in problems/papers/
python run_batch_experiments.py --max-iterations 10

# Skip already completed papers
python run_batch_experiments.py --skip-completed --max-iterations 10

# Start from a specific paper (useful for resuming)
python run_batch_experiments.py --start-from Waters2025 --max-iterations 10

# Dry run (test without executing)
python run_batch_experiments.py --papers Ashwin2025 --dry-run

# Custom progress tracking file
python run_batch_experiments.py --progress-file my_batch.json --max-iterations 10
```

Progress is automatically saved to `batch_progress/single.json` and the script can be resumed if interrupted.

### Batch Experiments (Grouped)

Use `run_batch_experiments_grouped.py` to run experiments where each can have multiple papers as context. Requires a JSON configuration file:

**Example config** (`batch_configs/my_experiments.json`):
```json
{
  "experiments": [
    {
      "name": "Pattern Formation Studies",
      "papers": ["Turing2024", "Swift2025", "Cross2024"],
      "max_iterations": 15
    },
    {
      "name": "Chaos Theory Analysis",
      "papers": ["Lorenz2024", "Feigenbaum2025"],
      "max_iterations": 10
    }
  ]
}
```

**Running grouped experiments:**
```bash
# Run experiments from config
python run_batch_experiments_grouped.py --config my_experiments.json

# Config file can be in batch_configs/ directory or full path
python run_batch_experiments_grouped.py --config batch_configs/my_experiments.json

# Skip completed experiments
python run_batch_experiments_grouped.py --config my_experiments.json --skip-completed

# Start from a specific experiment
python run_batch_experiments_grouped.py --config my_experiments.json --start-from "Chaos Theory Analysis"

# Dry run
python run_batch_experiments_grouped.py --config my_experiments.json --dry-run
```

Progress is automatically saved to `batch_progress/<config_name>_progress.json`.

### Python API

```python
from llm_maths_research import ScaffoldedResearcher

# Load your problem
with open('problems/problem.txt', 'r') as f:
    problem = f.read()

# Create and run researcher (basic)
researcher = ScaffoldedResearcher(
    session_name="my_session",
    max_iterations=20
)

researcher.run(problem)

# Or with reference papers (from problems/papers/ directory)
researcher = ScaffoldedResearcher(
    session_name="my_session",
    max_iterations=20,
    paper_ids=["2501.00123", "2501.00456"]  # Loads from problems/papers/*.txt
)

researcher.run(problem)
```

## Project Structure

```
llm-maths-research/
├── src/llm_maths_research/         # Main package
│   ├── core/                       # Core research logic
│   │   ├── session.py              # Research session management
│   │   └── researcher.py           # Generator-critic loop
│   ├── utils/                      # Utilities
│   │   ├── latex.py                # LaTeX compilation
│   │   └── code_execution.py       # Safe code execution
│   └── config.py                   # Configuration management
├── problems/                       # Research problem files
│   ├── papers/                     # Reference papers as .txt files (e.g., Turrini2024.txt)
│   └── open_research.txt           # Default problem for run_experiment.py
├── batch_configs/                  # JSON configs for grouped batch experiments
├── batch_progress/                 # Progress tracking for batch runs
├── outputs/                        # Generated outputs (papers, code, logs)
├── tests/                          # Test suite
├── config.yaml                     # Configuration file
├── run_experiment.py               # Run single experiment with reference papers
├── run_batch_experiments.py        # Run multiple papers sequentially
├── run_batch_experiments_grouped.py # Run grouped experiments from config
├── pyproject.toml                  # Package metadata
└── README.md                       # This file
```

## Output Structure

Each research session creates a directory in `outputs/` containing:

- `paper.tex` - Generated LaTeX paper
- `paper.pdf` - Compiled PDF (if successful)
- `code.py` - Python research code
- `session_log.txt` - Complete execution log
- `critiques.txt` - All critic feedback
- `plans.txt` - Research plans per iteration
- `metrics.json` - API usage and cost metrics
- Generated figures (PNG/PDF)

## Configuration

Edit `config.yaml` to customize:

```yaml
execution:
  timeout: 3600             # Code execution timeout (seconds)
  output_limit: 100000      # Output length limit (characters)

compilation:
  timeout: 30               # LaTeX compilation timeout
  error_limit: 500          # Error message length

api:
  model: claude-sonnet-4-5-20250929
  max_tokens: 64000
  thinking_budget: 32000
  rate_limit_wait: 20       # Wait time on rate limit
  costs:
    input_per_million: 3.0  # USD per million input tokens
    output_per_million: 15.0 # USD per million output tokens

research:
  max_iterations: 20        # Default iteration limit

output:
  figure_dpi: 300           # DPI for saved figures
```

## Development

### Installing for Development

```bash
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
# Format code
black .

# Or use ruff
ruff format .
```

### Linting

```bash
ruff check .
```

## Available Python Packages

The code execution environment includes:
- numpy
- scipy
- pandas
- matplotlib
- networkx
- scikit-learn

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Citation

If you use this work, please cite:

```bibtex
@software{llm_maths_research,
  title = {LLM Mathematics Research},
  author = {Allen G Hart},
  year = {2025},
  url = {https://github.com/yourusername/llm-maths-research}
}
```
