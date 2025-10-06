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

### Command Line Interface

```bash
# Run research on a problem file
llm-research problems/your_problem.txt

# Specify maximum iterations
llm-research problems/your_problem.txt --max-iterations 10

# Custom session name
llm-research problems/your_problem.txt --session-name my_experiment
```

### Python API

```python
from llm_maths_research import ScaffoldedResearcher

# Load your problem
with open('problems/problem.txt', 'r') as f:
    problem = f.read()

# Create and run researcher
researcher = ScaffoldedResearcher(
    session_name="my_session",
    max_iterations=20
)

researcher.run(problem)
```

### Direct Script Execution

```bash
python run_experiment.py
```

## Project Structure

```
llm-maths-research/
├── src/llm_maths_research/    # Main package
│   ├── core/                  # Core research logic
│   │   ├── session.py         # Research session management
│   │   └── researcher.py      # Generator-critic loop
│   ├── utils/                 # Utilities
│   │   ├── latex.py           # LaTeX compilation
│   │   └── code_execution.py  # Safe code execution
│   ├── config.py              # Configuration management
│   └── cli.py                 # Command-line interface
├── problems/                  # Research problem files
├── outputs/                   # Generated outputs (papers, code, logs)
├── tests/                     # Test suite
├── config.yaml                # Configuration file
├── pyproject.toml             # Package metadata
└── README.md                  # This file
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
  timeout: 600              # Code execution timeout (seconds)
  output_limit: 2000        # Output length limit (characters)

compilation:
  timeout: 30               # LaTeX compilation timeout
  error_limit: 500          # Error message length

api:
  model: claude-sonnet-4-5-20250929
  max_tokens: 64000
  thinking_budget: 8000
  rate_limit_wait: 10       # Wait time on rate limit

research:
  max_iterations: 20        # Default iteration limit
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
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/llm-maths-research}
}
```
