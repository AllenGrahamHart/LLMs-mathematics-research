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
- **Literature Integration**:
  - OpenAlex API for searching academic papers
  - ArXiv paper downloads with full-text extraction
  - Automatic inclusion of reference papers in prompts
- **Reference Papers**: Load papers as context from `problems/papers/` directory
- **Data File Integration**: Load datasets from `data/datasets/` directory for analysis
- **LaTeX Integration**: Automatic paper generation and PDF compilation
- **Code Execution**: Safe Python code execution with timeout protection
- **Template System**: Customizable prompts and LaTeX templates
- **Structured Response Format**: XML-based parsing for robust extraction of code, LaTeX, and plans
- **Comprehensive Logging**: Tracks all iterations, critiques, plans, and metrics
- **Cost Tracking**: Monitors API usage and costs
- **Unit Tests**: 67 tests covering core functionality, XML extraction, and literature integration
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

### Running Experiments

Use `run_experiment.py` to run experiments with reference papers and data files:

```bash
# Basic usage - provide paper IDs
python run_experiment.py --papers Turrini2024 --max-iterations 10

# Multiple reference papers as context
python run_experiment.py --papers Turrini2024 Smith2025 --max-iterations 10

# Custom problem file (default: problems/open_research.txt)
python run_experiment.py --papers Turrini2024 --problem problems/my_problem.txt

# With data files (from data/datasets/ directory)
python run_experiment.py --papers Turrini2024 --data mydata.csv

# Multiple data files
python run_experiment.py --papers Turrini2024 --data dataset1.csv dataset2.json

# Custom session name
python run_experiment.py --papers Turrini2024 --session-name my_experiment

# Resume from a specific iteration
python run_experiment.py --papers Turrini2024 --start-iteration 5

# Resume at critic phase (generator already completed)
python run_experiment.py --papers Turrini2024 --resume-at-critic 3
```

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

# With data files (from data/datasets/ directory)
researcher = ScaffoldedResearcher(
    session_name="my_session",
    max_iterations=20,
    paper_ids=["2501.00123"],
    data_ids=["mydata.csv", "timeseries.json"]  # Loads from data/datasets/
)

researcher.run(problem)
```

## Literature Search

The system includes integrated literature search capabilities using OpenAlex and ArXiv APIs.

### Agent Integration

Agents can use literature search through `<OPENALEX>` blocks in their responses:

```xml
<OPENALEX>
[
  {
    "function": "search_literature",
    "arguments": {
      "query": "pattern formation Turing",
      "filters": {"from_year": 2020, "min_citations": 10},
      "max_results": 15
    },
    "purpose": "Find recent highly-cited work on Turing patterns"
  },
  {
    "function": "get_paper",
    "arguments": {"identifier": "W2100837269"},
    "purpose": "Get full metadata for Turing's 1952 paper"
  },
  {
    "function": "get_arxiv_paper",
    "arguments": {"arxiv_id": "2211.02556"},
    "purpose": "Download full paper for detailed analysis"
  }
]
</OPENALEX>
```

### Python API

```python
from llm_maths_research.literature.tools import (
    search_literature,
    get_paper,
    get_arxiv_paper
)

# Search for papers with filters
results = search_literature(
    query="pattern formation Turing",
    filters={"from_year": 2020, "min_citations": 10},
    max_results=15
)

for paper in results['results']:
    print(f"{paper['title']} ({paper['publication_year']})")
    print(f"Citations: {paper['cited_by_count']}")

# Get paper metadata by OpenAlex ID or DOI
paper = get_paper("W2100837269")
print(paper['title'])
print(paper['abstract'])

# Download full ArXiv paper (LaTeX source)
arxiv_paper = get_arxiv_paper("2211.02556")
print(f"Downloaded {arxiv_paper['char_count']} characters")
```

### Available Functions

- `search_literature(query, filters, max_results)` - Search OpenAlex by keywords and filters
  - Filters: `from_year`, `to_year`, `min_citations`, `doi`, `cites`, `cited_by`, `is_open_access`, etc.
- `get_paper(identifier)` - Get full metadata for a paper (by OpenAlex ID or DOI)
  - Returns: title, authors, abstract, citations, references, formatted citations
- `get_arxiv_paper(arxiv_id)` - Download full LaTeX source from ArXiv
  - Returns: full text content (~15-30k+ tokens per paper)

## Data Files

The system supports loading external data files for analysis. Data files are stored in `data/datasets/` and can be provided to experiments via the `--data` parameter.

### Adding Data Files

1. Place your data file in `data/datasets/` (e.g., `mydata.csv`)
2. Optionally create a description file with the same name and `.txt` extension (e.g., `mydata.txt`)
3. Reference the file when running an experiment

**Example:**

```bash
# data/datasets/temperature_data.csv
# data/datasets/temperature_data.txt (optional description)

python run_experiment.py --papers Turrini2024 --data temperature_data.csv
```

### Description Files

Create a `.txt` file with the same name as your data file to provide context:

```
# data/datasets/temperature_data.txt
Daily temperature measurements from weather station Alpha.

Format: CSV with columns date, temperature_c, humidity_percent
Period: January 2024 - December 2024
Frequency: Daily readings at 12:00 UTC
Missing data: Coded as -999
```

The description is included in the prompt to help the LLM understand the data structure.

### How Data Files Work

1. **Copying**: Data files are copied from `data/datasets/` to `outputs/{session}/data/`
2. **Path Access**: In code, use `os.path.join(output_dir, "data", "filename.csv")` to load files
3. **Automatic**: The `output_dir` variable is pre-defined and figures are saved automatically

**Example code generated by the LLM:**

```python
import pandas as pd
import os

# Load data (output_dir is pre-defined)
df = pd.read_csv(os.path.join(output_dir, "data", "mydata.csv"))

# Analyze and visualize
# ... analysis code ...

# Save figure (automatically goes to correct directory)
plt.savefig("my_plot.png", dpi=300)
```

## Project Structure

```
llm-maths-research/
├── src/llm_maths_research/         # Main package
│   ├── core/                       # Core research logic
│   │   ├── session.py              # Research session management
│   │   └── researcher.py           # Generator-critic loop
│   ├── literature/                 # Literature search integration
│   │   ├── openalex_client.py      # OpenAlex API client
│   │   ├── arxiv_client.py         # ArXiv paper downloads
│   │   └── tools.py                # Literature search tools
│   ├── templates/                  # Prompt and LaTeX templates
│   │   ├── generator_prompt.txt    # Generator AI prompt template
│   │   ├── critic_prompt.txt       # Critic AI prompt template
│   │   └── initial_paper.tex       # Initial LaTeX document template
│   ├── utils/                      # Utilities
│   │   ├── xml_extraction.py       # XML-based response parsing
│   │   ├── latex.py                # LaTeX compilation
│   │   └── code_execution.py       # Safe code execution
│   └── config.py                   # Configuration management
├── problems/                       # Research problem files
│   ├── papers/                     # Reference papers as .txt files (e.g., Turrini2024.txt)
│   └── open_research.txt           # Default problem for run_experiment.py
├── data/                           # Data files for experiments
│   └── datasets/                   # Dataset files (CSV, JSON, etc.)
├── outputs/                        # Generated outputs (papers, code, logs)
├── tests/                          # Test suite
├── config.yaml                     # Configuration file
├── run_experiment.py               # Run experiments with reference papers and data
├── pyproject.toml                  # Package metadata
└── README.md                       # This file
```

## Output Structure

Each research session creates a directory in `outputs/` containing:

- `paper.tex` - Generated LaTeX paper
- `paper.pdf` - Compiled PDF (if successful)
- `code.py` - Python research code
- `data/` - Copies of data files used in the experiment
- `session_log.txt` - Complete execution log
- `current_plan.txt` - Latest plan (for easy resumption)
- `current_critique.txt` - Latest critique (for easy resumption)
- `current_researcher_openalex.txt` - Latest researcher literature searches
- `current_critic_openalex.txt` - Latest critic literature searches
- `plans.txt` - All research plans (append-only history)
- `critiques.txt` - All critic feedback (append-only history)
- `generator_responses.txt` - All generator outputs per iteration
- `metrics.json` - API usage and cost metrics
- Generated figures (PNG/PDF)
- `arxiv_cache/` - Downloaded ArXiv papers (if used)

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
# Run all tests
pytest

# Run only unit tests
pytest tests/unit/

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src/llm_maths_research
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
