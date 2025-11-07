# LLM Mathematics & Machine Learning Research

Automated research using Large Language Models with a scaffolded generator-critic loop for both mathematics and machine learning experiments.

## Overview

This project implements an automated research system where an LLM iteratively:
1. Generates research papers (LaTeX) with supporting experiments (Python code)
2. Receives critique from a critic AI
3. Refines its work based on feedback

The system manages the full research workflow including code execution, LaTeX compilation, GPU computation via Modal, and comprehensive logging.

## Features

- **Scaffolded Research Loop**: Generator-critic architecture with configurable iteration budgets
- **Literature Integration**:
  - OpenAlex API for searching academic papers
  - ArXiv paper downloads with full-text extraction
  - Automatic inclusion of reference papers in prompts
- **Code Context Integration**: Load entire codebases (e.g., nanoGPT) as context for ML research
- **Reference Papers**: Load papers as context from `problems/papers/` directory
- **Data File Integration**: Load datasets from `data/datasets/` directory for analysis
- **GPU Computation**: Modal integration for running training experiments on cloud GPUs
- **LaTeX Integration**: Automatic paper generation and PDF compilation
- **Code Execution**: Safe Python code execution with timeout protection
- **Template System**: Customizable prompts and LaTeX templates
- **Structured Response Format**: XML-based parsing for robust extraction of code, LaTeX, and plans
- **Comprehensive Logging**: Tracks all iterations, critiques, plans, and metrics
- **Cost Tracking**: Monitors API usage and costs
- **Unit Tests**: 74 tests covering core functionality, code loading, XML extraction, literature integration, and three-stage architecture
- **Configurable**: YAML-based configuration for timeouts, models, and limits
- **Prompt Caching**: Intelligent 1-hour caching reduces API costs by ~40%
- **Three-Stage Generator**: Sequential planning → coding → writing for proper information flow

## Three-Stage Generator Architecture

The generator phase uses a novel three-stage approach where each iteration consists of three sequential API calls:

1. **Planning Stage**: Creates detailed plan and performs optional literature search
2. **Coding Stage**: Generates and executes Python code (with access to plan and literature)
3. **Writing Stage**: Generates LaTeX paper (with access to plan, literature, code, and execution results)

**Key Benefit**: The LaTeX paper can now reference actual execution results instead of anticipated ones, ensuring papers accurately report real findings.

**Documentation**: See `docs/architecture/THREE_STAGE_IMPLEMENTATION_COMPLETE.md` for full details.

## Multi-Provider Support

The system supports multiple LLM providers, allowing you to choose between different models based on your needs:

- **Anthropic**: Claude Sonnet 4.5, Opus 4, etc. (with extended thinking & prompt caching)
- **OpenAI**: GPT-4o, GPT-4 Turbo, o1 series
- **Google**: Gemini 2.0 Flash, Gemini 1.5 Pro/Flash
- **xAI**: Grok Beta, Grok Vision
- **Moonshot**: Kimi (8k/32k/128k context)

Switch providers by editing one line in `config.yaml`:

```yaml
api:
  provider: anthropic  # or: openai, google, xai, moonshot
  model: claude-sonnet-4-5-20250929
```

When the AI writes research papers, it lists itself as an author using the appropriate model name (e.g., "GPT-4o", "Gemini 2.0 Flash"). The `{model}` placeholder in problem files is automatically replaced with the correct display name.

**Full Documentation**: See [`docs/providers/MULTI_PROVIDER_GUIDE.md`](docs/providers/MULTI_PROVIDER_GUIDE.md) for detailed configuration examples, cost comparisons, and feature matrix.

## Prompt Caching

The system implements Anthropic's 1-hour prompt caching to significantly reduce API costs during multi-iteration research sessions. By caching static prompt content (problem statements, reference papers, data descriptions), the system achieves approximately 40% cost savings on typical research runs.

**Note**: Prompt caching is currently only supported by Anthropic models.

### Cache Optimization

The three-stage generator architecture (planning → coding → writing) shares a single cache across all three stages within each iteration. This optimization reduces cache writes from 4 to 2 per iteration:

**Before optimization**:
- Planning stage: Creates cache (2× cost)
- Coding stage: Creates different cache (2× cost)
- Writing stage: Creates different cache (2× cost)
- Critic stage: Creates cache (2× cost)

**After optimization**:
- Planning stage: Creates cache (2× cost)
- Coding stage: **Reads from cache** (0.1× cost, 90% savings)
- Writing stage: **Reads from cache** (0.1× cost, 90% savings)
- Critic stage: Creates cache (2× cost)

This is achieved by ensuring all three generator stages share identical static content, with only the output format instructions varying in the dynamic portion.

### How It Works

Each API call (both generator and critic) is split into two parts:

1. **Static Content (Cached)**:
   - Problem statement
   - Reference papers
   - Code context (entire codebases like nanoGPT)
   - Data file descriptions
   - System instructions and templates
   - This content is cached for 1 hour and refreshed on each use

2. **Dynamic Content (Not Cached)**:
   - Current iteration number
   - Current stage (planning/coding/writing)
   - Output format instructions (stage-specific)
   - Latest code and LaTeX
   - Previous execution results
   - Recent critique feedback

### Cost Structure

- **Cache Write**: 2× base input cost (first call or after 1-hour expiry)
- **Cache Read**: 0.1× base input cost (subsequent calls within 1 hour)
- **Break-even**: After ~2-3 API calls (typical research sessions use 10-40 calls)

With a multi-iteration research session:
- Iteration 1: Cache write (2× cost on cached portion)
- Iteration 2+: Cache read (0.1× cost on cached portion, ~90% savings)
- Cache automatically refreshes on each use (resets 1-hour timer)

### Configuration

Caching is enabled by default but can be controlled:

```python
# Python API - disable caching
researcher = ScaffoldedResearcher(
    session_name="my_session",
    use_cache=False  # Disable caching
)

# Configuration (config.yaml)
api:
  costs:
    cache_write_multiplier: 2.0   # Cache creation cost (2× base)
    cache_read_multiplier: 0.1    # Cache read cost (0.1× base)
```

### Requirements

Caching requires static content ≥1,024 tokens. The system automatically:
- Splits prompts at optimal boundaries
- Only caches content above the threshold
- Tracks cache metrics in `metrics.json`

### Metrics

Cache performance is tracked in `outputs/{session}/metrics.json`:

```json
{
  "cache_creation_tokens": 2653,  // Tokens written to cache
  "cache_read_tokens": 1340,      // Tokens read from cache
  "cost": 0.1631                   // Cost including cache multipliers
}
```

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
- API key for at least one supported provider (Anthropic, OpenAI, Google, xAI, or Moonshot)

## Setup

1. Create a `.env` file with your API key(s):
```bash
# At least one of these (depending on which provider you want to use)
ANTHROPIC_API_KEY=your-anthropic-key-here
OPENAI_API_KEY=your-openai-key-here
GOOGLE_API_KEY=your-google-key-here
XAI_API_KEY=your-xai-key-here
MOONSHOT_API_KEY=your-moonshot-key-here
```

See `.env.example` for a template.

2. Configure `config.yaml` to adjust:
   - **LLM Provider**: Choose from anthropic, openai, google, xai, or moonshot
   - **Model**: Select specific model for your chosen provider
   - Code execution timeout
   - LaTeX compilation settings
   - API parameters (tokens, thinking budget, costs)
   - Output preferences

## Usage

### Running Experiments

Use `run_experiment.py` to run experiments with optional reference papers and data files:

```bash
# No papers - use literature search tools instead
python run_experiment.py --problem problems/my_problem.txt --max-iterations 10

# With paper IDs as context
python run_experiment.py --papers Turrini2024 --max-iterations 10

# Multiple reference papers as context
python run_experiment.py --papers Turrini2024 Smith2025 --max-iterations 10

# Custom problem file (default: problems/open_research.txt)
python run_experiment.py --papers Turrini2024 --problem problems/my_problem.txt

# With data files (from data/datasets/ directory)
python run_experiment.py --papers Turrini2024 --data mydata.csv

# Multiple data files
python run_experiment.py --papers Turrini2024 --data dataset1.csv dataset2.json

# With code context (for ML research)
python run_experiment.py --papers attention_paper --code nanogpt --problem problems/ml_research.txt

# Full ML research setup
python run_experiment.py --papers attention_paper --code nanogpt --data training_results.csv --max-iterations 10

# Custom session name
python run_experiment.py --papers Turrini2024 --session-name my_experiment
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

# With code context for ML research (from problems/code/ directory)
researcher = ScaffoldedResearcher(
    session_name="ml_research",
    max_iterations=20,
    paper_ids=["attention_paper"],
    code_ids=["nanogpt"],  # Loads from problems/code/nanogpt/
    data_ids=["training_data.csv"]
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
2. **Path Access**: In code, use relative paths from the session root directory
3. **Organized Output**: Figures go to `artifacts/figures/`, generated data to `artifacts/data/`

**Example code generated by the LLM:**

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load input data
df = pd.read_csv("data/mydata.csv")

# Analyze and visualize
# ... analysis code ...

# Save figure to artifacts/figures/
plt.savefig("artifacts/figures/my_plot.png", dpi=300)

# Save generated data to artifacts/data/
df.to_csv("artifacts/data/results.csv", index=False)
```

## Code Context

The system supports loading entire codebases as context for ML research experiments. This allows the LLM to understand, modify, and experiment with existing code (e.g., training frameworks like nanoGPT).

### Structure

Code contexts are stored in `problems/code/` with each codebase in its own subdirectory:

```
problems/code/
├── nanogpt/
│   ├── code.txt          # Consolidated source code
│   └── description.txt   # README(s) and documentation
```

### Adding Code Contexts

1. Create a directory: `problems/code/your_codebase/`
2. Create `code.txt` with consolidated source code:
   ```
   # ===== train.py =====
   [file content]

   # ===== model.py =====
   [file content]
   ```
3. Create `description.txt` with README(s) or overview
4. Use with `--code your_codebase`

### Usage

```bash
# ML research with code context
python run_experiment.py \
  --papers attention_is_all_you_need \
  --code nanogpt \
  --problem problems/ml_research.txt \
  --max-iterations 10

# With data files
python run_experiment.py \
  --papers ml_paper \
  --code nanogpt \
  --data training_results.csv \
  --max-iterations 10
```

### Available Codebases

**nanoGPT** - Andrej Karpathy's minimal GPT implementation (~15k tokens)
- Core files: model.py, train.py, sample.py, configurator.py
- Config files for Shakespeare and GPT-2 training
- Data preparation scripts

### GPU Computation with Modal

For GPU-intensive tasks (model training, large-scale inference), the system integrates with [Modal](https://modal.com) for serverless GPU computation.

#### Setup

1. **Install Modal**: Already included in dependencies
2. **Authenticate**: Run `modal token new` to set up your Modal account
3. **Ready to use**: The AI researcher can generate Modal code directly

#### How It Works

The AI researcher generates Python code that uses Modal to:
- Run model inference on cloud GPUs (T4, A10G, A100)
- Execute training jobs with automatic scaling
- Cache models and images for fast subsequent runs

**Key Pattern** (automatically used by the AI):

```python
import modal
import pandas as pd
import matplotlib.pyplot as plt

app = modal.App("experiment-name")

# Include ALL packages used in this file (even if only used locally)
image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers",
    "numpy",
    "pandas",      # Required even if only used locally!
    "matplotlib"   # Required even if only used locally!
)

@app.function(gpu="T4", timeout=1800, image=image)
def run_model_inference(problems):
    """GPU computation - returns data, NOT files"""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    # Load model (Modal caches this after first run)
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Run inference and return data structures
    return results_list  # Return data, not files!

# CRITICAL: Protect execution with if __name__ == "__main__"
if __name__ == "__main__":
    problems = generate_problems()

    # Single Modal session for all GPU work
    with app.run():
        results = run_model_inference.remote(problems)

    # ALL file I/O happens locally
    df = pd.DataFrame(results)
    df.to_csv("artifacts/data/results.csv", index=False)

    plt.plot(df['values'])
    plt.savefig("artifacts/figures/plot.png", dpi=300)
```

#### Performance

**Typical execution times for GPT-2 inference (431 problems)**:
- **First run**: ~2-3 minutes (Modal startup + model download + inference)
- **Subsequent runs**: ~1 minute (cached image + cached model + inference)
- **GPU time only**: ~30-60 seconds

**Caching** (automatic):
- **Image caching**: Container images cached by package hash
- **Model caching**: HuggingFace models cached after first download
- **Persistent**: Caches persist across experiments

#### Critical Patterns

The AI researcher has been trained to follow these Modal best practices:

1. **Include ALL imports in image**: Modal needs all packages you import at the file level, even if only used locally
2. **Use `if __name__ == "__main__":`**: Required to prevent code execution during Modal's import phase
3. **Return data, not files**: Modal functions return Python data structures; all file I/O happens locally
4. **Single Modal session**: Batch all GPU operations to minimize startup overhead

See `docs/MODAL_CODE_PATTERN.md` for complete documentation.

**Configuration** (in `config.yaml`):
```yaml
modal:
  timeout: 3600  # 1 hour for training
  gpu: "T4"      # GPU type (T4, A10G, A100)
```

### Token Efficiency

Code contexts are cached as part of static prompt content:
- **nanoGPT**: ~15k tokens (description + code)
- **First iteration**: 2× cost (cache write)
- **Subsequent iterations**: 0.1× cost (90% savings)
- Efficient for multi-iteration research sessions

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
│   ├── code/                       # Code contexts for ML research (e.g., nanogpt/)
│   ├── open_research.txt           # Default problem for mathematics research
│   └── ml_research.txt             # Default problem for ML research
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

Each research session creates an organized directory in `outputs/` with the following structure:

```
outputs/{session_name}/
├── paper.tex                    # Generated LaTeX paper (root level)
├── paper.pdf                    # Compiled PDF (root level)
├── experiment_code.py           # Python research code (root level)
├── artifacts/
│   ├── figures/                # Agent-generated figures (PNG/PDF)
│   └── data/                   # Agent-generated data files (CSV, pickle, npy)
├── logs/
│   ├── session_log.txt         # Complete execution log
│   ├── plans.txt               # All research plans (append-only history)
│   ├── critiques.txt           # All critic feedback (append-only history)
│   ├── generator_responses.txt # All generator outputs per iteration
│   ├── metrics.json            # API usage and cost metrics
│   ├── current_plan.txt        # Latest plan (for resuming)
│   ├── current_critique.txt    # Latest critique (for resuming)
│   ├── current_researcher_openalex.txt  # Latest researcher literature searches
│   └── current_critic_openalex.txt      # Latest critic literature searches
├── data/                       # Input data files (copied from data/datasets/)
└── literature_cache/           # Downloaded ArXiv papers (if used)
```

### Directory Organization

- **Root level**: Paper and code for easy compilation and execution
- **artifacts/**: Agent-generated outputs (figures, data files)
- **logs/**: System logs, plans, critiques, and metrics
- **data/**: Input data files provided to the experiment
- **literature_cache/**: Cached academic papers from ArXiv

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

**Core Scientific Computing:**
- numpy, scipy, pandas
- matplotlib, seaborn
- networkx, scikit-learn

**Deep Learning:**
- torch, torchvision
- transformer-lens (mechanistic interpretability)
- datasets (HuggingFace datasets including BLIMP)

**GPU-Accelerated Computing:**
- jax (scientific computing with autodiff and GPU support)
- cupy (NumPy-compatible GPU arrays)
- modal (serverless GPU computation)

**Advanced Mathematics:**
- sympy (symbolic mathematics)
- cvxpy (convex optimization)
- numba (JIT compilation)
- einops (tensor operations)

## Available Datasets

Pre-downloaded datasets in `data/datasets/`:
- **MNIST**: Handwritten digits (60k train, 10k test, 28×28 grayscale images)
  - Use with `--data mnist` to include in experiments
  - See `data/datasets/mnist.txt` for loading instructions
- **BLIMP**: 67 linguistic minimal pair datasets for evaluating grammatical knowledge
  - Load dynamically via HuggingFace: `load_dataset('blimp', 'phenomenon_name')`
  - See `data/datasets/blimp.txt` for the full list of phenomena

## Mechanistic Interpretability Research

The system supports mechanistic interpretability research using transformer-lens and the BLIMP dataset.

### Available Tools

**transformer-lens**: Analyze transformer model internals
- Hook access to activations, attention patterns, residual stream components
- Compatible with GPT-2, Pythia, and other HuggingFace models
- Supports activation patching, circuit analysis, and feature visualization

**BLIMP Dataset**: 67 linguistic minimal pair datasets
- Each dataset has 1,000 examples (sentence_good vs sentence_bad)
- Covers: agreement, islands, NPI licensing, binding, filler-gap, ellipsis
- Perfect for evaluating grammatical knowledge and finding circuits

### Example Usage

```bash
# Run mechanistic interpretability research
python run_experiment.py \
  --problem problems/open_mechanistic_interpretability.txt \
  --papers placeholder \
  --max-iterations 30
```

The AI researcher will:
1. Formulate a mechanistic interpretability research question
2. Design experiments using transformer-lens and/or BLIMP
3. Use Modal GPUs for efficient model analysis
4. Generate visualizations and write a research paper

### Example Research Questions

The system encourages autonomous exploration of questions like:
- What circuits detect subject-verb agreement violations?
- How do attention heads implement coreference resolution?
- Where are syntactic structures represented in the residual stream?
- How do models handle long-distance dependencies?

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
