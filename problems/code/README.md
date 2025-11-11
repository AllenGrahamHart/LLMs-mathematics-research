# Code Context Directory

This directory contains codebases that can be provided as context to ML research experiments.

## Structure

Each code context should be a subdirectory with:
- `code.txt` - Consolidated source code with file markers (e.g., `# ===== train.py =====`)
- `description.txt` - README(s) or high-level description of the codebase

## Usage

```bash
# Run experiment with code context
python run_experiment.py \
  --papers attention_is_all_you_need \
  --code nanogpt \
  --problem problems/ml_research.txt \
  --max-iterations 10
```

## Available Codebases

### nanogpt
Andrej Karpathy's nanoGPT - a minimal, educational implementation of GPT for training/finetuning medium-sized models.

- **Description**: ~13.5k chars (~3.4k tokens)
- **Code**: ~48k chars (~12k tokens)
- **Total**: ~15k tokens
- **Files included**: model.py, train.py, sample.py, configurator.py, config files, data preparation scripts

### proton-beam-sde
C++ implementation of stochastic differential equation (SDE) model for proton beam transport through matter. From Crossley et al. (2024) paper on proton therapy dose verification.

- **Description**: ~8.1k chars (~2k tokens) - Repository README
- **Code**: ~69k chars (~17k tokens) - C++ source files, Python cross-section generators, configuration files, material definitions
- **Total**: ~19k tokens
- **Files included**:
  - C++: simulate.cc, proton_beam.cc, material.cc, cross_sections.cc, grid.cc
  - Python: Cross-section generator scripts (elastic, non-elastic)
  - Config: Simulation configurations for water/bone at various energies
  - Materials: Atomic and material composition definitions
  - Note: Large cross-section data files (~130MB) excluded but available in repository

## Adding New Codebases

1. Create a directory: `problems/code/your_codebase/`
2. Create `code.txt` with consolidated source code:
   ```
   # ===== file1.py =====
   [code content]

   # ===== file2.py =====
   [code content]
   ```
3. Create `description.txt` with README(s) or overview
4. Use with `--code your_codebase`
