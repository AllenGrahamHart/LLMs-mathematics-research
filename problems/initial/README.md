# Custom Initial Files

This directory contains custom initial paper and code files that can be used to start a research session with pre-existing content.

## Usage

Use the `--initial-paper` and `--initial-code` flags when running experiments:

```bash
python run_experiment.py --problem problems/open_research.txt \
    --initial-paper my_paper.tex \
    --initial-code my_code.py
```

## File Types

- **Initial Paper** (`.tex`): A LaTeX document that will be used as the starting point for the paper instead of the default template
- **Initial Code** (`.py`): Python code that will be used as the starting point for the experiment code instead of an empty file

## Notes

- Files are optional - if not provided, the default template and empty code will be used
- Initial paper files should be valid LaTeX documents
- Initial code files should be valid Python code
