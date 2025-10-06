"""LaTeX compilation utilities."""

import subprocess
from typing import Dict, Any
from ..config import CONFIG


def compile_latex(output_dir: str) -> Dict[str, Any]:
    """
    Compile LaTeX to PDF.

    Args:
        output_dir: Directory containing paper.tex

    Returns:
        Dictionary with 'success' bool and optional 'error' message
    """
    try:
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', 'paper.tex'],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=CONFIG['compilation']['timeout']
        )

        if result.returncode == 0:
            return {'success': True}
        else:
            return {'success': False, 'error': result.stdout}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def extract_latex_content(text: str) -> str | None:
    """
    Extract LaTeX content from response text.

    Args:
        text: Response text that may contain LaTeX code blocks

    Returns:
        LaTeX content if found, None otherwise
    """
    if r'\documentclass' in text:
        if '```latex' in text or '```tex' in text:
            parts = text.split('```')
            for i in range(1, len(parts), 2):
                block = parts[i]
                if block.startswith('latex\n'):
                    return block[6:]  # Remove 'latex\n'
                elif block.startswith('tex\n'):
                    return block[4:]  # Remove 'tex\n'
        return text
    return None
