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
