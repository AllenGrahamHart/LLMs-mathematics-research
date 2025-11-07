"""LaTeX compilation utilities."""

import subprocess
from typing import Dict, Any, Union
from ..config import CONFIG


def compile_latex(output_dir: str) -> Dict[str, Union[bool, str]]:
    """
    Compile LaTeX to PDF using pdflatex.

    Runs pdflatex on paper.tex in the specified directory with non-interactive mode
    and a configurable timeout. The generated PDF will be named paper.pdf.

    Args:
        output_dir (str): Directory containing paper.tex. The pdflatex command will
            run in this directory, and paper.pdf will be generated here.

    Returns:
        Dict[str, Union[bool, str]]: Dictionary containing:
            - 'success' (bool): True if compilation succeeded (return code 0)
            - 'error' (str, optional): Error message from pdflatex stdout if compilation
              failed, or exception message if an error occurred. Only present when
              success is False.

    Examples:
        >>> result = compile_latex("/path/to/output")
        >>> if result['success']:
        ...     print("PDF generated successfully")
        >>> else:
        ...     print(f"Compilation failed: {result['error']}")

    Note:
        Requires pdflatex to be installed and available in PATH.
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
