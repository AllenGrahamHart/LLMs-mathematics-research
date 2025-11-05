"""Code execution utilities."""

import os
import subprocess
from typing import Dict, Any
from ..config import CONFIG


def execute_code(code: str, output_dir: str) -> Dict[str, Any]:
    """
    Execute Python code by writing to file and running as subprocess.

    This approach allows Modal and other tools that need to inspect source code
    to work correctly, since the code exists as a real file on disk.

    Args:
        code: Python code to execute
        output_dir: Directory for saving outputs/figures (code runs in this directory)

    Returns:
        Dictionary with 'success', 'output', and 'timeout' keys
    """
    # Write code to a file in the output directory
    # Use 'experiment_code.py' to avoid name collision with Python's stdlib 'code' module
    code_file = os.path.join(output_dir, 'experiment_code.py')

    with open(code_file, 'w', encoding='utf-8') as f:
        f.write(code)

    # Run the code as a subprocess with timeout
    timeout_seconds = CONFIG['execution']['timeout']

    try:
        result = subprocess.run(
            ['python', 'experiment_code.py'],
            cwd=output_dir,  # Run in the output directory
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )

        # Check for figures created
        try:
            imgs = sorted([p for p in os.listdir(output_dir)
                          if p.lower().endswith(('.png', '.pdf', '.jpg', '.jpeg', '.svg'))])
            diag = "\nFigures in output_dir: " + (", ".join(imgs) if imgs else "(none)")
        except Exception:
            diag = ""

        output = result.stdout + result.stderr + diag

        return {
            'success': result.returncode == 0,
            'output': output,
            'timeout': False
        }

    except subprocess.TimeoutExpired as e:
        # Collect any output that was produced before timeout
        stdout = e.stdout.decode('utf-8') if e.stdout else ''
        stderr = e.stderr.decode('utf-8') if e.stderr else ''

        return {
            'success': False,
            'output': stdout + stderr + f'\nCode did not finish running in {timeout_seconds} seconds',
            'timeout': True
        }
    except Exception as e:
        return {
            'success': False,
            'output': f'{type(e).__name__}: {str(e)}',
            'timeout': False
        }
