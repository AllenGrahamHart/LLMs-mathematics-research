"""Code execution utilities."""

import os
import subprocess
from typing import Dict, Any, Union
from ..config import CONFIG


def execute_code(code: str, output_dir: str) -> Dict[str, Union[bool, str]]:
    """
    Execute Python code by writing to file and running as subprocess.

    This approach allows Modal and other tools that need to inspect source code
    to work correctly, since the code exists as a real file on disk. The code is
    written to 'experiment_code.py' in the output directory and executed with a
    configurable timeout.

    Args:
        code (str): Python code to execute. Will be written to experiment_code.py
            before execution.
        output_dir (str): Directory for saving outputs/figures. The code runs in this
            directory with it as the current working directory, allowing relative paths
            like 'artifacts/figures/plot.png' to work correctly.

    Returns:
        Dict[str, Union[bool, str]]: Dictionary containing:
            - 'success' (bool): True if code executed without errors (return code 0)
            - 'output' (str): Combined stdout and stderr from execution, plus diagnostics
              about generated figures
            - 'timeout' (bool): True if execution exceeded the timeout limit

    Examples:
        >>> result = execute_code("print('Hello')", "/tmp/output")
        >>> result['success']
        True
        >>> result['output']
        'Hello\\n\\nFigures in output_dir: (none)'

        >>> result = execute_code("import time; time.sleep(10000)", "/tmp/output")
        >>> result['timeout']
        True
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
