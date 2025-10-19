"""Code execution utilities."""

import os
import threading
from io import StringIO
import contextlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Any
from ..config import CONFIG


def execute_code(code: str, output_dir: str) -> Dict[str, Any]:
    """
    Execute Python code and capture output with configurable timeout.

    Args:
        code: Python code to execute
        output_dir: Directory for saving outputs/figures

    Returns:
        Dictionary with 'success', 'output', and 'timeout' keys

    Note:
        Uses daemon threads for timeout enforcement. If timeout is reached, the thread
        continues running in the background until process exit (Python limitation - threads
        cannot be forcibly killed). This is acceptable for our use case since the process
        will eventually exit, but users should be aware that timeouts don't immediately
        terminate runaway computations.
    """
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    # --- Safe savefig helper & monkey-patch ---
    _abs_out = os.path.abspath(output_dir)
    _orig_savefig = plt.savefig

    def _safe_savefig(path: Any, *args: Any, **kwargs: Any) -> None:
        base = str(path)
        # Always save to output_dir, just use basename of the provided path
        final = os.path.join(_abs_out, os.path.basename(base))
        os.makedirs(os.path.dirname(final), exist_ok=True)
        _orig_savefig(final, *args, **kwargs)
        try:
            rel = os.path.relpath(final, _abs_out)
        except Exception:
            rel = final
        print(f"✓ Saved figure -> {final} (relative: {rel})")

    plt.savefig = _safe_savefig

    namespace = {
        '__name__': '__main__',
        'np': __import__('numpy'),
        'plt': plt,
        'matplotlib': matplotlib,
        'output_dir': output_dir,
        'savefig': _safe_savefig,
        'plt_savefig': _safe_savefig,
    }

    result = {'success': False, 'output': '', 'timeout': False}

    def run_code() -> None:
        try:
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):
                exec(code, namespace)

            for fig_num in plt.get_fignums():
                plt.close(fig_num)

            try:
                imgs = sorted([p for p in os.listdir(output_dir)
                               if p.lower().endswith(('.png', '.pdf', '.jpg', '.jpeg', '.svg'))])
                diag = "\nFigures in output_dir: " + (", ".join(imgs) if imgs else "(none)")
            except Exception:
                diag = ""

            result['success'] = True
            result['timeout'] = False
            result['output'] = stdout_capture.getvalue() + stderr_capture.getvalue() + diag
        except Exception as e:
            try:
                imgs = sorted([p for p in os.listdir(output_dir)
                               if p.lower().endswith(('.png', '.pdf', '.jpg', '.jpeg', '.svg'))])
                diag = "\nFigures in output_dir: " + (", ".join(imgs) if imgs else "(none)")
            except Exception:
                diag = ""
            result['success'] = False
            result['timeout'] = False
            result['output'] = stdout_capture.getvalue() + stderr_capture.getvalue() + \
                             f"\n{type(e).__name__}: {str(e)}" + diag

    thread = threading.Thread(target=run_code, daemon=True)
    thread.start()
    timeout_seconds = CONFIG['execution']['timeout']
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        return {
            'success': False,
            'output': stdout_capture.getvalue() + stderr_capture.getvalue() +
                     f'\nCode did not finish running in {timeout_seconds} seconds',
            'timeout': True
        }

    return result


def extract_code_blocks(text: str) -> list[str]:
    """
    Extract Python code blocks from text.

    Args:
        text: Text containing code blocks with ```python``` markers

    Returns:
        List of extracted code blocks
    """
    blocks = []
    parts = text.split('```')
    for i in range(1, len(parts), 2):
        block = parts[i]
        if block.startswith('python\n'):
            blocks.append(block[7:])
        elif block.startswith('python '):
            blocks.append(block[7:])
    return blocks
