import os
import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv
from io import StringIO
import contextlib
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import json
import time
import re
import yaml

load_dotenv()
client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Load configuration
with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

class ResearchSession:
    def __init__(self, session_name: str):
        self.session_name = session_name
        self.output_dir = f"outputs/{session_name}"
        os.makedirs(self.output_dir, exist_ok=True)

        self.latex_file = os.path.join(self.output_dir, "paper.tex")
        self.python_file = os.path.join(self.output_dir, "code.py")
        self.log_file = os.path.join(self.output_dir, "session_log.txt")
        self.metrics_file = os.path.join(self.output_dir, "metrics.json")

        self._initialize_files()
        self.log = []
        self.api_metrics = []
        self._last_call_time = None
        self.last_execution_output = ""
        self.current_plan = "Begin research"
        self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    
    def _initialize_files(self):
        """Create initial LaTeX and Python files"""
        initial_latex = r"""\documentclass{article}
\usepackage{amsmath}
\usepackage{amsthm}
\usepackage{graphicx}
\usepackage{hyperref}

\title{Research Paper}
\author{Claude}
\date{\today}

\begin{document}
\maketitle

% Content will be added here

\end{document}
"""
        with open(self.latex_file, 'w') as f:
            f.write(initial_latex)
        
        initial_python = "# Research code will be added here\n"
        with open(self.python_file, 'w') as f:
            f.write(initial_python)
    
    def get_state(self):
        """Get current state as dict"""
        # Read files
        with open(self.latex_file, 'r') as f:
            latex_content = f.read()
        
        with open(self.python_file, 'r') as f:
            python_content = f.read()
        
        # Get compilation status
        compile_result = self.compile_latex()
        if compile_result['success']:
            compilation_status = "✓ Compiled successfully"
        else:
            error_limit = CONFIG['compilation']['error_limit']
            compilation_status = f"✗ Compilation failed:\n{compile_result['error'][:error_limit]}"
        
        return {
            'latex': latex_content,
            'compilation': compilation_status,
            'python': python_content,
            'execution_output': self.last_execution_output,
            'plan': self.current_plan
        }
    
    def extract_code_blocks(self, text: str):
        """Extract Python code blocks"""
        blocks = []
        parts = text.split('```')
        for i in range(1, len(parts), 2):
            block = parts[i]
            if block.startswith('python\n'):
                blocks.append(block[7:])
            elif block.startswith('python '):
                blocks.append(block[7:])
        return blocks
    
    def extract_latex_content(self, text: str):
        """Extract LaTeX content from response"""
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
    
    def extract_plan(self, text: str):
        """Extract next iteration plan from response"""
        # Look for common patterns
        patterns = [
            r'PLAN:\s*(.+?)(?:\n|$)',
            r'Next iteration:\s*(.+?)(?:\n|$)',
            r'Next:\s*(.+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Default: extract last non-empty line
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            return lines[-1][:CONFIG['output']['plan_limit']]

        return "Continue research"
    
    def execute_code(self, code: str):
        """Execute code and capture output with configurable timeout"""
        import threading

        stdout_capture = StringIO()
        stderr_capture = StringIO()

        # --- Safe savefig helper & monkey-patch ---
        # Redirects any relative/incorrect paths into the current output_dir,
        # creates missing dirs, and prints where the figure actually went.
        import os as _os
        _abs_out = _os.path.abspath(self.output_dir)
        _orig_savefig = plt.savefig

        def _safe_savefig(path, *args, **kwargs):
            base = str(path)
            if _os.path.isabs(base):
                final = base
            else:
                # Any relative path -> drop directories and place into output_dir
                # (prevents writing into stale or wrong session folders)
                final = _os.path.join(_abs_out, _os.path.basename(base))
            _os.makedirs(_os.path.dirname(final), exist_ok=True)
            _orig_savefig(final, *args, **kwargs)
            try:
                rel = _os.path.relpath(final, _abs_out)
            except Exception:
                rel = final
            print(f"✓ Saved figure -> {final} (relative: {rel})")

        # Monkey-patch and expose helper in the user namespace
        plt.savefig = _safe_savefig  # catch plain plt.savefig(...)

        namespace = {
            'np': __import__('numpy'),
            'plt': plt,
            'matplotlib': matplotlib,
            'output_dir': self.output_dir,
            'savefig': _safe_savefig,     # preferred helper
            'plt_savefig': _safe_savefig, # alias, just in case
        }

        result = {'success': False, 'output': '', 'timeout': False}

        def run_code():
            try:
                with contextlib.redirect_stdout(stdout_capture), \
                     contextlib.redirect_stderr(stderr_capture):
                    exec(code, namespace)

                for fig_num in plt.get_fignums():
                    plt.close(fig_num)

                # Post-exec diagnostic: list figures in output_dir
                try:
                    imgs = sorted([p for p in os.listdir(self.output_dir)
                                   if p.lower().endswith(('.png', '.pdf', '.jpg', '.jpeg', '.svg'))])
                    diag = "\nFigures in output_dir: " + (", ".join(imgs) if imgs else "(none)")
                except Exception as _:
                    diag = ""

                result['success'] = True
                result['output'] = stdout_capture.getvalue() + stderr_capture.getvalue() + diag
            except Exception as e:
                # Post-exec diagnostic even on failure
                try:
                    imgs = sorted([p for p in os.listdir(self.output_dir)
                                   if p.lower().endswith(('.png', '.pdf', '.jpg', '.jpeg', '.svg'))])
                    diag = "\nFigures in output_dir: " + (", ".join(imgs) if imgs else "(none)")
                except Exception as _:
                    diag = ""
                result['success'] = False
                result['output'] = stdout_capture.getvalue() + stderr_capture.getvalue() + \
                                 f"\n{type(e).__name__}: {str(e)}" + diag

        thread = threading.Thread(target=run_code, daemon=True)
        thread.start()
        timeout_seconds = CONFIG['execution']['timeout']
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            # Timeout occurred
            return {
                'success': False,
                'output': stdout_capture.getvalue() + stderr_capture.getvalue() +
                         f'\nCode did not finish running in {timeout_seconds} seconds',
            }

        return result
    
    def compile_latex(self):
        """Compile LaTeX to PDF"""
        try:
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', 'paper.tex'],
                cwd=self.output_dir,
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
    
    def write_log(self, entry: str):
        """Append to session log"""
        self.log.append(entry)
        with open(self.log_file, 'a') as f:
            f.write(entry + "\n")
    
    def can_make_api_call(self) -> bool:
        """Test if we can make an API call with absolute minimal cost."""
        try:
            self.client.messages.create(
                model=CONFIG['api']['model'],
                max_tokens=1,  # Output limited to 1 token
                messages=[{"role": "user", "content": "x"}]  # Input: 1 token
            )
            return True
        except anthropic.RateLimitError:
            return False

    def call_claude(self, prompt: str):
        """Call Claude API"""

        # Test if we can make a call
        wait_time = CONFIG['api']['rate_limit_wait']
        while not self.can_make_api_call():
            print(f"  Rate limited, waiting {wait_time}s...")
            time.sleep(wait_time)

        params = {
            "model": CONFIG['api']['model'],
            "max_tokens": CONFIG['api']['max_tokens'],
            "thinking": {
                "type": "enabled",
                "budget_tokens": CONFIG['api']['thinking_budget']
            },
            "messages": [{"role": "user", "content": prompt}]  # Single message, no history
        }

        start_time = time.time()
        response = self.client.messages.create(**params)
        end_time = time.time()
        
        self._last_call_time = end_time
        response_time = end_time - start_time

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        input_cost = (input_tokens / 1_000_000) * CONFIG['api']['costs']['input_per_million']
        output_cost = (output_tokens / 1_000_000) * CONFIG['api']['costs']['output_per_million']
        total_cost = input_cost + output_cost

        metrics_entry = {
            'timestamp': datetime.now().isoformat(),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'response_time': response_time,
            'cost': total_cost
        }
        self.api_metrics.append(metrics_entry)

        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text += block.text

        print(f"  Input: {input_tokens:,} tokens | Output: {output_tokens:,} tokens | Cost: ${total_cost:.4f}")

        return response_text
    
    def process_response(self, response: str, iteration: int):
        """Process Claude's response and update state"""
        self.write_log(f"\n{'='*60}\nITERATION {iteration}\n{'='*60}")
        self.write_log(f"Response:\n{response}\n")
        
        # Extract and execute code
        code_blocks = self.extract_code_blocks(response)
        if code_blocks:
            self.write_log(f"Found {len(code_blocks)} code block(s)")
            
            # Concatenate all code blocks
            full_code = "\n\n".join(code_blocks)
            
            # Save to file
            with open(self.python_file, 'w') as f:
                f.write(full_code)
            
            # Execute
            exec_result = self.execute_code(full_code)
            output_limit = CONFIG['execution']['output_limit']
            if exec_result['success']:
                self.write_log("✓ Code executed successfully")
                self.last_execution_output = exec_result['output'][:output_limit]
            else:
                self.write_log("✗ Code execution failed")
                self.last_execution_output = exec_result['output'][:output_limit]

            self.write_log(f"Output:\n{self.last_execution_output}")
        else:
            self.last_execution_output = "No code executed this iteration"
        
        # Extract and save LaTeX
        latex_content = self.extract_latex_content(response)
        if latex_content:
            with open(self.latex_file, 'w') as f:
                f.write(latex_content)
            self.write_log("✓ LaTeX file updated")
        
        # Extract plan for next iteration
        self.current_plan = self.extract_plan(response)
        self.write_log(f"Next plan: {self.current_plan}")

    def get_metrics_summary(self):
        """Get summary of all API metrics"""
        if not self.api_metrics:
            return {
                'total_calls': 0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_time': 0.0,
                'total_cost': 0.0
            }

        total_input_tokens = sum(m['input_tokens'] for m in self.api_metrics)
        total_output_tokens = sum(m['output_tokens'] for m in self.api_metrics)
        total_time = sum(m['response_time'] for m in self.api_metrics)
        total_cost = sum(m['cost'] for m in self.api_metrics)

        summary = {
            'total_calls': len(self.api_metrics),
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_time': total_time,
            'total_cost': total_cost,
            'individual_calls': self.api_metrics
        }

        with open(self.metrics_file, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary


class ScaffoldedResearcher:
    def __init__(self, session_name: str, max_iterations: int = 20):
        self.session = ResearchSession(session_name)
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.problem_statement = ""

    def build_prompt(self, iteration: int, state: dict):
        """Build the prompt for current iteration"""
        experimental_design = f"""You are part way through the process of autonomously writing a research paper.

This is prompt and your reply together form 1 iteration in a multi-iteration research loop. 
The specific research problem you are working on is:

{self.problem_statement}

Each iteration, including this one:
1. You will receive the current state (LaTeX paper, code, execution output, your previous plan)
2. Based on the paper, code and previous plan, you will create a detailed plan for the remaining iterations
3. You will output ONLY your updated plan, python code and LaTeX
4. If you are have 0 remaining iterations, then the code and LaTeX created this iteration is final
5. If your code does not finish running after 10 minutes it will terminate.

When saving figures, ALWAYS use either `savefig("name.png", dpi={CONFIG['output']['figure_dpi']})`
   or `plt.savefig(os.path.join(output_dir, "name.png"), dpi={CONFIG['output']['figure_dpi']})`.
   Do NOT hard-code session paths. Figures must end up in the same directory as `paper.tex`.

OUTPUT FORMAT:
- PLAN: [detailed plan over the remaining iterations]
- Python code: ```python ... ```
- LaTeX: ```latex ... ``` (must be complete document with \\documentclass)
"""
        state_description = f"""
=== YOUR CURRENT STATE ===

Current iteration: {iteration} / {self.max_iterations}
Iterations remaining after this one: {self.max_iterations - iteration}

--- LaTeX Paper ---
{state['latex']}

--- LaTeX Compilation Status ---
{state['compilation']}

--- Python Code ---
{state['python']}

--- Last Execution Output ---
{state['execution_output']}

--- Plan from Previous Iteration ---
{state['plan']}
"""
        return experimental_design + state_description

    def run(self, problem: str):
        """Main research loop (no separate initial prompt)"""
        self.problem_statement = problem

        print("="*60)
        print("STATELESS SCAFFOLDED RESEARCH")
        print("="*60)
        print(f"Max iterations: {self.max_iterations}")
        print(f"Output directory: {self.session.output_dir}")
        print("="*60)

        # Jump straight into the uniform iteration loop
        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1

            print(f"\n{'='*60}")
            print(f"ITERATION {self.current_iteration}/{self.max_iterations}")
            print("="*60)

            # Get current state
            state = self.session.get_state()

            # Build the SAME style prompt every time
            prompt = self.build_prompt(self.current_iteration, state)

            # Call Claude (stateless - no history)
            response = self.session.call_claude(prompt)

            # Process response and update state
            self.session.process_response(response, self.current_iteration)

            print(f"[Iteration {self.current_iteration} complete]")

        # Final compilation
        print("\n" + "="*60)
        print("COMPILING FINAL PAPER")
        print("="*60)

        compile_result = self.session.compile_latex()
        if compile_result['success']:
            # Run twice for references
            compile_result = self.session.compile_latex()
            print(f"✓ PDF generated: {os.path.join(self.session.output_dir, 'paper.pdf')}")
        else:
            print(f"✗ PDF compilation failed")
            error_limit = CONFIG['compilation']['error_limit']
            print(f"Error: {compile_result.get('error', 'Unknown error')[:error_limit]}")

        # Print metrics
        print("\n" + "="*60)
        print("METRICS SUMMARY")
        print("="*60)
        metrics = self.session.get_metrics_summary()
        print(f"Total API calls: {metrics['total_calls']}")
        print(f"Total input tokens: {metrics['total_input_tokens']:,}")
        print(f"Total output tokens: {metrics['total_output_tokens']:,}")
        print(f"Total time: {metrics['total_time']:.2f}s")
        print(f"Total cost: ${metrics['total_cost']:.4f}")
        if metrics['total_calls'] > 0:
            print(f"Average input per call: {metrics['total_input_tokens'] // metrics['total_calls']:,} tokens")
            print(f"Average output per call: {metrics['total_output_tokens'] // metrics['total_calls']:,} tokens")
        print(f"\n=== EXPERIMENT COMPLETE ===")
        print(f"Output directory: {self.session.output_dir}")
