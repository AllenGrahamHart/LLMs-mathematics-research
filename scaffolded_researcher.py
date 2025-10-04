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
        self.critique_file = os.path.join(self.output_dir, "critiques.txt")
        self.plans_file = os.path.join(self.output_dir, "plans.txt")

        self._initialize_files()
        self.log = []
        self.api_metrics = []
        self._last_call_time = None
        self.last_execution_output = ""
        self.current_plan = "Begin research"
        self.current_critique = "Good luck!"
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
            'plan': self.current_plan,
            'critique': self.current_critique
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
        # Look for common patterns (multiline)
        patterns = [
            r'PLAN:\s*(.+?)(?:\n\n|\Z)',
            r'Next iteration:\s*(.+?)(?:\n\n|\Z)',
            r'Next:\s*(.+?)(?:\n\n|\Z)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        # Default: extract last non-empty line
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            return lines[-1]

        return "Continue research"
    
    def execute_code(self, code: str):
        """Execute code and capture output with configurable timeout"""
        import threading

        stdout_capture = StringIO()
        stderr_capture = StringIO()

        # --- Safe savefig helper & monkey-patch ---
        import os as _os
        _abs_out = _os.path.abspath(self.output_dir)
        _orig_savefig = plt.savefig

        def _safe_savefig(path, *args, **kwargs):
            base = str(path)
            if _os.path.isabs(base):
                final = base
            else:
                final = _os.path.join(_abs_out, _os.path.basename(base))
            _os.makedirs(_os.path.dirname(final), exist_ok=True)
            _orig_savefig(final, *args, **kwargs)
            try:
                rel = _os.path.relpath(final, _abs_out)
            except Exception:
                rel = final
            print(f"✓ Saved figure -> {final} (relative: {rel})")

        plt.savefig = _safe_savefig

        namespace = {
            'np': __import__('numpy'),
            'plt': plt,
            'matplotlib': matplotlib,
            'output_dir': self.output_dir,
            'savefig': _safe_savefig,
            'plt_savefig': _safe_savefig,
        }

        result = {'success': False, 'output': '', 'timeout': False}

        def run_code():
            try:
                with contextlib.redirect_stdout(stdout_capture), \
                     contextlib.redirect_stderr(stderr_capture):
                    exec(code, namespace)

                for fig_num in plt.get_fignums():
                    plt.close(fig_num)

                try:
                    imgs = sorted([p for p in os.listdir(self.output_dir)
                                   if p.lower().endswith(('.png', '.pdf', '.jpg', '.jpeg', '.svg'))])
                    diag = "\nFigures in output_dir: " + (", ".join(imgs) if imgs else "(none)")
                except Exception as _:
                    diag = ""

                result['success'] = True
                result['timeout'] = False
                result['output'] = stdout_capture.getvalue() + stderr_capture.getvalue() + diag
            except Exception as e:
                try:
                    imgs = sorted([p for p in os.listdir(self.output_dir)
                                   if p.lower().endswith(('.png', '.pdf', '.jpg', '.jpeg', '.svg'))])
                    diag = "\nFigures in output_dir: " + (", ".join(imgs) if imgs else "(none)")
                except Exception as _:
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
    
    def write_critique(self, iteration: int, critique: str):
        """Append critique to critique log"""
        with open(self.critique_file, 'a') as f:
            f.write(f"\n{'='*60}\nITERATION {iteration} CRITIQUE\n{'='*60}\n")
            f.write(critique + "\n")

    def write_plan(self, iteration: int, plan: str):
        """Append plan to plans log"""
        with open(self.plans_file, 'a') as f:
            f.write(f"\n{'='*60}\nITERATION {iteration} PLAN\n{'='*60}\n")
            f.write(plan + "\n")
    
    def can_make_api_call(self) -> bool:
        """Test if we can make an API call with absolute minimal cost."""
        try:
            self.client.messages.create(
                model=CONFIG['api']['model'],
                max_tokens=1,
                messages=[{"role": "user", "content": "x"}]
            )
            return True
        except anthropic.RateLimitError:
            return False

    def call_claude(self, prompt: str):
        """Call Claude API with streaming"""
        wait_time = CONFIG['api']['rate_limit_wait']

        params = {
            "model": CONFIG['api']['model'],
            "max_tokens": CONFIG['api']['max_tokens'],
            "thinking": {
                "type": "enabled",
                "budget_tokens": CONFIG['api']['thinking_budget']
            },
            "messages": [{"role": "user", "content": prompt}]
        }

        # Retry loop with rate limit handling
        while True:
            try:
                start_time = time.time()

                # Use streaming for long requests
                response_text = ""
                input_tokens = 0
                output_tokens = 0

                with self.client.messages.stream(**params) as stream:
                    for text in stream.text_stream:
                        response_text += text

                    # Get final message for usage stats
                    final_message = stream.get_final_message()
                    input_tokens = final_message.usage.input_tokens
                    output_tokens = final_message.usage.output_tokens

                end_time = time.time()
                break
            except anthropic.RateLimitError:
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

        self._last_call_time = end_time
        response_time = end_time - start_time

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
            
            full_code = "\n\n".join(code_blocks)
            
            with open(self.python_file, 'w') as f:
                f.write(full_code)
            
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
        self.write_plan(iteration, self.current_plan)

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

    def build_generator_prompt(self, iteration: int, state: dict):
        """Build the prompt for generator"""
        experimental_design = f"""You are part way through the process of autonomously writing a research paper.

This prompt, your reply, and comments from a critic AI, together form 1 iteration in a multi-iteration research loop. 
The specific research problem you are working on is:

{self.problem_statement}

Each iteration, including this one:
1. You will receive the current state (LaTeX paper, code, execution output, your previous plan, a critique)
2. Based on the paper, code, your previous plan, and external critique, you will create a detailed plan for the remaining iterations
3. You will output ONLY your updated plan, python code and LaTeX
4. If you have 0 remaining iterations, then the code and LaTeX created this iteration is final
5. If your code does not finish running after {CONFIG['execution']['timeout']} seconds it will terminate.

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

--- Critique from Critic ---
{state['critique']}
"""
        return experimental_design + state_description

    def build_critic_prompt(self, iteration: int, state: dict, generator_response: str):
        """Build the prompt for critic"""
        critic_prompt = f"""You are a research critic evaluating work in progress by an AI. Your role is to provide constructive, severity-graded feedback that helps improve the research.

The AI researcher is working on:
{self.problem_statement}

Your critique is part of an AI researcher-critic scaffolding loop with a fixed iteration budget.
Current iteration: {iteration} / {self.max_iterations}
Iterations remaining: {self.max_iterations - iteration}

Your job is to:
1. Identify errors and issues in the current work
2. Grade their severity (FATAL, SERIOUS, MINOR)
3. Provide actionable feedback appropriate to the remaining iteration budget
4. Decide whether work can proceed or must be revised

ERROR SEVERITY LEVELS - with examples:

FATAL (must be fixed or removed):
- Theorems with counterexamples to the core claim (not just edge cases)
- Proofs that are fundamentally wrong (key ideas are unworkable)
- Numerical experiments that are clearly nonesensical or irrelavent

SERIOUS (should fix, but core idea may be salvageable):
- Theorems that are incorrect but could be saved with small changes
- Proofs with fillable gaps or plausible sketch proofs
- Unclear but plausible connection between theory and experiment
- Experiments not explained clearly enough for replication

MINOR (could fix):
- Unclear or unecessary sentences
- Undefined terms, or unstated conditions that are arguably obvious from context
- Suboptimal presentation or style

=== CURRENT WORK TO CRITIQUE ===

--- LaTeX Paper ---
{state['latex']}

--- LaTeX Compilation ---
{state['compilation']}

--- Python Code ---
{state['python']}

--- Execution Output ---
{state['execution_output']}

--- Researcher's Plan ---
{state['plan']}

--- Researcher's Latest Response ---
{generator_response}

=== YOUR CRITIQUE ===

Provide your critique in this format:

FATAL ERRORS:
[List any fatal errors with clear explanations. If none, write "None identified."]

SERIOUS ISSUES:
[List serious issues with repair suggestions. If none, write "None identified."]

MINOR CONCERNS:
[List minor concerns. If none, write "None identified."]

RECOMMENDATION:
[Recommend revisions to the researcher's plan consistent with the remaining iteration budget.
If there are 0 iterations remaining, the paper is already complete and your critique is final.]
"""
        return critic_prompt

    def run(self, problem: str):
        """Main research loop with generator-critic interaction"""
        self.problem_statement = problem

        print("="*60)
        print("SCAFFOLDED RESEARCH WITH CRITIC")
        print("="*60)
        print(f"Max iterations: {self.max_iterations}")
        print(f"Output directory: {self.session.output_dir}")
        print("="*60)

        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1

            print(f"\n{'='*60}")
            print(f"ITERATION {self.current_iteration}/{self.max_iterations}")
            print("="*60)

            # Get current state
            state = self.session.get_state()

            # GENERATOR PHASE
            print("\n[GENERATOR]")
            generator_prompt = self.build_generator_prompt(self.current_iteration, state)
            generator_response = self.session.call_claude(generator_prompt)
            
            # Process generator response
            self.session.process_response(generator_response, self.current_iteration)

            # CRITIC PHASE
            print("\n[CRITIC]")
            # Get updated state after generator's changes
            state = self.session.get_state()
            critic_prompt = self.build_critic_prompt(self.current_iteration, state, generator_response)
            critic_response = self.session.call_claude(critic_prompt)

            # Store critique for next iteration
            self.session.current_critique = critic_response
            self.session.write_critique(self.current_iteration, critic_response)

            # Print critique summary
            print("\n--- Critique Summary ---")
            if "FATAL ERRORS:" in critic_response:
                parts = critic_response.split("FATAL ERRORS:")
                if len(parts) > 1:
                    fatal_parts = parts[1].split("SERIOUS ISSUES:")
                    fatal_section = fatal_parts[0].strip()
                    if fatal_section and "None identified" not in fatal_section:
                        print(f"⚠️  FATAL ERRORS FOUND")
                        print(fatal_section[:200] + "..." if len(fatal_section) > 200 else fatal_section)

            if "RECOMMENDATION:" in critic_response:
                parts = critic_response.split("RECOMMENDATION:")
                if len(parts) > 1:
                    rec_lines = parts[1].strip().split("\n")
                    rec = rec_lines[0].strip() if rec_lines else ""
                    if rec:
                        print(f"Recommendation: {rec}")

            print(f"\n[Iteration {self.current_iteration} complete]")

        # Final compilation
        print("\n" + "="*60)
        print("COMPILING FINAL PAPER")
        print("="*60)

        compile_result = self.session.compile_latex()
        if compile_result['success']:
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