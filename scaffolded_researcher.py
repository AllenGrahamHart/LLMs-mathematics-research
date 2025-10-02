import os
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

load_dotenv()
client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

class ResearchSession:
    def __init__(self, session_name: str):
        self.session_name = session_name
        self.output_dir = f"outputs/{session_name}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.latex_file = os.path.join(self.output_dir, "paper.tex")
        self.log_file = os.path.join(self.output_dir, "session_log.txt")
        
        self._initialize_latex()
        self.log = []
        self.conversation_history = []  # Add conversation history
    
    def _initialize_latex(self):
        """Create initial LaTeX structure"""
        initial_content = r"""\documentclass{article}
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
            f.write(initial_content)
    
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
            else:
                blocks.append(block.strip())
        return blocks
    
    def extract_latex_content(self, text: str):
        """Extract LaTeX content from response"""
        if r'\documentclass' in text:
            if '```latex' in text or '```tex' in text:
                parts = text.split('```')
                for i in range(1, len(parts), 2):
                    block = parts[i]
                    if block.startswith('latex\n') or block.startswith('tex\n'):
                        return block.split('\n', 1)[1] if '\n' in block else block
            return text
        return None
    
    def execute_code(self, code: str, block_num: int):
        """Execute code and save any plots"""
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        namespace = {
            'np': __import__('numpy'),
            'plt': plt,
            'matplotlib': matplotlib,
            'output_dir': self.output_dir,  # Expose output_dir to code
        }

        try:
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):
                exec(code, namespace)

            # Close any remaining figures without saving
            # (Claude should handle saving explicitly in the code)
            for fig_num in plt.get_fignums():
                plt.close(fig_num)

            return {
                'success': True,
                'stdout': stdout_capture.getvalue(),
                'stderr': stderr_capture.getvalue(),
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': stdout_capture.getvalue(),
                'stderr': stderr_capture.getvalue() + f"\n{type(e).__name__}: {str(e)}",
            }
    
    def update_latex(self, new_content: str):
        """Replace LaTeX file with new content"""
        with open(self.latex_file, 'w') as f:
            f.write(new_content)
    
    def compile_latex(self):
        """Compile LaTeX to PDF"""
        try:
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', 'paper.tex'],
                cwd=self.output_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Run second time for references
                subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', 'paper.tex'],
                    cwd=self.output_dir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
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
    
    def call_claude(self, prompt: str, use_thinking: bool = False):
        """Call Claude API with conversation history"""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": prompt
        })
        
        params = {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 16000,
            "messages": self.conversation_history  # Use full conversation history
        }
        
        if use_thinking:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": 10000
            }
        
        response = client.messages.create(**params)
        
        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text += block.text
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text
        })
        
        return response_text
    
    def process_response(self, response: str, iteration: int):
        """Process Claude's response"""
        self.write_log(f"\n{'='*60}\nITERATION {iteration}\n{'='*60}")
        self.write_log(f"Claude's response:\n{response}\n")
        
        results = {
            'code_executed': False,
            'latex_updated': False,
            'figures_generated': [],
            'execution_results': []
        }
        
        # Execute code blocks
        code_blocks = self.extract_code_blocks(response)
        if code_blocks:
            results['code_executed'] = True
            self.write_log(f"\nFound {len(code_blocks)} code block(s)")

            for i, code in enumerate(code_blocks):
                self.write_log(f"\n--- Executing code block {i+1} ---")
                exec_result = self.execute_code(code, iteration)
                results['execution_results'].append(exec_result)

                if exec_result['success']:
                    self.write_log(f"✓ Execution successful")
                    if exec_result['stdout']:
                        self.write_log(f"Output:\n{exec_result['stdout']}")
                else:
                    self.write_log(f"✗ Execution failed")
                    self.write_log(f"Error:\n{exec_result['stderr']}")
        
        # Update LaTeX
        latex_content = self.extract_latex_content(response)
        if latex_content:
            results['latex_updated'] = True
            self.update_latex(latex_content)
            self.write_log("\n✓ LaTeX file updated")
        
        return results


class ScaffoldedResearcher:
    def __init__(self, session_name: str, max_iterations: int = 20):
        self.session = ResearchSession(session_name)
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.prompts = self._load_prompts()

    def _load_prompts(self):
        """Load prompt templates from files"""
        prompts = {}
        prompt_files = {
            'introduce': 'prompts/introduce_experiment.txt',
            'iteration_step': 'prompts/iteration_step.txt',
            'iteration_feedback': 'prompts/iteration_feedback.txt'
        }
        for key, filepath in prompt_files.items():
            with open(filepath, 'r') as f:
                prompts[key] = f.read()
        return prompts
    
    def introduce_experiment(self, problem: str):
        """Explain scaffolding structure to Claude"""
        prompt = self.prompts['introduce'].format(
            problem=problem,
            max_iterations=self.max_iterations
        )

        strategy = self.session.call_claude(prompt, use_thinking=True)
        print("\n=== CLAUDE'S STRATEGY ===")
        print(strategy)
        self.session.write_log("=== STRATEGY ===\n" + strategy)
        return strategy

    def iteration_step(self):
        """Single iteration where Claude chooses and executes work"""
        prompt = self.prompts['iteration_step'].format(
            current_iteration=self.current_iteration,
            max_iterations=self.max_iterations,
            iterations_remaining=self.max_iterations - self.current_iteration
        )

        response = self.session.call_claude(prompt, use_thinking=True)
        results = self.session.process_response(response, self.current_iteration)

        return {
            'response': response,
            'results': results
        }

    def iteration_feedback(self, work: dict):
        """Provide execution feedback and ask Claude to self-assess"""
        exec_summary = ""
        if work['results']['execution_results']:
            for i, result in enumerate(work['results']['execution_results']):
                if result['success']:
                    exec_summary += f"\n✓ Code block {i+1}: Success"
                    if result['stdout']:
                        exec_summary += f"\n  Output: {result['stdout'][:200]}"
                else:
                    exec_summary += f"\n✗ Code block {i+1}: Failed"
                    exec_summary += f"\n  Error: {result['stderr'][:300]}"

        if work['results']['latex_updated']:
            exec_summary += "\n✓ LaTeX paper updated"

        prompt = self.prompts['iteration_feedback'].format(
            exec_summary=exec_summary if exec_summary else "No code executed or LaTeX updated this iteration.",
            iterations_remaining=self.max_iterations - self.current_iteration
        )

        assessment = self.session.call_claude(prompt)
        self.session.write_log(f"\n=== SELF-ASSESSMENT ===\n{assessment}")

        return assessment
    
    def run(self, problem: str):
        """Main scaffolding loop"""
        print("="*60)
        print("SCAFFOLDED RESEARCH EXPERIMENT")
        print("="*60)

        # Introduce experiment
        self.introduce_experiment(problem)
        input("\nPress Enter to begin iterations...")

        # Iteration loop
        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1

            print(f"\n{'='*60}")
            print(f"ITERATION {self.current_iteration}/{self.max_iterations}")
            print("="*60)

            # Claude chooses and does work
            work = self.iteration_step()

            # Provide feedback and self-assessment
            self.iteration_feedback(work)

            print(f"\n[Iteration {self.current_iteration} complete]")

        # Final compilation
        print("\n" + "="*60)
        print("COMPILING FINAL PAPER")
        print("="*60)

        compile_result = self.session.compile_latex()
        if compile_result['success']:
            print(f"✓ PDF generated: {os.path.join(self.session.output_dir, 'paper.pdf')}")
        else:
            print(f"✗ PDF compilation failed")
            print(f"Error: {compile_result.get('error', 'Unknown error')[:500]}")

        print(f"\n=== EXPERIMENT COMPLETE ===")
        print(f"Total iterations: {self.current_iteration}")
        print(f"Output directory: {self.session.output_dir}")


# Run experiment
if __name__ == "__main__":
    with open('problems/power_method.txt', 'r') as f:
        problem = f.read()

    researcher = ScaffoldedResearcher(
        session_name=f"power_method_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        max_iterations=2
    )

    researcher.run(problem)