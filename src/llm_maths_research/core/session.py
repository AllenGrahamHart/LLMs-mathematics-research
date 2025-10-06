"""Research session management."""

import os
import json
import time
import re
from datetime import datetime
from typing import Dict, Any, Optional
import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv

from ..config import CONFIG
from ..utils.latex import compile_latex, extract_latex_content
from ..utils.code_execution import execute_code, extract_code_blocks

load_dotenv()


class ResearchSession:
    """Manages a single research session including files, state, and API calls."""

    def __init__(self, session_name: str):
        """
        Initialize a new research session.

        Args:
            session_name: Unique name for this session
        """
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
        """Create initial LaTeX and Python files."""
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

    def get_state(self) -> Dict[str, str]:
        """
        Get current state as dictionary.

        Returns:
            Dictionary containing latex, compilation status, python code,
            execution output, plan, and critique
        """
        # Read files
        with open(self.latex_file, 'r') as f:
            latex_content = f.read()

        with open(self.python_file, 'r') as f:
            python_content = f.read()

        # Get compilation status
        compile_result = compile_latex(self.output_dir)
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

    def extract_plan(self, text: str) -> str:
        """
        Extract next iteration plan from response.

        Args:
            text: Response text containing plan

        Returns:
            Extracted plan text
        """
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

    def can_make_api_call(self) -> bool:
        """
        Test if we can make an API call with minimal cost.

        Returns:
            True if API is available, False if rate limited
        """
        try:
            self.client.messages.create(
                model=CONFIG['api']['model'],
                max_tokens=1,
                messages=[{"role": "user", "content": "x"}]
            )
            return True
        except anthropic.RateLimitError:
            return False

    def call_claude(self, prompt: str) -> str:
        """
        Call Claude API with streaming and rate limit handling.

        Args:
            prompt: Prompt to send to Claude

        Returns:
            Response text from Claude
        """
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
        """
        Process Claude's response and update state.

        Args:
            response: Response text from Claude
            iteration: Current iteration number
        """
        self.write_log(f"\n{'='*60}\nITERATION {iteration}\n{'='*60}")
        self.write_log(f"Response:\n{response}\n")

        # Extract and execute code
        code_blocks = extract_code_blocks(response)
        if code_blocks:
            self.write_log(f"Found {len(code_blocks)} code block(s)")

            full_code = "\n\n".join(code_blocks)

            with open(self.python_file, 'w') as f:
                f.write(full_code)

            exec_result = execute_code(full_code, self.output_dir)
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
        latex_content = extract_latex_content(response)
        if latex_content:
            with open(self.latex_file, 'w') as f:
                f.write(latex_content)
            self.write_log("✓ LaTeX file updated")

        # Extract plan for next iteration
        self.current_plan = self.extract_plan(response)
        self.write_log(f"Next plan: {self.current_plan}")
        self.write_plan(iteration, self.current_plan)

    def write_log(self, entry: str):
        """
        Append entry to session log.

        Args:
            entry: Log entry text
        """
        self.log.append(entry)
        with open(self.log_file, 'a') as f:
            f.write(entry + "\n")

    def write_critique(self, iteration: int, critique: str):
        """
        Append critique to critique log.

        Args:
            iteration: Current iteration number
            critique: Critique text
        """
        with open(self.critique_file, 'a') as f:
            f.write(f"\n{'='*60}\nITERATION {iteration} CRITIQUE\n{'='*60}\n")
            f.write(critique + "\n")

    def write_plan(self, iteration: int, plan: str):
        """
        Append plan to plans log.

        Args:
            iteration: Current iteration number
            plan: Plan text
        """
        with open(self.plans_file, 'a') as f:
            f.write(f"\n{'='*60}\nITERATION {iteration} PLAN\n{'='*60}\n")
            f.write(plan + "\n")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all API metrics.

        Returns:
            Dictionary containing metrics summary
        """
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
