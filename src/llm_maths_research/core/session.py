"""Research session management."""

import os
import json
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv

from ..config import CONFIG
from ..utils.latex import compile_latex
from ..utils.code_execution import execute_code
from ..utils.xml_extraction import extract_plan, extract_python_code, extract_latex_content
from ..utils.openalex_blocks import (
    extract_openalex_blocks,
    execute_openalex_calls,
    format_openalex_results,
    log_openalex_calls,
)

load_dotenv()

# Width of section separators in log files
SEPARATOR_WIDTH = 60


class ResearchSession:
    """Manages a single research session including files, state, and API calls."""

    def __init__(self, session_name: str, api_key: Optional[str] = None):
        """
        Initialize a new research session.

        Args:
            session_name: Unique name for this session
            api_key: Anthropic API key (if not provided, reads from ANTHROPIC_API_KEY env var)
        """
        self.session_name = session_name
        self.output_dir = f"outputs/{session_name}"
        os.makedirs(self.output_dir, exist_ok=True)

        self.data_dir = os.path.join(self.output_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.latex_file = os.path.join(self.output_dir, "paper.tex")
        self.python_file = os.path.join(self.output_dir, "experiment_code.py")
        self.log_file = os.path.join(self.output_dir, "session_log.txt")
        self.metrics_file = os.path.join(self.output_dir, "metrics.json")

        # Current state files (overwritten each iteration for easy loading)
        self.current_plan_file = os.path.join(self.output_dir, "current_plan.txt")
        self.current_critique_file = os.path.join(self.output_dir, "current_critique.txt")
        self.current_researcher_openalex_file = os.path.join(self.output_dir, "current_researcher_openalex.txt")
        self.current_critic_openalex_file = os.path.join(self.output_dir, "current_critic_openalex.txt")

        # Historical append-only logs (for review/debugging)
        self.critique_file = os.path.join(self.output_dir, "critiques.txt")
        self.plans_file = os.path.join(self.output_dir, "plans.txt")
        self.generator_responses_file = os.path.join(self.output_dir, "generator_responses.txt")

        self._initialize_files()
        self.log = []
        self.api_metrics = []
        self._last_call_time = None
        self.last_execution_output = ""
        self.current_plan = "No prior plan - beginning research"
        self.current_critique = "No prior critique - good luck!"
        self.current_researcher_openalex = "No literature searches performed yet"
        self.current_critic_openalex = "No literature searches performed yet"

        # Initialize Anthropic client with provided key or fallback to environment
        self.client = Anthropic(api_key=api_key or os.getenv('ANTHROPIC_API_KEY'))

    def load_last_state(self) -> None:
        """Load the last plan, critique, and OpenAlex results from current state files."""
        # Load current plan
        if os.path.exists(self.current_plan_file):
            try:
                with open(self.current_plan_file, 'r', encoding='utf-8') as f:
                    self.current_plan = f.read().strip()
            except (IOError, OSError) as e:
                print(f"Warning: Could not load plan from {self.current_plan_file}: {e}")
                print("  Using default plan: 'No prior plan - beginning research'")

        # Load current critique
        if os.path.exists(self.current_critique_file):
            try:
                with open(self.current_critique_file, 'r', encoding='utf-8') as f:
                    self.current_critique = f.read().strip()
            except (IOError, OSError) as e:
                print(f"Warning: Could not load critique from {self.current_critique_file}: {e}")
                print("  Using default critique: 'No prior critique - good luck!'")

        # Load current researcher OpenAlex results
        if os.path.exists(self.current_researcher_openalex_file):
            try:
                with open(self.current_researcher_openalex_file, 'r', encoding='utf-8') as f:
                    self.current_researcher_openalex = f.read().strip()
            except (IOError, OSError) as e:
                print(f"Warning: Could not load researcher OpenAlex from {self.current_researcher_openalex_file}: {e}")

        # Load current critic OpenAlex results
        if os.path.exists(self.current_critic_openalex_file):
            try:
                with open(self.current_critic_openalex_file, 'r', encoding='utf-8') as f:
                    self.current_critic_openalex = f.read().strip()
            except (IOError, OSError) as e:
                print(f"Warning: Could not load critic OpenAlex from {self.current_critic_openalex_file}: {e}")

    def load_last_generator_response(self) -> Optional[str]:
        """
        Load the last generator response from file for resuming at critic phase.

        Returns:
            The last generator response, or None if file doesn't exist or read fails
        """
        if os.path.exists(self.generator_responses_file):
            try:
                with open(self.generator_responses_file, 'r') as f:
                    content = f.read()
                    # Extract the last generator response (after the last separator)
                    parts = content.split("=" * SEPARATOR_WIDTH)
                    if len(parts) > 1:
                        # Get the last non-empty part that's not a header
                        # Headers look like "\nITERATION X GENERATOR RESPONSE\n"
                        for part in reversed(parts):
                            stripped = part.strip()
                            # Skip empty parts and header-only parts
                            if stripped and "GENERATOR RESPONSE" not in stripped[:100]:
                                return stripped
            except (IOError, OSError) as e:
                print(f"Warning: Could not load generator response from {self.generator_responses_file}: {e}")
                return None
        return None

    def _initialize_files(self) -> None:
        """Create initial LaTeX and Python files if they don't exist."""
        # Load LaTeX template from file
        template_path = os.path.join(os.path.dirname(__file__), "..", "templates", "initial_paper.tex")
        with open(template_path, 'r') as f:
            initial_latex = f.read()

        if not os.path.exists(self.latex_file):
            with open(self.latex_file, 'w') as f:
                f.write(initial_latex)

        initial_python = "# Research code will be added here\n"
        if not os.path.exists(self.python_file):
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
            'critique': self.current_critique,
            'researcher_openalex': self.current_researcher_openalex,
            'critic_openalex': self.current_critic_openalex
        }

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

    def call_claude(self, prompt: str, cache_static_content: bool = False, static_content: str = None) -> str:
        """
        Call Claude API with streaming, rate limit handling, retry logic, and optional prompt caching.

        Args:
            prompt: Full prompt OR dynamic content (if cache_static_content=True)
            cache_static_content: If True, use static_content as cached portion with 1-hour TTL
            static_content: Static prompt content to cache (papers, instructions, etc.)

        Returns:
            Response text from Claude
        """
        import httpx

        wait_time = CONFIG['api']['rate_limit_wait']
        max_retries = 5

        # Build messages with caching if requested
        if cache_static_content and static_content:
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": static_content,
                        "cache_control": {"type": "ephemeral", "ttl": "1h"}
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        else:
            # Original behavior - simple string prompt
            messages = [{"role": "user", "content": prompt}]

        params = {
            "model": CONFIG['api']['model'],
            "max_tokens": CONFIG['api']['max_tokens'],
            "thinking": {
                "type": "enabled",
                "budget_tokens": CONFIG['api']['thinking_budget']
            },
            "messages": messages
        }

        # Add beta header for 1-hour cache
        extra_headers = {}
        if cache_static_content:
            extra_headers["anthropic-beta"] = "prompt-caching-2024-07-31,extended-cache-ttl-2025-04-11"

        # Retry loop with rate limit and timeout handling
        for retry_attempt in range(max_retries):
            try:
                start_time = time.time()

                # Use streaming for long requests
                response_text = ""
                input_tokens = 0
                output_tokens = 0
                cache_creation_tokens = 0
                cache_read_tokens = 0

                with self.client.messages.stream(**params, extra_headers=extra_headers if cache_static_content else None) as stream:
                    for text in stream.text_stream:
                        response_text += text

                    # Get final message for usage stats
                    final_message = stream.get_final_message()
                    input_tokens = final_message.usage.input_tokens
                    output_tokens = final_message.usage.output_tokens

                    # Track cache metrics if available
                    if hasattr(final_message.usage, 'cache_creation_input_tokens'):
                        cache_creation_tokens = final_message.usage.cache_creation_input_tokens
                    if hasattr(final_message.usage, 'cache_read_input_tokens'):
                        cache_read_tokens = final_message.usage.cache_read_input_tokens

                end_time = time.time()
                break
            except anthropic.RateLimitError:
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            except (anthropic.APIStatusError, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError, anthropic.APIConnectionError) as e:
                # Handle API errors, overloaded errors, and connection errors
                if retry_attempt < max_retries - 1:
                    backoff_time = (2 ** retry_attempt) * 60  # 1min, 2min, 4min, 8min, 16min
                    error_msg = f"{type(e).__name__}"
                    if hasattr(e, 'status_code'):
                        error_msg += f" ({e.status_code}): {str(e)}"
                    print(f"  API/Connection error: {error_msg}")
                    print(f"  Attempt {retry_attempt + 1}/{max_retries}, retrying in {backoff_time}s...")
                    time.sleep(backoff_time)
                else:
                    print(f"  Failed after {max_retries} attempts")
                    raise

        self._last_call_time = end_time
        response_time = end_time - start_time

        # Calculate costs with cache multipliers
        base_input_cost_per_million = CONFIG['api']['costs']['input_per_million']
        cache_write_multiplier = CONFIG['api']['costs'].get('cache_write_multiplier', 2.0)
        cache_read_multiplier = CONFIG['api']['costs'].get('cache_read_multiplier', 0.1)

        # Regular input tokens (not cached)
        regular_input_tokens = input_tokens - cache_creation_tokens - cache_read_tokens
        regular_input_cost = (regular_input_tokens / 1_000_000) * base_input_cost_per_million

        # Cache creation tokens cost cache_write_multiplier × base price
        cache_creation_cost = (cache_creation_tokens / 1_000_000) * (base_input_cost_per_million * cache_write_multiplier)

        # Cache read tokens cost cache_read_multiplier × base price
        cache_read_cost = (cache_read_tokens / 1_000_000) * (base_input_cost_per_million * cache_read_multiplier)

        input_cost = regular_input_cost + cache_creation_cost + cache_read_cost
        output_cost = (output_tokens / 1_000_000) * CONFIG['api']['costs']['output_per_million']
        total_cost = input_cost + output_cost

        metrics_entry = {
            'timestamp': datetime.now().isoformat(),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cache_creation_tokens': cache_creation_tokens,
            'cache_read_tokens': cache_read_tokens,
            'response_time': response_time,
            'cost': total_cost
        }
        self.api_metrics.append(metrics_entry)

        # Print token usage with cache info
        cache_info = ""
        if cache_creation_tokens > 0:
            cache_info += f" | Cache write: {cache_creation_tokens:,}"
        if cache_read_tokens > 0:
            cache_info += f" | Cache read: {cache_read_tokens:,}"
        print(f"  Input: {input_tokens:,} tokens | Output: {output_tokens:,} tokens{cache_info} | Cost: ${total_cost:.4f}")

        return response_text

    def update_state_from_response(self, response: str) -> None:
        """
        Update in-memory state from a generator response without file I/O.
        Used when resuming at critic with already-saved files.

        Args:
            response: Generator response text
        """
        # Extract full plan for in-memory state
        self.current_plan = extract_plan(response)

        # Re-execute code to get execution output
        python_code = extract_python_code(response)
        if python_code:
            exec_result = execute_code(python_code, self.output_dir)
            output_limit = CONFIG['execution']['output_limit']
            self.last_execution_output = exec_result['output'][:output_limit]
        else:
            self.last_execution_output = "No code executed this iteration"

    def process_response(self, response: str, iteration: int) -> None:
        """
        Process Claude's response and update state.

        Args:
            response: Response text from Claude
            iteration: Current iteration number
        """
        self.write_log(f"\n{'='*60}\nITERATION {iteration}\n{'='*60}")
        self.write_log(f"Response:\n{response}\n")

        # Extract and execute code
        python_code = extract_python_code(response)
        if python_code:
            self.write_log("Found Python code block")

            with open(self.python_file, 'w', encoding='utf-8') as f:
                f.write(python_code)

            exec_result = execute_code(python_code, self.output_dir)
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
            with open(self.latex_file, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            self.write_log("✓ LaTeX file updated")

        # Extract plan for next iteration
        self.current_plan = extract_plan(response)
        self.write_log(f"Next plan: {self.current_plan}")

        # Write full plan section to plans file
        self.write_plan(iteration, self.current_plan)

    def process_openalex(self, response: str, role: str = 'researcher') -> None:
        """
        Process OpenAlex blocks from response and update state.

        Args:
            response: Response text containing potential <OPENALEX> blocks
            role: Either 'researcher' or 'critic' to determine which state to update
        """
        # Extract OpenAlex API calls
        calls = extract_openalex_blocks(response)

        if calls:
            self.write_log(f"Found {len(calls)} OpenAlex API call(s) from {role}")

            # Execute calls
            results = execute_openalex_calls(
                calls,
                session_dir=Path(self.output_dir),
                email=None  # Could be made configurable
            )

            # Format results for prompt inclusion
            formatted_results = format_openalex_results(results)

            # Update appropriate state
            if role == 'researcher':
                self.current_researcher_openalex = formatted_results
                # Save to current file
                with open(self.current_researcher_openalex_file, 'w', encoding='utf-8') as f:
                    f.write(formatted_results)
            else:  # critic
                self.current_critic_openalex = formatted_results
                # Save to current file
                with open(self.current_critic_openalex_file, 'w', encoding='utf-8') as f:
                    f.write(formatted_results)

            # Log summary
            log_summary = log_openalex_calls(results)
            self.write_log(f"OpenAlex calls summary:\n{log_summary}")

        else:
            # No calls made
            no_search_msg = f"No literature searches performed by {role} this iteration"
            if role == 'researcher':
                self.current_researcher_openalex = no_search_msg
                with open(self.current_researcher_openalex_file, 'w', encoding='utf-8') as f:
                    f.write(no_search_msg)
            else:
                self.current_critic_openalex = no_search_msg
                with open(self.current_critic_openalex_file, 'w', encoding='utf-8') as f:
                    f.write(no_search_msg)

    def process_planning_response(self, response: str, iteration: int) -> None:
        """
        Process planning phase response (Stage 1: Plan + Literature Search).

        Args:
            response: Response text from Claude containing <PLAN> and optional <OPENALEX>
            iteration: Current iteration number
        """
        self.write_log(f"\n{'='*60}\nITERATION {iteration} - PLANNING PHASE\n{'='*60}")
        self.write_log(f"Response:\n{response}\n")

        # Extract plan
        self.current_plan = extract_plan(response)
        if self.current_plan:
            self.write_log(f"✓ Plan extracted: {self.current_plan[:100]}...")
            self.write_plan(iteration, self.current_plan)
        else:
            self.write_log("✗ No plan found in response")
            self.current_plan = "No plan provided"

        # Process literature search if present
        self.process_openalex(response, role='researcher')

    def process_code_response(self, response: str, iteration: int) -> None:
        """
        Process code generation phase response (Stage 2: Code Generation + Execution).

        Args:
            response: Response text from Claude containing <PYTHON>
            iteration: Current iteration number
        """
        self.write_log(f"\n{'='*60}\nITERATION {iteration} - CODE GENERATION PHASE\n{'='*60}")
        self.write_log(f"Response:\n{response}\n")

        # Extract and execute code
        python_code = extract_python_code(response)
        if python_code:
            self.write_log("✓ Found Python code block")

            with open(self.python_file, 'w', encoding='utf-8') as f:
                f.write(python_code)

            exec_result = execute_code(python_code, self.output_dir)
            output_limit = CONFIG['execution']['output_limit']
            if exec_result['success']:
                self.write_log("✓ Code executed successfully")
                self.last_execution_output = exec_result['output'][:output_limit]
            else:
                self.write_log("✗ Code execution failed")
                self.last_execution_output = exec_result['output'][:output_limit]

            self.write_log(f"Output:\n{self.last_execution_output}")
        else:
            self.write_log("✗ No code found in response")
            self.last_execution_output = "No code executed this iteration"

    def process_latex_response(self, response: str, iteration: int) -> None:
        """
        Process LaTeX generation phase response (Stage 3: LaTeX Generation).

        Args:
            response: Response text from Claude containing <LATEX>
            iteration: Current iteration number
        """
        self.write_log(f"\n{'='*60}\nITERATION {iteration} - LATEX GENERATION PHASE\n{'='*60}")
        self.write_log(f"Response:\n{response}\n")

        # Extract and save LaTeX
        latex_content = extract_latex_content(response)
        if latex_content:
            with open(self.latex_file, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            self.write_log("✓ LaTeX file updated")
        else:
            self.write_log("✗ No LaTeX content found in response")

    def write_log(self, entry: str) -> None:
        """
        Append entry to session log.

        Args:
            entry: Log entry text
        """
        self.log.append(entry)
        with open(self.log_file, 'a') as f:
            f.write(entry + "\n")

    def write_critique(self, iteration: int, critique: str) -> None:
        """
        Save critique to current file and append to history log.

        Args:
            iteration: Current iteration number
            critique: Critique text
        """
        # Save to current file (overwrite)
        with open(self.current_critique_file, 'w', encoding='utf-8') as f:
            f.write(critique)

        # Append to history log
        with open(self.critique_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*SEPARATOR_WIDTH}\nITERATION {iteration} CRITIQUE\n{'='*SEPARATOR_WIDTH}\n")
            f.write(critique + "\n")

    def write_plan(self, iteration: int, plan: str) -> None:
        """
        Save plan to current file and append to history log.

        Args:
            iteration: Current iteration number
            plan: Plan text
        """
        # Save to current file (overwrite)
        with open(self.current_plan_file, 'w', encoding='utf-8') as f:
            f.write(plan)

        # Append to history log
        with open(self.plans_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*SEPARATOR_WIDTH}\nITERATION {iteration} PLAN\n{'='*SEPARATOR_WIDTH}\n")
            f.write(plan + "\n")

    def write_generator_response(self, iteration: int, response: str) -> None:
        """
        Append generator response to generator responses log.

        Args:
            iteration: Current iteration number
            response: Generator response text
        """
        with open(self.generator_responses_file, 'a') as f:
            f.write(f"\n{'='*SEPARATOR_WIDTH}\nITERATION {iteration} GENERATOR RESPONSE\n{'='*SEPARATOR_WIDTH}\n")
            f.write(response + "\n")

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
