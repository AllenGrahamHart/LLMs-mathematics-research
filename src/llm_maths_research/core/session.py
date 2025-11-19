"""Research session management."""

import os
import json
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
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
from .llm_provider import create_provider, LLMProvider
from .reasoning_logger import ReasoningLogger

load_dotenv()

# Width of section separators in log files
SEPARATOR_WIDTH = 60


class ResearchSession:
    """Manages a single research session including files, state, and API calls."""

    def __init__(self, session_name: str, api_key: Optional[str] = None, initial_paper: Optional[str] = None, initial_code: Optional[str] = None):
        """
        Initialize a new research session.

        Args:
            session_name: Unique name for this session
            api_key: LLM provider API key (if not provided, reads from appropriate env var)
            initial_paper: Custom initial paper filename from problems/initial/ directory (optional)
            initial_code: Custom initial code filename from problems/initial/ directory (optional)
        """
        self.session_name = session_name
        self.initial_paper = initial_paper
        self.initial_code = initial_code
        self.output_dir = f"outputs/{session_name}"
        os.makedirs(self.output_dir, exist_ok=True)

        # Create subdirectories for organized output
        self.artifacts_dir = os.path.join(self.output_dir, "artifacts")
        self.figures_dir = os.path.join(self.artifacts_dir, "figures")
        self.data_generated_dir = os.path.join(self.artifacts_dir, "data")
        self.derivations_dir = os.path.join(self.artifacts_dir, "derivations")
        self.code_dir = os.path.join(self.artifacts_dir, "code")
        self.notes_dir = os.path.join(self.artifacts_dir, "notes")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        self.data_dir = os.path.join(self.output_dir, "data")  # Input data

        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.data_generated_dir, exist_ok=True)
        os.makedirs(self.derivations_dir, exist_ok=True)
        os.makedirs(self.code_dir, exist_ok=True)
        os.makedirs(self.notes_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        # Paper and code files at root for easy access
        self.latex_file = os.path.join(self.output_dir, "paper.tex")
        self.python_file = os.path.join(self.output_dir, "experiment_code.py")

        # Log files in logs/ directory
        self.log_file = os.path.join(self.logs_dir, "session_log.txt")
        self.metrics_file = os.path.join(self.logs_dir, "metrics.json")

        # Current state files (overwritten each iteration for easy loading)
        self.current_plan_file = os.path.join(self.logs_dir, "current_plan.txt")
        self.current_critique_file = os.path.join(self.logs_dir, "current_critique.txt")
        self.current_researcher_openalex_file = os.path.join(self.logs_dir, "current_researcher_openalex.txt")
        self.current_critic_openalex_file = os.path.join(self.logs_dir, "current_critic_openalex.txt")

        # Historical append-only logs (for review/debugging)
        self.critique_file = os.path.join(self.logs_dir, "critiques.txt")
        self.plans_file = os.path.join(self.logs_dir, "plans.txt")
        self.generator_responses_file = os.path.join(self.logs_dir, "generator_responses.txt")

        self._initialize_files()
        self.log = []
        self.api_metrics = []
        self._last_call_time = None
        self.last_execution_output = ""
        self.current_plan = "No prior plan - beginning research"
        self.current_critique = "No prior critique - good luck!"
        self.current_researcher_openalex = "No literature searches performed yet"
        self.current_critic_openalex = "No literature searches performed yet"

        # Initialize LLM provider from config
        provider_name = CONFIG['api'].get('provider', 'anthropic')
        model = CONFIG['api']['model']

        # Get API key from parameter or environment
        if api_key is None:
            env_var_map = {
                'anthropic': 'ANTHROPIC_API_KEY',
                'openai': 'OPENAI_API_KEY',
                'google': 'GOOGLE_API_KEY',
                'xai': 'XAI_API_KEY',
                'moonshot': 'MOONSHOT_API_KEY',
            }
            api_key = os.getenv(env_var_map.get(provider_name, 'ANTHROPIC_API_KEY'))

        # Get provider-specific parameters from config (e.g., reasoning_effort for OpenAI)
        provider_kwargs = {}
        if provider_name == 'openai' and 'reasoning_effort' in CONFIG['api']:
            provider_kwargs['reasoning_effort'] = CONFIG['api']['reasoning_effort']

        self.provider = create_provider(provider_name, api_key, model, **provider_kwargs)

        # Initialize reasoning logger
        self.reasoning_logger = ReasoningLogger(self.logs_dir)

        # Track current iteration and stage for reasoning logging
        self.current_iteration = 0
        self.current_stage = None

    def load_last_state(self) -> None:
        """
        Load the last plan, critique, and OpenAlex results from current state files.

        This method is used when resuming a research session from disk. It reads the
        most recent state from the following files:
        - current_plan.txt: The latest research plan
        - current_critique.txt: The latest critic feedback
        - current_researcher_openalex.txt: Latest literature search results from researcher
        - current_critic_openalex.txt: Latest literature search results from critic

        If any files don't exist or can't be read, default values are used without
        raising an error (with a warning printed to stdout).

        Raises:
            None: All errors are caught and logged as warnings with default fallbacks.
        """
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
        # Determine LaTeX source
        if self.initial_paper:
            # Load custom initial paper from problems/initial/
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            custom_paper_path = os.path.join(project_root, "problems", "initial", self.initial_paper)
            with open(custom_paper_path, 'r') as f:
                initial_latex = f.read()
        else:
            # Load default LaTeX template
            template_path = os.path.join(os.path.dirname(__file__), "..", "templates", "initial_paper.tex")
            with open(template_path, 'r') as f:
                initial_latex = f.read()

        if not os.path.exists(self.latex_file):
            with open(self.latex_file, 'w') as f:
                f.write(initial_latex)

        # Determine Python source
        if self.initial_code:
            # Load custom initial code from problems/initial/
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            custom_code_path = os.path.join(project_root, "problems", "initial", self.initial_code)
            with open(custom_code_path, 'r') as f:
                initial_python = f.read()
        else:
            # Use default initial code
            initial_python = "# Research code will be added here\n"

        if not os.path.exists(self.python_file):
            with open(self.python_file, 'w') as f:
                f.write(initial_python)

    def _update_artifact_manifest(self) -> None:
        """Generate MANIFEST.json cataloging all artifacts."""
        import numpy as np

        manifest = {
            "last_updated": f"iteration_{self.current_iteration}",
            "artifacts": {}
        }

        # Walk through artifacts directory
        for root, dirs, files in os.walk(self.artifacts_dir):
            for file in files:
                if file == "MANIFEST.json":
                    continue  # Skip the manifest itself

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, self.artifacts_dir)

                info = {
                    "size_kb": round(os.path.getsize(full_path) / 1024, 1)
                }

                # Auto-detect npz array names
                if file.endswith('.npz'):
                    try:
                        with np.load(full_path) as data:
                            info["arrays"] = list(data.keys())
                    except Exception:
                        pass

                manifest["artifacts"][rel_path] = info

        # Write manifest
        manifest_path = os.path.join(self.artifacts_dir, "MANIFEST.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    def get_artifact_manifest_summary(self) -> str:
        """Get formatted summary of artifact manifest for prompt."""
        manifest_path = os.path.join(self.artifacts_dir, "MANIFEST.json")

        if not os.path.exists(manifest_path):
            return "No artifacts saved yet."

        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception:
            return "No artifacts saved yet."

        if not manifest.get("artifacts"):
            return "No artifacts saved yet."

        # Group by directory
        by_dir = {}
        for path, info in sorted(manifest["artifacts"].items()):
            dir_name = os.path.dirname(path) or "root"
            file_name = os.path.basename(path)
            by_dir.setdefault(dir_name, []).append((file_name, info))

        # Format output
        lines = [f"Last updated: {manifest.get('last_updated', 'unknown')}"]
        lines.append(f"Total files: {len(manifest['artifacts'])}\n")

        for dir_name in sorted(by_dir.keys()):
            lines.append(f"{dir_name}/")
            for file_name, info in by_dir[dir_name]:
                size_kb = info.get("size_kb", 0)
                arrays = info.get("arrays", [])
                if arrays:
                    lines.append(f"  - {file_name} ({size_kb}KB, arrays: {', '.join(arrays)})")
                else:
                    lines.append(f"  - {file_name} ({size_kb}KB)")

        return "\n".join(lines)

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
            'critic_openalex': self.current_critic_openalex,
            'artifact_manifest': self.get_artifact_manifest_summary()
        }

    def set_reasoning_context(self, iteration: int, stage: str) -> None:
        """
        Set the current iteration and stage context for reasoning logging.

        Args:
            iteration: Current iteration number (1-indexed)
            stage: Current stage ("planning", "coding", "writing", or "critique")
        """
        self.current_iteration = iteration
        self.current_stage = stage

    def can_make_api_call(self) -> bool:
        """
        Test if we can make an API call with minimal cost.

        Returns:
            True if API is available, False if rate limited
        """
        try:
            self.provider.create_message(
                messages=[{"role": "user", "content": "Reply with only the character 'x' and nothing else."}],
                max_tokens=1  # Minimal tokens - even "max_tokens reached" means API works
            )
            return True
        except Exception as e:
            # Check if it's a rate limit error (provider-specific)
            error_type = type(e).__name__
            error_msg = str(e).lower()

            if 'RateLimit' in error_type or 'rate_limit' in error_msg:
                return False

            # "max_tokens reached" means API call succeeded (GPT-5/o1 used tokens for thinking)
            if 'max_tokens' in error_msg or 'max_completion_tokens' in error_msg:
                return True

            # For other errors, assume API is available (may be temporary issue)
            return True

    def call_claude(
        self,
        prompt: str,
        cache_static_content: bool = False,
        static_content: Optional[str] = None
    ) -> str:
        """
        Call LLM API with streaming, rate limit handling, retry logic, and optional prompt caching.

        Note: Manual prompt caching (cache_static_content/static_content) is only supported
        by Anthropic. Other providers (OpenAI, Google, xAI, Moonshot) use automatic caching
        that detects repeated prompt prefixes without requiring explicit cache markers.

        Args:
            prompt: Full prompt OR dynamic content (if cache_static_content=True)
            cache_static_content: If True, use static_content as cached portion (Anthropic only)
            static_content: Static prompt content to cache (Anthropic only)

        Returns:
            Response text from LLM
        """
        wait_time = CONFIG['api']['rate_limit_wait']
        max_retries = 5

        # Check if we can make API call (proactive rate limit detection)
        max_rate_limit_checks = 10  # Prevent infinite loop
        for check_attempt in range(max_rate_limit_checks):
            if self.can_make_api_call():
                break
            print(f"  Rate limit detected (pre-check), waiting {wait_time}s... (attempt {check_attempt + 1}/{max_rate_limit_checks})")
            time.sleep(wait_time)
        else:
            # Exhausted all rate limit checks
            raise Exception(f"API rate limit persists after {max_rate_limit_checks} checks ({max_rate_limit_checks * wait_time}s)")

        # Build messages with caching if requested (only for Anthropic)
        provider_name = CONFIG['api'].get('provider', 'anthropic')
        if cache_static_content and static_content and provider_name == 'anthropic':
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
            # Add beta header for 1-hour cache
            extra_headers = {"anthropic-beta": "prompt-caching-2024-07-31,extended-cache-ttl-2025-04-11"}
        elif cache_static_content and static_content:
            # Non-Anthropic provider with caching: combine static + dynamic content
            # (automatic caching will handle repeated prefixes)
            messages = [{"role": "user", "content": static_content + prompt}]
            extra_headers = None
        else:
            # Original behavior - simple string prompt
            messages = [{"role": "user", "content": prompt}]
            extra_headers = None

        # Get thinking budget (only supported by Anthropic)
        thinking_budget = None
        if provider_name == 'anthropic':
            thinking_budget = CONFIG['api'].get('thinking_budget')

        # Retry loop with rate limit and timeout handling
        response = None
        response_text = ""

        for retry_attempt in range(max_retries):
            try:
                start_time = time.time()

                # Make a single API call for all providers
                response = self.provider.create_message(
                    messages=messages,
                    max_tokens=CONFIG['api']['max_tokens'],
                    thinking_budget=thinking_budget,
                    extra_headers=extra_headers,
                )
                response_text = response.content

                end_time = time.time()
                break
            except Exception as e:
                error_type = type(e).__name__
                is_rate_limit = 'RateLimit' in error_type or 'rate_limit' in str(e).lower()

                if is_rate_limit:
                    print(f"  Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif retry_attempt < max_retries - 1:
                    # Handle API errors and connection errors
                    backoff_time = (2 ** retry_attempt) * 60  # 1min, 2min, 4min, 8min, 16min
                    error_msg = f"{error_type}"
                    if hasattr(e, 'status_code'):
                        error_msg += f" ({e.status_code}): {str(e)}"
                    else:
                        error_msg += f": {str(e)}"
                    print(f"  API/Connection error: {error_msg}")
                    print(f"  Attempt {retry_attempt + 1}/{max_retries}, retrying in {backoff_time}s...")
                    time.sleep(backoff_time)
                else:
                    print(f"  Failed after {max_retries} attempts")
                    raise

        self._last_call_time = end_time
        response_time = end_time - start_time

        # Extract token counts from response
        input_tokens = response.input_tokens
        output_tokens = response.output_tokens
        cache_creation_tokens = response.cache_creation_tokens
        cache_read_tokens = response.cache_read_tokens

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

        # Log reasoning content if available and context is set
        if response.reasoning_content and self.current_iteration and self.current_stage:
            self.reasoning_logger.log_reasoning(
                iteration=self.current_iteration,
                stage=self.current_stage,
                model=response.model or CONFIG['api']['model'],
                reasoning_content=response.reasoning_content,
                reasoning_tokens=None,  # We don't have separate reasoning token count yet
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        # Return the streamed response (which should match the non-streaming one)
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

            # Update artifact manifest after code execution
            try:
                self._update_artifact_manifest()
                self.write_log("✓ Artifact manifest updated")
            except Exception as e:
                self.write_log(f"✗ Failed to update artifact manifest: {e}")
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

        Extracts the research plan from <PLAN> tags and executes any literature search
        requests in <OPENALEX> tags. This is the first stage of the three-stage
        generator architecture where the AI creates a detailed plan for the current
        iteration and optionally searches for relevant papers.

        Args:
            response (str): Response text from Claude containing <PLAN> and optional
                <OPENALEX> blocks
            iteration (int): Current iteration number (used for logging)

        Side Effects:
            - Updates self.current_plan with extracted plan
            - Writes plan to current_plan.txt and appends to plans.txt
            - Executes literature searches and updates self.current_researcher_openalex
            - Writes log entries to session_log.txt
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

        Extracts Python code from <PYTHON> tags, saves it to experiment_code.py, and
        executes it with timeout protection. This is the second stage of the three-stage
        generator architecture where the AI writes experimental code based on the plan
        and literature search results from Stage 1.

        Args:
            response (str): Response text from Claude containing <PYTHON> block
            iteration (int): Current iteration number (used for logging)

        Side Effects:
            - Writes extracted code to experiment_code.py in output directory
            - Executes code and updates self.last_execution_output
            - Writes log entries to session_log.txt with execution results
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

        Extracts LaTeX document from <LATEX> tags and saves it to paper.tex. This is the
        final stage of the three-stage generator architecture where the AI writes the
        research paper based on the plan (Stage 1), code (Stage 2), and actual execution
        results. This ensures the paper accurately reports real findings rather than
        anticipated ones.

        Args:
            response (str): Response text from Claude containing <LATEX> block
            iteration (int): Current iteration number (used for logging)

        Side Effects:
            - Writes extracted LaTeX to paper.tex in output directory
            - Writes log entries to session_log.txt
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
