"""Scaffolded researcher with generator-critic loop."""

import os
import shutil
from typing import Dict, List, Optional
from .session import ResearchSession, SEPARATOR_WIDTH
from ..config import CONFIG
from ..utils.xml_extraction import extract_critique


class ScaffoldedResearcher:
    """
    Manages a scaffolded research loop with generator and critic phases.

    "Scaffolded" refers to the iterative support structure where:
    - A generator AI produces research outputs (LaTeX papers + Python code)
    - A critic AI provides feedback to guide improvements
    - The process repeats for multiple iterations with an explicit plan
    - Each iteration builds upon previous work with structured feedback

    This architecture provides AI researchers with guidance and error correction,
    similar to how scaffolding supports construction work.
    """

    def __init__(
        self,
        session_name: str,
        max_iterations: int = 20,
        api_key: Optional[str] = None,
        paper_ids: Optional[List[str]] = None,
        paper_paths: Optional[List[str]] = None,
        papers: Optional[Dict[str, str]] = None,
        data_ids: Optional[List[str]] = None,
        data_paths: Optional[List[str]] = None,
        start_iteration: int = 1,
        resume_at_critic: Optional[int] = None,
        use_cache: bool = True
    ):
        """
        Initialize scaffolded researcher.

        Args:
            session_name: Unique name for this research session
            max_iterations: Maximum number of iterations to run
            api_key: Anthropic API key (if not provided, reads from ANTHROPIC_API_KEY env var)
            paper_ids: List of paper file names (without .txt) from problems/papers/ directory
                      (for use when running from repo - legacy/backward compatible)
            paper_paths: List of full paths to paper .txt files (for pip users)
            papers: Dict mapping paper names to content strings (for programmatic use)
            data_ids: List of data file names from data/datasets/ directory
                     (for use when running from repo - legacy/backward compatible)
            data_paths: List of full paths to data files (for pip users)
            start_iteration: Starting iteration number (for resuming sessions, default: 1)
            resume_at_critic: If set, resume at critic phase of this iteration (generator already completed)
            use_cache: Enable 1-hour prompt caching for static content (default: True)
        """
        self.session = ResearchSession(session_name, api_key=api_key)
        self.max_iterations = max_iterations
        self.problem_statement = ""
        self.paper_ids = paper_ids or []
        self.data_ids = data_ids or []
        self.resume_at_critic = resume_at_critic
        self.use_cache = use_cache

        # Validate that only one resume mode is set
        if start_iteration > 1 and resume_at_critic:
            raise ValueError("Cannot use both --start-iteration and --resume-at-critic. Choose one.")

        # Set iteration counter and load state based on resume mode
        if resume_at_critic:
            self.current_iteration = resume_at_critic - 1  # Will be incremented at start of loop
            self.session.load_last_state()
        elif start_iteration > 1:
            self.current_iteration = start_iteration - 1  # Will be incremented at start of loop
            self.session.load_last_state()
        else:
            self.current_iteration = 0  # Will be incremented to 1 at start of loop

        # Load papers content with priority: papers dict > paper_paths > paper_ids
        self.papers_content = {}
        if papers:
            # Direct content provided (highest priority)
            self.papers_content = papers
        elif paper_paths:
            # Full paths provided (for pip users)
            for paper_path in paper_paths:
                if os.path.exists(paper_path):
                    paper_name = os.path.splitext(os.path.basename(paper_path))[0]
                    with open(paper_path, 'r') as f:
                        self.papers_content[paper_name] = f.read()
                else:
                    print(f"Warning: Paper file not found: {paper_path}")
        elif paper_ids:
            # Paper IDs provided (backward compatible - looks in problems/papers/)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            for paper_id in paper_ids:
                paper_path = os.path.join(project_root, "problems", "papers", f"{paper_id}.txt")
                if os.path.exists(paper_path):
                    with open(paper_path, 'r') as f:
                        self.papers_content[paper_id] = f.read()
                else:
                    print(f"Warning: Paper file not found: {paper_path}")

        # Load and copy data files with priority: data_paths > data_ids
        self.data_files = {}

        data_sources = []
        if data_paths:
            # Full paths provided (for pip users)
            data_sources = [(path, os.path.basename(path)) for path in data_paths]
        elif data_ids:
            # Data IDs provided (backward compatible - looks in data/datasets/)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            data_sources = [(os.path.join(project_root, "data", "datasets", data_id), data_id)
                           for data_id in data_ids]

        for data_path, original_id in data_sources:
            if os.path.exists(data_path):
                # Copy file to session data directory
                filename = os.path.basename(data_path)
                dest_path = os.path.join(self.session.data_dir, filename)
                shutil.copy(data_path, dest_path)

                # Try to load optional description file
                # Skip if data file is already .txt (rare edge case - would read itself as description)
                description = None
                if not data_path.endswith('.txt'):
                    desc_path = os.path.splitext(data_path)[0] + ".txt"
                    if os.path.exists(desc_path):
                        with open(desc_path, 'r') as f:
                            description = f.read()

                self.data_files[filename] = {
                    'original_id': original_id,
                    'filename': filename,
                    'description': description
                }

                print(f"✓ Loaded data file: {original_id} → {filename}")
            else:
                print(f"Warning: Data file not found: {data_path}")

    def _build_papers_section(self) -> str:
        """
        Build the papers section for prompts.

        Returns:
            Formatted papers section string
        """
        papers_section = ""
        if self.papers_content:
            papers_section = "\n\n=== REFERENCE PAPERS ===\n\n"
            for paper_id, content in self.papers_content.items():
                papers_section += f"--- Paper {paper_id} ---\n{content}\n\n"
        return papers_section

    def _build_data_section(self, include_paths: bool = True) -> str:
        """
        Build the data files section for prompts.

        Args:
            include_paths: If True, include path instructions (for generator).
                          If False, use simpler format (for critic).

        Returns:
            Formatted data section string
        """
        data_section = ""
        if self.data_files:
            data_section = "\n\n=== AVAILABLE DATA FILES ===\n\n"

            if include_paths:
                data_section += "The following data files have been loaded for your analysis.\n"
                data_section += "IMPORTANT: Files are in the data/ subdirectory. Use the EXACT paths shown below.\n\n"
            else:
                data_section += "The researcher has access to these data files:\n\n"

            for filename, info in self.data_files.items():
                data_section += f"--- {filename} ---\n"
                if info['description']:
                    data_section += f"{info['description']}\n"
                else:
                    data_section += f"Data file: {filename}\n"

                if include_paths:
                    data_section += f"EXACT PATH: os.path.join(output_dir, 'data', '{filename}')\n\n"
                else:
                    data_section += "\n"

        return data_section

    def build_generator_prompt(self, iteration: int, state: Dict[str, str]) -> tuple:
        """
        Build the prompt for the generator phase, split into static (cacheable) and dynamic parts.

        Args:
            iteration: Current iteration number
            state: Current session state

        Returns:
            Tuple of (static_content, dynamic_content) where static_content is cacheable
        """
        # Load template from file
        template_path = os.path.join(os.path.dirname(__file__), "..", "templates", "generator_prompt.txt")
        with open(template_path, 'r') as f:
            template = f.read()

        # Split template at the "=== YOUR CURRENT STATE ===" marker
        # Everything before this is static (instructions, papers, data)
        # Everything from this point on is dynamic (current state)
        split_marker = "=== YOUR CURRENT STATE ==="
        parts = template.split(split_marker)

        if len(parts) != 2:
            # Fallback: if template structure changed, return entire prompt as dynamic
            filled = template.format(
                problem_statement=self.problem_statement,
                papers_section=self._build_papers_section(),
                data_section=self._build_data_section(include_paths=True),
                timeout=CONFIG['execution']['timeout'],
                figure_dpi=CONFIG['output']['figure_dpi'],
                iteration=iteration,
                max_iterations=self.max_iterations,
                iterations_remaining=self.max_iterations - iteration,
                latex=state['latex'],
                compilation=state['compilation'],
                python=state['python'],
                execution_output=state['execution_output'],
                plan=state['plan'],
                critique=state['critique'],
                researcher_openalex=state['researcher_openalex'],
                critic_openalex=state['critic_openalex']
            )
            return ("", filled)

        static_template = parts[0]
        dynamic_template = split_marker + parts[1]

        # Fill in static content (same across all generator calls)
        static_content = static_template.format(
            problem_statement=self.problem_statement,
            papers_section=self._build_papers_section(),
            data_section=self._build_data_section(include_paths=True),
            timeout=CONFIG['execution']['timeout'],
            figure_dpi=CONFIG['output']['figure_dpi']
        )

        # Fill in dynamic content (changes each iteration)
        dynamic_content = dynamic_template.format(
            iteration=iteration,
            max_iterations=self.max_iterations,
            iterations_remaining=self.max_iterations - iteration,
            latex=state['latex'],
            compilation=state['compilation'],
            python=state['python'],
            execution_output=state['execution_output'],
            plan=state['plan'],
            critique=state['critique'],
            researcher_openalex=state['researcher_openalex'],
            critic_openalex=state['critic_openalex']
        )

        return (static_content, dynamic_content)

    def build_critic_prompt(self, iteration: int, state: Dict[str, str], generator_response: str) -> tuple:
        """
        Build the prompt for the critic phase, split into static (cacheable) and dynamic parts.

        Args:
            iteration: Current iteration number
            state: Current session state
            generator_response: Response from generator phase

        Returns:
            Tuple of (static_content, dynamic_content) where static_content is cacheable
        """
        # Load template from file
        template_path = os.path.join(os.path.dirname(__file__), "..", "templates", "critic_prompt.txt")
        with open(template_path, 'r') as f:
            template = f.read()

        # Load survey content if this is the final iteration
        survey_section = ""
        if iteration >= self.max_iterations:
            survey_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "survey.txt")
            if os.path.exists(survey_path):
                with open(survey_path, 'r') as f:
                    survey_content = f.read()
                survey_section = f"""
There are 0 iterations remaining so the paper is already complete and your critique is a final evaluation of the completed work.
In addition to your critique - please complete this survey:

{survey_content}
"""

        # Split template right after papers/data section, before iteration info
        # The iteration numbers change each iteration and should NOT be cached
        split_marker = "Your critique is part of an AI researcher-critic agentic loop"
        parts = template.split(split_marker)

        if len(parts) != 2:
            # Fallback: if template structure changed, return entire prompt as dynamic
            filled = template.format(
                problem_statement=self.problem_statement,
                papers_section=self._build_papers_section(),
                data_section=self._build_data_section(include_paths=False),
                iteration=iteration,
                max_iterations=self.max_iterations,
                iterations_remaining=self.max_iterations - iteration,
                latex=state['latex'],
                compilation=state['compilation'],
                python=state['python'],
                execution_output=state['execution_output'],
                plan=state['plan'],
                generator_response=generator_response,
                researcher_openalex=state['researcher_openalex'],
                critic_openalex=state['critic_openalex'],
                survey_section=survey_section
            )
            return ("", filled)

        static_template = parts[0]
        dynamic_template = split_marker + parts[1]

        # Fill in static content (same across all critic calls)
        # Only includes: role description, problem statement, papers, and data
        static_content = static_template.format(
            problem_statement=self.problem_statement,
            papers_section=self._build_papers_section(),
            data_section=self._build_data_section(include_paths=False)
        )

        # Fill in dynamic content (changes each iteration)
        # Includes: iteration info, current work, and survey (if final iteration)
        dynamic_content = dynamic_template.format(
            iteration=iteration,
            max_iterations=self.max_iterations,
            iterations_remaining=self.max_iterations - iteration,
            latex=state['latex'],
            compilation=state['compilation'],
            python=state['python'],
            execution_output=state['execution_output'],
            plan=state['plan'],
            generator_response=generator_response,
            researcher_openalex=state['researcher_openalex'],
            critic_openalex=state['critic_openalex'],
            survey_section=survey_section
        )

        return (static_content, dynamic_content)

    def run(self, problem: str) -> None:
        """
        Main research loop with generator-critic interaction.

        Args:
            problem: Research problem statement
        """
        self.problem_statement = problem

        print("="*SEPARATOR_WIDTH)
        print("SCAFFOLDED RESEARCH WITH CRITIC")
        print("="*SEPARATOR_WIDTH)
        print(f"Max iterations: {self.max_iterations}")
        print(f"Output directory: {self.session.output_dir}")
        print("="*SEPARATOR_WIDTH)

        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1

            print(f"\n{'='*SEPARATOR_WIDTH}")
            print(f"ITERATION {self.current_iteration}/{self.max_iterations}")
            print("="*SEPARATOR_WIDTH)

            # Get current state
            state = self.session.get_state()

            # Check if resuming at critic for this iteration
            if self.resume_at_critic and self.current_iteration == self.resume_at_critic:
                print("\n[RESUMING AT CRITIC - skipping generator API call, using saved response]")
                # Load the saved generator response
                generator_response = self.session.load_last_generator_response()
                if not generator_response:
                    print("ERROR: Could not load generator response. Falling back to normal execution.")
                    # Fall through to normal generator execution
                    print("\n[GENERATOR]")
                    static_content, dynamic_content = self.build_generator_prompt(self.current_iteration, state)

                    if self.use_cache:
                        generator_response = self.session.call_claude(
                            prompt=dynamic_content,
                            cache_static_content=True,
                            static_content=static_content
                        )
                    else:
                        # No caching - combine prompts
                        generator_response = self.session.call_claude(static_content + dynamic_content)

                    self.session.process_response(generator_response, self.current_iteration)
                    self.session.write_generator_response(self.current_iteration, generator_response)
                else:
                    print(f"✓ Loaded generator response from file ({len(generator_response)} chars)")
                    # Update in-memory state without overwriting files (they're already correct)
                    self.session.update_state_from_response(generator_response)

                    # Also process OpenAlex calls from the loaded response
                    self.session.process_openalex(generator_response, role='researcher')

                # Clear flag so we don't skip generator in subsequent iterations
                self.resume_at_critic = None
            else:
                # GENERATOR PHASE
                print("\n[GENERATOR]")
                static_content, dynamic_content = self.build_generator_prompt(self.current_iteration, state)

                if self.use_cache:
                    generator_response = self.session.call_claude(
                        prompt=dynamic_content,
                        cache_static_content=True,
                        static_content=static_content
                    )
                else:
                    # No caching - combine prompts
                    generator_response = self.session.call_claude(static_content + dynamic_content)

                # Process generator response
                self.session.process_response(generator_response, self.current_iteration)

                # Process OpenAlex calls from generator
                self.session.process_openalex(generator_response, role='researcher')

                # Save generator response for potential resume at critic
                self.session.write_generator_response(self.current_iteration, generator_response)

            # CRITIC PHASE
            print("\n[CRITIC]")
            # Get updated state after generator's changes
            state = self.session.get_state()
            static_content, dynamic_content = self.build_critic_prompt(self.current_iteration, state, generator_response)

            if self.use_cache:
                critic_response = self.session.call_claude(
                    prompt=dynamic_content,
                    cache_static_content=True,
                    static_content=static_content
                )
            else:
                # No caching - combine prompts
                critic_response = self.session.call_claude(static_content + dynamic_content)

            # Process OpenAlex calls from critic
            self.session.process_openalex(critic_response, role='critic')

            # Extract critique content from XML tags
            critique_content = extract_critique(critic_response)

            # Store critique for next iteration
            self.session.current_critique = critique_content
            self.session.write_critique(self.current_iteration, critique_content)

            # Print critique summary
            print("\n--- Critique Summary ---")
            if "FATAL ERRORS:" in critique_content:
                parts = critique_content.split("FATAL ERRORS:")
                if len(parts) > 1:
                    fatal_parts = parts[1].split("SERIOUS ISSUES:")
                    fatal_section = fatal_parts[0].strip()
                    if fatal_section and "None identified" not in fatal_section:
                        print(f"⚠️  FATAL ERRORS FOUND")
                        print(fatal_section[:200] + "..." if len(fatal_section) > 200 else fatal_section)

            if "RECOMMENDATION:" in critique_content:
                parts = critique_content.split("RECOMMENDATION:")
                if len(parts) > 1:
                    rec_lines = parts[1].strip().split("\n")
                    rec = rec_lines[0].strip() if rec_lines else ""
                    if rec:
                        print(f"Recommendation: {rec}")

            print(f"\n[Iteration {self.current_iteration} complete]")

        # Final compilation
        print("\n" + "="*SEPARATOR_WIDTH)
        print("COMPILING FINAL PAPER")
        print("="*SEPARATOR_WIDTH)

        from ..utils.latex import compile_latex
        compile_result = compile_latex(self.session.output_dir)
        if compile_result['success']:
            print(f"✓ PDF generated: {os.path.join(self.session.output_dir, 'paper.pdf')}")
        else:
            print(f"✗ PDF compilation failed")
            error_limit = CONFIG['compilation']['error_limit']
            print(f"Error: {compile_result.get('error', 'Unknown error')[:error_limit]}")

        # Print metrics
        print("\n" + "="*SEPARATOR_WIDTH)
        print("METRICS SUMMARY")
        print("="*SEPARATOR_WIDTH)
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
