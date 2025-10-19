"""Scaffolded researcher with generator-critic loop."""

import os
import shutil
from typing import Dict, List, Optional
from .session import ResearchSession, SEPARATOR_WIDTH
from ..config import CONFIG


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
        paper_ids: Optional[List[str]] = None,
        data_ids: Optional[List[str]] = None,
        start_iteration: int = 1,
        resume_at_critic: Optional[int] = None
    ):
        """
        Initialize scaffolded researcher.

        Args:
            session_name: Unique name for this research session
            max_iterations: Maximum number of iterations to run
            paper_ids: List of paper file names (without .txt) from problems/papers/ directory
            data_ids: List of data file names from data/datasets/ directory
            start_iteration: Starting iteration number (for resuming sessions, default: 1)
            resume_at_critic: If set, resume at critic phase of this iteration (generator already completed)
        """
        self.session = ResearchSession(session_name)
        self.max_iterations = max_iterations
        self.problem_statement = ""
        self.paper_ids = paper_ids or []
        self.data_ids = data_ids or []
        self.resume_at_critic = resume_at_critic

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

        # Find project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        # Load papers content from problems/papers/
        self.papers_content = {}
        for paper_id in self.paper_ids:
            paper_path = os.path.join(project_root, "problems", "papers", f"{paper_id}.txt")

            if os.path.exists(paper_path):
                with open(paper_path, 'r') as f:
                    self.papers_content[paper_id] = f.read()
            else:
                print(f"Warning: Paper file not found: {paper_path}")

        # Load and copy data files from data/datasets/
        self.data_files = {}
        for data_id in self.data_ids:
            # Check if data_id is a path or just a filename
            data_path = os.path.join(project_root, "data", "datasets", data_id)

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
                    'original_id': data_id,
                    'filename': filename,
                    'description': description
                }

                print(f"✓ Loaded data file: {data_id} → {filename}")
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

    def build_generator_prompt(self, iteration: int, state: Dict[str, str]) -> str:
        """
        Build the prompt for the generator phase.

        Args:
            iteration: Current iteration number
            state: Current session state

        Returns:
            Generator prompt string
        """
        # Load template from file
        template_path = os.path.join(os.path.dirname(__file__), "..", "templates", "generator_prompt.txt")
        with open(template_path, 'r') as f:
            template = f.read()

        # Build sections using helper methods
        papers_section = self._build_papers_section()
        data_section = self._build_data_section(include_paths=True)

        # Fill in template
        return template.format(
            problem_statement=self.problem_statement,
            papers_section=papers_section,
            data_section=data_section,
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

    def build_critic_prompt(self, iteration: int, state: Dict[str, str], generator_response: str) -> str:
        """
        Build the prompt for the critic phase.

        Args:
            iteration: Current iteration number
            state: Current session state
            generator_response: Response from generator phase

        Returns:
            Critic prompt string
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

        # Build sections using helper methods
        papers_section = self._build_papers_section()
        data_section = self._build_data_section(include_paths=False)

        # Fill in template
        return template.format(
            problem_statement=self.problem_statement,
            papers_section=papers_section,
            data_section=data_section,
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
                    generator_prompt = self.build_generator_prompt(self.current_iteration, state)
                    generator_response = self.session.call_claude(generator_prompt)
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
                generator_prompt = self.build_generator_prompt(self.current_iteration, state)
                generator_response = self.session.call_claude(generator_prompt)

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
            critic_prompt = self.build_critic_prompt(self.current_iteration, state, generator_response)
            critic_response = self.session.call_claude(critic_prompt)

            # Process OpenAlex calls from critic
            self.session.process_openalex(critic_response, role='critic')

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
        print("\n" + "="*SEPARATOR_WIDTH)
        print("COMPILING FINAL PAPER")
        print("="*SEPARATOR_WIDTH)

        from ..utils.latex import compile_latex
        compile_result = compile_latex(self.session.output_dir)
        if compile_result['success']:
            compile_result = compile_latex(self.session.output_dir)
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
