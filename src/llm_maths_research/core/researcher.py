"""Scaffolded researcher with generator-critic loop."""

import os
from typing import Dict
from .session import ResearchSession
from ..config import CONFIG


class ScaffoldedResearcher:
    """Manages the research loop with generator and critic phases."""

    def __init__(self, session_name: str, max_iterations: int = 20):
        """
        Initialize scaffolded researcher.

        Args:
            session_name: Unique name for this research session
            max_iterations: Maximum number of iterations to run
        """
        self.session = ResearchSession(session_name)
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.problem_statement = ""

    def build_generator_prompt(self, iteration: int, state: Dict[str, str]) -> str:
        """
        Build the prompt for the generator phase.

        Args:
            iteration: Current iteration number
            state: Current session state

        Returns:
            Generator prompt string
        """
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

When writing code, note the Pip-installed packages are:
- numpy
- scipy
- pandas
- matplotlib
- networkx
- scikit-learn

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

ERROR SEVERITY LEVELS - with examples:

FATAL (must be fixed or removed):
- Theorems with counterexamples to the core claim (not just edge cases)
- Proofs that are fundamentally wrong (key ideas are unworkable)
- Numerical experiments that are clearly nonesensical or irrelavent

SERIOUS (should fix):
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
If there are 0 iterations remaining after this one, the paper is already complete and your critique is a final evaluation of the completed work.]
"""
        return critic_prompt

    def run(self, problem: str):
        """
        Main research loop with generator-critic interaction.

        Args:
            problem: Research problem statement
        """
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
