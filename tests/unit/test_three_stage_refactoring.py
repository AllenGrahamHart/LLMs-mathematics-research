"""Unit tests for three-stage generator refactoring."""

import pytest
import tempfile
import shutil
from pathlib import Path
from llm_maths_research.core.session import ResearchSession
from llm_maths_research.core.researcher import ScaffoldedResearcher


@pytest.fixture
def temp_outputs_dir():
    """Create a temporary outputs directory."""
    temp_dir = tempfile.mkdtemp()
    outputs_dir = Path(temp_dir) / "outputs"
    outputs_dir.mkdir()

    yield outputs_dir

    # Cleanup
    shutil.rmtree(temp_dir)


class TestSessionStageMethods:
    """Test the new stage-specific session methods."""

    def test_process_planning_response_with_plan(self, temp_outputs_dir, monkeypatch):
        """Test processing planning response with valid plan."""
        monkeypatch.chdir(temp_outputs_dir.parent)
        session = ResearchSession("test_session")

        response = """
<PLAN>
This is my detailed plan for the remaining iterations.
Step 1: Do something
Step 2: Do something else
</PLAN>
"""
        session.process_planning_response(response, 1)

        # Check plan was extracted and stored
        assert "This is my detailed plan" in session.current_plan
        assert "Step 1" in session.current_plan

        # Check plan was written to file
        plan_file = Path(session.current_plan_file)
        assert plan_file.exists()
        plan_content = plan_file.read_text()
        assert "This is my detailed plan" in plan_content

    def test_process_planning_response_no_plan(self, temp_outputs_dir, monkeypatch):
        """Test processing planning response without plan tag."""
        monkeypatch.chdir(temp_outputs_dir.parent)
        session = ResearchSession("test_session")

        response = "Some response without plan tags"
        session.process_planning_response(response, 1)

        # Should set default message (from extract_plan function)
        assert session.current_plan == "No plan provided in response"

    def test_process_planning_response_with_literature(self, temp_outputs_dir, monkeypatch):
        """Test processing planning response with literature search."""
        monkeypatch.chdir(temp_outputs_dir.parent)
        session = ResearchSession("test_session")

        response = """
<PLAN>
My plan with literature
</PLAN>

<OPENALEX>
[
  {
    "function": "search_literature",
    "arguments": {
      "query": "test query",
      "max_results": 5
    },
    "purpose": "Find relevant papers"
  }
]
</OPENALEX>
"""
        # Note: This will attempt to actually call OpenAlex API
        # In a real test, we'd mock this, but for now just verify it doesn't crash
        session.process_planning_response(response, 1)

        # Plan should be extracted
        assert "My plan with literature" in session.current_plan

    def test_process_code_response_with_code(self, temp_outputs_dir, monkeypatch):
        """Test processing code response with valid Python code."""
        monkeypatch.chdir(temp_outputs_dir.parent)
        session = ResearchSession("test_session")

        response = """
<PYTHON>
import numpy as np
print("Hello from test")
result = np.array([1, 2, 3])
print(f"Result: {result}")
</PYTHON>
"""
        session.process_code_response(response, 1)

        # Check code was saved
        code_file = Path(session.python_file)
        assert code_file.exists()
        code_content = code_file.read_text()
        assert "import numpy" in code_content
        assert "Hello from test" in code_content

        # Check code was executed and output captured
        assert session.last_execution_output != "No code executed this iteration"
        assert "Hello from test" in session.last_execution_output

    def test_process_code_response_no_code(self, temp_outputs_dir, monkeypatch):
        """Test processing code response without code tag."""
        monkeypatch.chdir(temp_outputs_dir.parent)
        session = ResearchSession("test_session")

        response = "Response without code tags"
        session.process_code_response(response, 1)

        # Should set default message
        assert session.last_execution_output == "No code executed this iteration"

    def test_process_code_response_with_error(self, temp_outputs_dir, monkeypatch):
        """Test processing code response with code that errors."""
        monkeypatch.chdir(temp_outputs_dir.parent)
        session = ResearchSession("test_session")

        response = """
<PYTHON>
# This will cause an error
raise ValueError("Test error")
</PYTHON>
"""
        session.process_code_response(response, 1)

        # Check error was captured in output
        assert "ValueError" in session.last_execution_output
        assert "Test error" in session.last_execution_output

    def test_process_latex_response_with_latex(self, temp_outputs_dir, monkeypatch):
        """Test processing LaTeX response with valid LaTeX."""
        monkeypatch.chdir(temp_outputs_dir.parent)
        session = ResearchSession("test_session")

        response = """
<LATEX>
\\documentclass{article}
\\begin{document}
This is my test paper.
\\end{document}
</LATEX>
"""
        session.process_latex_response(response, 1)

        # Check LaTeX was saved
        latex_file = Path(session.latex_file)
        assert latex_file.exists()
        latex_content = latex_file.read_text()
        assert "\\documentclass{article}" in latex_content
        assert "This is my test paper" in latex_content

    def test_process_latex_response_no_latex(self, temp_outputs_dir, monkeypatch):
        """Test processing LaTeX response without LaTeX tag."""
        monkeypatch.chdir(temp_outputs_dir.parent)
        session = ResearchSession("test_session")

        # Get initial latex content
        initial_latex = Path(session.latex_file).read_text()

        response = "Response without LaTeX tags"
        session.process_latex_response(response, 1)

        # LaTeX file should remain unchanged
        final_latex = Path(session.latex_file).read_text()
        assert initial_latex == final_latex

    def test_stage_logging_separate(self, temp_outputs_dir, monkeypatch):
        """Test that each stage logs separately."""
        monkeypatch.chdir(temp_outputs_dir.parent)
        session = ResearchSession("test_session")

        # Process all three stages
        session.process_planning_response("<PLAN>Plan text</PLAN>", 1)
        session.process_code_response("<PYTHON>print('test')</PYTHON>", 1)
        session.process_latex_response("<LATEX>\\documentclass{article}</LATEX>", 1)

        # Check log file contains all three phase markers
        log_file = Path(session.log_file)
        log_content = log_file.read_text()

        assert "PLANNING PHASE" in log_content
        assert "CODE GENERATION PHASE" in log_content
        assert "LATEX GENERATION PHASE" in log_content


class TestResearcherStageMethods:
    """Test the new stage-specific researcher methods."""

    def test_build_generator_prompt_for_stage_planning(self, temp_outputs_dir, monkeypatch):
        """Test building prompt for planning stage."""
        monkeypatch.chdir(temp_outputs_dir.parent)

        researcher = ScaffoldedResearcher(
            session_name="test_session",
            max_iterations=1
        )
        # Set problem statement (normally done in run() method)
        researcher.problem_statement = "Test problem"

        state = researcher.session.get_state()
        static_content, dynamic_content = researcher.build_generator_prompt_for_stage(
            1, state, "planning"
        )

        # Check that planning-specific instructions are in the prompt
        combined = static_content + dynamic_content
        assert "<PLAN>" in combined
        assert "<OPENALEX>" in combined
        assert "ONLY output the <PLAN> and optional <OPENALEX>" in combined

        # Should NOT mention Python or LaTeX in planning stage
        assert "Do NOT generate code or LaTeX yet" in combined

    def test_build_generator_prompt_for_stage_coding(self, temp_outputs_dir, monkeypatch):
        """Test building prompt for coding stage."""
        monkeypatch.chdir(temp_outputs_dir.parent)

        researcher = ScaffoldedResearcher(
            session_name="test_session",
            max_iterations=1
        )
        researcher.problem_statement = "Test problem"

        state = researcher.session.get_state()
        static_content, dynamic_content = researcher.build_generator_prompt_for_stage(
            1, state, "coding"
        )

        # Check that coding-specific instructions are in the prompt
        combined = static_content + dynamic_content
        assert "<PYTHON>" in combined
        assert "ONLY output the <PYTHON> tag" in combined
        assert "plan and literature search have already been completed" in combined

    def test_build_generator_prompt_for_stage_writing(self, temp_outputs_dir, monkeypatch):
        """Test building prompt for writing stage."""
        monkeypatch.chdir(temp_outputs_dir.parent)

        researcher = ScaffoldedResearcher(
            session_name="test_session",
            max_iterations=1
        )
        researcher.problem_statement = "Test problem"

        state = researcher.session.get_state()
        static_content, dynamic_content = researcher.build_generator_prompt_for_stage(
            1, state, "writing"
        )

        # Check that writing-specific instructions are in the prompt
        combined = static_content + dynamic_content
        assert "<LATEX>" in combined
        assert "ONLY output the <LATEX> tag" in combined
        assert "plan, literature search, and code execution have already been completed" in combined

    def test_stage_prompt_includes_state(self, temp_outputs_dir, monkeypatch):
        """Test that stage prompts include current state."""
        monkeypatch.chdir(temp_outputs_dir.parent)

        researcher = ScaffoldedResearcher(
            session_name="test_session",
            max_iterations=1
        )
        researcher.problem_statement = "Test problem"

        # Update state with some values
        researcher.session.current_plan = "This is my current plan"
        researcher.session.last_execution_output = "Code output here"

        state = researcher.session.get_state()
        static_content, dynamic_content = researcher.build_generator_prompt_for_stage(
            1, state, "writing"
        )

        # Check that state is included in dynamic content
        combined = static_content + dynamic_content
        assert "This is my current plan" in combined
        assert "Code output here" in combined

    def test_all_three_stages_use_same_static_content(self, temp_outputs_dir, monkeypatch):
        """Test that all three stages use the same static (cacheable) content."""
        monkeypatch.chdir(temp_outputs_dir.parent)

        researcher = ScaffoldedResearcher(
            session_name="test_session",
            max_iterations=1
        )
        researcher.problem_statement = "Test problem statement for caching"

        state = researcher.session.get_state()

        # Get static content for all three stages
        static1, _ = researcher.build_generator_prompt_for_stage(1, state, "planning")
        static2, _ = researcher.build_generator_prompt_for_stage(1, state, "coding")
        static3, _ = researcher.build_generator_prompt_for_stage(1, state, "writing")

        # Static content should be identical except for output format instructions
        # Check that problem statement is in all static content
        assert "Test problem statement for caching" in static1
        assert "Test problem statement for caching" in static2
        assert "Test problem statement for caching" in static3

        # But output format instructions should differ
        assert "<PLAN>" in static1
        assert "<PYTHON>" in static2
        assert "<LATEX>" in static3

    def test_stage_name_displayed_in_prompt(self, temp_outputs_dir, monkeypatch):
        """Test that stage name is displayed in the dynamic prompt."""
        monkeypatch.chdir(temp_outputs_dir.parent)

        researcher = ScaffoldedResearcher(
            session_name="test_session",
            max_iterations=1
        )
        researcher.problem_statement = "Test problem"

        state = researcher.session.get_state()

        # Test all three stages
        for stage_name in ["planning", "coding", "writing"]:
            static_content, dynamic_content = researcher.build_generator_prompt_for_stage(
                1, state, stage_name
            )

            # Check that stage name appears in dynamic content
            assert f"Current stage: {stage_name}" in dynamic_content

            # Check it appears after iteration info and before the state details
            lines = dynamic_content.split("\n")
            stage_line_idx = None
            iteration_line_idx = None

            for idx, line in enumerate(lines):
                if "Current iteration:" in line:
                    iteration_line_idx = idx
                if f"Current stage: {stage_name}" in line:
                    stage_line_idx = idx

            # Stage line should come after iteration line
            assert stage_line_idx is not None, f"Stage line not found for {stage_name}"
            assert iteration_line_idx is not None, "Iteration line not found"
            assert stage_line_idx > iteration_line_idx, "Stage should come after iteration"


class TestIntegrationStateFlow:
    """Integration tests for state flow through three stages."""

    def test_state_updates_between_stages(self, temp_outputs_dir, monkeypatch):
        """Test that state updates correctly between stages."""
        monkeypatch.chdir(temp_outputs_dir.parent)
        session = ResearchSession("test_session")

        # Stage 1: Planning
        planning_response = "<PLAN>Initial plan for testing</PLAN>"
        session.process_planning_response(planning_response, 1)

        state1 = session.get_state()
        assert "Initial plan for testing" in state1['plan']

        # Stage 2: Coding (should see the plan)
        code_response = "<PYTHON>print('Using plan from stage 1')</PYTHON>"
        session.process_code_response(code_response, 1)

        state2 = session.get_state()
        assert "Initial plan for testing" in state2['plan']  # Plan still available
        assert "Using plan from stage 1" in state2['python']  # Code now available
        assert session.last_execution_output != ""  # Code was executed

        # Stage 3: LaTeX (should see plan and code results)
        latex_response = "<LATEX>\\documentclass{article}\\begin{document}Results\\end{document}</LATEX>"
        session.process_latex_response(latex_response, 1)

        state3 = session.get_state()
        assert "Initial plan for testing" in state3['plan']  # Plan still available
        assert "Using plan from stage 1" in state3['python']  # Code still available
        assert "Results" in state3['latex']  # LaTeX now available

    def test_complete_three_stage_flow(self, temp_outputs_dir, monkeypatch):
        """Test a complete three-stage flow."""
        monkeypatch.chdir(temp_outputs_dir.parent)
        session = ResearchSession("test_session")

        # Simulate complete three-stage flow
        planning_response = """
<PLAN>
1. Generate data
2. Analyze data
3. Write paper
</PLAN>
"""
        session.process_planning_response(planning_response, 1)

        code_response = """
<PYTHON>
import numpy as np
data = np.random.randn(100)
mean = np.mean(data)
print(f"Mean: {mean}")
</PYTHON>
"""
        session.process_code_response(code_response, 1)

        latex_response = """
<LATEX>
\\documentclass{article}
\\begin{document}
\\section{Results}
We analyzed the data.
\\end{document}
</LATEX>
"""
        session.process_latex_response(latex_response, 1)

        # Verify all files were created and contain expected content
        assert Path(session.current_plan_file).exists()
        assert Path(session.python_file).exists()
        assert Path(session.latex_file).exists()

        plan_content = Path(session.current_plan_file).read_text()
        assert "Generate data" in plan_content

        code_content = Path(session.python_file).read_text()
        assert "import numpy" in code_content

        latex_content = Path(session.latex_file).read_text()
        assert "We analyzed the data" in latex_content

        # Verify execution output contains results
        assert "Mean:" in session.last_execution_output


class TestBackwardCompatibility:
    """Test backward compatibility with old process_response method."""

    def test_old_process_response_still_works(self, temp_outputs_dir, monkeypatch):
        """Test that the old process_response method still works."""
        monkeypatch.chdir(temp_outputs_dir.parent)
        session = ResearchSession("test_session")

        # Old-style response with all tags
        response = """
<PLAN>
Complete plan here
</PLAN>

<PYTHON>
print("test")
</PYTHON>

<LATEX>
\\documentclass{article}
\\begin{document}
Test
\\end{document}
</LATEX>
"""
        # This should still work for backward compatibility
        session.process_response(response, 1)

        # Verify all components were processed
        assert "Complete plan here" in session.current_plan
        assert Path(session.python_file).exists()
        assert Path(session.latex_file).exists()
