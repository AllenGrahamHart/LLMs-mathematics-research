"""Tests for ResearchSession functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path
from llm_maths_research.core.session import ResearchSession


@pytest.fixture
def temp_outputs_dir():
    """Create a temporary outputs directory."""
    temp_dir = tempfile.mkdtemp()
    outputs_dir = Path(temp_dir) / "outputs"
    outputs_dir.mkdir()

    yield outputs_dir

    # Cleanup
    shutil.rmtree(temp_dir)


class TestSessionInitialization:
    """Test session initialization and directory creation."""

    def test_session_creates_directories(self, temp_outputs_dir, monkeypatch):
        """Test that session creates required directories."""
        monkeypatch.chdir(temp_outputs_dir.parent)

        session = ResearchSession("test_session")

        # Check directories exist
        assert Path(session.output_dir).exists()
        assert Path(session.data_dir).exists()
        assert Path(session.data_dir).name == "data"

    def test_session_creates_files(self, temp_outputs_dir, monkeypatch):
        """Test that session creates required files."""
        monkeypatch.chdir(temp_outputs_dir.parent)

        session = ResearchSession("test_session")

        # Check files exist
        assert Path(session.latex_file).exists()
        assert Path(session.python_file).exists()

        # Check initial content
        latex_content = Path(session.latex_file).read_text()
        assert "\\documentclass{article}" in latex_content

        python_content = Path(session.python_file).read_text()
        assert "# Research code will be added here" in python_content

    def test_session_paths_are_correct(self, temp_outputs_dir, monkeypatch):
        """Test that session paths are correctly constructed."""
        monkeypatch.chdir(temp_outputs_dir.parent)

        session = ResearchSession("my_session")

        assert session.session_name == "my_session"
        assert session.output_dir == "outputs/my_session"
        assert session.latex_file == "outputs/my_session/paper.tex"
        assert session.python_file == "outputs/my_session/code.py"
        assert session.log_file == "outputs/my_session/session_log.txt"
        assert session.data_dir == "outputs/my_session/data"

        # Test new current state file paths
        assert session.current_plan_file == "outputs/my_session/current_plan.txt"
        assert session.current_critique_file == "outputs/my_session/current_critique.txt"
        assert session.current_researcher_openalex_file == "outputs/my_session/current_researcher_openalex.txt"
        assert session.current_critic_openalex_file == "outputs/my_session/current_critic_openalex.txt"


class TestSessionState:
    """Test session state management."""

    def test_get_state_returns_dict(self, temp_outputs_dir, monkeypatch):
        """Test that get_state returns a dictionary with expected keys."""
        monkeypatch.chdir(temp_outputs_dir.parent)

        session = ResearchSession("test_session")
        state = session.get_state()

        assert isinstance(state, dict)
        assert 'latex' in state
        assert 'compilation' in state
        assert 'python' in state
        assert 'execution_output' in state
        assert 'plan' in state
        assert 'critique' in state

    def test_initial_state_values(self, temp_outputs_dir, monkeypatch):
        """Test initial state values."""
        monkeypatch.chdir(temp_outputs_dir.parent)

        session = ResearchSession("test_session")

        assert session.current_plan == "No prior plan - beginning research"
        assert session.current_critique == "No prior critique - good luck!"
        assert session.last_execution_output == ""


class TestLogging:
    """Test logging functionality."""

    def test_write_log_creates_file(self, temp_outputs_dir, monkeypatch):
        """Test that write_log creates log file."""
        monkeypatch.chdir(temp_outputs_dir.parent)

        session = ResearchSession("test_session")
        session.write_log("Test log entry")

        log_path = Path(session.log_file)
        assert log_path.exists()
        content = log_path.read_text()
        assert "Test log entry" in content

    def test_write_plan(self, temp_outputs_dir, monkeypatch):
        """Test writing plans to file."""
        monkeypatch.chdir(temp_outputs_dir.parent)

        session = ResearchSession("test_session")
        session.write_plan(1, "This is my plan for iteration 1")

        # Test history file (append-only)
        plan_path = Path(session.plans_file)
        assert plan_path.exists()
        content = plan_path.read_text()
        assert "ITERATION 1 PLAN" in content
        assert "This is my plan" in content

        # Test current file (overwritten each time)
        current_plan_path = Path(session.current_plan_file)
        assert current_plan_path.exists()
        current_content = current_plan_path.read_text()
        assert "This is my plan for iteration 1" in current_content
        assert "ITERATION" not in current_content  # No header in current file

    def test_write_critique(self, temp_outputs_dir, monkeypatch):
        """Test writing critiques to file."""
        monkeypatch.chdir(temp_outputs_dir.parent)

        session = ResearchSession("test_session")
        session.write_critique(1, "This is my critique")

        # Test history file (append-only)
        critique_path = Path(session.critique_file)
        assert critique_path.exists()
        content = critique_path.read_text()
        assert "ITERATION 1 CRITIQUE" in content
        assert "This is my critique" in content

        # Test current file (overwritten each time)
        current_critique_path = Path(session.current_critique_file)
        assert current_critique_path.exists()
        current_content = current_critique_path.read_text()
        assert "This is my critique" in current_content
        assert "ITERATION" not in current_content  # No header in current file

    def test_write_generator_response(self, temp_outputs_dir, monkeypatch):
        """Test writing generator responses to file."""
        monkeypatch.chdir(temp_outputs_dir.parent)

        session = ResearchSession("test_session")
        session.write_generator_response(1, "Generator output here")

        response_path = Path(session.generator_responses_file)
        assert response_path.exists()
        content = response_path.read_text()
        assert "ITERATION 1 GENERATOR RESPONSE" in content
        assert "Generator output here" in content


class TestMetrics:
    """Test metrics tracking."""

    def test_api_metrics_initialization(self, temp_outputs_dir, monkeypatch):
        """Test API metrics are initialized empty."""
        monkeypatch.chdir(temp_outputs_dir.parent)

        session = ResearchSession("test_session")
        assert session.api_metrics == []

    def test_get_metrics_summary_empty(self, temp_outputs_dir, monkeypatch):
        """Test metrics summary with no calls."""
        monkeypatch.chdir(temp_outputs_dir.parent)

        session = ResearchSession("test_session")
        summary = session.get_metrics_summary()

        assert summary['total_calls'] == 0
        assert summary['total_input_tokens'] == 0
        assert summary['total_output_tokens'] == 0
        assert summary['total_cost'] == 0.0

    def test_metrics_file_created_with_data(self, temp_outputs_dir, monkeypatch):
        """Test that metrics file is created when there are metrics."""
        monkeypatch.chdir(temp_outputs_dir.parent)

        session = ResearchSession("test_session")
        # Add a mock metric
        session.api_metrics.append({
            'role': 'generator',
            'iteration': 1,
            'input_tokens': 100,
            'output_tokens': 50,
            'response_time': 1.5,
            'cost': 0.001
        })
        session.get_metrics_summary()

        metrics_path = Path(session.metrics_file)
        assert metrics_path.exists()


class TestLoadState:
    """Test loading state from files for resuming."""

    def test_load_last_state_no_files(self, temp_outputs_dir, monkeypatch):
        """Test loading state when no files exist."""
        monkeypatch.chdir(temp_outputs_dir.parent)

        session = ResearchSession("test_session")
        session.load_last_state()

        # Should use defaults
        assert session.current_plan == "No prior plan - beginning research"
        assert session.current_critique == "No prior critique - good luck!"

    def test_load_last_state_with_files(self, temp_outputs_dir, monkeypatch):
        """Test loading state from existing files."""
        monkeypatch.chdir(temp_outputs_dir.parent)

        session = ResearchSession("test_session")

        # Write some state
        session.write_plan(1, "Saved plan content")
        session.write_critique(1, "Saved critique content")

        # Create new session and load state
        new_session = ResearchSession("test_session")
        new_session.load_last_state()

        assert "Saved plan content" in new_session.current_plan
        assert "Saved critique content" in new_session.current_critique

    def test_load_last_generator_response(self, temp_outputs_dir, monkeypatch):
        """Test loading last generator response."""
        monkeypatch.chdir(temp_outputs_dir.parent)

        session = ResearchSession("test_session")
        session.write_generator_response(1, "Generator response text")

        # Create new session and load
        new_session = ResearchSession("test_session")
        response = new_session.load_last_generator_response()

        assert response is not None
        assert "Generator response text" in response
