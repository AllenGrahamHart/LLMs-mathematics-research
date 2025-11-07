"""Unit tests for code context loading functionality."""

import pytest
import shutil
from pathlib import Path
from llm_maths_research.core.researcher import ScaffoldedResearcher


@pytest.fixture
def test_code_context():
    """Create test code context directory and clean up after."""
    project_root = Path(__file__).parent.parent.parent

    # Create test code context directory
    test_code_dir = project_root / "problems" / "code" / "unittest_codebase"
    test_code_dir.mkdir(parents=True, exist_ok=True)

    # Create code.txt
    code_content = """# ===== main.py =====
def hello():
    print("Hello, World!")

# ===== utils.py =====
def helper():
    return 42
"""
    (test_code_dir / "code.txt").write_text(code_content)

    # Create description.txt
    desc_content = """# Test Codebase

This is a test codebase for unit testing.

## Features
- Simple hello function
- Helper utilities
"""
    (test_code_dir / "description.txt").write_text(desc_content)

    # Session for cleanup
    test_output = project_root / "outputs" / "unittest_code_session"

    yield {
        'code_id': 'unittest_codebase',
        'code_dir': test_code_dir,
        'session': 'unittest_code_session',
        'output': test_output
    }

    # Cleanup
    if test_code_dir.exists():
        shutil.rmtree(test_code_dir)
    if test_output.exists():
        shutil.rmtree(test_output)


def test_code_context_loading(test_code_context):
    """Test that code context is loaded correctly."""
    researcher = ScaffoldedResearcher(
        session_name=test_code_context['session'],
        max_iterations=1,
        code_ids=[test_code_context['code_id']]
    )

    assert test_code_context['code_id'] in researcher.code_content
    code_data = researcher.code_content[test_code_context['code_id']]

    # Verify both description and code were loaded
    assert 'description' in code_data
    assert 'code' in code_data
    assert "Test Codebase" in code_data['description']
    assert "def hello():" in code_data['code']
    assert "def helper():" in code_data['code']


def test_code_context_with_only_code_file(test_code_context):
    """Test code context loading when only code.txt exists."""
    # Remove description.txt
    desc_file = test_code_context['code_dir'] / "description.txt"
    desc_file.unlink()

    researcher = ScaffoldedResearcher(
        session_name=test_code_context['session'],
        max_iterations=1,
        code_ids=[test_code_context['code_id']]
    )

    assert test_code_context['code_id'] in researcher.code_content
    code_data = researcher.code_content[test_code_context['code_id']]

    # Should have code but not description
    assert 'code' in code_data
    assert 'description' not in code_data or code_data['description'] is None


def test_code_context_with_only_description_file(test_code_context):
    """Test code context loading when only description.txt exists."""
    # Remove code.txt
    code_file = test_code_context['code_dir'] / "code.txt"
    code_file.unlink()

    researcher = ScaffoldedResearcher(
        session_name=test_code_context['session'],
        max_iterations=1,
        code_ids=[test_code_context['code_id']]
    )

    assert test_code_context['code_id'] in researcher.code_content
    code_data = researcher.code_content[test_code_context['code_id']]

    # Should have description but not code
    assert 'description' in code_data
    assert 'code' not in code_data or code_data['code'] is None


def test_missing_code_context_directory():
    """Test handling of non-existent code context directory."""
    import pytest

    # Should raise FileNotFoundError for missing code context
    with pytest.raises(FileNotFoundError, match="Code context directory not found"):
        researcher = ScaffoldedResearcher(
            session_name="test_missing_code",
            max_iterations=1,
            code_ids=["nonexistent_codebase"]
        )


def test_code_context_initialization_empty():
    """Test that code_content initializes as empty dict when no code provided."""
    researcher = ScaffoldedResearcher(
        session_name="test_no_code",
        max_iterations=1
    )

    assert researcher.code_content == {}
    assert researcher.code_ids == []


def test_multiple_code_contexts(test_code_context):
    """Test loading multiple code contexts."""
    project_root = Path(__file__).parent.parent.parent

    # Create second test code context
    test_code_dir2 = project_root / "problems" / "code" / "unittest_codebase2"
    test_code_dir2.mkdir(parents=True, exist_ok=True)
    (test_code_dir2 / "code.txt").write_text("# Second codebase\nprint('test')")
    (test_code_dir2 / "description.txt").write_text("Second test codebase")

    try:
        researcher = ScaffoldedResearcher(
            session_name=test_code_context['session'],
            max_iterations=1,
            code_ids=[test_code_context['code_id'], 'unittest_codebase2']
        )

        assert test_code_context['code_id'] in researcher.code_content
        assert 'unittest_codebase2' in researcher.code_content
        assert len(researcher.code_content) == 2
    finally:
        if test_code_dir2.exists():
            shutil.rmtree(test_code_dir2)


def test_code_section_building(test_code_context):
    """Test that _build_code_section() formats code correctly."""
    researcher = ScaffoldedResearcher(
        session_name=test_code_context['session'],
        max_iterations=1,
        code_ids=[test_code_context['code_id']]
    )

    code_section = researcher._build_code_section()

    # Check structure
    assert "=== AVAILABLE CODE CONTEXT ===" in code_section
    assert "--- unittest_codebase ---" in code_section
    assert "DESCRIPTION:" in code_section
    assert "SOURCE CODE:" in code_section
    assert "Test Codebase" in code_section
    assert "def hello():" in code_section
    assert "Modal" in code_section  # Should mention Modal


def test_code_section_empty_when_no_code():
    """Test that _build_code_section() returns empty string when no code."""
    researcher = ScaffoldedResearcher(
        session_name="test_no_code",
        max_iterations=1
    )

    code_section = researcher._build_code_section()
    assert code_section == ""


def test_code_context_with_papers_and_data(test_code_context):
    """Test that code context works alongside papers and data."""
    project_root = Path(__file__).parent.parent.parent

    # Create test paper
    test_paper = project_root / "problems" / "papers" / "UnitTestCodePaper.txt"
    test_paper.write_text("Test paper content")

    # Create test data
    test_csv = project_root / "data" / "datasets" / "unittest_code_data.csv"
    test_csv.write_text("a,b\n1,2")

    try:
        researcher = ScaffoldedResearcher(
            session_name=test_code_context['session'],
            max_iterations=1,
            paper_ids=['UnitTestCodePaper'],
            data_ids=['unittest_code_data.csv'],
            code_ids=[test_code_context['code_id']]
        )

        # All three should be loaded
        assert 'UnitTestCodePaper' in researcher.papers_content
        assert 'unittest_code_data.csv' in researcher.data_files
        assert test_code_context['code_id'] in researcher.code_content
    finally:
        test_paper.unlink(missing_ok=True)
        test_csv.unlink(missing_ok=True)


def test_code_context_programmatic_dict():
    """Test providing code context directly as dict."""
    code_dict = {
        'test_code': {
            'description': 'Test description',
            'code': 'print("test")'
        }
    }

    researcher = ScaffoldedResearcher(
        session_name="test_programmatic",
        max_iterations=1,
        code_contexts=code_dict
    )

    assert 'test_code' in researcher.code_content
    assert researcher.code_content['test_code']['description'] == 'Test description'
    assert researcher.code_content['test_code']['code'] == 'print("test")'
