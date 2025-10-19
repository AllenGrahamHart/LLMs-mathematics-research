"""Simplified tests for core data loading functionality."""

import pytest
import shutil
from pathlib import Path
from llm_maths_research.core.researcher import ScaffoldedResearcher


@pytest.fixture
def test_files():
    """Create test files in actual project directories and clean up after."""
    project_root = Path(__file__).parent.parent.parent

    # Create test files with unique names
    test_paper = project_root / "problems" / "papers" / "UnitTestPaper.txt"
    test_csv = project_root / "data" / "datasets" / "unittest_data.csv"
    test_csv_desc = project_root / "data" / "datasets" / "unittest_data.txt"
    test_txt = project_root / "data" / "datasets" / "unittest_notes.txt"
    test_output = project_root / "outputs" / "unittest_session"

    # Write test files
    test_paper.write_text("Test paper for unit tests.\n\nMathematics research.")
    test_csv.write_text("x,y\n1,2\n3,4")
    test_csv_desc.write_text("Unit test CSV file\n\nTest data for validation.")
    test_txt.write_text("Test notes")

    yield {
        'paper': 'UnitTestPaper',
        'csv': 'unittest_data.csv',
        'txt': 'unittest_notes.txt',
        'session': 'unittest_session'
    }

    # Cleanup
    for f in [test_paper, test_csv, test_csv_desc, test_txt]:
        f.unlink(missing_ok=True)
    if test_output.exists():
        shutil.rmtree(test_output)


def test_data_loading_with_description(test_files):
    """Test that CSV data files load with descriptions."""
    researcher = ScaffoldedResearcher(
        session_name=test_files['session'],
        max_iterations=1,
        data_ids=[test_files['csv']]
    )

    assert test_files['csv'] in researcher.data_files
    data_info = researcher.data_files[test_files['csv']]
    assert data_info['description'] is not None
    assert "Unit test CSV file" in data_info['description']
    assert data_info['filename'] == test_files['csv']


def test_txt_data_file_no_self_description(test_files):
    """Test that .txt data files don't use themselves as description."""
    researcher = ScaffoldedResearcher(
        session_name=test_files['session'],
        max_iterations=1,
        data_ids=[test_files['txt']]
    )

    assert test_files['txt'] in researcher.data_files
    # Should NOT load itself as description
    assert researcher.data_files[test_files['txt']]['description'] is None


def test_data_files_copied_to_session(test_files):
    """Test that data files are copied to session data directory."""
    researcher = ScaffoldedResearcher(
        session_name=test_files['session'],
        max_iterations=1,
        data_ids=[test_files['csv']]
    )

    project_root = Path(__file__).parent.parent.parent
    copied_file = project_root / "outputs" / test_files['session'] / "data" / test_files['csv']

    assert copied_file.exists()
    content = copied_file.read_text()
    assert "x,y" in content


def test_paper_loading(test_files):
    """Test that paper files are loaded correctly."""
    researcher = ScaffoldedResearcher(
        session_name=test_files['session'],
        max_iterations=1,
        paper_ids=[test_files['paper']]
    )

    assert test_files['paper'] in researcher.papers_content
    assert "Test paper for unit tests" in researcher.papers_content[test_files['paper']]


def test_initialization_with_no_data(test_files):
    """Test researcher initializes correctly with no data or papers."""
    researcher = ScaffoldedResearcher(
        session_name=test_files['session'],
        max_iterations=5
    )

    assert researcher.max_iterations == 5
    assert researcher.paper_ids == []
    assert researcher.data_ids == []
    assert researcher.papers_content == {}
    assert researcher.data_files == {}


def test_resume_mode_validation():
    """Test that conflicting resume modes raise error."""
    with pytest.raises(ValueError, match="Cannot use both"):
        ScaffoldedResearcher(
            session_name="test",
            max_iterations=10,
            start_iteration=5,
            resume_at_critic=3
        )
