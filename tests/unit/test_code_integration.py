"""Integration tests for code context functionality."""

import pytest
import shutil
from pathlib import Path
from llm_maths_research.core.researcher import ScaffoldedResearcher


@pytest.fixture
def full_test_environment():
    """Create complete test environment with code, paper, and data."""
    project_root = Path(__file__).parent.parent.parent

    # Create test code context
    test_code_dir = project_root / "problems" / "code" / "unittest_integration"
    test_code_dir.mkdir(parents=True, exist_ok=True)

    code_content = """# ===== model.py =====
class SimpleModel:
    def __init__(self):
        self.value = 42

    def forward(self, x):
        return x + self.value

# ===== train.py =====
def train(model, data):
    for epoch in range(10):
        loss = model.forward(epoch)
        print(f"Epoch {epoch}: loss={loss}")
"""
    (test_code_dir / "code.txt").write_text(code_content)

    desc_content = """# SimpleModel Framework

A minimal ML framework for testing.

## Components
- SimpleModel: Basic model class
- train: Training loop
"""
    (test_code_dir / "description.txt").write_text(desc_content)

    # Create test paper
    test_paper = project_root / "problems" / "papers" / "UnitTestIntegration.txt"
    test_paper.write_text("Research paper on simple models.\n\nIntroduction goes here.")

    # Create test data
    test_data = project_root / "data" / "datasets" / "unittest_integration.csv"
    test_data.write_text("epoch,loss\n1,0.5\n2,0.3")

    test_data_desc = project_root / "data" / "datasets" / "unittest_integration.txt"
    test_data_desc.write_text("Training loss data from experiments")

    output_dir = project_root / "outputs" / "unittest_integration"

    yield {
        'code_dir': test_code_dir,
        'paper': test_paper,
        'data': test_data,
        'data_desc': test_data_desc,
        'output': output_dir,
        'code_id': 'unittest_integration',
        'paper_id': 'UnitTestIntegration',
        'data_id': 'unittest_integration.csv'
    }

    # Cleanup
    for item in [test_code_dir, test_paper, test_data, test_data_desc]:
        if item.exists():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    if output_dir.exists():
        shutil.rmtree(output_dir)


def test_full_initialization_with_all_contexts(full_test_environment):
    """Test initializing researcher with code, papers, and data."""
    researcher = ScaffoldedResearcher(
        session_name="unittest_integration",
        max_iterations=2,
        paper_ids=[full_test_environment['paper_id']],
        data_ids=[full_test_environment['data_id']],
        code_ids=[full_test_environment['code_id']]
    )

    # Verify all contexts loaded
    assert full_test_environment['paper_id'] in researcher.papers_content
    assert full_test_environment['data_id'] in researcher.data_files
    assert full_test_environment['code_id'] in researcher.code_content

    # Verify content
    assert "simple models" in researcher.papers_content[full_test_environment['paper_id']].lower()
    assert researcher.data_files[full_test_environment['data_id']]['description'] is not None
    assert "SimpleModel" in researcher.code_content[full_test_environment['code_id']]['code']


def test_all_sections_in_generator_prompt(full_test_environment):
    """Test that generator prompt includes all sections."""
    researcher = ScaffoldedResearcher(
        session_name="unittest_integration",
        max_iterations=2,
        paper_ids=[full_test_environment['paper_id']],
        data_ids=[full_test_environment['data_id']],
        code_ids=[full_test_environment['code_id']]
    )
    researcher.problem_statement = "Implement and test the SimpleModel"

    state = researcher.session.get_state()
    static_content, dynamic_content = researcher.build_generator_prompt(1, state)

    # Verify all sections present
    assert "=== REFERENCE PAPERS ===" in static_content
    assert "=== AVAILABLE DATA FILES ===" in static_content
    assert "=== AVAILABLE CODE CONTEXT ===" in static_content

    # Verify content
    assert "simple models" in static_content.lower()
    assert "unittest_integration.csv" in static_content
    assert "SimpleModel" in static_content


def test_all_sections_in_critic_prompt(full_test_environment):
    """Test that critic prompt includes all sections."""
    researcher = ScaffoldedResearcher(
        session_name="unittest_integration",
        max_iterations=2,
        paper_ids=[full_test_environment['paper_id']],
        data_ids=[full_test_environment['data_id']],
        code_ids=[full_test_environment['code_id']]
    )
    researcher.problem_statement = "Implement and test the SimpleModel"

    state = researcher.session.get_state()
    static_content, dynamic_content = researcher.build_critic_prompt(
        1, state, "generator response here"
    )

    # Verify all sections present
    assert "=== REFERENCE PAPERS ===" in static_content
    assert "=== AVAILABLE DATA FILES ===" in static_content
    assert "=== AVAILABLE CODE CONTEXT ===" in static_content


def test_data_file_copied_with_code_context(full_test_environment):
    """Test that data files are copied correctly when using code context."""
    researcher = ScaffoldedResearcher(
        session_name="unittest_integration",
        max_iterations=1,
        data_ids=[full_test_environment['data_id']],
        code_ids=[full_test_environment['code_id']]
    )

    # Data should be copied to session directory
    copied_file = full_test_environment['output'] / "data" / full_test_environment['data_id']
    assert copied_file.exists()
    content = copied_file.read_text()
    assert "epoch,loss" in content


def test_papers_section_builder(full_test_environment):
    """Test _build_papers_section with code context."""
    researcher = ScaffoldedResearcher(
        session_name="unittest_integration",
        max_iterations=1,
        paper_ids=[full_test_environment['paper_id']],
        code_ids=[full_test_environment['code_id']]
    )

    papers_section = researcher._build_papers_section()

    assert "=== REFERENCE PAPERS ===" in papers_section
    assert full_test_environment['paper_id'] in papers_section
    assert "simple models" in papers_section.lower()


def test_data_section_builder_with_code(full_test_environment):
    """Test _build_data_section with code context."""
    researcher = ScaffoldedResearcher(
        session_name="unittest_integration",
        max_iterations=1,
        data_ids=[full_test_environment['data_id']],
        code_ids=[full_test_environment['code_id']]
    )

    data_section = researcher._build_data_section(include_paths=True)

    assert "=== AVAILABLE DATA FILES ===" in data_section
    assert full_test_environment['data_id'] in data_section
    assert "EXACT PATH:" in data_section


def test_code_section_order_in_prompt(full_test_environment):
    """Test that sections appear in correct order in prompt."""
    researcher = ScaffoldedResearcher(
        session_name="unittest_integration",
        max_iterations=1,
        paper_ids=[full_test_environment['paper_id']],
        data_ids=[full_test_environment['data_id']],
        code_ids=[full_test_environment['code_id']]
    )
    researcher.problem_statement = "Test order"

    state = researcher.session.get_state()
    static_content, _ = researcher.build_generator_prompt(1, state)

    # Find indices of each section
    problem_idx = static_content.find("Test order")
    papers_idx = static_content.find("=== REFERENCE PAPERS ===")
    data_idx = static_content.find("=== AVAILABLE DATA FILES ===")
    code_idx = static_content.find("=== AVAILABLE CODE CONTEXT ===")

    # Verify order: problem -> papers -> data -> code
    assert problem_idx < papers_idx < data_idx < code_idx


def test_empty_sections_dont_interfere(full_test_environment):
    """Test that empty sections don't break prompt building."""
    # Only code, no papers or data
    researcher = ScaffoldedResearcher(
        session_name="unittest_integration_minimal",
        max_iterations=1,
        code_ids=[full_test_environment['code_id']]
    )
    researcher.problem_statement = "Test"

    state = researcher.session.get_state()
    static_content, dynamic_content = researcher.build_generator_prompt(1, state)

    # Should have code section but not papers/data
    assert "=== AVAILABLE CODE CONTEXT ===" in static_content
    assert "=== REFERENCE PAPERS ===" not in static_content
    assert "=== AVAILABLE DATA FILES ===" not in static_content

    # Should still be valid prompt
    assert "Test" in static_content
    assert "=== YOUR CURRENT STATE ===" in dynamic_content

    # Cleanup
    output_dir = Path(__file__).parent.parent.parent / "outputs" / "unittest_integration_minimal"
    if output_dir.exists():
        shutil.rmtree(output_dir)


def test_code_ids_attribute(full_test_environment):
    """Test that code_ids attribute is set correctly."""
    researcher = ScaffoldedResearcher(
        session_name="unittest_integration",
        max_iterations=1,
        code_ids=[full_test_environment['code_id']]
    )

    assert researcher.code_ids == [full_test_environment['code_id']]


def test_code_context_priority():
    """Test that code_contexts dict takes priority over code_ids."""
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "outputs" / "unittest_priority"

    # Create a code directory
    test_code_dir = project_root / "problems" / "code" / "unittest_priority"
    test_code_dir.mkdir(parents=True, exist_ok=True)
    (test_code_dir / "code.txt").write_text("File version")

    try:
        # Provide both code_contexts dict and code_ids
        # Dict should take priority
        researcher = ScaffoldedResearcher(
            session_name="unittest_priority",
            max_iterations=1,
            code_contexts={'priority_test': {'code': 'Dict version'}},
            code_ids=['unittest_priority']
        )

        # Should use dict version, not file version
        assert 'priority_test' in researcher.code_content
        assert researcher.code_content['priority_test']['code'] == 'Dict version'
        assert 'unittest_priority' not in researcher.code_content
    finally:
        if test_code_dir.exists():
            shutil.rmtree(test_code_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
