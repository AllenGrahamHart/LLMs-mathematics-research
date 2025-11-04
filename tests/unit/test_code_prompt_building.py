"""Unit tests for prompt building with code context."""

import pytest
import shutil
from pathlib import Path
from llm_maths_research.core.researcher import ScaffoldedResearcher


@pytest.fixture
def researcher_with_code():
    """Create researcher with test code context."""
    project_root = Path(__file__).parent.parent.parent

    # Create test code context
    test_code_dir = project_root / "problems" / "code" / "unittest_prompt_test"
    test_code_dir.mkdir(parents=True, exist_ok=True)

    code_content = "def test():\n    return True"
    (test_code_dir / "code.txt").write_text(code_content)

    desc_content = "Test code for prompt building"
    (test_code_dir / "description.txt").write_text(desc_content)

    researcher = ScaffoldedResearcher(
        session_name="unittest_prompt_build",
        max_iterations=2,
        code_ids=['unittest_prompt_test']
    )
    researcher.problem_statement = "Test research problem"

    yield researcher

    # Cleanup
    if test_code_dir.exists():
        shutil.rmtree(test_code_dir)
    output_dir = project_root / "outputs" / "unittest_prompt_build"
    if output_dir.exists():
        shutil.rmtree(output_dir)


def test_generator_prompt_includes_code_section(researcher_with_code):
    """Test that generator prompt includes code section."""
    state = researcher_with_code.session.get_state()
    static_content, dynamic_content = researcher_with_code.build_generator_prompt(1, state)

    # Code section should be in static content (cacheable)
    assert "=== AVAILABLE CODE CONTEXT ===" in static_content
    assert "unittest_prompt_test" in static_content
    assert "def test():" in static_content
    assert "Test code for prompt building" in static_content


def test_critic_prompt_includes_code_section(researcher_with_code):
    """Test that critic prompt includes code section."""
    state = researcher_with_code.session.get_state()
    static_content, dynamic_content = researcher_with_code.build_critic_prompt(
        1, state, "test generator response"
    )

    # Code section should be in static content (cacheable)
    assert "=== AVAILABLE CODE CONTEXT ===" in static_content
    assert "unittest_prompt_test" in static_content
    assert "def test():" in static_content


def test_code_section_in_static_content_generator(researcher_with_code):
    """Test that code section is in static (cacheable) part of generator prompt."""
    state = researcher_with_code.session.get_state()
    static_content, dynamic_content = researcher_with_code.build_generator_prompt(1, state)

    # Code should be in static, not dynamic
    assert "=== AVAILABLE CODE CONTEXT ===" in static_content
    assert "=== AVAILABLE CODE CONTEXT ===" not in dynamic_content


def test_code_section_in_static_content_critic(researcher_with_code):
    """Test that code section is in static (cacheable) part of critic prompt."""
    state = researcher_with_code.session.get_state()
    static_content, dynamic_content = researcher_with_code.build_critic_prompt(
        1, state, "test response"
    )

    # Code should be in static, not dynamic
    assert "=== AVAILABLE CODE CONTEXT ===" in static_content
    assert "=== AVAILABLE CODE CONTEXT ===" not in dynamic_content


def test_generator_prompt_structure_with_code(researcher_with_code):
    """Test overall structure of generator prompt with code."""
    state = researcher_with_code.session.get_state()
    static_content, dynamic_content = researcher_with_code.build_generator_prompt(1, state)

    # Check that all major sections are present in correct order
    problem_idx = static_content.find("Test research problem")
    code_idx = static_content.find("=== AVAILABLE CODE CONTEXT ===")

    # Problem should come before code section
    assert problem_idx < code_idx

    # Dynamic content should have state info
    assert "=== YOUR CURRENT STATE ===" in dynamic_content
    assert "Iteration" in dynamic_content


def test_critic_prompt_structure_with_code(researcher_with_code):
    """Test overall structure of critic prompt with code."""
    state = researcher_with_code.session.get_state()
    static_content, dynamic_content = researcher_with_code.build_critic_prompt(
        1, state, "test response"
    )

    # Static should have problem and code
    assert "Test research problem" in static_content
    assert "=== AVAILABLE CODE CONTEXT ===" in static_content

    # Dynamic should have iteration info
    assert "Current iteration:" in dynamic_content


def test_prompt_without_code_context():
    """Test that prompts work correctly when no code context provided."""
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "outputs" / "unittest_no_code_prompt"

    try:
        researcher = ScaffoldedResearcher(
            session_name="unittest_no_code_prompt",
            max_iterations=1
        )
        researcher.problem_statement = "Test problem"

        state = researcher.session.get_state()
        static_content, dynamic_content = researcher.build_generator_prompt(1, state)

        # Should NOT have code section
        assert "=== AVAILABLE CODE CONTEXT ===" not in static_content
        assert "=== AVAILABLE CODE CONTEXT ===" not in dynamic_content

        # But should still work
        assert "Test problem" in static_content
        assert "=== YOUR CURRENT STATE ===" in dynamic_content
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)


def test_modal_mention_in_code_section(researcher_with_code):
    """Test that code section mentions Modal for GPU tasks."""
    code_section = researcher_with_code._build_code_section()

    assert "Modal" in code_section
    assert "GPU" in code_section or "gpu" in code_section.lower()


def test_prompt_caching_structure(researcher_with_code):
    """Test that static/dynamic split is correct for caching."""
    state = researcher_with_code.session.get_state()

    # Generator
    gen_static, gen_dynamic = researcher_with_code.build_generator_prompt(1, state)

    # Static should have problem, code, but NOT iteration-specific info
    assert "Test research problem" in gen_static
    assert "=== AVAILABLE CODE CONTEXT ===" in gen_static
    assert "Iteration 1" not in gen_static
    assert "iteration: 1" not in gen_static.lower()

    # Dynamic should have iteration-specific info
    assert "Iteration" in gen_dynamic or "iteration" in gen_dynamic

    # Critic
    crit_static, crit_dynamic = researcher_with_code.build_critic_prompt(
        1, state, "test"
    )

    # Same pattern for critic
    assert "Test research problem" in crit_static
    assert "=== AVAILABLE CODE CONTEXT ===" in crit_static
    assert "Current iteration: 1" not in crit_static


def test_multiple_code_contexts_in_prompt():
    """Test prompt building with multiple code contexts."""
    project_root = Path(__file__).parent.parent.parent

    # Create two test code contexts
    test_dirs = []
    for i in [1, 2]:
        test_dir = project_root / "problems" / "code" / f"unittest_multi_{i}"
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "code.txt").write_text(f"# Code {i}")
        (test_dir / "description.txt").write_text(f"Description {i}")
        test_dirs.append(test_dir)

    output_dir = project_root / "outputs" / "unittest_multi_code"

    try:
        researcher = ScaffoldedResearcher(
            session_name="unittest_multi_code",
            max_iterations=1,
            code_ids=['unittest_multi_1', 'unittest_multi_2']
        )
        researcher.problem_statement = "Test"

        state = researcher.session.get_state()
        static_content, _ = researcher.build_generator_prompt(1, state)

        # Both code contexts should be in prompt
        assert "unittest_multi_1" in static_content
        assert "unittest_multi_2" in static_content
        assert "# Code 1" in static_content
        assert "# Code 2" in static_content
    finally:
        for test_dir in test_dirs:
            if test_dir.exists():
                shutil.rmtree(test_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
