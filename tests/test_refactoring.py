#!/usr/bin/env python3
"""Quick test to verify the three-stage refactoring works."""

import sys
import os

# Test that all the new methods exist and have correct signatures
def test_session_methods():
    """Test that session.py has the new methods."""
    from src.llm_maths_research.core.session import ResearchSession

    # Check that new methods exist
    assert hasattr(ResearchSession, 'process_planning_response'), "Missing process_planning_response"
    assert hasattr(ResearchSession, 'process_code_response'), "Missing process_code_response"
    assert hasattr(ResearchSession, 'process_latex_response'), "Missing process_latex_response"

    print("✓ Session methods exist")

def test_researcher_methods():
    """Test that researcher.py has the new methods."""
    from src.llm_maths_research.core.researcher import ScaffoldedResearcher

    # Check that new method exists
    assert hasattr(ScaffoldedResearcher, 'build_generator_prompt_for_stage'), "Missing build_generator_prompt_for_stage"

    print("✓ Researcher methods exist")

def test_template_updated():
    """Test that template has been updated with the placeholder."""
    template_path = "src/llm_maths_research/templates/generator_prompt.txt"
    with open(template_path, 'r') as f:
        content = f.read()

    assert '{output_format_instructions}' in content, "Template missing output_format_instructions placeholder"

    print("✓ Template updated correctly")

if __name__ == "__main__":
    try:
        test_session_methods()
        test_researcher_methods()
        test_template_updated()
        print("\n✅ ALL TESTS PASSED - Refactoring looks good!")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
