"""Tests for XML extraction utilities."""

import pytest
from llm_maths_research.utils.xml_extraction import (
    extract_xml_tag,
    extract_plan,
    extract_python_code,
    extract_latex_content,
    extract_critique,
)


class TestXMLTagExtraction:
    """Test basic XML tag extraction."""

    def test_extract_simple_tag(self):
        """Test extracting content from simple XML tags."""
        text = "<PLAN>This is my plan</PLAN>"
        result = extract_xml_tag(text, "PLAN")
        assert result == "This is my plan"

    def test_extract_multiline_tag(self):
        """Test extracting multiline content."""
        text = """
<PLAN>
Line 1
Line 2
Line 3
</PLAN>
"""
        result = extract_xml_tag(text, "PLAN")
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_extract_tag_with_code(self):
        """Test extracting tags containing code."""
        text = """
<PYTHON>
import numpy as np
print("hello")
</PYTHON>
"""
        result = extract_xml_tag(text, "PYTHON")
        assert "import numpy" in result
        assert "print" in result

    def test_extract_tag_case_insensitive(self):
        """Test that extraction is case insensitive."""
        text = "<plan>content</PLAN>"
        result = extract_xml_tag(text, "PLAN")
        assert result == "content"

    def test_extract_tag_not_found(self):
        """Test that None is returned when tag not found."""
        text = "No tags here"
        result = extract_xml_tag(text, "PLAN")
        assert result is None

    def test_extract_tag_with_nested_content(self):
        """Test extracting tag with complex nested content."""
        text = """
<LATEX>
\\documentclass{article}
\\begin{document}
Test \\textbf{bold} text
\\end{document}
</LATEX>
"""
        result = extract_xml_tag(text, "LATEX")
        assert "\\documentclass" in result
        assert "\\begin{document}" in result
        assert "\\textbf{bold}" in result


class TestPlanExtraction:
    """Test plan extraction."""

    def test_extract_plan_basic(self):
        """Test extracting a basic plan."""
        text = """
<PLAN>
I will prove theorem X and run experiments.
</PLAN>
"""
        result = extract_plan(text)
        assert "prove theorem X" in result

    def test_extract_plan_default_when_missing(self):
        """Test that default message is returned when plan not found."""
        text = "No plan tags here"
        result = extract_plan(text)
        assert result == "No plan provided in response"

    def test_extract_plan_with_other_tags(self):
        """Test extracting plan when other tags are present."""
        text = """
<PLAN>
This is the plan
</PLAN>

<PYTHON>
code here
</PYTHON>
"""
        result = extract_plan(text)
        assert "This is the plan" in result
        assert "code here" not in result


class TestPythonCodeExtraction:
    """Test Python code extraction."""

    def test_extract_python_basic(self):
        """Test extracting Python code."""
        text = """
<PYTHON>
import numpy as np
x = np.array([1, 2, 3])
print(x)
</PYTHON>
"""
        result = extract_python_code(text)
        assert "import numpy" in result
        assert "x = np.array" in result
        assert "print(x)" in result

    def test_extract_python_none_when_missing(self):
        """Test that None is returned when Python tag not found."""
        text = "No Python tags here"
        result = extract_python_code(text)
        assert result is None

    def test_extract_python_strips_whitespace(self):
        """Test that extracted code has surrounding whitespace stripped."""
        text = """
<PYTHON>

import numpy as np

</PYTHON>
"""
        result = extract_python_code(text)
        assert result == "import numpy as np"


class TestLatexExtraction:
    """Test LaTeX content extraction."""

    def test_extract_latex_basic(self):
        """Test extracting LaTeX content."""
        text = """
<LATEX>
\\documentclass{article}
\\begin{document}
Hello world
\\end{document}
</LATEX>
"""
        result = extract_latex_content(text)
        assert "\\documentclass{article}" in result
        assert "Hello world" in result

    def test_extract_latex_none_when_missing(self):
        """Test that None is returned when LaTeX tag not found."""
        text = "No LaTeX tags here"
        result = extract_latex_content(text)
        assert result is None

    def test_extract_latex_with_special_chars(self):
        """Test extracting LaTeX with special characters."""
        text = """
<LATEX>
\\section{Introduction}
Test $\\alpha = \\beta$ equation.
</LATEX>
"""
        result = extract_latex_content(text)
        assert "\\alpha = \\beta" in result


class TestCritiqueExtraction:
    """Test critique extraction."""

    def test_extract_critique_basic(self):
        """Test extracting a basic critique."""
        text = """
<CRITIQUE>
FATAL ERRORS:
None identified.

SERIOUS ISSUES:
Fix the proof in section 2.

RECOMMENDATION:
Address the serious issues.
</CRITIQUE>
"""
        result = extract_critique(text)
        assert "FATAL ERRORS:" in result
        assert "SERIOUS ISSUES:" in result
        assert "Fix the proof" in result

    def test_extract_critique_fallback_when_missing(self):
        """Test that whole text is returned when critique tag not found."""
        text = "This is a critique without tags"
        result = extract_critique(text)
        assert result == "This is a critique without tags"

    def test_extract_critique_with_openalex(self):
        """Test that critique extraction ignores OPENALEX tags."""
        text = """
<OPENALEX>
[{"function": "search"}]
</OPENALEX>

<CRITIQUE>
The work is good.
</CRITIQUE>
"""
        result = extract_critique(text)
        assert "The work is good" in result
        assert "OPENALEX" not in result


class TestRealWorldExamples:
    """Test with realistic multi-tag responses."""

    def test_complete_generator_response(self):
        """Test extracting all parts from a complete generator response."""
        response = """
<PLAN>
In this iteration, I will:
1. Implement the algorithm
2. Run experiments
3. Update the paper
</PLAN>

<PYTHON>
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.savefig("plot.png")
</PYTHON>

<LATEX>
\\documentclass{article}
\\begin{document}
\\section{Results}
The results are shown in Figure 1.
\\end{document}
</LATEX>
"""
        # Extract each part
        plan = extract_plan(response)
        code = extract_python_code(response)
        latex = extract_latex_content(response)

        # Verify plan
        assert "Implement the algorithm" in plan
        assert "Run experiments" in plan

        # Verify code
        assert "import numpy" in code
        assert "plt.savefig" in code

        # Verify LaTeX
        assert "\\section{Results}" in latex
        assert "Figure 1" in latex

    def test_complete_critic_response(self):
        """Test extracting all parts from a complete critic response."""
        response = """
<OPENALEX>
[
  {
    "function": "get_paper",
    "arguments": {"identifier": "W2100837269"}
  }
]
</OPENALEX>

<CRITIQUE>
FATAL ERRORS:
None identified.

SERIOUS ISSUES:
The proof in Theorem 2.1 needs clarification.

MINOR CONCERNS:
Some typos in the abstract.

RECOMMENDATION:
Revise the proof and fix typos before final submission.
</CRITIQUE>
"""
        critique = extract_critique(response)

        assert "FATAL ERRORS:" in critique
        assert "None identified" in critique
        assert "Theorem 2.1" in critique
        assert "typos" in critique
        assert "RECOMMENDATION:" in critique

        # Should not include OPENALEX block
        assert "<OPENALEX>" not in critique
