"""XML tag extraction utilities for parsing structured responses."""

import re
from typing import Optional


def extract_xml_tag(text: str, tag_name: str) -> Optional[str]:
    """
    Extract content between XML-style tags.

    Args:
        text: Text containing XML tags
        tag_name: Name of the tag (without < >)

    Returns:
        Content between tags, or None if tag not found

    Examples:
        >>> extract_xml_tag("<PLAN>Do this</PLAN>", "PLAN")
        'Do this'
        >>> extract_xml_tag("No tags here", "PLAN")
        None
    """
    # Use DOTALL flag to match across newlines
    pattern = rf'<{tag_name}>(.*?)</{tag_name}>'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()
    return None


def extract_plan(text: str) -> str:
    """
    Extract plan from <PLAN> tags.

    Args:
        text: Response text containing plan

    Returns:
        Plan text, or default message if not found
    """
    plan = extract_xml_tag(text, 'PLAN')
    if plan:
        return plan

    # Fallback: if no PLAN tag found, return a default
    return "No plan provided in response"


def extract_python_code(text: str) -> Optional[str]:
    """
    Extract Python code from <PYTHON> tags.

    Args:
        text: Response text containing code

    Returns:
        Python code, or None if not found
    """
    return extract_xml_tag(text, 'PYTHON')


def extract_latex_content(text: str) -> Optional[str]:
    """
    Extract LaTeX content from <LATEX> tags.

    Args:
        text: Response text containing LaTeX

    Returns:
        LaTeX document, or None if not found
    """
    return extract_xml_tag(text, 'LATEX')


def extract_critique(text: str) -> str:
    """
    Extract critique from <CRITIQUE> tags.

    Args:
        text: Response text containing critique

    Returns:
        Critique text, or default message if not found
    """
    critique = extract_xml_tag(text, 'CRITIQUE')
    if critique:
        return critique

    # Fallback: if no CRITIQUE tag found, return the whole response
    # (for backwards compatibility or if critic doesn't use tags)
    return text.strip()
