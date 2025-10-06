"""Test that all modules can be imported successfully."""

import pytest


def test_import_main_package():
    """Test importing main package."""
    from llm_maths_research import ScaffoldedResearcher, ResearchSession
    assert ScaffoldedResearcher is not None
    assert ResearchSession is not None


def test_import_config():
    """Test importing config module."""
    from llm_maths_research import CONFIG, load_config
    assert CONFIG is not None
    assert load_config is not None
    assert isinstance(CONFIG, dict)


def test_import_core_modules():
    """Test importing core modules."""
    from llm_maths_research.core import ResearchSession, ScaffoldedResearcher
    assert ResearchSession is not None
    assert ScaffoldedResearcher is not None


def test_import_utils():
    """Test importing utility modules."""
    from llm_maths_research.utils import (
        compile_latex,
        extract_latex_content,
        execute_code,
        extract_code_blocks
    )
    assert compile_latex is not None
    assert extract_latex_content is not None
    assert execute_code is not None
    assert extract_code_blocks is not None


def test_config_structure():
    """Test that config has expected structure."""
    from llm_maths_research import CONFIG

    # Check top-level keys
    assert 'execution' in CONFIG
    assert 'compilation' in CONFIG
    assert 'api' in CONFIG
    assert 'output' in CONFIG
    assert 'research' in CONFIG

    # Check some specific values
    assert 'timeout' in CONFIG['execution']
    assert 'model' in CONFIG['api']
