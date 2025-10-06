"""LLM Mathematics Research - Automated research with LLMs."""

from .core.session import ResearchSession
from .core.researcher import ScaffoldedResearcher
from .config import CONFIG, load_config

__version__ = "0.1.0"

__all__ = [
    "ResearchSession",
    "ScaffoldedResearcher",
    "CONFIG",
    "load_config",
]
