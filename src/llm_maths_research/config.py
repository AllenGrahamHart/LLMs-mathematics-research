"""Configuration management for LLM Mathematics Research."""

import os
from pathlib import Path
import yaml
from typing import Dict, Any
from .provider_defaults import get_provider_config, PROVIDER_DEFAULTS


def get_config_path() -> Path:
    """Get the path to the configuration file."""
    # Look for config.yaml in the current working directory first
    cwd_config = Path.cwd() / "config.yaml"
    if cwd_config.exists():
        return cwd_config

    # Fall back to package default location (project root)
    package_root = Path(__file__).parent.parent.parent
    default_config = package_root / "config.yaml"
    if default_config.exists():
        return default_config

    raise FileNotFoundError(
        "config.yaml not found. Please ensure config.yaml exists in your "
        "current working directory or the project root."
    )


def load_config() -> Dict[str, Any]:
    """
    Load configuration from YAML file and merge with provider defaults.

    If no provider-specific settings are in config.yaml, uses defaults from
    provider_defaults.py. User can override specific settings if needed.

    Returns:
        Complete configuration dictionary with provider defaults merged
    """
    config_path = get_config_path()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get provider name (default to 'anthropic' if not specified)
    provider_name = config.get('api', {}).get('provider', 'anthropic')

    # Get overrides from config.yaml (if any)
    api_config = config.get('api', {})

    # Build list of overrides (only include if explicitly set in config)
    overrides = {}
    if 'model' in api_config:
        overrides['model'] = api_config['model']
    if 'max_tokens' in api_config:
        overrides['max_tokens'] = api_config['max_tokens']
    if 'thinking_budget' in api_config:
        overrides['thinking_budget'] = api_config['thinking_budget']
    if 'costs' in api_config:
        overrides['costs'] = api_config['costs']

    # Get provider defaults with overrides
    provider_config = get_provider_config(provider_name, overrides if overrides else None)

    # Merge provider config into api section
    config['api']['provider'] = provider_name
    config['api']['model'] = provider_config['model']
    config['api']['max_tokens'] = provider_config['max_tokens']
    config['api']['thinking_budget'] = provider_config.get('thinking_budget')

    # Add OpenAI-specific parameters if present
    if 'reasoning_effort' in provider_config:
        config['api']['reasoning_effort'] = provider_config['reasoning_effort']

    # Merge costs (allow partial overrides)
    if 'costs' not in config['api']:
        config['api']['costs'] = {}
    config['api']['costs'] = {**provider_config['costs'], **config['api'].get('costs', {})}

    return config


def set_provider(provider_name: str) -> None:
    """
    Update the global CONFIG to use a different provider.

    This is useful for runtime provider switching (e.g., from CLI).

    Args:
        provider_name: Name of provider to switch to
    """
    global CONFIG

    # Validate provider exists
    if provider_name not in PROVIDER_DEFAULTS:
        available = ', '.join(PROVIDER_DEFAULTS.keys())
        raise ValueError(
            f"Unsupported provider: {provider_name}. "
            f"Available providers: {available}"
        )

    # Load fresh config from file
    config_path = get_config_path()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override provider in config
    if 'api' not in config:
        config['api'] = {}
    config['api']['provider'] = provider_name

    # Get provider config with any existing overrides
    api_config = config.get('api', {})
    overrides = {}
    if 'model' in api_config:
        overrides['model'] = api_config['model']
    if 'max_tokens' in api_config:
        overrides['max_tokens'] = api_config['max_tokens']
    if 'thinking_budget' in api_config:
        overrides['thinking_budget'] = api_config['thinking_budget']
    if 'costs' in api_config:
        overrides['costs'] = api_config['costs']

    # Get provider defaults with overrides
    provider_config = get_provider_config(provider_name, overrides if overrides else None)

    # Update config with provider defaults
    config['api']['provider'] = provider_name
    config['api']['model'] = provider_config['model']
    config['api']['max_tokens'] = provider_config['max_tokens']
    config['api']['thinking_budget'] = provider_config.get('thinking_budget')

    # Add OpenAI-specific parameters if present
    if 'reasoning_effort' in provider_config:
        config['api']['reasoning_effort'] = provider_config['reasoning_effort']

    # Merge costs
    if 'costs' not in config['api']:
        config['api']['costs'] = {}
    config['api']['costs'] = {**provider_config['costs'], **config['api'].get('costs', {})}

    # Update global CONFIG in-place (don't reassign, update the dict)
    CONFIG.clear()
    CONFIG.update(config)


# Global config instance
CONFIG = load_config()
