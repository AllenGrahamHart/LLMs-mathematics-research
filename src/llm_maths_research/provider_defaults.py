"""Default configurations for each LLM provider.

These are sensible defaults for the most powerful/recommended model from each provider.
Users can override these in config.yaml if needed.
"""

PROVIDER_DEFAULTS = {
    'anthropic': {
        'model': 'claude-sonnet-4-5-20250929',
        'display_name': 'Claude Sonnet 4.5',
        'max_tokens': 64000,
        'thinking_budget': 32000,  # Extended thinking for complex reasoning
        'supports_thinking': True,
        'supports_caching': True,
        'costs': {
            'input_per_million': 3.0,
            'output_per_million': 15.0,
            'cache_write_multiplier': 2.0,
            'cache_read_multiplier': 0.1,
        },
        'notes': 'Best for complex reasoning tasks. Supports extended thinking and prompt caching.',
    },

    'openai': {
        'model': 'gpt-4o',
        'display_name': 'GPT-4o',
        'max_tokens': 16000,
        'thinking_budget': None,  # Not supported
        'supports_thinking': False,
        'supports_caching': False,
        'costs': {
            'input_per_million': 2.5,
            'output_per_million': 10.0,
            'cache_write_multiplier': 1.0,  # No caching
            'cache_read_multiplier': 1.0,
        },
        'notes': 'Fast and cost-effective. Good balance of performance and price.',
    },

    'google': {
        'model': 'gemini-2.5-pro',
        'display_name': 'Gemini 2.5 Pro',
        'max_tokens': 65536,  # 65.5K output, 1M input context
        'thinking_budget': None,
        'supports_thinking': True,  # State-of-the-art thinking model
        'supports_caching': True,  # Context caching with 90% savings
        'costs': {
            'input_per_million': 1.25,  # For prompts <= 200K tokens
            'output_per_million': 10.0,
            'cache_write_multiplier': 0.1,  # $0.125/M (90% savings from $1.25)
            'cache_read_multiplier': 0.1,
        },
        'notes': 'Most capable Gemini model. State-of-the-art reasoning for code, math, and STEM. Context caching provides 90% cost savings on repeated prompts.',
    },

    'xai': {
        'model': 'grok-4-0709',
        'display_name': 'Grok 4',
        'max_tokens': 128000,  # Output limit; 256K input context window
        'thinking_budget': None,
        'supports_thinking': False,
        'supports_caching': True,  # 75% cost savings on cached tokens
        'costs': {
            'input_per_million': 3.0,
            'output_per_million': 15.0,
            'cache_write_multiplier': 1.0,  # Full price for cache writes
            'cache_read_multiplier': 0.25,  # $0.75/M (75% savings from $3.00)
        },
        'notes': 'Latest Grok model. 256K context window with caching support for cost-effective research.',
    },

    'moonshot': {
        'model': 'kimi-k2-thinking',
        'display_name': 'Kimi K2 Thinking',
        'max_tokens': 128000,  # Output limit; 256K input context window
        'thinking_budget': None,
        'supports_thinking': True,  # Extended thinking capabilities
        'supports_caching': True,  # Automatic token caching
        'costs': {
            'input_per_million': 0.60,
            'output_per_million': 2.50,
            'cache_write_multiplier': 1.0,  # Automatic caching
            'cache_read_multiplier': 0.1,  # Estimated 90% savings (typical for auto-caching)
        },
        'notes': 'Very cost-effective thinking model. 256K context with automatic caching for budget-conscious research.',
    },
}


def get_provider_config(provider_name: str, overrides: dict = None) -> dict:
    """
    Get configuration for a provider with optional overrides.

    Args:
        provider_name: Name of provider ('anthropic', 'openai', etc.)
        overrides: Dictionary of settings to override defaults

    Returns:
        Complete provider configuration

    Raises:
        ValueError: If provider_name is not supported
    """
    if provider_name not in PROVIDER_DEFAULTS:
        available = ', '.join(PROVIDER_DEFAULTS.keys())
        raise ValueError(
            f"Unsupported provider: {provider_name}. "
            f"Available providers: {available}"
        )

    # Start with defaults
    config = PROVIDER_DEFAULTS[provider_name].copy()

    # Deep copy nested dicts
    config['costs'] = config['costs'].copy()

    # Apply overrides if provided
    if overrides:
        for key, value in overrides.items():
            if key == 'costs' and isinstance(value, dict):
                # Merge cost overrides
                config['costs'].update(value)
            else:
                config[key] = value

    return config


def list_providers() -> list:
    """
    Get list of supported provider names.

    Returns:
        List of provider names
    """
    return list(PROVIDER_DEFAULTS.keys())


def get_provider_info(provider_name: str) -> str:
    """
    Get human-readable information about a provider.

    Args:
        provider_name: Name of provider

    Returns:
        Formatted string with provider info
    """
    if provider_name not in PROVIDER_DEFAULTS:
        return f"Unknown provider: {provider_name}"

    config = PROVIDER_DEFAULTS[provider_name]
    info = [
        f"{provider_name.upper()}:",
        f"  Model: {config['model']} ({config['display_name']})",
        f"  Max Tokens: {config['max_tokens']:,}",
        f"  Cost: ${config['costs']['input_per_million']:.2f} / ${config['costs']['output_per_million']:.2f} per million tokens",
    ]

    if config['supports_thinking']:
        if config['thinking_budget']:
            info.append(f"  Extended Thinking: Yes (budget: {config['thinking_budget']:,} tokens)")
        else:
            info.append("  Extended Thinking: Yes")

    if config['supports_caching']:
        info.append("  Prompt Caching: Yes")

    info.append(f"  Notes: {config['notes']}")

    return '\n'.join(info)


if __name__ == '__main__':
    """Print information about all providers."""
    print("Available LLM Providers")
    print("=" * 80)
    print()

    for provider in list_providers():
        print(get_provider_info(provider))
        print()
