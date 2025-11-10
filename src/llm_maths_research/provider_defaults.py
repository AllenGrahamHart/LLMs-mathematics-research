"""Default configurations for each LLM provider.

These are sensible defaults for the most powerful/recommended model from each provider.
Users can override these in config.yaml if needed.

IMPORTANT: The 'max_tokens' key is a generic configuration value used by the system.
Each provider implementation maps this to the appropriate provider-specific parameter:
- Anthropic: max_tokens
- OpenAI (GPT-5, o1): max_completion_tokens
- Google (Gemini): max_output_tokens
- xAI (Grok): max_tokens (OpenAI-compatible)
- Moonshot (Kimi): max_tokens (OpenAI-compatible)

Each provider dict only includes fields that are meaningful for that specific provider.
"""

PROVIDER_DEFAULTS = {
    'anthropic': {
        'model': 'claude-sonnet-4-5-20250929',
        'display_name': 'Claude Sonnet 4.5',
        'max_tokens': 64000,
        'thinking_budget': 32000,  # Anthropic-specific: explicit thinking token budget
        'costs': {
            'input_per_million': 3.0,
            'output_per_million': 15.0,
            'cache_write_multiplier': 2.0,
            'cache_read_multiplier': 0.1,
        },
        'notes': 'Extended thinking with explicit budget control. Prompt caching with 90% savings on reads. ✓ EXPOSES reasoning content via ThinkingBlock.',
    },

    'openai': {
        'model': 'gpt-5',
        'display_name': 'GPT-5',
        'max_tokens': 128000,
        'max_completion_tokens': 128000,  # OpenAI-specific parameter name for GPT-5/o1
        'reasoning_effort': 'high',  # GPT-5 reasoning effort: minimal, low, medium, high
        'costs': {
            'input_per_million': 1.25,
            'output_per_million': 10.0,
            'cache_write_multiplier': 1.0,
            'cache_read_multiplier': 0.1,  # 90% savings on cached tokens
        },
        'notes': 'GPT-5 with high reasoning effort for maximum performance. Uses max_completion_tokens parameter. Prompt caching provides 90% cost savings.',
    },

    'google': {
        'model': 'gemini-2.5-pro',
        'display_name': 'Gemini 2.5 Pro',
        'max_tokens': 65536,
        'max_output_tokens': 65536,  # Google-specific parameter name
        'thinking_budget': -1,  # -1 = adaptive thinking (model adjusts automatically), 0 = disabled, or up to 32000 tokens
        'costs': {
            'input_per_million': 1.25,  # For prompts <= 200K tokens
            'output_per_million': 10.0,
            'cache_write_multiplier': 1.0,  # Automatic caching
            'cache_read_multiplier': 0.1,   # 90% savings on cached tokens
        },
        'notes': 'Adaptive thinking model with automatic context caching. Note: Gemini does NOT expose reasoning/thought content via API - thinking_budget is for documentation only. 1M input context, 65.5K output.',
    },

    'xai': {
        'model': 'grok-4-0709',
        'display_name': 'Grok 4',
        'max_tokens': 128000,  # 256K input context, 128K output
        'costs': {
            'input_per_million': 3.0,
            'output_per_million': 15.0,
            'cache_write_multiplier': 1.0,
            'cache_read_multiplier': 0.25,  # 75% savings on cached tokens
        },
        'notes': 'OpenAI-compatible API. 256K context window with prompt caching (75% cost savings). Note: Grok does NOT expose reasoning content via API.',
    },

    'moonshot': {
        'model': 'kimi-k2-thinking',
        'display_name': 'Kimi K2 Thinking',
        'max_tokens': 128000,  # 256K input context, 128K output
        'costs': {
            'input_per_million': 0.60,
            'output_per_million': 2.50,
            'cache_write_multiplier': 1.0,
            'cache_read_multiplier': 0.1,  # Estimated 90% savings
        },
        'notes': 'Extended thinking model with automatic caching. OpenAI-compatible API. Very cost-effective for budget-conscious research. ✓ EXPOSES reasoning content via reasoning_content field.',
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
        if config.get('thinking_budget'):
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
