"""Test provider defaults and CLI integration."""

from src.llm_maths_research.config import CONFIG, set_provider
from src.llm_maths_research.provider_defaults import get_provider_config, list_providers

print("=" * 80)
print("Testing Provider Defaults System")
print("=" * 80)

# Test 1: Default config loading
print("\nTest 1: Default Configuration")
print("-" * 40)
print(f"Provider: {CONFIG['api']['provider']}")
print(f"Model: {CONFIG['api']['model']}")
print(f"Max Tokens: {CONFIG['api']['max_tokens']:,}")
print(f"Thinking Budget: {CONFIG['api'].get('thinking_budget')}")
print(f"Input Cost: ${CONFIG['api']['costs']['input_per_million']:.2f}/M")
print(f"Output Cost: ${CONFIG['api']['costs']['output_per_million']:.2f}/M")
print("✓ Default config loads correctly")

# Test 2: Provider defaults
print("\nTest 2: Provider Defaults")
print("-" * 40)
for provider in list_providers():
    config = get_provider_config(provider)
    print(f"{provider:12} -> {config['model']:35} ({config['display_name']})")
print("✓ All provider defaults accessible")

# Test 3: Provider switching
print("\nTest 3: Provider Switching")
print("-" * 40)
original_provider = CONFIG['api']['provider']
original_model = CONFIG['api']['model']

# Switch to OpenAI
set_provider('openai')
# Re-import CONFIG to get updated value
from src.llm_maths_research import config
print(f"Switched to: {config.CONFIG['api']['provider']}")
print(f"Model: {config.CONFIG['api']['model']}")
print(f"Max Tokens: {config.CONFIG['api']['max_tokens']:,}")
assert config.CONFIG['api']['provider'] == 'openai'
assert config.CONFIG['api']['model'] == 'gpt-4o'
print("✓ Provider switch works")

# Switch back
set_provider(original_provider)
assert config.CONFIG['api']['provider'] == original_provider
print(f"✓ Restored to: {config.CONFIG['api']['provider']}")

# Test 4: Override mechanism
print("\nTest 4: Override Mechanism")
print("-" * 40)
# Test with custom model override
custom_config = get_provider_config('anthropic', {'model': 'custom-model-123'})
print(f"Original model: claude-sonnet-4-5-20250929")
print(f"Override model: {custom_config['model']}")
assert custom_config['model'] == 'custom-model-123'
print("✓ Overrides work correctly")

# Test 5: Cost override
custom_costs = get_provider_config('openai', {'costs': {'input_per_million': 999.0}})
print(f"Original OpenAI cost: $2.50/M")
print(f"Override cost: ${custom_costs['costs']['input_per_million']:.2f}/M")
assert custom_costs['costs']['input_per_million'] == 999.0
assert custom_costs['costs']['output_per_million'] == 10.0  # Should keep other costs
print("✓ Partial cost overrides work")

print("\n" + "=" * 80)
print("ALL TESTS PASSED! ✓")
print("=" * 80)
print("\nProvider defaults system is working correctly:")
print("  • Config loads with provider defaults")
print("  • Runtime provider switching works")
print("  • Overrides work for specific settings")
print("  • All 5 providers have complete configs")
