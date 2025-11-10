"""Test script to verify LLM provider implementations."""

import os
from llm_maths_research.core.llm_provider import create_provider

def test_provider(provider_name: str, model: str, api_key: str):
    """Test a provider with a simple message."""
    print(f"\n{'='*60}")
    print(f"Testing {provider_name} with model {model}")
    print('='*60)

    try:
        # Create provider
        provider = create_provider(provider_name, api_key, model)
        print(f"✓ Provider created successfully")

        # Get display name
        display_name = provider.get_model_display_name()
        print(f"✓ Display name: {display_name}")

        # Test simple message
        messages = [{"role": "user", "content": "Say 'Hello, I am working!' and nothing else."}]
        response = provider.create_message(
            messages=messages,
            max_tokens=50,
            temperature=0.7
        )

        print(f"✓ Response: {response.content[:100]}...")
        print(f"✓ Input tokens: {response.input_tokens}")
        print(f"✓ Output tokens: {response.output_tokens}")
        print(f"✓ Model: {response.model}")

        # Test token counting
        token_count = provider.count_tokens("This is a test message for token counting.")
        print(f"✓ Token counting works: {token_count} tokens")

        print(f"\n✓ {provider_name} provider test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ {provider_name} provider test FAILED: {type(e).__name__}: {str(e)}")
        return False


if __name__ == "__main__":
    print("LLM Provider Integration Test")
    print("="*60)

    # Test configurations
    tests = [
        {
            'provider': 'anthropic',
            'model': 'claude-sonnet-4-5-20250929',
            'env_var': 'ANTHROPIC_API_KEY'
        },
        # Uncomment to test other providers (requires API keys)
        # {
        #     'provider': 'openai',
        #     'model': 'gpt-4o-mini',
        #     'env_var': 'OPENAI_API_KEY'
        # },
        # {
        #     'provider': 'google',
        #     'model': 'gemini-2.0-flash-exp',
        #     'env_var': 'GOOGLE_API_KEY'
        # },
        # {
        #     'provider': 'xai',
        #     'model': 'grok-beta',
        #     'env_var': 'XAI_API_KEY'
        # },
        # {
        #     'provider': 'moonshot',
        #     'model': 'moonshot-v1-8k',
        #     'env_var': 'MOONSHOT_API_KEY'
        # },
    ]

    results = {}
    for test_config in tests:
        provider = test_config['provider']
        model = test_config['model']
        env_var = test_config['env_var']

        api_key = os.getenv(env_var)
        if not api_key:
            print(f"\n⊘ Skipping {provider}: {env_var} not set")
            results[provider] = 'skipped'
            continue

        success = test_provider(provider, model, api_key)
        results[provider] = 'passed' if success else 'failed'

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    for provider, result in results.items():
        emoji = {'passed': '✓', 'failed': '✗', 'skipped': '⊘'}[result]
        print(f"{emoji} {provider}: {result.upper()}")

    print("\nTo test additional providers, set their API keys as environment variables")
    print("and uncomment the corresponding test configuration in this script.")
