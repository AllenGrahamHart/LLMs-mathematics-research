# Multi-Provider LLM Support

This guide explains how to use different LLM providers (Anthropic, OpenAI, Google, xAI, Moonshot) with the AI researcher system.

## Overview

The codebase now supports multiple LLM providers through a unified interface. You can easily switch between different models by updating the configuration file.

## Supported Providers

| Provider | Models | Features |
|----------|--------|----------|
| **Anthropic** | Claude Sonnet 4.5, Claude Opus 4, Claude 3.7 Sonnet | Extended thinking, prompt caching |
| **OpenAI** | GPT-4o, GPT-4 Turbo, o1, o1-mini | Full support |
| **Google** | Gemini 2.0 Flash, Gemini 1.5 Pro | Full support |
| **xAI** | Grok Beta, Grok Vision Beta | Full support |
| **Moonshot** | Kimi (8k/32k/128k) | Full support |

## Quick Start

### 1. Set up API Keys

Create a `.env` file in the project root with your API key(s):

```bash
# Anthropic (default)
ANTHROPIC_API_KEY=your_anthropic_key_here

# OpenAI
OPENAI_API_KEY=your_openai_key_here

# Google
GOOGLE_API_KEY=your_google_key_here

# xAI
XAI_API_KEY=your_xai_key_here

# Moonshot
MOONSHOT_API_KEY=your_moonshot_key_here
```

### 2. Configure Your Provider

Edit `config.yaml` to select your provider and model:

```yaml
api:
  provider: anthropic  # or: openai, google, xai, moonshot
  model: claude-sonnet-4-5-20250929
  max_tokens: 64000
  # ... other settings
```

### 3. Run Your Experiment

```bash
python run_experiment.py --problem problems/open_research.txt --max-iterations 5
```

## Configuration Examples

### Anthropic Claude Sonnet 4.5 (Default)

```yaml
api:
  provider: anthropic
  model: claude-sonnet-4-5-20250929
  max_tokens: 64000
  thinking_budget: 32000  # Extended thinking (Anthropic-only)
  costs:
    input_per_million: 3.0
    output_per_million: 15.0
```

**Display Name in Papers**: "Claude Sonnet 4.5"

**Special Features**:
- Extended thinking mode for complex reasoning
- Prompt caching (1-hour TTL) for cost savings

### OpenAI GPT-4o

```yaml
api:
  provider: openai
  model: gpt-4o
  max_tokens: 16000
  costs:
    input_per_million: 2.5
    output_per_million: 10.0
```

**Display Name in Papers**: "GPT-4o"

**Notes**:
- Thinking budget not supported
- Prompt caching not supported

### Google Gemini 2.0 Flash

```yaml
api:
  provider: google
  model: gemini-2.0-flash-exp
  max_tokens: 8192
  costs:
    input_per_million: 0.0  # Check current pricing
    output_per_million: 0.0
```

**Display Name in Papers**: "Gemini 2.0 Flash"

**Notes**:
- Experimental model, pricing may change
- Token counting is approximate

### xAI Grok Beta

```yaml
api:
  provider: xai
  model: grok-beta
  max_tokens: 32768
  costs:
    input_per_million: 5.0
    output_per_million: 15.0
```

**Display Name in Papers**: "Grok Beta"

**Notes**:
- Uses OpenAI-compatible API
- Token counting is approximate

### Moonshot Kimi

```yaml
api:
  provider: moonshot
  model: moonshot-v1-128k  # or moonshot-v1-8k, moonshot-v1-32k
  max_tokens: 16000
  costs:
    input_per_million: 1.0
    output_per_million: 1.0
```

**Display Name in Papers**: "Kimi (128k)"

**Notes**:
- Uses OpenAI-compatible API
- Supports very long context windows

## Model Display Names

When the AI writes research papers, it lists itself as an author. The display name is automatically determined by the model:

- `claude-sonnet-4-5-20250929` → "Claude Sonnet 4.5"
- `gpt-4o` → "GPT-4o"
- `gemini-2.0-flash-exp` → "Gemini 2.0 Flash"
- `grok-beta` → "Grok Beta"
- `moonshot-v1-128k` → "Kimi (128k)"

You can customize these in `src/llm_maths_research/core/llm_provider.py` by editing the `get_model_display_name()` method for each provider.

## Testing Different Providers

Use the included test script to verify your provider setup:

```bash
python test_providers.py
```

This will test:
- Provider initialization
- Message creation
- Token counting
- Model display names

To test additional providers, uncomment them in `test_providers.py` and ensure you have the corresponding API key set.

## Feature Comparison

| Feature | Anthropic | OpenAI | Google | xAI | Moonshot |
|---------|-----------|--------|--------|-----|----------|
| Extended Thinking | ✓ | ✗ | ✗ | ✗ | ✗ |
| Prompt Caching | ✓ | ✗ | ✗ | ✗ | ✗ |
| Streaming | ✓ | ✓ | ✓ | ✓ | ✓ |
| Token Counting | ✓ (exact) | ✓ (exact) | ≈ | ≈ | ≈ |
| Long Context | 200K+ | 128K | 1M+ | 128K | 128K |

## Cost Optimization Tips

### 1. Use Prompt Caching (Anthropic Only)

The system automatically caches static content (problem statement, papers, instructions) for 1 hour:
- First call: Cache creation (2x cost)
- Subsequent calls: Cache read (0.1x cost)
- Net savings: ~40% per iteration after first call

### 2. Choose Appropriate Models

For faster iteration during development:
- OpenAI: `gpt-4o-mini` (cheaper, faster)
- Google: `gemini-1.5-flash` (free tier available)
- Anthropic: `claude-3-5-sonnet` (good balance)

For final papers:
- Anthropic: `claude-sonnet-4-5-20250929` (best reasoning)
- OpenAI: `o1` (strong on mathematics)
- Google: `gemini-2.0-flash-thinking` (strong reasoning)

### 3. Adjust Max Tokens

Reduce `max_tokens` if your experiments don't need long outputs:

```yaml
api:
  max_tokens: 8000  # Instead of 64000
```

## Troubleshooting

### "Provider not found" Error

Make sure the provider name in `config.yaml` exactly matches one of:
- `anthropic`
- `openai`
- `google`
- `xai`
- `moonshot`

### API Key Not Found

1. Check your `.env` file exists and contains the key
2. Verify the environment variable name:
   - `ANTHROPIC_API_KEY`
   - `OPENAI_API_KEY`
   - `GOOGLE_API_KEY`
   - `XAI_API_KEY`
   - `MOONSHOT_API_KEY`
3. Restart your Python environment to load new env vars

### Rate Limit Errors

The system automatically handles rate limits with exponential backoff. To reduce rate limiting:

1. Increase `rate_limit_wait` in config:
   ```yaml
   api:
     rate_limit_wait: 60  # Wait 60s instead of 20s
   ```

2. Reduce iteration frequency
3. Use a higher-tier API plan

### Import Errors

Install missing provider dependencies:

```bash
pip install -e .
```

This installs all required packages:
- `anthropic>=0.39.0`
- `openai>=1.0.0`
- `google-generativeai>=0.3.0`

## Architecture

The multi-provider support is implemented through:

1. **Abstract Interface** (`LLMProvider`): Defines standard methods all providers must implement
2. **Provider Implementations**: Each provider (Anthropic, OpenAI, etc.) has its own class
3. **Factory Function** (`create_provider`): Creates the appropriate provider based on config
4. **Unified Response** (`LLMResponse`): Standardizes responses across providers

See `src/llm_maths_research/core/llm_provider.py` for implementation details.

## Adding New Providers

To add a new provider:

1. Create a new class in `llm_provider.py` that inherits from `LLMProvider`
2. Implement all abstract methods:
   - `create_message()`
   - `create_message_stream()`
   - `count_tokens()`
   - `get_model_display_name()`
3. Add the provider to the factory function
4. Add the API key environment variable to the map in `session.py`
5. Update this guide with configuration examples

## FAQ

**Q: Can I use multiple providers in the same experiment?**
A: Not currently. The provider is set at the session level and applies to all API calls.

**Q: Does prompt caching work with non-Anthropic providers?**
A: No, prompt caching is Anthropic-specific. For other providers, the cache parameters are ignored.

**Q: What about extended thinking for other providers?**
A: Extended thinking is Anthropic-specific. The system automatically disables it for other providers.

**Q: How accurate is token counting for non-Anthropic providers?**
A: OpenAI provides exact counts. Google, xAI, and Moonshot use approximate estimates based on tiktoken.

**Q: Can I switch providers mid-experiment?**
A: No, but you can resume an experiment with a different provider by loading the last state and continuing.

## Additional Resources

- [Anthropic API Docs](https://docs.anthropic.com/)
- [OpenAI API Docs](https://platform.openai.com/docs/)
- [Google AI Studio](https://ai.google.dev/)
- [xAI API Docs](https://docs.x.ai/)
- [Moonshot API Docs](https://platform.moonshot.cn/)
