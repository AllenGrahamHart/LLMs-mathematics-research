# Gemini 2.5 Pro Update

## Summary

Updated the Google provider to use **Gemini 2.5 Pro**, Google's latest and most capable model released in 2025.

## Changes Made

### 1. Updated `provider_defaults.py`

**Before:**
```python
'google': {
    'model': 'gemini-2.0-flash-exp',
    'display_name': 'Gemini 2.0 Flash',
    'max_tokens': 8192,
    'supports_caching': False,
    # ...
}
```

**After:**
```python
'google': {
    'model': 'gemini-2.5-pro',
    'display_name': 'Gemini 2.5 Pro',
    'max_tokens': 65536,  # 65.5K output, 1M input context
    'supports_thinking': True,
    'supports_caching': True,  # 90% cost savings
    'costs': {
        'input_per_million': 1.25,
        'output_per_million': 10.0,
        'cache_write_multiplier': 0.1,
        'cache_read_multiplier': 0.1,
    },
}
```

### 2. Updated `llm_provider.py`

Added Gemini 2.5 series to display name mapping:
```python
model_map = {
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    # ... existing models
}
```

### 3. Fixed `provider_defaults.py`

Fixed bug in `get_provider_info()` that crashed when `thinking_budget` was `None`.

## Gemini 2.5 Pro Features

| Feature | Details |
|---------|---------|
| **Context Window** | 1M tokens input, 65.5K tokens output |
| **Capabilities** | State-of-the-art reasoning for code, math, STEM |
| **Thinking** | Yes (built-in reasoning model) |
| **Caching** | Yes (90% cost savings via context caching) |
| **Input Cost** | $1.25/M tokens (≤200K prompts) |
| **Output Cost** | $10.00/M tokens |
| **Cached Cost** | $0.125/M tokens (90% savings) |
| **Multimodal** | Text, images, video, audio, PDF |

## Usage

### CLI
```bash
# Use Gemini 2.5 Pro
python run_experiment.py --provider google --problem problems/open_research.txt

# See details
python run_experiment.py --list-providers
```

### Config File
```yaml
api:
  provider: google
```

### Python
```python
from src.llm_maths_research.config import set_provider
set_provider('google')
```

## Cost Comparison

For a typical research session with 50K input (40K cached), 10K output:

| Scenario | Cost | Savings |
|----------|------|---------|
| **Without caching** | $0.1625 | - |
| **With caching** | $0.1175 | 28% |

*Note: Savings increase with more cached content and longer sessions.*

## Testing

Created comprehensive test suite:

```bash
python test_gemini_config.py
```

**Tests:**
- ✓ Provider defaults loaded correctly
- ✓ Config switching works
- ✓ Provider instantiation
- ✓ Cost calculations accurate

## Next Steps (Optional)

To complete the multi-provider update for other providers:

1. **OpenAI**: Update to latest model (gpt-5-thinking-high or current best)
2. **xAI**: Update to Grok 3/4 with caching support
3. **Moonshot**: Update to Kimi K2 with caching support

For now, users can use:
- **Anthropic**: Claude Sonnet 4.5 (fully configured ✓)
- **Google**: Gemini 2.5 Pro (fully configured ✓)
- **OpenAI/xAI/Moonshot**: Current defaults (functional but may not be latest models)

## Documentation

- Full multi-provider guide: `MULTI_PROVIDER_GUIDE.md`
- Provider defaults guide: `PROVIDER_DEFAULTS_GUIDE.md`
- Implementation details: `CHANGES.md`

## Verification

```bash
# Check Gemini is available
python run_experiment.py --list-providers | grep -A 6 "GOOGLE:"

# Output should show:
# GOOGLE:
#   Model: gemini-2.5-pro (Gemini 2.5 Pro)
#   Max Tokens: 65,536
#   Cost: $1.25 / $10.00 per million tokens
#   Extended Thinking: Yes
#   Prompt Caching: Yes
#   Notes: Most capable Gemini model...
```

## Status

✓ **Complete and tested**
- Gemini 2.5 Pro configured
- Caching support enabled
- Display names updated
- Tests passing
- Ready for production use
