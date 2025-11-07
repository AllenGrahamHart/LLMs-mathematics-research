# Provider Defaults System

## Overview

The hybrid provider defaults system gives you the best of both worlds:
- **Simple**: Just `--provider openai` and you're done
- **Flexible**: Override any setting in `config.yaml` if needed
- **Maintainable**: Provider specs live in code, easy to update

## Quick Start

### Method 1: CLI (Easiest)

```bash
# Use OpenAI instead of default
python run_experiment.py --provider openai --problem problems/open_research.txt

# Use Gemini
python run_experiment.py --provider google --problem problems/open_research.txt

# List all available providers
python run_experiment.py --list-providers
```

### Method 2: Config File

Edit `config.yaml`:

```yaml
api:
  provider: openai  # Change this line
```

That's it! The system automatically uses sensible defaults for each provider.

### Method 3: Python API

```python
from src.llm_maths_research.config import set_provider

# Switch provider at runtime
set_provider('openai')
```

## Provider Defaults

Each provider has pre-configured defaults for the most powerful/recommended model:

| Provider | Model | Max Tokens | Special Features |
|----------|-------|------------|------------------|
| **anthropic** | claude-sonnet-4-5-20250929 | 64,000 | Extended thinking (32k), Prompt caching |
| **openai** | gpt-4o | 16,000 | Fast and balanced |
| **google** | gemini-2.0-flash-exp | 8,192 | Free tier available |
| **xai** | grok-beta | 32,768 | Large context |
| **moonshot** | moonshot-v1-128k | 16,000 | Very cost-effective |

## Overriding Defaults

Sometimes you want to customize specific settings. Just uncomment and edit in `config.yaml`:

```yaml
api:
  provider: anthropic  # Use Anthropic

  # Override specific settings:
  model: claude-opus-4-20250514  # Use a different Claude model
  max_tokens: 32000              # Reduce token limit
  thinking_budget: 16000         # Reduce thinking budget
  costs:
    input_per_million: 15.0      # Update pricing (if changed)
```

You only need to specify what you want to change. Everything else uses the provider defaults.

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     config.yaml                          │
│                  (User preferences)                      │
│                                                          │
│   api:                                                   │
│     provider: openai  ← Simple provider selection       │
│     # Optional overrides only if needed                 │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              provider_defaults.py                        │
│            (System-maintained specs)                     │
│                                                          │
│  PROVIDER_DEFAULTS = {                                   │
│    'openai': {                                           │
│      'model': 'gpt-4o',                                  │
│      'max_tokens': 16000,                                │
│      'costs': {...},                                     │
│    },                                                    │
│    ...                                                   │
│  }                                                       │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   config.py                              │
│              (Merge logic)                               │
│                                                          │
│  CONFIG = defaults + user_overrides                      │
└─────────────────────────────────────────────────────────┘
```

### Loading Process

1. **Read config.yaml** - Get user's provider choice
2. **Load provider defaults** - Get specs from `provider_defaults.py`
3. **Apply overrides** - Merge any custom settings from config.yaml
4. **Create CONFIG** - Final configuration ready to use

### Example

**config.yaml**:
```yaml
api:
  provider: openai
  max_tokens: 8000  # Override just this
```

**Result**:
```python
CONFIG['api'] = {
    'provider': 'openai',
    'model': 'gpt-4o',            # From defaults
    'max_tokens': 8000,            # From config.yaml (override)
    'thinking_budget': None,       # From defaults
    'costs': {                     # From defaults
        'input_per_million': 2.5,
        'output_per_million': 10.0,
    }
}
```

## Adding a New Provider

To add support for a new provider (e.g., "cohere"):

1. **Add to `provider_defaults.py`**:

```python
PROVIDER_DEFAULTS = {
    # ... existing providers ...
    'cohere': {
        'model': 'command-r-plus',
        'display_name': 'Command R+',
        'max_tokens': 4096,
        'thinking_budget': None,
        'supports_thinking': False,
        'supports_caching': False,
        'costs': {
            'input_per_million': 3.0,
            'output_per_million': 15.0,
            'cache_write_multiplier': 1.0,
            'cache_read_multiplier': 1.0,
        },
        'notes': 'Great for RAG and enterprise use cases.',
    },
}
```

2. **Implement provider class in `llm_provider.py`**:

```python
class CohereProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        import cohere
        self.client = cohere.Client(api_key)

    # Implement abstract methods...
```

3. **Add to factory function**:

```python
def create_provider(provider_name: str, api_key: str, model: str):
    providers = {
        # ... existing providers ...
        'cohere': CohereProvider,
    }
    # ...
```

4. **Add environment variable mapping in `session.py`**:

```python
env_var_map = {
    # ... existing mappings ...
    'cohere': 'COHERE_API_KEY',
}
```

That's it! Users can now use `--provider cohere`.

## Command Reference

### List Providers

```bash
python run_experiment.py --list-providers
```

Output:
```
ANTHROPIC:
  Model: claude-sonnet-4-5-20250929 (Claude Sonnet 4.5)
  Max Tokens: 64,000
  Cost: $3.00 / $15.00 per million tokens
  Extended Thinking: Yes (budget: 32,000 tokens)
  Prompt Caching: Yes
  Notes: Best for complex reasoning tasks.

OPENAI:
  Model: gpt-4o (GPT-4o)
  Max Tokens: 16,000
  Cost: $2.50 / $10.00 per million tokens
  Notes: Fast and cost-effective.

...
```

### Test Provider Info

```bash
# From Python
python -c "from src.llm_maths_research.provider_defaults import get_provider_info; print(get_provider_info('openai'))"
```

### Check Current Config

```bash
python -c "from src.llm_maths_research.config import CONFIG; print(f'Provider: {CONFIG[\"api\"][\"provider\"]}'); print(f'Model: {CONFIG[\"api\"][\"model\"]}')"
```

## Benefits

### For Users

1. **Zero configuration** - Just pick a provider
2. **Automatic updates** - When costs change, we update `provider_defaults.py`
3. **Flexibility** - Can still override anything if needed
4. **Discovery** - `--list-providers` shows what's available

### For Maintainers

1. **Single source of truth** - Provider specs in one file
2. **Version controlled** - Changes tracked in git
3. **Testable** - Easy to verify all providers have complete configs
4. **Documented** - Code is the documentation

## FAQ

**Q: What if I want to use a different OpenAI model?**

A: Just override in config.yaml:
```yaml
api:
  provider: openai
  model: gpt-4-turbo  # Different model
```

**Q: Can I change costs without editing Python?**

A: Yes! Override in config.yaml:
```yaml
api:
  provider: openai
  costs:
    input_per_million: 5.0  # Updated price
```

**Q: What if my provider adds a new feature?**

A: We update `provider_defaults.py`. Your existing config keeps working.

**Q: Can I use CLI and config.yaml together?**

A: Yes! CLI overrides config.yaml. For example:
```bash
# config.yaml says anthropic, but this uses openai
python run_experiment.py --provider openai ...
```

**Q: How do I know what options each provider supports?**

A: Run `python run_experiment.py --list-providers` or check `provider_defaults.py`.

## See Also

- `MULTI_PROVIDER_GUIDE.md` - Full multi-provider documentation
- `provider_defaults.py` - Provider specifications
- `config.py` - Configuration loading logic
- `llm_provider.py` - Provider implementations
