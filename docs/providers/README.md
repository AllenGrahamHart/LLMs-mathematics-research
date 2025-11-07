# Multi-Provider Documentation

This directory contains documentation for the multi-provider LLM support system.

## Documents

### [MULTI_PROVIDER_GUIDE.md](MULTI_PROVIDER_GUIDE.md)
Comprehensive guide to using multiple LLM providers (Anthropic, OpenAI, Google, xAI, Moonshot).

**Contents:**
- Overview of supported providers
- Setup instructions for each provider
- Configuration examples
- Feature comparison matrix
- Cost optimization tips
- Troubleshooting guide
- Architecture overview

### [PROVIDER_DEFAULTS_GUIDE.md](PROVIDER_DEFAULTS_GUIDE.md)
Guide to the hybrid provider defaults system.

**Contents:**
- Quick start with CLI flags
- How provider defaults work
- Overriding default settings
- Adding new providers
- Architecture explanation

### [GEMINI_UPDATE.md](GEMINI_UPDATE.md)
Details about the Gemini 2.5 Pro integration (latest update).

**Contents:**
- What changed
- Gemini 2.5 Pro features
- Pricing and caching information
- Usage examples
- Testing information

### [IMPLEMENTATION_CHANGES.md](IMPLEMENTATION_CHANGES.md)
Technical implementation details and change log.

**Contents:**
- Complete file-by-file changes
- Architecture decisions
- Breaking changes (none!)
- Migration guide
- Verification checklist

## Quick Links

- **Main README**: [`../../README.md`](../../README.md)
- **Provider Tests**: [`../../tests/integration/`](../../tests/integration/)
- **Configuration**: [`../../config.yaml`](../../config.yaml)
- **Run Script**: [`../../run_experiment.py`](../../run_experiment.py)

## Quick Start

### Use a Different Provider

```bash
# CLI (easiest)
python run_experiment.py --provider google --problem problems/open_research.txt

# Or edit config.yaml
api:
  provider: google
```

### See Available Providers

```bash
python run_experiment.py --list-providers
```

### Test Your Setup

```bash
cd tests/integration
python test_providers.py
python test_gemini_config.py
```

## Support

For issues or questions:
1. Check the relevant guide above
2. See the main README for general usage
3. Report issues at: https://github.com/AllenGrahamHart/llm-maths-research/issues
