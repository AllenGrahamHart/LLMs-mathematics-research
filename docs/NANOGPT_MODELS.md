# NanoGPT Pretrained Models Guide

This guide explains how to use pretrained GPT-2 models with the nanoGPT codebase for ML research experiments.

## Available Models

We have downloaded two pretrained GPT-2 models optimized for budget-conscious research:

| Model | Parameters | Size | VRAM | Fine-tune Time* | Use Case |
|-------|-----------|------|------|----------------|----------|
| **gpt2** | 124M | ~500 MB | 2-4 GB | Few minutes | Quick experiments, debugging |
| **gpt2-medium** | 350M | ~1.5 GB | 6-8 GB | 10-30 minutes | Final experiments, quality results |

*Fine-tuning time on Shakespeare dataset (~1MB) with single GPU (T4/V100 class)

### Other Available Models (not downloaded)

| Model | Parameters | Size | VRAM | Fine-tune Time* |
|-------|-----------|------|------|----------------|
| gpt2-large | 774M | ~3 GB | 10-14 GB | 30-60 minutes |
| gpt2-xl | 1.5B | ~6 GB | 16+ GB | 1-2 hours |

## Quick Start

### Verify Models Are Working

```bash
python test_nanogpt_weights.py
```

This will test that both models load correctly and can generate text.

### Download Additional Models

```bash
# Download a specific model
python download_nanogpt_weights.py --model gpt2-large

# Download all models (requires ~11 GB)
python download_nanogpt_weights.py --all
```

## Usage in Research Experiments

### 1. Using with the Research Agent

The research agent can use these models automatically when working with nanoGPT code:

```bash
# Run ML research with nanoGPT context
python run_experiment.py \
  --code nanogpt \
  --problem problems/ml_research.txt \
  --max-iterations 10
```

The agent will have access to the nanoGPT codebase and can use these pretrained models for:
- Fine-tuning experiments
- Evaluation tasks
- Mechanistic interpretability studies
- Behavioral experiments

### 2. Direct Usage in Python Code

#### Loading a Model

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Or use gpt2-medium
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
```

#### Text Generation

```python
import torch

# Prepare input
prompt = "The meaning of life is"
inputs = tokenizer.encode(prompt, return_tensors='pt')

# Generate text
with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_length=100,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### 3. Using with nanoGPT Training Scripts

#### Fine-tuning on Custom Data

```bash
# Prepare your data (example: Shakespeare)
python data/shakespeare_char/prepare.py

# Fine-tune from gpt2 checkpoint
python train.py config/finetune_shakespeare.py
```

In your config file (e.g., `config/finetune_shakespeare.py`):

```python
# Start from pretrained GPT-2
init_from = 'gpt2'  # or 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

# Training settings for fine-tuning
learning_rate = 3e-5  # Lower LR for fine-tuning
max_iters = 1000      # Fewer iterations needed
```

#### Sampling from Pretrained Models

```bash
# Sample from gpt2
python sample.py --init_from=gpt2

# Sample from gpt2-medium with custom prompt
python sample.py \
  --init_from=gpt2-medium \
  --start="Once upon a time" \
  --num_samples=5 \
  --max_new_tokens=200
```

## Research Experiment Ideas

### 1. Baseline Evaluation
- Measure perplexity on various datasets
- Compare quality across model sizes
- Benchmark generation speed

### 2. Fine-tuning Studies
- Domain adaptation (code, scientific text, dialogue)
- Few-shot learning
- Hyperparameter sensitivity

### 3. Mechanistic Interpretability
- Attention pattern analysis
- Neuron activation studies
- Layer-wise representation analysis
- Probing classifiers

### 4. Behavioral Experiments
- Prompting strategies
- Instruction following
- Reasoning capabilities
- Factual knowledge extraction

## Cost Estimates (Cloud GPU)

Based on Modal/cloud GPU pricing:

### gpt2 (124M)
- **Hardware**: T4 GPU ($0.20-0.60/hour)
- **Fine-tuning**: ~5-10 minutes = $0.02-0.10 per experiment
- **Evaluation**: ~1-2 minutes = $0.01-0.02
- **Total for 10 experiments**: ~$0.30-1.20

### gpt2-medium (350M)
- **Hardware**: T4 GPU ($0.20-0.60/hour) or A10G ($0.80-1.50/hour)
- **Fine-tuning**: ~15-30 minutes = $0.05-0.75 per experiment
- **Evaluation**: ~2-5 minutes = $0.01-0.05
- **Total for 10 experiments**: ~$0.60-8.00

## Recommended Workflow

1. **Initial Development** (use gpt2):
   - Debug code and pipelines
   - Quick feasibility tests
   - Hyperparameter search
   - Development is fast and cheap

2. **Final Experiments** (use gpt2-medium):
   - Publication-quality results
   - Final benchmarks
   - Detailed analysis
   - Better quality justifies slightly higher cost

3. **Avoid for Budget Research** (gpt2-xl):
   - 3-10x more expensive
   - Much slower to run
   - Use only if quality difference is critical

## Technical Details

### Model Architecture

All GPT-2 models share the same architecture:
- Transformer decoder with causal attention
- Vocabulary size: 50,257 tokens
- Context length: 1,024 tokens
- BPE tokenization

### Differences by Size

| Model | Layers | Heads | Embedding Dim |
|-------|--------|-------|---------------|
| gpt2 | 12 | 12 | 768 |
| gpt2-medium | 24 | 16 | 1024 |
| gpt2-large | 36 | 20 | 1280 |
| gpt2-xl | 48 | 25 | 1600 |

### Storage Location

Models are cached in: `~/.cache/huggingface/hub/`

This directory is:
- Outside your git repository (safe from accidental commits)
- Shared across all your projects
- Automatically managed by HuggingFace transformers

## Troubleshooting

### Out of Memory Errors

If you get OOM errors during fine-tuning:

```python
# Reduce batch size
batch_size = 4  # or smaller

# Use gradient accumulation
gradient_accumulation_steps = 8

# Reduce context length
block_size = 512  # instead of 1024

# Use a smaller model
init_from = 'gpt2'  # instead of 'gpt2-medium'
```

### Slow Generation

For faster sampling:

```python
# Use greedy decoding instead of sampling
outputs = model.generate(inputs, do_sample=False)

# Reduce max_length
outputs = model.generate(inputs, max_length=50)

# Use cache (enabled by default)
outputs = model.generate(inputs, use_cache=True)
```

### Model Not Found

If you get "model not found" errors:

```bash
# Re-download the model
python download_nanogpt_weights.py --model gpt2
```

## Additional Resources

- [nanoGPT Repository](https://github.com/karpathy/nanoGPT)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [HuggingFace GPT-2 Documentation](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [OpenAI GPT-2 Blog Post](https://openai.com/blog/better-language-models/)

## License

The GPT-2 models are released by OpenAI under the MIT License. You are free to use them for research and commercial purposes.
