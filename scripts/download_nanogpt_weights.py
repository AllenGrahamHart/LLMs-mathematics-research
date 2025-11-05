#!/usr/bin/env python3
"""
Download and cache pretrained GPT-2 weights for nanoGPT experiments.

This script pre-downloads GPT-2 checkpoints from HuggingFace so that they're
available immediately when the research agent needs them, avoiding delays and
ensuring experiments can run offline if needed.

The weights are cached in the standard transformers cache directory
(~/.cache/huggingface/hub by default).
"""

import argparse
from transformers import GPT2LMHeadModel


# Available GPT-2 model sizes with their parameter counts and disk sizes
AVAILABLE_MODELS = {
    'gpt2': {'params': '124M', 'size': '~500 MB', 'time': 'few minutes'},
    'gpt2-medium': {'params': '350M', 'size': '~1.5 GB', 'time': '10-30 minutes'},
    'gpt2-large': {'params': '774M', 'size': '~3 GB', 'time': '30-60 minutes'},
    'gpt2-xl': {'params': '1558M', 'size': '~6 GB', 'time': '1-2 hours'},
}


def download_model(model_name: str):
    """Download and cache a GPT-2 model."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {', '.join(AVAILABLE_MODELS.keys())}"
        )

    info = AVAILABLE_MODELS[model_name]
    print(f"\n{'='*70}")
    print(f"Downloading: {model_name}")
    print(f"  Parameters: {info['params']}")
    print(f"  Disk size: {info['size']}")
    print(f"  Fine-tune time (Shakespeare): {info['time']}")
    print(f"{'='*70}\n")

    # Load model (this will download and cache it)
    print("Downloading from HuggingFace... (this may take a moment)")
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Get model size info
    total_params = sum(p.numel() for p in model.parameters())
    total_params_m = total_params / 1_000_000

    print(f"\n✓ Successfully downloaded {model_name}")
    print(f"  Total parameters: {total_params:,} ({total_params_m:.1f}M)")
    print(f"  Model is cached and ready for use in nanoGPT\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained GPT-2 weights for nanoGPT research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  gpt2         124M parameters,  ~500 MB  (fastest, good for initial experiments)
  gpt2-medium  350M parameters,  ~1.5 GB  (balanced size/performance) ★ RECOMMENDED
  gpt2-large   774M parameters,  ~3 GB    (larger, better quality)
  gpt2-xl      1558M parameters, ~6 GB    (largest, best quality, slower)

Examples:
  # Download recommended models for budget research
  python download_nanogpt_weights.py --model gpt2 gpt2-medium

  # Download a single model
  python download_nanogpt_weights.py --model gpt2

  # Download all models (may take a while and use ~11 GB!)
  python download_nanogpt_weights.py --all

Recommended for budget-conscious research:
  Start with gpt2 (124M) for quick experiments and debugging,
  then use gpt2-medium (350M) for final evaluations.
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        nargs='+',
        choices=list(AVAILABLE_MODELS.keys()),
        help='Model(s) to download'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all available models'
    )

    args = parser.parse_args()

    # Determine which models to download
    if args.all:
        models_to_download = list(AVAILABLE_MODELS.keys())
    elif args.model:
        models_to_download = args.model
    else:
        # Default: download the two recommended models
        print("No model specified. Downloading gpt2 and gpt2-medium by default.")
        print("Use --help to see all options.\n")
        models_to_download = ['gpt2', 'gpt2-medium']

    # Download models
    print(f"\n{'='*70}")
    print(f"  Will download {len(models_to_download)} model(s)")
    print(f"{'='*70}")

    for i, model_name in enumerate(models_to_download, 1):
        print(f"\n[{i}/{len(models_to_download)}] Processing {model_name}...")
        try:
            download_model(model_name)
        except Exception as e:
            print(f"✗ Error downloading {model_name}: {e}\n")
            continue

    print(f"\n{'='*70}")
    print("  DOWNLOAD COMPLETE!")
    print(f"{'='*70}\n")
    print("The models are now cached and ready to use in your experiments.\n")
    print("Usage in nanoGPT:")
    print("  • In train.py: set init_from='gpt2' or init_from='gpt2-medium'")
    print("  • In sample.py: python sample.py --init_from=gpt2-medium")
    print("  • In your research code: model = GPT.from_pretrained('gpt2')\n")
    print("What you can do with these models:")
    print("  ✓ Fast fine-tuning experiments (instead of training from scratch)")
    print("  ✓ Immediate chatbot/generation capabilities (via sampling)")
    print("  ✓ Baseline evaluations (perplexity, generation quality)")
    print("  ✓ Mechanistic interpretability (attention patterns, neurons)")
    print("  ✓ Efficient hyperparameter tuning\n")
    print("Recommended workflow:")
    print("  1. Use gpt2 (124M) for initial testing and debugging")
    print("  2. Use gpt2-medium (350M) for final experiments and results\n")


if __name__ == '__main__':
    main()
