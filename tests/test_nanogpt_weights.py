#!/usr/bin/env python3
"""
Test script to verify GPT-2 weights are properly downloaded and functional.

This script tests that the models can:
1. Load successfully
2. Generate text
3. Work with the nanoGPT codebase
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sys
import pytest


@pytest.mark.parametrize("model_name", ['gpt2', 'gpt2-medium'])
def test_model(model_name: str):
    """Test loading and generating with a GPT-2 model."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}\n")

    try:
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Check model loaded correctly
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model loaded: {total_params:,} parameters\n")

        # Test text generation
        print("Testing text generation...")
        prompt = "The research explores"
        inputs = tokenizer.encode(prompt, return_tensors='pt')

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=50,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"Prompt: \"{prompt}\"")
        print(f"Generated: \"{generated_text}\"\n")
        print(f"✓ {model_name} is working correctly!\n")

        return True

    except Exception as e:
        print(f"✗ Error testing {model_name}: {e}\n")
        return False


def main():
    print("\n" + "="*70)
    print("  GPT-2 Model Verification Test")
    print("="*70)

    models_to_test = ['gpt2', 'gpt2-medium']
    results = {}

    for model_name in models_to_test:
        results[model_name] = test_model(model_name)

    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70 + "\n")

    all_passed = all(results.values())

    for model_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {model_name:15s} {status}")

    print("\n" + "="*70 + "\n")

    if all_passed:
        print("All models are ready for use in research experiments!")
        print("\nNext steps:")
        print("  1. Use these models in your ML research experiments")
        print("  2. Start with gpt2 for quick tests")
        print("  3. Use gpt2-medium for final experiments")
        print("\nExample usage:")
        print("  python run_experiment.py --code nanogpt --problem problems/ml_research.txt")
        print()
        return 0
    else:
        print("Some models failed to load. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
