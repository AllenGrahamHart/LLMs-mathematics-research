#!/usr/bin/env python3
"""Test Modal image build with verbose output to diagnose build failures."""

import modal

# Enable verbose output to see build logs
modal.enable_output()

app = modal.App("test-image-build")

# Test the exact same image configuration from the generated code
image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn"
)

@app.function(image=image)
def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    import torch
    print(f"✓ torch {torch.__version__}")

    import transformers
    print(f"✓ transformers {transformers.__version__}")

    import numpy
    print(f"✓ numpy {numpy.__version__}")

    import pandas
    print(f"✓ pandas {pandas.__version__}")

    import matplotlib
    print(f"✓ matplotlib {matplotlib.__version__}")

    import seaborn
    print(f"✓ seaborn {seaborn.__version__}")

    print("\nAll imports successful!")
    return "SUCCESS"

if __name__ == "__main__":
    print("Building Modal image with verbose output...")
    print("=" * 60)

    with app.run():
        result = test_imports.remote()
        print(f"\nResult: {result}")
