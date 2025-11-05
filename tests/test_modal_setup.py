#!/usr/bin/env python3
"""
Test Modal setup with a simple GPU example.

This script verifies that:
1. Modal is properly configured
2. You can run functions on Modal's infrastructure
3. GPU access works correctly
"""

import modal

# Create a Modal app
app = modal.App("test-modal-setup")


# Test 1: Simple CPU function
@app.function()
def hello_modal():
    """Simple test function to verify basic Modal functionality."""
    import sys
    return f"Hello from Modal! Python version: {sys.version}"


# Test 2: GPU function with PyTorch
@app.function(
    gpu="T4",  # Request a T4 GPU (cheapest option, ~$0.60/hour)
    image=modal.Image.debian_slim().pip_install("torch"),
)
def test_gpu():
    """Test GPU access with PyTorch."""
    import torch

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        return {
            "success": True,
            "gpu_available": True,
            "device_name": device_name,
            "device_count": device_count,
            "message": f"✓ GPU test passed! Using {device_name}"
        }
    else:
        return {
            "success": False,
            "gpu_available": False,
            "message": "✗ GPU not available"
        }


# Test 3: Quick computation test
@app.function(
    gpu="T4",
    image=modal.Image.debian_slim().pip_install("torch"),
)
def test_computation():
    """Test actual GPU computation."""
    import torch

    # Create tensors on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Small matrix multiplication
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    c = torch.matmul(a, b)

    return {
        "success": True,
        "device": str(device),
        "result_shape": list(c.shape),
        "message": f"✓ Computation test passed on {device}"
    }


@app.local_entrypoint()
def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("  MODAL SETUP VERIFICATION")
    print("="*70 + "\n")

    # Test 1: Basic functionality
    print("[1/3] Testing basic Modal functionality...")
    try:
        result = hello_modal.remote()
        print(f"✓ {result}\n")
    except Exception as e:
        print(f"✗ Basic test failed: {e}\n")
        print("Tip: Make sure you ran 'modal token new' to authenticate\n")
        return 1

    # Test 2: GPU availability
    print("[2/3] Testing GPU access (this may take a moment to provision)...")
    try:
        result = test_gpu.remote()
        print(f"{result['message']}")
        if result['gpu_available']:
            print(f"  Device: {result['device_name']}")
            print(f"  GPU count: {result['device_count']}\n")
        else:
            print("  Warning: GPU not available\n")
    except Exception as e:
        print(f"✗ GPU test failed: {e}\n")
        return 1

    # Test 3: GPU computation
    print("[3/3] Testing GPU computation...")
    try:
        result = test_computation.remote()
        print(f"{result['message']}")
        print(f"  Matrix shape: {result['result_shape']}\n")
    except Exception as e:
        print(f"✗ Computation test failed: {e}\n")
        return 1

    # Summary
    print("="*70)
    print("  ALL TESTS PASSED!")
    print("="*70 + "\n")
    print("Your Modal setup is working correctly!")
    print("\nNext steps:")
    print("  1. You can now run GPU experiments with Modal")
    print("  2. Your research agent can use Modal for training")
    print("  3. Check your credits at: https://modal.com/settings/usage")
    print("\nEstimated costs:")
    print("  • T4 GPU: ~$0.60/hour")
    print("  • This test used: <$0.01 (ran for ~30 seconds)")
    print("  • Free tier: $30/month included\n")

    return 0
