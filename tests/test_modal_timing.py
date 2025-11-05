import modal
import time

# Test: Measure actual Modal + GPT-2 timing
app = modal.App("test-gpt2-timing")

image = modal.Image.debian_slim().pip_install("transformers", "torch")

@app.function(gpu="T4", timeout=300, image=image)
def test_gpt2_load_time():
    """Test how long it takes to load GPT-2"""
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import time

    start = time.time()
    print("Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")

    # Try one inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model on {device}")

    prompt = "5 × 3 ="
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    inf_start = time.time()
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=5)
    inf_time = time.time() - inf_start

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Inference took {inf_time:.2f}s")
    print(f"Result: {result}")

    return {"load_time": load_time, "inference_time": inf_time, "result": result}

if __name__ == "__main__":
    print("Running Modal timing test...")
    overall_start = time.time()

    try:
        with app.run():
            result = test_gpt2_load_time.remote()

        overall_time = time.time() - overall_start
        print(f"\n✓ SUCCESS!")
        print(f"Total wall time: {overall_time:.2f}s")
        print(f"Model load time: {result['load_time']:.2f}s")
        print(f"Inference time: {result['inference_time']:.2f}s")
        print(f"Result: {result['result']}")

    except Exception as e:
        overall_time = time.time() - overall_start
        print(f"\n✗ FAILED after {overall_time:.2f}s")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
