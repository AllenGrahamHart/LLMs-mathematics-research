import modal
import time

# OPTIMIZED: Pre-download the model into the image itself
app = modal.App("test-cached-gpt2")

# This function runs ONCE when building the image, not every time the function runs
def download_model():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("Downloading GPT-2 into image cache...")
    GPT2Tokenizer.from_pretrained("gpt2")
    GPT2LMHeadModel.from_pretrained("gpt2")
    print("Model downloaded and cached in image!")

# Build image with model pre-downloaded
image = (
    modal.Image.debian_slim()
    .pip_install("transformers", "torch")
    .run_function(download_model)  # This downloads the model INTO the image
)

@app.function(gpu="T4", timeout=300, image=image)
def test_cached_inference(prompt):
    """With cached model, this should be much faster!"""
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import time

    # Model is already cached in the image, so this should be fast
    start = time.time()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s (from image cache)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    inf_start = time.time()
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=5)
    inf_time = time.time() - inf_start

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "load_time": load_time,
        "inference_time": inf_time,
        "result": result
    }

if __name__ == "__main__":
    print("Testing CACHED Modal image with pre-downloaded model...")
    print("First run will be slow (building image), subsequent runs will be fast!\n")

    overall_start = time.time()

    with app.run():
        result = test_cached_inference.remote("5 × 3 =")

    overall_time = time.time() - overall_start

    print(f"\n✓ SUCCESS!")
    print(f"Total wall time: {overall_time:.2f}s")
    print(f"Model load time: {result['load_time']:.2f}s")
    print(f"Inference time: {result['inference_time']:.2f}s")
    print(f"Result: {result['result']}")
    print(f"\nSavings: Image caching reduces model load from 6.5s to ~{result['load_time']:.2f}s")
    print(f"And subsequent runs will use the cached image (much faster startup!)")
