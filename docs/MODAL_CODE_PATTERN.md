# Modal Code Pattern for LLM Math Research

## Problem

When running code that uses Modal for GPU computation within the research framework:
- Modal tries to serialize the entire project directory
- This causes import errors and timeouts
- Need GPU code remote, but file I/O local
- Must respect session output directory structure

## Solution: Data-Only Returns Pattern

### Correct Pattern

```python
import modal
import os

# 1. Create Modal app
app = modal.App("my-experiment")

# 2. Define image with ONLY the packages needed for GPU computation
image = modal.Image.debian_slim().pip_install(
    "transformers",
    "torch",
    "numpy"
)

# 3. Modal functions return DATA, not files
@app.function(gpu="T4", timeout=1800, image=image)
def run_model_evaluation(problems):
    """Run GPU computation and return results as data structures.

    NO file I/O here - just return the data!
    """
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    # Load model (happens on GPU)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = model.to("cuda")

    # Run computation
    results = []
    for problem in problems:
        # ... do inference ...
        results.append({
            "input": problem,
            "output": predicted_text,
            "correct": correct_answer,
            "time": inference_time
        })

    # Return data structures (lists, dicts, primitives)
    return results

# 4. Main execution: Modal for compute, local for I/O
if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    import time

    print("Generating test problems...")
    problems = generate_test_problems()

    print("Running evaluation on Modal GPU...")
    start = time.time()

    # This is the ONLY Modal interaction - get data back
    with app.run():
        results = run_model_evaluation.remote(problems)

    eval_time = time.time() - start
    print(f"Evaluation completed in {eval_time:.2f}s")

    # 5. ALL file I/O happens locally using output_dir
    # (output_dir is pre-defined by the research framework)

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
    print(f"✓ Saved results.csv")

    # Generate figures locally
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['time'])
    ax.set_xlabel("Problem")
    ax.set_ylabel("Inference Time (s)")
    plt.savefig(os.path.join(output_dir, "timing.png"), dpi=300)
    plt.close()
    print(f"✓ Saved timing.png")

    # Calculate metrics locally
    accuracy = sum(1 for r in results if r['output'] == r['correct']) / len(results)
    print(f"Final accuracy: {accuracy:.1%}")
```

### Key Principles

1. **Modal functions are pure computation**
   - Take data in (primitives, lists, dicts, numpy arrays)
   - Return data out (same types)
   - NO file operations (no `open()`, no `pd.to_csv()`, no `plt.savefig()`)

2. **Local code handles all I/O**
   - Use the pre-defined `output_dir` variable
   - All CSV, PNG, JSON files saved locally
   - Use `os.path.join(output_dir, "filename")`

3. **Minimal Modal image**
   - Only include packages needed for GPU computation
   - Don't include pandas, matplotlib if not needed on GPU

4. **Single Modal session when possible**
   - Avoid multiple `with app.run():` blocks
   - Batch all GPU work into one session
   - This saves ~2-3 minutes of Modal startup overhead per session

5. **CRITICAL: Use `if __name__ == "__main__":` guard**
   - Modal imports your file to serialize functions
   - All execution code MUST be inside this block
   - Without it: `InvalidError: Can not run an app in global scope within a container`

### Anti-Patterns (DON'T DO THIS)

```python
# ❌ BAD: No if __name__ == "__main__" guard
import modal

app = modal.App("my-app")
image = modal.Image.debian_slim().pip_install("torch")

@app.function(gpu="T4", image=image)
def my_function(data):
    return process(data)

# This runs during Modal import and causes errors!
with app.run():  # ❌ InvalidError: Can not run an app in global scope within a container
    results = my_function.remote(data)

# ✅ CORRECT: Use if __name__ == "__main__" guard
if __name__ == "__main__":
    with app.run():
        results = my_function.remote(data)

# ❌ BAD: File I/O inside Modal function
@app.function(gpu="T4", image=image)
def bad_function(problems):
    results = run_inference(problems)

    # DON'T DO THIS - file paths won't work correctly
    pd.DataFrame(results).to_csv("results.csv")  # ❌
    plt.savefig("figure.png")  # ❌

    return results

# ❌ BAD: Multiple Modal sessions
with app.run():
    pilot = run_pilot.remote(small_set)

with app.run():  # This starts a NEW session (adds 2-3 min overhead)
    results = run_full.remote(full_set)

# ❌ BAD: Trying to pass output_dir to Modal
@app.function(gpu="T4", image=image)
def bad_function(output_dir):  # output_dir is local path!
    # This won't work - Modal container has different filesystem
    with open(os.path.join(output_dir, "file.txt"), "w") as f:
        f.write("data")
```

### Good Patterns

```python
# ✅ GOOD: Combine multiple operations in one Modal function
@app.function(gpu="T4", image=image)
def run_all_evaluations(pilot_problems, full_problems):
    """Do all GPU work in one function to avoid multiple sessions."""
    model = load_model()  # Load once

    pilot_results = [run_inference(model, p) for p in pilot_problems]
    full_results = [run_inference(model, p) for p in full_problems]

    return {
        "pilot": pilot_results,
        "full": full_results
    }

# ✅ GOOD: Single Modal session
with app.run():
    all_results = run_all_evaluations.remote(pilot_set, full_set)

# Then handle all I/O locally
save_results(all_results, output_dir)
create_figures(all_results, output_dir)
```

### Debugging Modal Issues

If you get `ModuleNotFoundError` or import errors:

1. Check Modal logs: `modal app logs <app-id>`
2. Modal is probably trying to serialize files it shouldn't
3. Make sure Modal functions don't import local project code
4. Keep Modal functions completely self-contained

### Critical: Include ALL Packages in Modal Image

**IMPORTANT**: Modal imports your entire file to serialize functions. You must include ALL packages you import, even if they're only used locally:

```python
import modal
import pandas as pd          # Used locally
import matplotlib.pyplot as plt  # Used locally

app = modal.App("experiment")

# Include ALL packages - even those only used locally!
image = modal.Image.debian_slim().pip_install(
    "transformers",
    "torch",
    "pandas",      # Must include!
    "matplotlib",  # Must include!
    "numpy"
)

@app.function(gpu="T4", timeout=1800, image=image)
def run_inference(problems):
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    # Modal caches model automatically after first download
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # ... inference code ...
    return results  # Return data, not files
```

**Why?** When Modal serializes your functions, it imports the entire Python file. If you have `import pandas` at the top of your file, Modal needs pandas in the image to import successfully, even if pandas is never used inside the Modal function itself.

**Why not pre-download in image build?**
- Image builds have strict timeouts and can fail for large models
- Model downloads during inference have longer timeouts
- Modal caches models automatically anyway after first run
- Simpler and more reliable to let Modal handle it

**Performance:**
- First run: ~3-5 minutes (includes model download)
- Subsequent runs: ~10-30 seconds (uses cached model)
- Modal handles all caching transparently

### Example: Refactoring the GPT-2 Evaluation

Instead of:
- Pilot test in Modal (load model, save results)
- Full eval in Modal (load model again, save results)
- Post-processing locally

Do:
- Single Modal function: Load model once, return all inference data
- All CSV/PNG generation locally with proper paths
- Let Modal handle model caching automatically

This reduces:
- 30+ minutes timeout → ~5-10 minutes completion
- 2 Modal sessions → 1 Modal session
- 2 model loads → 1 model load
- Modal file I/O issues → Pure data passing
- Multiple image builds → Simple, reliable pattern
