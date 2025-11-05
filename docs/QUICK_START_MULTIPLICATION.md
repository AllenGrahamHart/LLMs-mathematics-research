# Quick Start: GPT-2 Multiplication Research

This guide helps you run the GPT-2 multiplication reasoning experiment.

## What This Experiment Does

Investigates whether GPT-2 can learn to multiply numbers accurately:
1. **Baseline evaluation**: Test pretrained GPT-2 on multiplication problems
2. **Fine-tuning**: Train the model to improve multiplication performance
3. **Analysis**: Measure improvements and understand what was learned
4. **Paper**: Generate a complete research paper with results

## Prerequisites

✅ **Already set up:**
- GPT-2 models downloaded (gpt2 124M, gpt2-medium 350M)
- Modal configured for GPU training
- nanoGPT codebase available as context

## Running the Experiment

### Basic Usage

```bash
python run_experiment.py \
  --code nanogpt \
  --problem problems/gpt2_multiplication.txt \
  --max-iterations 15
```

This will:
- Load the nanoGPT codebase as context
- Run the multiplication research task
- Generate code, train models, and write a paper
- Iterate based on critique feedback

### With Custom Session Name

```bash
python run_experiment.py \
  --code nanogpt \
  --problem problems/gpt2_multiplication.txt \
  --session-name multiplication_study_v1 \
  --max-iterations 15
```

### Resume from Previous Run

```bash
# Resume from iteration 5
python run_experiment.py \
  --code nanogpt \
  --problem problems/gpt2_multiplication.txt \
  --session-name multiplication_study_v1 \
  --start-iteration 5
```

## Expected Workflow

The research agent will autonomously:

1. **Design evaluation protocol**
   - Create test sets with varying multiplication difficulty
   - Set up prompting format (e.g., "Q: What is 7 × 8? A:")

2. **Run baseline evaluation**
   - Load pretrained GPT-2
   - Test on multiplication problems
   - Measure accuracy by difficulty level

3. **Design training intervention**
   - Create training dataset (correct multiplication examples)
   - Choose approach (supervised fine-tuning, chain-of-thought, etc.)
   - Set up Modal GPU training if needed

4. **Fine-tune the model**
   - Run training (10-30 minutes on T4 GPU)
   - Monitor training loss

5. **Post-training evaluation**
   - Re-test on same problems
   - Measure improvements
   - Analyze what was learned

6. **Write paper**
   - Introduction and motivation
   - Methodology
   - Results with tables and figures
   - Discussion and conclusions

## Expected Outputs

The session directory (`outputs/[session-name]/`) will contain:

### Code Files
- `code.py` - Complete experimental code
  - Data generation
  - Baseline evaluation
  - Training setup
  - Post-training evaluation

### Paper
- `paper.tex` - LaTeX source
- `paper.pdf` - Compiled PDF (if LaTeX compilation succeeds)

### Figures
- Accuracy comparison plots (before/after)
- Training loss curves
- Error analysis visualizations

### Data
- `multiplication_test.txt` - Test problems
- `multiplication_train.txt` - Training data
- `results_baseline.json` - Baseline accuracy
- `results_finetuned.json` - Post-training accuracy

### Logs
- `session_log.txt` - Complete execution log
- `metrics.json` - API costs and token usage
- `plans.txt` - Research plans per iteration
- `critiques.txt` - Critic feedback per iteration

## Cost Estimates

**Total expected cost: $0.50 - $2.00**

Breakdown:
- API calls (Claude): $0.20-0.80 (with prompt caching)
- Modal GPU training: $0.10-0.30 (10-30 min on T4)
- Evaluation: ~$0.01 (inference is cheap)

With 15 iterations and prompt caching, should stay well under $2.

## Monitoring Progress

### Check Current Iteration
```bash
ls -lt outputs/[session-name]/ | head -20
```

### View Latest Plan
```bash
cat outputs/[session-name]/current_plan.txt
```

### View Latest Critique
```bash
cat outputs/[session-name]/current_critique.txt
```

### Check Costs So Far
```bash
cat outputs/[session-name]/metrics.json | grep cost
```

### View Modal GPU Usage
Visit: https://modal.com/settings/usage

## Tips for Success

### Let the Agent Work
- The agent will design the experiment autonomously
- Trust the iterative refinement process
- Critiques help improve quality each iteration

### Budget Management
- Start with max 10-15 iterations
- Monitor costs in `metrics.json`
- Agent uses prompt caching (saves ~40% on API costs)

### Fine-tuning Tips
The agent will likely:
- Use supervised fine-tuning (simplest, most effective)
- Create balanced training data across difficulty levels
- Use Modal for GPU training (automatic in the code)
- Train for a small number of epochs (avoid overfitting)

### Common Patterns
The agent typically:
- Iteration 1-3: Setup and baseline evaluation
- Iteration 4-6: Design and run training
- Iteration 7-10: Post-training evaluation
- Iteration 11-15: Paper writing and refinement

## Example Results to Expect

**Baseline GPT-2 Accuracy:**
- 1-digit × 1-digit: 60-80% (some basic memorization)
- 1-digit × 2-digit: 10-30% (struggles)
- 2-digit × 2-digit: <5% (essentially random)

**After Fine-tuning:**
- 1-digit × 1-digit: 95-100% (should master)
- 1-digit × 2-digit: 60-90% (significant improvement)
- 2-digit × 2-digit: 20-60% (may still struggle, but better)

## Troubleshooting

### "Out of memory" during training
The agent should automatically use gpt2 (124M), but if it uses gpt2-medium:
- Smaller batch size will be needed
- May need gradient accumulation
- Agent should handle this automatically

### Training takes too long
- Should complete in 10-30 minutes
- Check Modal dashboard: https://modal.com/apps
- Agent may need to reduce training epochs

### Accuracy doesn't improve
This is actually a valid research finding! The paper would discuss:
- Why multiplication is hard for language models
- What architectures might work better
- Limitations of the approach

### Paper won't compile
- Check `session_log.txt` for LaTeX errors
- Often minor issues (missing packages, bad formatting)
- Agent will fix in next iteration based on critic feedback

## Advanced: Modifying the Problem

Edit `problems/gpt2_multiplication.txt` to:
- Change difficulty levels (3-digit, 4-digit numbers)
- Try different operations (addition, division)
- Request different training approaches
- Adjust paper length or focus areas

## Next Steps

After a successful run:
1. Review the paper in `outputs/[session-name]/paper.pdf`
2. Examine the code in `outputs/[session-name]/code.py`
3. Check figures for insights
4. Consider running variations:
   - Different training approaches
   - Different difficulty levels
   - Comparison with gpt2-medium

## Research Ideas to Explore

This problem can be extended:
- **Addition vs multiplication**: Which is easier to learn?
- **Model size**: Does gpt2-medium learn faster/better?
- **Training data size**: How much data is needed?
- **Chain-of-thought**: Does showing work help?
- **Transfer learning**: Does learning multiplication help with division?

Good luck with your experiment! 🚀
