# Stage Information Placement in Generator Prompt

## Question
Where does the stage information (i.e., are we planning, coding, writing) get passed to the generator relative to other information?

## Answer

The stage information is passed via the `{output_format_instructions}` placeholder in the **static (cacheable) content** at **line 108** of the template, which comes **BEFORE** the dynamic current state information.

---

## Prompt Structure

### Full Template Flow (generator_prompt.txt)

```
┌─────────────────────────────────────────────────────────────┐
│ STATIC CONTENT (Cacheable - Same across all 3 stages)      │
├─────────────────────────────────────────────────────────────┤
│ Lines 1-106: Instructions                                   │
│   - Role description                                        │
│   - Problem statement                                       │
│   - Reference papers                                        │
│   - Data files available                                    │
│   - Code context                                            │
│   - Pip packages                                            │
│   - Execution timeout                                       │
│   - Literature search instructions                          │
├─────────────────────────────────────────────────────────────┤
│ Line 107: "OUTPUT FORMAT:"                                  │
│ Line 108: {output_format_instructions}  ← STAGE INFO HERE!  │
│          (Different for planning/coding/writing)            │
├─────────────────────────────────────────────────────────────┤
│ Lines 110-114: General formatting instructions              │
│   - Use XML-style tags                                      │
│   - No markdown headers or code fences                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ DYNAMIC CONTENT (Changes each iteration)                    │
├─────────────────────────────────────────────────────────────┤
│ Line 116: "=== YOUR CURRENT STATE ==="                      │
│ Line 118: Current iteration / Max iterations                │
│ Line 122: LaTeX Paper (current version)                     │
│ Line 125: LaTeX Compilation Status                          │
│ Line 128: Python Code (current version)                     │
│ Line 131: Last Execution Output                             │
│ Line 134: Plan from Previous Iteration                      │
│ Line 137: Critique from Critic                              │
│ Line 140: Researcher's Literature Searches                  │
│ Line 143: Critic's Literature Searches                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Stage-Specific Instructions (Line 108)

### Planning Stage
```
<PLAN>
[detailed plan over the remaining iterations]
</PLAN>

<OPENALEX>
[JSON array of API calls - OPTIONAL, only include if you need to search literature]
</OPENALEX>

For this stage, ONLY output the <PLAN> and optional <OPENALEX> tags. Do NOT generate code or LaTeX yet.
```

### Coding Stage
```
<PYTHON>
[Python code without markdown code fences - just the raw code]
</PYTHON>

For this stage, ONLY output the <PYTHON> tag with your code. The plan and literature search have already been completed.
You have access to the plan and literature results in the current state below.
```

### Writing Stage
```
<LATEX>
[LaTeX document - must be complete with \documentclass - without markdown code fences]
</LATEX>

For this stage, ONLY output the <LATEX> tag with your complete paper. The plan, literature search, and code execution have already been completed.
You have access to the plan, literature results, code, and execution output in the current state below.
```

---

## Code Location

**File**: `src/llm_maths_research/core/researcher.py`

**Method**: `build_generator_prompt_for_stage(iteration, state, stage)`

**Lines 383-408**: Output format definitions
**Line 454**: Stage info inserted into static content

```python
# Line 447-455: Fill in static content
static_content = static_template.format(
    problem_statement=self.problem_statement,
    papers_section=self._build_papers_section(),
    data_section=self._build_data_section(include_paths=True),
    code_section=self._build_code_section(),
    timeout=CONFIG['execution']['timeout'],
    figure_dpi=CONFIG['output']['figure_dpi'],
    output_format_instructions=output_formats[stage]  # ← STAGE INFO HERE
)
```

---

## Key Points

### 1. Stage Info is in STATIC Content
- ✅ Part of the cacheable static content
- ✅ Comes BEFORE the dynamic state
- ✅ Same position across all three stages (line 108)
- ✅ Only the *content* changes, not the position

### 2. Placement Relative to Other Info

**Order of Information**:
1. **Problem statement** (static)
2. **Reference papers** (static)
3. **Data files** (static)
4. **Code context** (static)
5. **Execution instructions** (static)
6. **Literature search instructions** (static)
7. **→ STAGE-SPECIFIC OUTPUT FORMAT ←** (static, but changes between stages)
8. **General formatting rules** (static)
9. **--- SPLIT POINT ---**
10. **Current state** (dynamic):
    - LaTeX paper
    - Python code
    - Execution output
    - Plan
    - Critique
    - Literature results

### 3. Why This Placement?

**Strategic Reasons**:

1. **Before Current State**: The model sees what format to output *before* seeing the current state, so it understands how to structure its response

2. **After Instructions**: The model has already read all the problem context and instructions, so it knows *what* to do before learning *how* to format it

3. **In Static Content**: Enables caching benefits - the stage instruction is cached along with all other instructions

4. **Clear Separation**: The "OUTPUT FORMAT:" header (line 107) makes it obvious that this is about response structure

---

## Information Flow Example

### When Building Coding Stage Prompt:

```
Static Content (Cached):
  ├─ Problem: "Analyze temperature-humidity data..."
  ├─ Papers: [Reference papers]
  ├─ Data: "example_timeseries.csv"
  ├─ Instructions: [How to load data, track time, etc.]
  ├─ OUTPUT FORMAT:
  │   <PYTHON>
  │   [Python code without markdown code fences]
  │   </PYTHON>
  │
  │   For this stage, ONLY output the <PYTHON> tag.
  │   The plan and literature search have already been completed.
  │   You have access to the plan and literature results in the current state below.
  └─ [General formatting rules]

Dynamic Content (Not Cached):
  ├─ Current iteration: 1/3
  ├─ LaTeX: [current paper]
  ├─ Python: [previous code]
  ├─ Execution Output: [previous output]
  ├─ Plan: "ITERATION 1: Load data, compute correlations..."  ← FROM STAGE 1
  ├─ Critique: [critic feedback]
  └─ Literature: [search results]  ← FROM STAGE 1
```

The model reads top-to-bottom and understands:
1. What the problem is (static)
2. What resources are available (static)
3. **What format to output in** (static, stage-specific)
4. What the current state is (dynamic)
5. What previous stages produced (dynamic)

---

## Comparison: Static vs Dynamic Placement

### Current Design (Static Placement) ✅
```
[All Instructions]
OUTPUT FORMAT: <PYTHON>  ← Stage info
=== YOUR CURRENT STATE ===
Plan: [from stage 1]
Code: [previous]
```

**Benefits**:
- Clear separation of "what to do" vs "current state"
- Cacheable across multiple calls
- Model sees output format before processing state
- Consistent placement across stages

### Alternative (Dynamic Placement) ❌
```
[All Instructions]
OUTPUT FORMAT: <PLAN> or <PYTHON> or <LATEX>
=== YOUR CURRENT STATE ===
Current Stage: "coding"  ← Would need to go here
Plan: [from stage 1]
Code: [previous]
```

**Why NOT Used**:
- Would break caching (stage changes every call)
- Mixes "instructions" with "state"
- Less clear distinction between format and content
- More tokens per call (can't cache format instructions)

---

## Summary

**Answer**: The stage information is inserted at **line 108** of the template via `{output_format_instructions}`, which is in the **static (cacheable) section**, positioned:

- **AFTER**: Problem statement, papers, data, general instructions
- **BEFORE**: Current state, plan, code, execution output, critique

This placement ensures the model knows how to format its response before seeing the current state, while keeping the instruction cacheable for efficiency.
