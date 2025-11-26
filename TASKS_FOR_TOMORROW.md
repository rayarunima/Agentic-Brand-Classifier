# Tasks for Tomorrow - Accuracy Evaluation & Testing

## 📋 What You Have

### 1. **`prompts .csv`** - Test Prompts
- Contains 25 test prompts with IDs and timestamps
- These are the INPUT prompts you'll run through your agents
- Format: `id, timestamp, text`

### 2. **`Golden_Records_Brand_Agent.json`** - Ground Truth (Expected Results)
- Contains the CORRECT/EXPECTED answers for evaluation
- Each record has:
  - `prompt`: The input text
  - `name`: Expected product names
  - `brand`: Expected brands (this is what you'll compare against!)
  - `category`: Expected category

## 🎯 Goal

**Calculate how accurate your Brand Agent is** by:
1. Running your Brand Agent on all prompts from the CSV
2. Comparing results against the "golden records" (expected answers)
3. Calculating accuracy metrics (precision, recall, F1-score)

## ✅ Tasks for Tomorrow

### Task 1: Create Evaluation Script ⏱️ 30 mins

**File to create**: `scripts/evaluate_accuracy.py`

**What it should do**:
```python
1. Load prompts from CSV
2. Load golden records from JSON
3. Run Brand Agent on each prompt
4. Compare predicted brands vs expected brands
5. Calculate accuracy metrics
6. Generate a report
```

**Key Metrics to Calculate**:
- **Exact Match**: % of prompts where ALL brands match exactly
- **Partial Match**: % of prompts where at least one brand matches
- **Precision**: (Correct brands predicted) / (Total brands predicted)
- **Recall**: (Correct brands predicted) / (Total expected brands)
- **F1-Score**: Harmonic mean of precision and recall

### Task 2: Run Evaluation ⏱️ 15 mins

```bash
# Make sure Ollama is running
ollama serve

# Run evaluation
python scripts/evaluate_accuracy.py
```

**Expected Output**:
- Console output showing progress
- Accuracy report saved to file (e.g., `results/accuracy_report.json`)
- Summary printed at the end

### Task 3: Analyze Results ⏱️ 20 mins

**Check**:
- Which prompts did the agent get wrong?
- Are there common patterns in failures?
- Confidence scores for correct vs incorrect predictions

**Create**:
- `results/error_analysis.md` - Analysis of mistakes
- Suggestions for improvement

### Task 4: Improve Agent (If Needed) ⏱️ 30-60 mins

**If accuracy is low**:
1. Check if missing brands are in `KNOWN_BRANDS` config
2. Review confidence scores - are they calibrated?
3. Look at error cases - what patterns emerge?
4. Update agent logic if needed

### Task 5: Create Report for Manager ⏱️ 20 mins

**Create**: `results/evaluation_report.md`

**Include**:
- Overall accuracy percentage
- Key metrics (precision, recall, F1)
- Examples of correct predictions
- Examples of incorrect predictions
- Recommendations for improvement

## 🔧 Quick Implementation Guide

### Step-by-Step Script Outline

```python
# scripts/evaluate_accuracy.py
import json
import csv
from pathlib import Path
from agent.brand_agent import BrandAgent

# 1. Load golden records
with open("Golden_Records_Brand_Agent.json") as f:
    golden_records = json.load(f)

# 2. Create a lookup: prompt -> expected brands
golden_lookup = {record["prompt"]: record["brand"] for record in golden_records}

# 3. Load test prompts
test_prompts = []
with open("prompts .csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["text"]:  # Skip empty rows
            test_prompts.append({
                "id": row["id"],
                "prompt": row["text"]
            })

# 4. Initialize agent
agent = BrandAgent()

# 5. Run evaluation
results = []
for test in test_prompts:
    prompt = test["prompt"]
    
    # Get prediction
    prediction = agent.extract_brand(prompt)
    predicted_brands = prediction.get("brands", []) if isinstance(prediction, dict) else prediction
    
    # Get expected brands
    expected_brands = golden_lookup.get(prompt, [])
    
    # Compare
    correct = set(predicted_brands) == set(expected_brands)
    
    results.append({
        "prompt": prompt,
        "predicted": predicted_brands,
        "expected": expected_brands,
        "correct": correct,
        "confidence": prediction.get("confidence", 0.0) if isinstance(prediction, dict) else 0.0
    })

# 6. Calculate metrics
# (add precision, recall, F1 calculation here)

# 7. Save results
with open("results/accuracy_report.json", "w") as f:
    json.dump(results, f, indent=2)

# 8. Print summary
print(f"Total prompts: {len(results)}")
print(f"Correct: {sum(1 for r in results if r['correct'])}")
print(f"Accuracy: {sum(1 for r in results if r['correct']) / len(results) * 100:.2f}%")
```

## 📊 Success Criteria

**Target Accuracy** (based on your demo needs):
- **Minimum Acceptable**: 70% exact match
- **Good**: 80% exact match
- **Excellent**: 90%+ exact match

**If below 70%**:
- Check if all brands in golden records are in your config
- Review confidence scoring
- Look for patterns in failures

## 🚀 Quick Start Tomorrow

1. **Morning** (30 mins):
   ```bash
   # Create results directory
   mkdir -p results
   
   # Run the evaluation script
   python scripts/evaluate_accuracy.py
   ```

2. **Mid-morning** (30 mins):
   - Review accuracy results
   - Identify common failure patterns
   - Update config if needed

3. **Afternoon** (60 mins):
   - Fix issues if accuracy is low
   - Re-run evaluation
   - Create report for manager

## 📝 Notes

- **Don't panic if accuracy isn't perfect** - this is normal for first runs
- **Focus on understanding WHY it fails** - that's more valuable than just the number
- **Your manager wants to see**:
  1. You understand the evaluation process
  2. You can identify issues
  3. You can improve iteratively

## 🎯 Expected Output Files

After running evaluation, you should have:
- `results/accuracy_report.json` - Detailed results
- `results/evaluation_report.md` - Human-readable report
- `results/error_analysis.md` - Analysis of failures

## 💡 Tips

1. **Start simple**: Just compare exact brand lists first
2. **Add metrics later**: Once basic comparison works, add precision/recall
3. **Save everything**: Keep all results for analysis
4. **Document issues**: Note any problems you encounter

---

**You've got this!** 🚀
Start with Task 1 tomorrow morning - create the evaluation script step by step.
If you get stuck, the script outline above will guide you.

**Good night! Sleep well!** 😴

