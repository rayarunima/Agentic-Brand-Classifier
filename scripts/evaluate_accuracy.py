import json
import csv
import sys
import os
import dspy
from pathlib import Path

# Add the project root to sys.path to allow importing from agent
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from agent.brand_agent import BrandAgent
except ImportError:
    print("Error: Could not import BrandAgent. Make sure you are running this script from the project root.")
    print("Usage: python scripts/evaluate_accuracy.py")
    sys.exit(1)

def calculate_metrics(results):
    """Calculate precision, recall, and F1 score."""
    total_predicted = 0
    total_expected = 0
    correct_predicted = 0
    
    exact_matches = 0
    partial_matches = 0
    
    for r in results:
        predicted = set(r['predicted'])
        expected = set(r['expected'])
        
        # Exact match (all brands match exactly)
        if predicted == expected:
            exact_matches += 1
            
        # Partial match (at least one brand matches)
        if predicted.intersection(expected):
            partial_matches += 1
            
        # For precision/recall
        total_predicted += len(predicted)
        total_expected += len(expected)
        correct_predicted += len(predicted.intersection(expected))
        
    precision = correct_predicted / total_predicted if total_predicted > 0 else 0
    recall = correct_predicted / total_expected if total_expected > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "exact_match_rate": exact_matches / len(results) if results else 0,
        "partial_match_rate": partial_matches / len(results) if results else 0,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def main():
    print("Starting Brand Agent Accuracy Evaluation...")
    
    # 1. Load golden records
    golden_path = project_root / "Golden_Records_Brand_Agent.json"
    if not golden_path.exists():
        print(f"Error: Golden records file not found at {golden_path}")
        return

    with open(golden_path) as f:
        golden_records = json.load(f)
    print(f"Loaded {len(golden_records)} golden records.")

    # 2. Create a lookup: prompt -> expected brands
    # Normalize keys to be safe
    golden_lookup = {record["prompt"].strip(): record["brand"] for record in golden_records}

    # 3. Load test prompts
    # NOTE: The prompts in 'prompts .csv' do not match the keys in 'Golden_Records_Brand_Agent.json'.
    # To calculate accuracy, we must use the prompts for which we have ground truth.
    # We will use the prompts from the golden records file.
    
    test_prompts = []
    for i, record in enumerate(golden_records):
        test_prompts.append({
            "id": str(i),
            "prompt": record["prompt"]
        })
    print(f"Using {len(test_prompts)} prompts from Golden Records for evaluation.")

    # 4. Initialize agent
    print("Initializing BrandAgent...")
    try:
        # Configure DSPy
        dspy.configure(lm=dspy.LM(model="ollama/phi3"))
        agent = BrandAgent()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return

    # 5. Run evaluation
    print("\nRunning evaluation...")
    results = []
    
    # Create results directory if it doesn't exist
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    for i, test in enumerate(test_prompts):
        prompt = test["prompt"]
        print(f"Processing {i+1}/{len(test_prompts)}: {prompt[:50]}...")
        
        try:
            # Get prediction
            prediction = agent.extract_brand(prompt)
            predicted_brands = prediction.get("brands", []) if isinstance(prediction, dict) else prediction
            
            # Ensure predicted_brands is a list
            if isinstance(predicted_brands, str):
                predicted_brands = [predicted_brands]
            
            # Get expected brands
            # Try exact match first, then stripped
            expected_brands = golden_lookup.get(prompt, golden_lookup.get(prompt.strip(), []))
            
            # Compare
            # Normalize for comparison (case insensitive, stripped)
            pred_set = set(b.lower().strip() for b in predicted_brands)
            exp_set = set(b.lower().strip() for b in expected_brands)
            
            correct = pred_set == exp_set
            
            results.append({
                "id": test["id"],
                "prompt": prompt,
                "predicted": predicted_brands,
                "expected": expected_brands,
                "correct": correct,
                "confidence": prediction.get("confidence", 0.0) if isinstance(prediction, dict) else 0.0
            })
        except Exception as e:
            print(f"Error processing prompt '{prompt}': {e}")
            results.append({
                "id": test["id"],
                "prompt": prompt,
                "error": str(e),
                "correct": False,
                "predicted": [],
                "expected": golden_lookup.get(prompt, [])
            })

    # 6. Calculate metrics
    metrics = calculate_metrics(results)

    # 7. Save results
    output_path = results_dir / "accuracy_report.json"
    report = {
        "metrics": metrics,
        "details": results
    }
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nResults saved to {output_path}")

    # 8. Print summary
    print("\n" + "="*30)
    print("EVALUATION SUMMARY")
    print("="*30)
    print(f"Total Prompts: {len(results)}")
    print(f"Exact Matches: {metrics['exact_match_rate']*100:.1f}%")
    print(f"Partial Matches: {metrics['partial_match_rate']*100:.1f}%")
    print(f"Precision:     {metrics['precision']:.3f}")
    print(f"Recall:        {metrics['recall']:.3f}")
    print(f"F1 Score:      {metrics['f1_score']:.3f}")
    print("="*30)
    
    if metrics['exact_match_rate'] < 0.7:
        print("\n⚠️  Accuracy is below 70%. Recommendations:")
        print("1. Check 'results/accuracy_report.json' for mismatches.")
        print("2. Verify if missing brands are in KNOWN_BRANDS config.")
        print("3. Review confidence scores.")
    else:
        print("\n✅ Good job! Accuracy is acceptable.")

if __name__ == "__main__":
    main()
