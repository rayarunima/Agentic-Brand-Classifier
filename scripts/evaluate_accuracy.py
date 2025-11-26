"""
Accuracy Evaluation Script for Brand Agent

Compares agent predictions against golden records to calculate accuracy metrics.
"""
import json
import csv
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict

from agent.brand_agent import BrandAgent


def load_golden_records(filepath: str) -> Dict[str, Dict]:
    """Load golden records and create lookup by prompt."""
    with open(filepath, 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    # Create lookup: prompt -> expected results
    lookup = {}
    for record in records:
        prompt = record.get("prompt", "").strip()
        if prompt:
            lookup[prompt] = {
                "expected_brands": [b.strip() for b in record.get("brand", []) if b.strip()],
                "expected_names": [n.strip() for n in record.get("name", []) if n.strip()],
                "expected_category": record.get("category", "").strip()
            }
    
    return lookup


def load_test_prompts(filepath: str) -> List[Dict]:
    """Load test prompts from CSV."""
    prompts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            if text:
                prompts.append({
                    "id": row.get("id", "").strip(),
                    "timestamp": row.get("timestamp", "").strip(),
                    "prompt": text
                })
    return prompts


def normalize_brands(brands: List[str]) -> Set[str]:
    """Normalize brand names for comparison (lowercase, strip)."""
    return {b.lower().strip() for b in brands if b.strip()}


def calculate_exact_match(predicted: List[str], expected: List[str]) -> bool:
    """Check if predicted brands exactly match expected brands."""
    pred_set = normalize_brands(predicted)
    exp_set = normalize_brands(expected)
    return pred_set == exp_set


def calculate_partial_match(predicted: List[str], expected: List[str]) -> bool:
    """Check if at least one brand matches."""
    pred_set = normalize_brands(predicted)
    exp_set = normalize_brands(expected)
    return len(pred_set.intersection(exp_set)) > 0


def calculate_precision_recall(predicted: List[str], expected: List[str]) -> Tuple[float, float]:
    """Calculate precision and recall."""
    pred_set = normalize_brands(predicted)
    exp_set = normalize_brands(expected)
    
    if not pred_set:
        return (0.0, 0.0) if exp_set else (1.0, 1.0)
    
    intersection = pred_set.intersection(exp_set)
    
    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(exp_set) if exp_set else 0.0
    
    return (precision, recall)


def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_agent(
    agent: BrandAgent,
    test_prompts: List[Dict],
    golden_lookup: Dict[str, Dict]
) -> List[Dict]:
    """Evaluate agent on test prompts."""
    results = []
    
    print(f"Evaluating {len(test_prompts)} prompts...")
    
    for i, test in enumerate(test_prompts, 1):
        prompt = test["prompt"]
        print(f"[{i}/{len(test_prompts)}] Processing: {prompt[:50]}...")
        
        # Get prediction
        try:
            prediction = agent.extract_brand(prompt)
            if isinstance(prediction, dict):
                predicted_brands = prediction.get("brands", [])
                confidence = prediction.get("confidence", 0.0)
            else:
                predicted_brands = prediction if isinstance(prediction, list) else []
                confidence = 0.0
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            predicted_brands = []
            confidence = 0.0
        
        # Get expected results
        expected_data = golden_lookup.get(prompt, {})
        expected_brands = expected_data.get("expected_brands", [])
        
        # Calculate metrics
        exact_match = calculate_exact_match(predicted_brands, expected_brands)
        partial_match = calculate_partial_match(predicted_brands, expected_brands)
        precision, recall = calculate_precision_recall(predicted_brands, expected_brands)
        f1 = calculate_f1_score(precision, recall)
        
        results.append({
            "id": test["id"],
            "prompt": prompt,
            "predicted_brands": predicted_brands,
            "expected_brands": expected_brands,
            "confidence": confidence,
            "exact_match": exact_match,
            "partial_match": partial_match,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        
        # Show result
        status = "✅" if exact_match else "❌"
        print(f"  {status} Predicted: {predicted_brands} | Expected: {expected_brands}")
    
    return results


def generate_summary(results: List[Dict]) -> Dict:
    """Generate summary statistics."""
    total = len(results)
    exact_matches = sum(1 for r in results if r["exact_match"])
    partial_matches = sum(1 for r in results if r["partial_match"])
    
    avg_precision = sum(r["precision"] for r in results) / total if total > 0 else 0.0
    avg_recall = sum(r["recall"] for r in results) / total if total > 0 else 0.0
    avg_f1 = sum(r["f1_score"] for r in results) / total if total > 0 else 0.0
    avg_confidence = sum(r["confidence"] for r in results) / total if total > 0 else 0.0
    
    # Confidence analysis
    correct_confidences = [r["confidence"] for r in results if r["exact_match"]]
    incorrect_confidences = [r["confidence"] for r in results if not r["exact_match"]]
    
    avg_correct_conf = sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0.0
    avg_incorrect_conf = sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0.0
    
    return {
        "total_prompts": total,
        "exact_matches": exact_matches,
        "partial_matches": partial_matches,
        "exact_match_accuracy": exact_matches / total if total > 0 else 0.0,
        "partial_match_accuracy": partial_matches / total if total > 0 else 0.0,
        "average_precision": avg_precision,
        "average_recall": avg_recall,
        "average_f1_score": avg_f1,
        "average_confidence": avg_confidence,
        "average_confidence_correct": avg_correct_conf,
        "average_confidence_incorrect": avg_incorrect_conf
    }


def save_results(results: List[Dict], summary: Dict, output_dir: Path):
    """Save evaluation results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_file = output_dir / "accuracy_report.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": summary,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Detailed results saved to: {results_file}")
    
    # Save summary report
    report_file = output_dir / "evaluation_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Brand Agent Evaluation Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Total Prompts**: {summary['total_prompts']}\n")
        f.write(f"- **Exact Matches**: {summary['exact_matches']} ({summary['exact_match_accuracy']*100:.2f}%)\n")
        f.write(f"- **Partial Matches**: {summary['partial_matches']} ({summary['partial_match_accuracy']*100:.2f}%)\n")
        f.write(f"- **Average Precision**: {summary['average_precision']:.3f}\n")
        f.write(f"- **Average Recall**: {summary['average_recall']:.3f}\n")
        f.write(f"- **Average F1-Score**: {summary['average_f1_score']:.3f}\n")
        f.write(f"- **Average Confidence**: {summary['average_confidence']:.3f}\n")
        f.write(f"- **Avg Confidence (Correct)**: {summary['average_confidence_correct']:.3f}\n")
        f.write(f"- **Avg Confidence (Incorrect)**: {summary['average_confidence_incorrect']:.3f}\n\n")
        
        f.write("## Incorrect Predictions\n\n")
        incorrect = [r for r in results if not r["exact_match"]]
        for r in incorrect:
            f.write(f"### Prompt {r['id']}\n")
            f.write(f"**Prompt**: {r['prompt']}\n\n")
            f.write(f"**Predicted**: {r['predicted_brands']}\n")
            f.write(f"**Expected**: {r['expected_brands']}\n")
            f.write(f"**Confidence**: {r['confidence']:.3f}\n")
            f.write(f"**Precision**: {r['precision']:.3f} | **Recall**: {r['recall']:.3f} | **F1**: {r['f1_score']:.3f}\n\n")
    
    print(f"✅ Summary report saved to: {report_file}")


def main():
    """Main evaluation function."""
    # File paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    csv_file = project_root / "prompts .csv"
    golden_file = project_root / "Golden_Records_Brand_Agent.json"
    results_dir = project_root / "results"
    
    # Check files exist
    if not csv_file.exists():
        print(f"❌ Error: CSV file not found at {csv_file}")
        return
    
    if not golden_file.exists():
        print(f"❌ Error: Golden records file not found at {golden_file}")
        return
    
    print("=" * 60)
    print("Brand Agent Accuracy Evaluation")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading data...")
    golden_lookup = load_golden_records(str(golden_file))
    test_prompts = load_test_prompts(str(csv_file))
    
    print(f"✅ Loaded {len(golden_lookup)} golden records")
    print(f"✅ Loaded {len(test_prompts)} test prompts")
    
    # Find prompts that exist in both
    matched_prompts = []
    for test in test_prompts:
        if test["prompt"] in golden_lookup:
            matched_prompts.append(test)
        else:
            print(f"⚠️  Warning: Prompt not found in golden records: {test['prompt'][:50]}...")
    
    if not matched_prompts:
        print("❌ Error: No matching prompts found between CSV and golden records!")
        return
    
    print(f"\n✅ Found {len(matched_prompts)} matching prompts to evaluate")
    
    # Initialize agent
    print("\n🤖 Initializing Brand Agent...")
    agent = BrandAgent()
    
    # Run evaluation
    print("\n🔍 Running evaluation...\n")
    results = evaluate_agent(agent, matched_prompts, golden_lookup)
    
    # Calculate summary
    print("\n📊 Calculating metrics...")
    summary = generate_summary(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Prompts: {summary['total_prompts']}")
    print(f"Exact Matches: {summary['exact_matches']} ({summary['exact_match_accuracy']*100:.2f}%)")
    print(f"Partial Matches: {summary['partial_matches']} ({summary['partial_match_accuracy']*100:.2f}%)")
    print(f"\nMetrics:")
    print(f"  Average Precision: {summary['average_precision']:.3f}")
    print(f"  Average Recall: {summary['average_recall']:.3f}")
    print(f"  Average F1-Score: {summary['average_f1_score']:.3f}")
    print(f"\nConfidence Analysis:")
    print(f"  Average Confidence: {summary['average_confidence']:.3f}")
    print(f"  Avg Confidence (Correct): {summary['average_confidence_correct']:.3f}")
    print(f"  Avg Confidence (Incorrect): {summary['average_confidence_incorrect']:.3f}")
    print("=" * 60)
    
    # Save results
    print("\n💾 Saving results...")
    save_results(results, summary, results_dir)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()

