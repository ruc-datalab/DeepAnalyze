import json
import os
import argparse
from datetime import datetime
from tqdm import tqdm


def evaluate_feverous_results(input_file, output_file=None):
    """
    Evaluate FEVEROUS results by comparing ground_truth and model_prediction fields.

    Args:
        input_file: Path to the JSON file containing FEVEROUS results
        output_file: Path to save evaluation results (default: automatically generated)
        verbose: Whether to print evaluation results to console

    Returns:
        Evaluation results dictionary
    """
    if output_file is None:
        # Generate default output filename
        dirname = os.path.dirname(input_file)
        basename = os.path.basename(input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            dirname, f"eval_{basename.split('.')[0]}_{timestamp}.json"
        )

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(
            f"Error: Could not parse input file {input_file}. Please ensure it is valid JSON."
        )
        return None

    # Handle different result formats
    results = []
    if "results" in data:
        # Format: {"results": [item1, item2, ...]}
        results = data["results"]
    elif isinstance(data, list):
        # Format: [item1, item2, ...]
        results = data
    else:
        print(
            f"Error: Unexpected data format. Expected a list or an object with 'results' field."
        )
        return None

    total_samples = len(results)

    # Initialize statistics
    stats = {
        "total_samples": total_samples,
        "correct_predictions": 0,
        "incorrect_predictions": 0,
        "accuracy": 0.0,
        "samples_with_missing_truth": 0,
        "samples_with_missing_prediction": 0,
    }

    # For label-specific metrics
    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    label_counts = {label: {"total": 0, "correct": 0} for label in labels}

    # Process each sample
    evaluation_results = []
    for i, item in tqdm(enumerate(results), total=total_samples, desc="Evaluating"):
        # Extract ground truth and model prediction
        ground_truth = item.get("ground_truth")
        model_prediction = item.get("model_prediction")

        # Create result item
        result_item = {
            "id": item.get("id", f"item-{i}"),
            "claim": item.get("claim", ""),
            "ground_truth": ground_truth,
            "model_prediction": model_prediction,
            "prompt": item.get("prompt", ""),
            "full_response": item.get("full_response", ""),
        }

        # Evaluate correctness
        if ground_truth is not None and model_prediction is not None:
            is_correct = ground_truth == model_prediction
            result_item["is_correct"] = is_correct

            if is_correct:
                stats["correct_predictions"] += 1
            else:
                stats["incorrect_predictions"] += 1

            # Update label-specific metrics
            if ground_truth in labels:
                label_counts[ground_truth]["total"] += 1
                if is_correct:
                    label_counts[ground_truth]["correct"] += 1

        elif ground_truth is None:
            stats["samples_with_missing_truth"] += 1
            result_item["is_correct"] = None
        elif model_prediction is None:
            stats["samples_with_missing_prediction"] += 1
            result_item["is_correct"] = False

        evaluation_results.append(result_item)

    # Calculate accuracy
    valid_samples = (
        total_samples
        - stats["samples_with_missing_truth"]
        - stats["samples_with_missing_prediction"]
    )
    if valid_samples > 0:
        stats["accuracy"] = stats["correct_predictions"] / valid_samples

    # Calculate label-specific accuracy
    label_stats = {}
    for label in labels:
        if label_counts[label]["total"] > 0:
            accuracy = label_counts[label]["correct"] / label_counts[label]["total"]
        else:
            accuracy = 0.0
        label_stats[label] = {
            "total": label_counts[label]["total"],
            "correct": label_counts[label]["correct"],
            "accuracy": accuracy,
        }

    # Calculate confusion matrix
    confusion_matrix = {
        "true_supports": {"pred_supports": 0, "pred_refutes": 0, "pred_nei": 0},
        "true_refutes": {"pred_supports": 0, "pred_refutes": 0, "pred_nei": 0},
        "true_nei": {"pred_supports": 0, "pred_refutes": 0, "pred_nei": 0},
    }

    for item in evaluation_results:
        if item["ground_truth"] == "SUPPORTS":
            if item["model_prediction"] == "SUPPORTS":
                confusion_matrix["true_supports"]["pred_supports"] += 1
            elif item["model_prediction"] == "REFUTES":
                confusion_matrix["true_supports"]["pred_refutes"] += 1
            elif item["model_prediction"] == "NOT ENOUGH INFO":
                confusion_matrix["true_supports"]["pred_nei"] += 1
        elif item["ground_truth"] == "REFUTES":
            if item["model_prediction"] == "SUPPORTS":
                confusion_matrix["true_refutes"]["pred_supports"] += 1
            elif item["model_prediction"] == "REFUTES":
                confusion_matrix["true_refutes"]["pred_refutes"] += 1
            elif item["model_prediction"] == "NOT ENOUGH INFO":
                confusion_matrix["true_refutes"]["pred_nei"] += 1
        elif item["ground_truth"] == "NOT ENOUGH INFO":
            if item["model_prediction"] == "SUPPORTS":
                confusion_matrix["true_nei"]["pred_supports"] += 1
            elif item["model_prediction"] == "REFUTES":
                confusion_matrix["true_nei"]["pred_refutes"] += 1
            elif item["model_prediction"] == "NOT ENOUGH INFO":
                confusion_matrix["true_nei"]["pred_nei"] += 1

    # Prepare final result
    evaluation_result = {
        "summary": stats,
        "label_stats": label_stats,
        "confusion_matrix": confusion_matrix,
        "results": evaluation_results,
    }

    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_result, f, indent=4, ensure_ascii=False)

    # Print main metrics
    print("\n===== Evaluation Score =====")
    print(f"{stats['accuracy']:.4f}")

    print("\n===== Evaluation Summary =====")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Correct predictions: {stats['correct_predictions']}")
    print(f"Incorrect predictions: {stats['incorrect_predictions']}")
    print(f"Samples with missing ground truth: {stats['samples_with_missing_truth']}")
    print(
        f"Samples with missing model prediction: {stats['samples_with_missing_prediction']}"
    )
    print(f"Accuracy: {stats['accuracy']:.4f}")

    print("\n===== Label-specific Metrics =====")
    for label in labels:
        total = label_stats[label]["total"]
        correct = label_stats[label]["correct"]
        acc = label_stats[label]["accuracy"]
        print(f"{label}: {correct}/{total} correct, Accuracy: {acc:.4f}")

    print("\n===== Confusion Matrix =====")
    print(
        "                   | Predicted SUPPORTS | Predicted REFUTES | Predicted NOT ENOUGH INFO"
    )
    print("-" * 80)
    print(
        f"True SUPPORTS      | {confusion_matrix['true_supports']['pred_supports']:<19} | {confusion_matrix['true_supports']['pred_refutes']:<18} | {confusion_matrix['true_supports']['pred_nei']}"
    )
    print(
        f"True REFUTES       | {confusion_matrix['true_refutes']['pred_supports']:<19} | {confusion_matrix['true_refutes']['pred_refutes']:<18} | {confusion_matrix['true_refutes']['pred_nei']}"
    )
    print(
        f"True NOT ENOUGH INFO | {confusion_matrix['true_nei']['pred_supports']:<19} | {confusion_matrix['true_nei']['pred_refutes']:<18} | {confusion_matrix['true_nei']['pred_nei']}"
    )

    return evaluation_result


def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description="FEVEROUS prediction results evaluation"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Input file path (JSON file containing prediction results)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file path (evaluation results will be saved to this file)",
    )
    parser.add_argument(
        "--base_path", type=str, help="Base path for the project (optional)"
    )

    args = parser.parse_args()

    # Run evaluation
    result = evaluate_feverous_results(
        input_file=args.results_file,
        output_file=args.output_file,
    )

    if result is None:
        exit(1)


if __name__ == "__main__":
    main()
