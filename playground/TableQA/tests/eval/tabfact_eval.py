import json
import os
import argparse
from datetime import datetime


def evaluate_tabfact_results(input_file, output_file=None, verbose=True):
    """
    Evaluate TabFact results by comparing ground_truth and model_label fields.

    Args:
        input_file: Path to the JSON file containing TabFact results
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
    if verbose:
        print(f"Starting evaluation of {total_samples} samples...")

    # Initialize statistics
    stats = {
        "total_samples": total_samples,
        "correct_predictions": 0,
        "incorrect_predictions": 0,
        "accuracy": 0.0,
        "samples_with_missing_truth": 0,
        "samples_with_missing_prediction": 0,
    }

    # Process each sample
    evaluation_results = []
    for i, item in enumerate(results):
        # Extract ground truth and model prediction
        ground_truth = item.get("ground_truth")
        model_label = item.get("model_label")

        # Create result item
        result_item = {
            "id": item.get("id", f"item-{i}"),
            "claim": item.get("claim", ""),
            "ground_truth": ground_truth,
            "prompt": item.get("pronpt", ""),
            "full_response": item.get("model_answer", ""),
            "model_label": model_label,
        }

        # Evaluate correctness
        if ground_truth is not None and model_label is not None:
            is_correct = ground_truth == model_label
            result_item["is_correct"] = is_correct

            if is_correct:
                stats["correct_predictions"] += 1
            else:
                stats["incorrect_predictions"] += 1
        elif ground_truth is None:
            stats["samples_with_missing_truth"] += 1
            result_item["is_correct"] = None
        elif model_label is None:
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

    # Prepare final result
    evaluation_result = {"summary": stats, "results": evaluation_results}

    # Save to file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_result, f, indent=4, ensure_ascii=False)
        if verbose:
            print(f"Evaluation results saved to {output_file}")
    except Exception as e:
        if verbose:
            print(f"Error saving evaluation results: {e}")

    # Print main metrics
    if verbose:
        print("\n===== Evaluation Summary =====")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Correct predictions: {stats['correct_predictions']}")
        print(f"Incorrect predictions: {stats['incorrect_predictions']}")
        print(
            f"Samples with missing ground truth: {stats['samples_with_missing_truth']}"
        )
        print(
            f"Samples with missing model prediction: {stats['samples_with_missing_prediction']}"
        )
        print(f"Accuracy: {stats['accuracy']:.4f}")

    return evaluation_result


def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description="TabFact prediction results evaluation"
    )
    parser.add_argument(
        "--results_file",
        required=True,
        help="Input file path with prediction results (absolute path)",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output file path for evaluation results (absolute path)",
    )
    parser.add_argument(
        "--base_path", type=str, help="Base path for the project (optional)"
    )

    args = parser.parse_args()

    # Run evaluation
    result = evaluate_tabfact_results(
        input_file=args.results_file, output_file=args.output_file, verbose=True
    )

    if result is None:
        exit(1)


if __name__ == "__main__":
    main()
