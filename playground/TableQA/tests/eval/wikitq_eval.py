import json
import re
import os
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Union


# def extract_predicted_answer(model_answer):
#     """Extract predicted answer from model's response"""
#     if not model_answer:
#         return None

#     # Try to match content wrapped in <answer> tags
#     answer_tag_pattern = r'<answer>(.*?)</answer>'
#     answer_tag_match = re.search(answer_tag_pattern, model_answer, re.DOTALL)

#     if answer_tag_match:
#         model_answer = answer_tag_match.group(1).strip()
#         if 'Answer:' in model_answer:
#             match = re.search(r'Answer:\s*(.+?)(?:\n|$|\.|")', model_answer)
#             if match:
#                 return match.group(1).strip()
#         else:
#             return model_answer.strip() if model_answer else None
#     else:
#         match = re.search(r'Answer:\s*(.+?)(?:\n|$|\.|")', model_answer)
#         if match:
#             return match.group(1).strip()


def extract_predicted_answer(model_answer):
    """Extract predicted answer from model's response"""
    if not model_answer:
        return None
    model_answer = model_answer.replace("**", "")
    # Try to match content wrapped in <answer> tags
    answer_tag_pattern = r"<Answer>(.*?)</Answer>"
    answer_tag_match = re.findall(r"<Answer>(.*?)</Answer>", model_answer, re.DOTALL)

    if answer_tag_match:
        model_answer = answer_tag_match[-1].strip()

        return model_answer.split("Answer:")[-1].strip(".").strip()
        # if 'Answer:' in model_answer:
        #     match = re.findall(r'Answer:\s*(.+?)(?:\n|$|\.|")', model_answer)
        #     if match:
        #         return match[-1].strip()
        # else:
        #     return model_answer.strip() if model_answer else None
    else:
        match = re.search(r'Answer:\s*(.+?)(?:\n|$|\.|")', model_answer)
        if match:
            return match.group(1).strip()


def normalize_answer(answer):
    """
    Normalize answer for robust comparison, handling:
    - Numbers with/without commas
    - Units (km/h, pages, etc.)
    - Lists with different separators
    - Accents and special characters
    - Different date/time formats
    """
    if not answer:
        return ""

    # Convert to lowercase and strip spaces
    answer = str(answer).strip().lower()

    # Handle numerical values with commas or units
    numeric_with_units_match = re.match(
        r"^([\d,]+)\s*(days|years|pages|km\/h|mph)$", answer
    )
    if numeric_with_units_match:
        # Extract the numeric part and remove commas
        numeric_value = numeric_with_units_match.group(1).replace(",", "")
        unit = numeric_with_units_match.group(2)
        # Return standardized format
        return f"{numeric_value} {unit}"

    # Handle simple numbers with commas
    numeric_match = re.match(r"^[\d,]+$", answer)
    if numeric_match:
        return answer.replace(",", "")  # Remove commas

    # Handle time periods
    time_period_mapping = {
        "1 week": "7 days",
        "2 weeks": "14 days",
        "1 year": "12 months",
        # Add more mappings as needed
    }
    if answer in time_period_mapping:
        return time_period_mapping[answer]

    # Try to convert to float for numerical comparison
    try:
        num = float(answer.replace(",", ""))
        if num.is_integer():
            return str(int(num))
        return str(num)
    except:
        # Not a number, continue with further normalization
        pass

    # Handle lists with different separators
    if "|" in answer or "," in answer:
        items = re.split(r"[|,]\s*", answer)
        # Sort the items to handle different orders
        return "|".join(sorted([item.strip() for item in items if item.strip()]))

    # Remove accents for better character matching
    import unicodedata

    answer = "".join(
        c for c in unicodedata.normalize("NFKD", answer) if not unicodedata.combining(c)
    )

    # Final cleanup - remove unnecessary characters
    answer = re.sub(r"[^\w\s]", "", answer)  # Remove punctuation
    answer = " ".join(answer.split())  # Normalize whitespace

    return answer


def exact_match_enhanced(prediction, reference):
    """Enhanced exact match function with various normalization techniques"""
    if prediction is None or reference is None:
        return 0

    # First try direct comparison after simple normalization
    pred_norm = normalize_answer(prediction)
    ref_norm = normalize_answer(reference)

    if pred_norm == ref_norm:
        return 1

    # Special case for numbers with units
    # Check if prediction is just the number part of reference with units
    pred_num_match = re.match(r"^(\d+)$", pred_norm)
    ref_units_match = re.match(r"^(\d+)\s+([a-z/]+)$", ref_norm)

    if (
        pred_num_match
        and ref_units_match
        and pred_num_match.group(1) == ref_units_match.group(1)
    ):
        return 1

    # Check if both are lists but with different separators
    pred_items = set(re.split(r"[|,]\s*", prediction.lower()))
    ref_items = set(re.split(r"[|,]\s*", reference.lower()))

    if len(pred_items) > 1 and len(ref_items) > 1 and pred_items == ref_items:
        return 1

    return 0


def evaluate_answers(input_file, output_file=None, verbose=True):
    """Evaluate model answers against expected answers"""
    # Set output file if not provided
    if output_file is None:
        dirname = os.path.dirname(input_file)
        basename = os.path.basename(input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            dirname, f"eval_simple_{basename.split('.')[0]}_{timestamp}.json"
        )

    # Load data
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(
            f"Error: Cannot parse input file {input_file}, please ensure it is valid JSON."
        )
        return None

    # Handle case where data is a single object instead of a list
    if not isinstance(data, list):
        if isinstance(data, dict):
            data = [data]
        else:
            print(
                f"Error: Input data format is incorrect, should be a list of JSON objects or a single JSON object."
            )
            return None

    total_samples = len(data)
    if verbose:
        print(f"Starting evaluation of {total_samples} samples...")

    # Initialize statistics
    stats = {
        "total_samples": total_samples,
        "answered_samples": 0,
        "exact_matches": 0,
        "exact_match_rate": 0.0,
        "no_answer_extracted": 0,
        "samples_with_missing_truth": 0,
    }

    # Process each sample
    results_list = []
    for i, item in enumerate(data):
        # Find expected answer
        expected_answer = None
        if "answer" in item:
            expected_answer = item["answer"]
        elif "truth_answer" in item:
            expected_answer = item["truth_answer"]
        elif "true_answer" in item:
            expected_answer = item["true_answer"]

        # Get model answer
        model_answer = item.get("model_answer", "")
        predicted_answer = extract_predicted_answer(model_answer)

        # Create result item
        result_item = {
            "id": item.get("id", f"item-{i}"),
            "prompt": item.get("prompt", ""),
            "full_response": item.get("full_response", ""),
            "question": item.get("question", ""),
            "expected_answer": expected_answer,
            "predicted_answer": predicted_answer,
        }

        # Evaluate correctness
        is_match = False
        if expected_answer is not None and predicted_answer is not None:
            is_match = exact_match_enhanced(predicted_answer, expected_answer)

            stats["answered_samples"] += 1
            stats["exact_matches"] += is_match
            result_item["is_exact_match"] = bool(is_match)
        elif expected_answer is None:
            stats["samples_with_missing_truth"] += 1
            result_item["is_exact_match"] = None
        elif predicted_answer is None:
            stats["no_answer_extracted"] += 1
            result_item["is_exact_match"] = False

        results_list.append(result_item)

    # Calculate final statistics
    answered_samples = stats["total_samples"] - stats["samples_with_missing_truth"]
    if answered_samples > 0:
        stats["exact_match_rate"] = stats["exact_matches"] / answered_samples

    evaluation_result = {"summary": stats, "results": results_list}

    # Save results
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
        print(f"Samples with answers: {stats['answered_samples']}")
        print(f"Samples without extracted answers: {stats['no_answer_extracted']}")
        print(f"Samples missing ground truth: {stats['samples_with_missing_truth']}")
        print(f"Exact matches: {stats['exact_matches']}")
        if answered_samples > 0:
            print(f"Exact match rate: {stats['exact_matches'] / answered_samples:.4f}")
        else:
            print("Exact match rate: N/A (no valid samples)")

    return evaluation_result


def main():
    """Main function that handles command line arguments"""
    parser = argparse.ArgumentParser(description="WikiTQ prediction evaluation")

    # Add parameters to match the script
    parser.add_argument(
        "--results_file",
        required=True,
        help="Input file path with predictions (absolute path)",
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
    result = evaluate_answers(
        input_file=args.results_file, output_file=args.output_file, verbose=True
    )

    if result is None:
        exit(1)


if __name__ == "__main__":
    main()
