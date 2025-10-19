import json
import os
import re
import argparse
import sys
import math
from typing import Dict, List, Union, Any
from collections import Counter


def str_to_num(text):
    """Convert text to number, handling various formats and cleaning characters"""
    if text is None:
        return "n/a"

    if isinstance(text, (int, float)):
        return float(text)

    text = str(text)
    text = text.replace("$", "").replace(",", "").replace("_", "")

    # Handle percentages
    if "%" in text:
        text = text.replace("%", "")
        try:
            return float(text) / 100
        except ValueError:
            return "n/a"

    # Handle regular numbers
    try:
        return float(text)
    except ValueError:
        return "n/a"


import re


def extract_answer_from_response(model_answer):
    """Extract final answer from the model response, supporting multiple formats"""
    if not model_answer:
        return ""

    # Case 1: <answer>...</answer> tag
    answer_tag_pattern = re.search(r"<answer>(.*?)</answer>", model_answer, re.DOTALL)
    if answer_tag_pattern:
        answer_content = answer_tag_pattern.group(1).strip()
        if "Answer:" in answer_content:
            return answer_content.split("Answer:")[-1].strip()
        return answer_content

    # Case 2: find all "Answer: ..." and return the last one
    answer_matches = re.findall(r"Answer:\s*(.*?)(?=(?:\n|$))", model_answer, re.DOTALL)
    if answer_matches:
        return answer_matches[-1].strip()

    # Case 3: fallback to last line
    lines = model_answer.strip().split("\n")
    return lines[-1].strip()


def normalize_answer(s):
    """Normalize answer text, remove punctuation and extra whitespace"""
    if s is None:
        return ""

    # If it's a number, return original format
    if isinstance(s, (int, float)):
        return str(s)

    s = str(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^\w\s]", "", s)
    return s


def exact_match_score(prediction, ground_truth):
    """Calculate exact match score"""
    if prediction is None or ground_truth is None:
        return 0

    # Handle numeric type exact matching
    pred_num = str_to_num(prediction)
    truth_num = str_to_num(ground_truth)

    if pred_num != "n/a" and truth_num != "n/a":
        # Use relative tolerance for comparison
        if isinstance(pred_num, (int, float)) and isinstance(truth_num, (int, float)):
            # Handle special case of zero value
            if truth_num == 0:
                return 1.0 if abs(pred_num) < 1e-5 else 0.0

            # Use relative error
            relative_diff = abs(pred_num - truth_num) / max(abs(truth_num), 1e-10)
            return 1.0 if relative_diff < 1e-5 else 0.0

    # Handle text type exact matching
    return (
        1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0
    )


def arithmetic_exact_match(prediction, ground_truth, tolerance=1e-5):
    """
    Exact match for arithmetic problems, allowing some tolerance
    """
    pred_num = str_to_num(prediction)
    truth_num = str_to_num(ground_truth)

    if pred_num == "n/a" or truth_num == "n/a":
        return 0.0

    # Handle special case of zero value
    if abs(truth_num) < 1e-10:
        return 1.0 if abs(pred_num) < tolerance else 0.0

    # Use relative or absolute error, whichever is smaller
    rel_tol = min(
        abs(truth_num) * 0.01, 0.1
    )  # Allow 1% relative error, but not exceeding 0.1
    abs_tol = min(
        abs(truth_num) / 1000, 0.1
    )  # Allow one-thousandth absolute error, but not exceeding 0.1

    return 1.0 if abs(pred_num - truth_num) <= max(rel_tol, abs_tol) else 0.0


def get_tokens(s):
    """Split text into tokens"""
    if not s:
        return []
    s = normalize_answer(s)
    return s.split()


def compute_f1(prediction, ground_truth):
    """Calculate F1 score"""
    if prediction is None or ground_truth is None:
        return 0.0

    # First try numeric comparison
    pred_num = str_to_num(prediction)
    truth_num = str_to_num(ground_truth)

    if pred_num != "n/a" and truth_num != "n/a":
        # If both are numeric, use exact match as F1
        if abs(truth_num) < 1e-10:
            return 1.0 if abs(pred_num) < 1e-5 else 0.0

        rel_tol = min(abs(truth_num) * 0.01, 0.1)
        abs_tol = min(abs(truth_num) / 1000, 0.1)

        return 1.0 if abs(pred_num - truth_num) <= max(rel_tol, abs_tol) else 0.0

    # Text F1 calculation
    prediction_tokens = get_tokens(prediction)
    ground_truth_tokens = get_tokens(ground_truth)

    # If both are empty, return F1=1.0
    if len(prediction_tokens) == 0 and len(ground_truth_tokens) == 0:
        return 1.0

    # Calculate common tokens
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    # If no common tokens, return F1=0
    if num_same == 0:
        return 0.0

    # Calculate precision and recall
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)

    # Calculate F1
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate_predictions(input_file, output_file=None):
    """Evaluate prediction results, extracting predictions and ground truth from a single file"""
    # Load input file, which contains both predictions and ground truth answers
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Calculate metrics
    total_examples = len(data)
    exact_match_count = 0
    arithmetic_match_count = 0
    f1_score_sum = 0
    results = []

    for item in data:
        # Ensure data item contains necessary fields
        if "ground_truth" not in item and "answer" not in item:
            print(
                f"Warning: No ground truth found for example {item.get('uid', 'unknown')}"
            )
            continue

        # Get ground truth answer
        ground_truth = (
            item.get("ground_truth") if "ground_truth" in item else item.get("answer")
        )

        # Get prediction answer, preferring extracted_answer
        # prediction = item.get('extracted_answer')
        # if not prediction:
        #     prediction = item.get('model_answer', '')

        prediction = extract_answer_from_response(item.get("model_answer", ""))
        # import pdb;pdb.set_trace()

        # Calculate exact match score
        em_score = exact_match_score(prediction, ground_truth)
        exact_match_count += em_score

        # Try calculating arithmetic exact match score
        arith_score = arithmetic_exact_match(prediction, ground_truth)
        arithmetic_match_count += arith_score

        # Calculate F1 score
        f1 = compute_f1(prediction, ground_truth)
        f1_score_sum += f1

        prompt = item.get("prompt", "")
        question_type = item.get("question_type", "")

        # Save evaluation results
        result = {
            "uid": item.get("uid", "unknown"),
            "prompt": prompt,
            "full_response": item.get("model_answer", ""),
            "question_type": question_type,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "exact_match": em_score == 1.0,
            "arithmetic_match": arith_score == 1.0,
            "f1_score": f1,
        }
        results.append(result)

    # Calculate overall metrics
    exact_match = exact_match_count / total_examples if total_examples > 0 else 0
    arithmetic_accuracy = (
        arithmetic_match_count / total_examples if total_examples > 0 else 0
    )
    avg_f1 = f1_score_sum / total_examples if total_examples > 0 else 0

    # Output evaluation results
    print(f"Total examples: {total_examples}")
    print(f"Exact Match: {exact_match:.4f}")
    print(f"Arithmetic Accuracy: {arithmetic_accuracy:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")

    # Save detailed evaluation results
    if output_file:
        evaluation_results = {
            "metrics": {
                "exact_match": exact_match,
                "arithmetic_accuracy": arithmetic_accuracy,
                "f1_score": avg_f1,
                "total_examples": total_examples,
            },
            "results": results,
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

        print(f"Detailed evaluation results saved to {output_file}")

    return exact_match, arithmetic_accuracy, avg_f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate MultHier-TT predictions")
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to input file containing predictions and ground truth answers",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output evaluation results",
    )
    parser.add_argument(
        "--base_path", type=str, help="Base path for the project (optional)"
    )

    args = parser.parse_args()

    evaluate_predictions(args.results_file, args.output_file)


if __name__ == "__main__":
    main()
