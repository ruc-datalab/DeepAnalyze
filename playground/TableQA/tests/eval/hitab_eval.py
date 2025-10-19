import json
import re
import argparse
from tqdm import tqdm
from typing import Dict, List, Union, Any, Tuple
import pandas as pd
import numpy as np
import os


def process_result_file(results_file: str) -> Dict[str, Dict]:
    """
    Process results file into a dictionary with question IDs as keys
    """
    with open(results_file, "r", encoding="utf-8") as f:
        results_data = json.load(f)

    # Check if results are in a nested structure
    if "results" in results_data:
        results = results_data["results"]
    else:
        results = results_data

    processed_results = {}
    for item in results:
        if "id" not in item:
            continue

        question_id = item["id"]
        model_answer = item.get("model_answer", "")
        full_response = item.get("full_response", "")
        expected_answer = item.get("expected_answer", [])

        # Ensure model_answer is a string
        if isinstance(model_answer, list) and len(model_answer) > 0:
            model_answer = str(model_answer[0])

        # Ensure expected_answer is properly formatted
        if isinstance(expected_answer, list):
            expected_answer = [str(ans) for ans in expected_answer]
        else:
            expected_answer = [str(expected_answer)]

        processed_results[question_id] = {
            "model_answer": model_answer,
            "full_response": full_response,
            "expected_answer": expected_answer,
            "question": item.get("question", ""),
            "prompt": item.get("prompt", ""),
        }

    return processed_results


def extract_final_answer(response):
    """Extract final answer from response, supporting multiple formats"""
    if not response:
        return ""

    # Try to extract answer from <answer> tags
    answer_tag_pattern = re.search(r"<Answer>(.*?)</Answer>", response, re.DOTALL)
    if answer_tag_pattern:
        answer_content = answer_tag_pattern.group(1).strip()

        # Check if there's an "Answer:" prefix inside the tags
        if "Answer:" in answer_content:
            return answer_content.split("Answer:")[-1].strip()
        return answer_content

    # Try to use regex to match "Answer: xxx" pattern
    answer_pattern = re.findall(
        r"Answer:\s*(.*?)(?:$|\n|\.(?:\s|$))", response, re.IGNORECASE | re.DOTALL
    )
    if answer_pattern:
        return answer_pattern[-1].strip()

    # If all extraction methods fail, return the original response
    return response


def normalize_answer(answer):
    """Normalize the answer for better matching"""
    if answer is None:
        return ""

    answer = str(answer).lower()

    # Remove unnecessary characters but preserve spaces
    answer = re.sub(r"[^\w\s\.]", "", answer).strip()

    # Normalize whitespace
    answer = re.sub(r"\s+", " ", answer)

    # Remove common prefixes like "the answer is", etc.
    prefixes = ["the answer is", "answer:", "=", "approximately", "about", "around"]
    for prefix in prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix) :].strip()

    return answer


def is_number(s):
    """Check if string can be converted to a number"""
    try:
        float(s.replace(",", ""))
        return True
    except (ValueError, TypeError):
        return False


def match_values(predicted, actual):
    """
    Check if predicted value matches with any of the actual values
    Supporting both numeric and text matching with special cases handling
    """
    if not predicted or not actual:
        return False

    normalized_predicted = normalize_answer(predicted)

    # Case 1: Handle answers with units
    # If the predicted answer contains a number with a unit, strip the unit
    if (
        len(normalized_predicted.split()) > 0
        and is_number(normalized_predicted.split()[0])
        and len(normalized_predicted.split()) > 1
    ):
        # Get just the numeric part
        numeric_part = normalized_predicted.split()[0]
        if any(match_numeric(numeric_part, act) for act in actual):
            return True

    # Case 2: Handle multiple expected answers (all must be present)
    if len(actual) > 1:
        # Check if all expected answers are present in the predicted answer
        normalized_actuals = [normalize_answer(act) for act in actual]

        # For each expected answer, check if it's in the prediction
        matches = []
        for expected_val in normalized_actuals:
            # Check either direct string containment or numeric match
            found = expected_val in normalized_predicted or (
                is_number(expected_val)
                and any(
                    match_numeric(expected_val, part)
                    for part in normalized_predicted.split()
                )
            )
            matches.append(found)

        # If all expected answers are found, return True
        if all(matches):
            return True

    # Handle numeric answers with different formats
    if is_number(normalized_predicted):
        for act in actual:
            if match_numeric(normalized_predicted, act):
                return True

    # Direct string matching
    return normalized_predicted in [normalize_answer(act) for act in actual]


def match_numeric(pred, actual):
    """Match numeric values with tolerance for small differences"""
    norm_pred = normalize_answer(pred)
    norm_actual = normalize_answer(actual)

    if is_number(norm_pred) and is_number(norm_actual):
        pred_num = float(norm_pred.replace(",", ""))
        act_num = float(norm_actual.replace(",", ""))
        # Allow small difference for floating point numbers
        return abs(pred_num - act_num) < 1e-6

    return False


def evaluate(results_dict: Dict[str, Dict]) -> Tuple[float, List[Dict]]:
    """
    Evaluate model answers against expected answers using exact accuracy
    """
    total = 0
    correct = 0
    details = []

    for qid, item in tqdm(results_dict.items(), desc="Evaluating"):
        total += 1
        model_answer = item["model_answer"]
        expected_answer = item["expected_answer"]
        # expected_answer = extract_final_answer(model_answer)

        # Check if model answer matches any of the expected answers
        is_correct = match_values(model_answer, expected_answer)

        if is_correct:
            correct += 1

        # Save detailed information for this example
        detail = {
            "id": qid,
            "question": item["question"],
            "prompt": item["prompt"],
            "model_answer": model_answer,
            "expected_answer": expected_answer,
            "is_correct": is_correct,
            "full_response": item["full_response"],
        }
        details.append(detail)

    exact_accuracy = correct / total if total > 0 else 0
    return exact_accuracy, details


def save_results(exact_accuracy: float, details: List[Dict], output_file: str):
    """
    Save evaluation results to a file
    """
    results = {"exact_accuracy": exact_accuracy * 100, "details": details}

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")


def analyze_results(details: List[Dict]) -> Dict:
    """
    Perform additional analysis on results
    """
    df = pd.DataFrame(details)

    # Calculate accuracy by converting correctness to numeric values
    accuracy = df["is_correct"].mean() * 100

    # Analyze error patterns if needed
    error_examples = df[~df["is_correct"]]

    # Additional breakdowns can be added here

    analysis = {
        "total_examples": len(details),
        "correct_examples": int(df["is_correct"].sum()),
        "accuracy": accuracy,
        "error_count": len(error_examples),
    }

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Evaluate HiTAB results")
    parser.add_argument(
        "--results_file", type=str, required=True, help="Path to the results JSON file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the evaluation results",
    )
    parser.add_argument(
        "--base_path", type=str, help="Base path for the project (optional)"
    )

    args = parser.parse_args()

    # Process the results file
    results_dict = process_result_file(args.results_file)

    # Evaluate
    exact_accuracy, details = evaluate(results_dict)

    # Analyze results
    analysis = analyze_results(details)

    # Print summary results
    print("=" * 40)
    print(f"Exact Accuracy (EA): {exact_accuracy * 100:.2f}%")
    print(f"Total examples: {analysis['total_examples']}")
    print(f"Correct examples: {analysis['correct_examples']}")
    print("=" * 40)

    # Save detailed results
    save_results(exact_accuracy, details, args.output_file)


if __name__ == "__main__":
    main()
