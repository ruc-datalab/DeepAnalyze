import json
import re
import os
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Union, Any, Tuple


def normalize_answer(answer: str) -> Tuple[float, str, bool]:
    """
    Normalize answer format, identify value type and standardize format

    Returns:
        (numeric_value, unit/type, is_percentage)
    """
    if answer is None:
        return None, "", False

    original = answer
    answer = answer.strip().lower()

    # Detect units and format
    is_percentage = "%" in answer
    has_dollar = "$" in answer
    has_million = "million" in answer.lower()
    has_billion = "billion" in answer.lower()

    # Extract numeric part
    num_pattern = r"[-+]?[0-9]*\.?[0-9]+"
    num_match = re.search(num_pattern, answer)

    if not num_match:
        return None, original, False

    try:
        # Extract and convert number
        num_value = float(num_match.group(0))

        # Prepare return value
        if is_percentage:
            return num_value, "%", True
        elif has_million:
            return num_value, "million", False
        elif has_billion:
            return num_value, "billion", False
        elif has_dollar:
            return num_value, "$", False
        else:
            return num_value, "", False
    except:
        return None, original, False


def extract_answer_from_response(model_answer: str) -> str:
    """Extract answer from model response"""
    if not model_answer:
        return ""

    # Try to match content wrapped in <answer> tags
    answer_tag_pattern = r"<Answer>(.*?)</Answer>"

    tag_match = re.search(answer_tag_pattern, model_answer, re.DOTALL)
    if tag_match:
        model_answer = tag_match.group(1).strip()

    # Try to match content after "Answer:"
    answer_pattern = r"Answer:\s*(.*?)(?:(?:\n\s*\n)|(?:\n\s*Used cells:)|$)"
    match = re.findall(answer_pattern, model_answer, re.DOTALL)
    if match:
        # Extract match content and clean it
        answer_text = match[-1].strip()
        answer_text = (
            answer_text.replace("</answer>", "")
            .replace("<answer>", "")
            .replace("Answer:", "")
        )
        return answer_text

    # Fallback approaches: try various answer formats

    # 1. Look for "Answer:" or similar patterns
    for pattern in [r"Answer:\s*(.*?)(?:\n|$)", r"answer:\s*(.*?)(?:\n|$)"]:
        # match = re.search(pattern, model_answer)
        # if match:
        #     return match.group(1).strip()

        answer_matches = re.findall(pattern, model_answer, re.DOTALL)
        if answer_matches:
            return answer_matches[-1].strip()

    # 2. Look for formatted numbers (possibly with currency symbols, percentages or units)
    number_patterns = [
        r"([$¥€£]?\s*[-+]?[0-9,]*\.?[0-9]+\s*(?:million|billion|thousand|M|B|K)?%?)",
        r"([-+]?[0-9,]*\.?[0-9]+\s*(?:million|billion|thousand|M|B|K)?%?)",
        r"([-+]?[0-9,]*\.?[0-9]+\%)",
    ]

    for pattern in number_patterns:
        match = re.findall(pattern, model_answer)
        if match:
            return match[-1].strip()

    # 3. Extract the last line that might contain an answer
    lines = model_answer.strip().split("\n")
    for line in reversed(lines):
        if line.strip() and not line.lower().startswith(
            ("thus", "therefore", "so", "hence")
        ):
            # Try to extract numeric expression from this line
            number_match = re.findall(
                r"([$¥€£]?\s*[-+]?[0-9,]*\.?[0-9]+\s*(?:million|billion|thousand|M|B|K)?%?)",
                line,
            )
            if number_match:
                return number_match[-1].strip()
            break

    return ""


def check_answer_correctness(
    model_answer: str, expected_answer: str
) -> Tuple[bool, str]:
    """Check if model answer is correct, supporting various numeric format comparisons"""
    # Extract model's answer
    extracted_answer = extract_answer_from_response(model_answer)
    if not extracted_answer:
        return False, ""

    # Normalize and extract numeric values
    model_value, model_unit, model_is_percent = normalize_answer(extracted_answer)
    expected_value, expected_unit, expected_is_percent = normalize_answer(
        expected_answer
    )

    if model_value is None or expected_value is None:
        # If numbers cannot be extracted, fall back to exact string matching
        return (
            extracted_answer.strip().lower() == expected_answer.strip().lower(),
            extracted_answer,
        )

    # Check if values and units are equivalent (considering unit variations)
    # For identical values, don't require exactly matching units
    if abs(model_value - expected_value) < 0.01:
        # For very close values, consider correct
        return True, extracted_answer

    # Unit conversion and comparison
    is_same_unit_type = False

    # Check unit type consistency
    if model_is_percent == expected_is_percent:
        is_same_unit_type = True
    # Special handling for "$X" and "X million" cases
    elif model_unit == "$" and expected_unit == "million":
        model_value = model_value * 1000000
        is_same_unit_type = True
    elif expected_unit == "$" and model_unit == "million":
        expected_value = expected_value * 1000000
        is_same_unit_type = True
    # Special handling for "$X" and "X billion" cases
    elif model_unit == "$" and expected_unit == "billion":
        model_value = model_value * 1000000000
        is_same_unit_type = True
    elif expected_unit == "$" and model_unit == "billion":
        expected_value = expected_value * 1000000000
        is_same_unit_type = True
    # Special handling for "$X million" and "X" cases (assuming X alone means millions)
    elif model_unit == "$" and model_unit == "million" and expected_unit == "":
        is_same_unit_type = True
    elif expected_unit == "$" and expected_unit == "million" and model_unit == "":
        is_same_unit_type = True

    if not is_same_unit_type:
        # Unit types inconsistent, but if values are identical, still consider a match
        if abs(model_value - expected_value) < 0.01:
            return True, extracted_answer
        return False, extracted_answer

    # Allow percentage precision errors (round to 1 decimal place)
    if model_is_percent and expected_is_percent:
        # Round to 1 decimal place before comparing
        model_rounded = round(model_value, 1)
        expected_rounded = round(expected_value, 1)
        return abs(model_rounded - expected_rounded) < 0.15, extracted_answer

    # Handle monetary amounts
    if (
        model_unit == "$"
        or model_unit == "million"
        or model_unit == "billion"
        or expected_unit == "$"
        or expected_unit == "million"
        or expected_unit == "billion"
    ):
        # Compare after converting to the same unit
        if abs(model_value - expected_value) / max(abs(expected_value), 1) < 0.05:
            return True, extracted_answer
    else:
        # General number comparison, allow small error margin
        relative_error = abs(model_value - expected_value) / max(abs(expected_value), 1)
        if relative_error < 0.05:  # 5% relative error margin
            return True, extracted_answer

    return False, extracted_answer


def process_result_file(results_file: str) -> Dict[str, Dict]:
    """
    Read data from results file
    """
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    processed_results = {}
    for item in results:
        question_id = item.get("id", "")
        if not question_id:
            continue

        full_response = item.get("full_response", "")
        # Use model_answer if it exists, otherwise extract from full_response
        model_answer = item.get(
            "model_answer", extract_answer_from_response(full_response)
        )
        model_answer = (
            model_answer.replace("</answer>", "")
            .replace("<answer>", "")
            .replace("Answer:", "")
        )

        gold_answer = item.get("gold_answer", "")

        processed_results[question_id] = {
            "model_answer": model_answer,
            "prompt": item.get("prompt", ""),
            "full_response": full_response,
            "gold_answer": gold_answer,
            "question": item.get("question", ""),
            "filename": item.get("filename", ""),
        }

    return processed_results


def evaluate_finqa(results_dict: Dict[str, Dict]) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate accuracy of FinQA results

    Returns:
        Exact match accuracy, detailed results
    """
    correct_count = 0
    total_count = len(results_dict)
    details = []

    for qid, item in tqdm(results_dict.items(), desc="Evaluating FinQA results"):
        model_answer = item.get("full_response", "")  # Use full response for evaluation
        gold_answer = item.get("gold_answer", "")

        is_correct, extracted_answer = check_answer_correctness(
            model_answer, gold_answer
        )

        if is_correct:
            correct_count += 1

        # Record detailed information
        detail = {
            "id": qid,
            "question": item.get("question", ""),
            "prompt": item.get("prompt", ""),
            "full_response": item.get("full_response", ""),
            "gold_answer": gold_answer,
            "model_answer": item.get("model_answer", ""),
            "extracted_answer": extracted_answer,
            "is_correct": is_correct,
            "filename": item.get("filename", ""),
        }
        details.append(detail)

    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    return accuracy, details


def save_results(
    accuracy: float, details: List[Dict], output_file: str, results_file: str
):
    """
    Save evaluation results to file
    """
    with open(results_file, "r", encoding="utf-8") as f:
        original_results = json.load(f)

    # Create mapping from ID to original data
    original_data_map = {}
    for item in original_results:
        if "id" in item:
            original_data_map[item["id"]] = item

    enhanced_details = []
    for detail in details:
        detail_copy = detail.copy()

        # Add prompts and full responses from original data
        uid = detail.get("id", "")
        if uid in original_data_map:
            original_item = original_data_map[uid]
            if "prompt" not in detail_copy and "prompt" in original_item:
                detail_copy["prompt"] = original_item["prompt"]
            if "full_response" not in detail_copy and "full_response" in original_item:
                detail_copy["full_response"] = original_item["full_response"]

        enhanced_details.append(detail_copy)

    correct_count = sum(1 for d in details if d.get("is_correct", False))
    total_count = len(details)

    results = {
        "accuracy": accuracy * 100,
        "correct_count": correct_count,
        "total_count": total_count,
        "details": enhanced_details,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Evaluation results saved to {output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Evaluate model answers for FinQA task"
    )
    parser.add_argument(
        "--results_file", type=str, help="Path to JSON file with prediction results"
    )
    parser.add_argument(
        "--output_file", type=str, help="Path to save evaluation results"
    )
    parser.add_argument(
        "--base_path", type=str, help="Base path for the project (optional)"
    )

    args = parser.parse_args()

    # Process results file
    print(f"Reading results file: {args.results_file}")
    results_dict = process_result_file(args.results_file)
    print(f"Read results for {len(results_dict)} questions")

    # Evaluate
    print("Evaluating...")
    accuracy, details = evaluate_finqa(results_dict)

    # Print summary results
    print("=" * 50)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(
        f"Correct answers: {sum(1 for d in details if d.get('is_correct', False))}/{len(details)}"
    )
    print("=" * 50)

    # Save detailed results
    save_results(accuracy, details, args.output_file, args.results_file)


if __name__ == "__main__":
    main()
