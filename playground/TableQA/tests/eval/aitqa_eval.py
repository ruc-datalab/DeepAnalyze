import json
import pandas as pd
import numpy as np
import re
import os
import argparse
from tqdm import tqdm
from typing import Dict, List, Union, Any, Tuple

# def extract_answer_from_response(response: str) -> str:
#     """Extract final answer from model response"""
#     # Look for answer in <answer> tags
#     match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
#     if match:
#         return match.group(1).strip()
#     return ""


def extract_answer_from_response(response: str) -> str:
    """Extract the final <answer>...</answer> content from the model response (last one if multiple)"""

    last_answer_start = response.rfind("<Answer>")
    if last_answer_start == -1:
        return ""

    # 截取从最后一个 <answer> 开始的内容
    truncated_response = response[last_answer_start:]

    matches = re.findall(r"<Answer>(.*?)</Answer>", truncated_response, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return ""


def process_result_file(results_file: str) -> Dict[str, Dict]:
    """
    Process results file into a dictionary with question IDs as keys
    """
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    processed_results = {}
    for item in results:
        if "id" not in item:
            continue

        question_id = item["id"]
        model_answer = item.get("model_answer", "")
        full_response = item.get("full_response", "")
        gold_answer = item.get("gold_answer", "")
        answer_type = item.get("answer_type", "")
        scale = item.get("scale", "")
        question_type = item.get("type", "unknown")  # Add question type field

        processed_results[question_id] = {
            "model_answer": model_answer,
            "full_response": full_response,
            "gold_answer": gold_answer,
            "answer_type": answer_type,
            "scale": scale,
            "type": question_type,  # Store question type
        }

    return processed_results


def normalize_answer(answer):
    """
    Normalize answers for comparison with improved numeric handling
    """
    if isinstance(answer, list):
        return [normalize_answer(a) for a in answer]

    # Convert to string and lowercase
    answer = str(answer).lower().strip()

    # Handle common units
    answer = re.sub(r"million(s)?", "", answer)
    answer = re.sub(r"billion(s)?", "", answer)
    answer = re.sub(r"thousand(s)?", "", answer)
    answer = re.sub(r"gallons?", "", answer)

    # Handle negative numbers in parentheses (1,234) -> -1234
    if answer.startswith("(") and answer.endswith(")"):
        answer = "-" + answer[1:-1]

    # Remove currency symbols and commas
    answer = answer.replace("$", "")
    answer = answer.replace(",", "")

    # Remove percent sign (critical fix)
    answer = answer.replace("%", "")

    # Try numeric conversion for standardization
    try:
        value = float(answer)
        if value == int(value):
            # Return as integer if it's a whole number
            return str(int(value))
        return str(value)
    except:
        # Not numeric, just clean and return
        return "".join(c for c in answer if c.isdigit() or c in ".-")


def compute_answer_match(ground_truth, prediction):
    """
    Calculate if answers match with robust numeric comparison
    """
    if isinstance(ground_truth, list):
        return any(compute_answer_match(gt, prediction) for gt in ground_truth)

    # Normalize both answers
    norm_gt = normalize_answer(ground_truth)
    norm_pred = normalize_answer(prediction)

    # Try numeric comparison first
    try:
        gt_num = float(norm_gt)
        pred_num = float(norm_pred)
        # Use small epsilon for floating point comparison
        return abs(gt_num - pred_num) < 1e-6
    except:
        # Fall back to string comparison
        return norm_gt == norm_pred


def evaluate(results_dict: Dict[str, Dict]):
    """
    Evaluate model answers against gold answers with type breakdown
    """
    # Overall scoring
    score_list = []
    score_dict = {}

    # Type-based scoring
    type_scores = {"Table-driven": [], "KPI-driven": [], "unknown": []}

    for qid, item in tqdm(results_dict.items(), desc="Evaluating"):
        # import pdb;pdb.set_trace()
        model_answer = extract_answer_from_response(item["full_response"])
        gold_answer = item["gold_answer"]
        question_type = item.get("type", "unknown")

        if model_answer and gold_answer:
            # Calculate answer match
            is_correct = compute_answer_match(gold_answer, model_answer)
            score = 1.0 if is_correct else 0.0
        else:
            score = 0.0

        # Add to overall scores
        score_list.append(score)
        score_dict[qid] = score

        # Add to type-based scores
        if question_type in type_scores:
            type_scores[question_type].append(score)
        else:
            type_scores["unknown"].append(score)

    # Calculate overall accuracy
    acc_score = np.mean(score_list) if score_list else 0.0

    # Calculate type-based accuracies
    type_acc_scores = {}
    for qtype, scores in type_scores.items():
        if scores:
            type_acc_scores[qtype] = np.mean(scores)
        else:
            type_acc_scores[qtype] = 0.0

    return acc_score, score_dict, type_acc_scores


def save_results(
    acc_score, details, type_acc_scores, output_file: str, results_file: str
):
    """
    Save evaluation results to a file with type breakdown
    """
    with open(results_file, "r", encoding="utf-8") as f:
        original_results = json.load(f)

    enhanced_details = []
    for item in original_results:
        detail_copy = {}
        id = item.get("id", "")
        detail_copy["id"] = id
        detail_copy["model_answer"] = item.get("model_answer", "")
        detail_copy["full_response"] = item.get("full_response", "")
        detail_copy["gold_answer"] = item.get("gold_answer", "")
        detail_copy["question"] = item.get("question", "")
        detail_copy["type"] = item.get("type", "unknown")  # Store question type
        detail_copy["acc_score"] = details.get(id, 0)
        enhanced_details.append(detail_copy)

    results = {
        "accuracy": acc_score * 100,
        "type_accuracy": {
            qtype: score * 100 for qtype, score in type_acc_scores.items() if score > 0
        },
        "correct_count": sum(1 for score in details.values() if score > 0),
        "total_count": len(details),
        "details": enhanced_details,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AIT QA results")
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to the JSON file with prediction results",
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
    acc_score, details, type_acc_scores = evaluate(results_dict)

    # Print summary results
    print("--------------------------------------")
    print(f"Overall Accuracy: {acc_score * 100:.2f}%")
    print(
        f"Correct Answers: {sum(1 for score in details.values() if score > 0)}/{len(details)}"
    )

    # Print type-based accuracies
    for qtype, score in type_acc_scores.items():
        if score > 0 and qtype != "unknown":
            # Fixed: Corrected the list comprehension to count items with matching type
            count = len(
                [item for item in results_dict.values() if item.get("type") == qtype]
            )
            if count > 0:
                print(
                    f"{qtype} Accuracy: {score * 100:.2f}% ({int(score * count)}/{count})"
                )
    print("--------------------------------------")

    # Save detailed results
    save_results(
        acc_score, details, type_acc_scores, args.output_file, args.results_file
    )


if __name__ == "__main__":
    main()
