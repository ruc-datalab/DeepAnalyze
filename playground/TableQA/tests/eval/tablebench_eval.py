import json
import pandas as pd
import numpy as np

import re
import os
import argparse
from tqdm import tqdm
from typing import Dict, List, Union, Any, Tuple
import string


# def extract_answer_from_response(response: str) -> str:
#     """Extract final answer from model response"""
#     # Look for "Answer: " pattern at the end of the text
#     match = re.search(r'Answer:\s*(.*?)$', response, re.DOTALL)
#     if match:
#         return match.group(1).strip()
#     return ""


def extract_answer_from_response(response: str) -> str:
    """Extract final answer from model response"""
    # Look for all "Answer: ..." patterns and return the last one
    matches = re.findall(r"Answer:\s*(.*?)$", response, re.DOTALL)
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

        processed_results[question_id] = {
            "model_answer": model_answer,
            "full_response": full_response,
            "gold_answer": gold_answer,
            "answer_type": answer_type,
            "scale": scale,
        }

    return processed_results


def normalize_answer(s):
    if not s:
        return ""

    s = str(s).lower().strip()

    number_match = re.search(r"([-+]?[\d,]+\.?\d*)", s)
    if number_match:
        num_str = number_match.group(1).replace(",", "")

        try:
            num_val = float(num_str)
            # Return as integer if it's a whole number
            if num_val == int(num_val):
                return str(int(num_val))
            return str(num_val)
        except:
            return num_str

    # Non-numeric answers - standard text normalization
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s)))


def compute_exact_match(ground_truth, prediction):

    if isinstance(ground_truth, list):

        return any(compute_exact_match(gt, prediction) for gt in ground_truth)

    norm_gt = normalize_answer(ground_truth)
    norm_pred = normalize_answer(prediction)

    try:
        gt_float = float(norm_gt)
        pred_float = float(norm_pred)

        return abs(gt_float - pred_float) < 1e-6
    except (ValueError, TypeError):

        return norm_gt == norm_pred


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
        qtype = item.get("qtype", "")

        processed_results[question_id] = {
            "model_answer": model_answer,
            "prompt": item.get("prompt", ""),
            "full_response": full_response,
            "gold_answer": gold_answer,
            "answer_type": answer_type,
            "scale": scale,
            "qtype": qtype,  # 保存问题类型
        }

    return processed_results


def evaluate(results_dict: Dict[str, Dict]) -> Tuple[float, Dict, Dict]:

    score_list = []
    score_dict = {}

    qtype_scores = {"NumericalReasoning": [], "FactChecking": []}

    qtype_counts = {"NumericalReasoning": 0, "FactChecking": 0, "Other": 0}

    for qid, item in tqdm(results_dict.items(), desc="Evaluating"):
        model_answer = item["model_answer"]
        gold_answer = item["gold_answer"]
        qtype = item.get("qtype", "Other")

        if model_answer and gold_answer:
            model_answer = normalize_answer(model_answer)
            gold_answer = normalize_answer(gold_answer)
            exact_match = compute_exact_match(gold_answer, model_answer)
        else:
            exact_match = 0.0

        score_list.append(exact_match)
        score_dict[qid] = exact_match

        if qtype in qtype_scores:
            qtype_scores[qtype].append(exact_match)
            qtype_counts[qtype] += 1
        else:
            qtype_counts["Other"] += 1

    overall_em_score = np.mean(score_list) if score_list else 0.0

    qtype_em_scores = {}
    for qtype, scores in qtype_scores.items():
        if scores:
            qtype_em_scores[qtype] = np.mean(scores)
        else:
            qtype_em_scores[qtype] = 0.0

    return overall_em_score, score_dict, qtype_em_scores


def save_results(
    em_score, details, qtype_em_scores, output_file: str, results_file: str
):
    """
    Save evaluation results to a file, including breakdown by question type
    """
    with open(results_file, "r", encoding="utf-8") as f:
        original_results = json.load(f)

    enhanced_details = []
    for item in original_results:
        detail_copy = {}
        id = item.get("id", "")
        detail_copy["id"] = id
        detail_copy["model_answer"] = item.get("model_answer", "")
        detail_copy["prompt"] = item.get("prompt", "")
        detail_copy["full_response"] = item.get("full_response", "")
        detail_copy["gold_answer"] = item.get("gold_answer", "")
        detail_copy["question"] = item.get("question", "")
        detail_copy["qtype"] = item.get("qtype", "")
        detail_copy["em_score"] = details.get(id, 0)
        enhanced_details.append(detail_copy)

    results = {
        "em_score": em_score * 100,
        "qtype_em_scores": {
            qtype: score * 100 for qtype, score in qtype_em_scores.items()
        },
        "details": enhanced_details,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Tablebench results")
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

    results_dict = process_result_file(args.results_file)

    em_score, details, qtype_em_scores = evaluate(results_dict)

    print("--------------------------------------")
    print(f"Overall Exact Match Score: {em_score * 100:.2f}")

    for qtype, score in qtype_em_scores.items():
        if score > 0:
            print(f"{qtype} Exact Match Score: {score * 100:.2f}")
    print("--------------------------------------")

    save_results(
        em_score, details, qtype_em_scores, args.output_file, args.results_file
    )


if __name__ == "__main__":
    main()
