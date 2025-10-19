import json
import pandas as pd
import numpy as np
from utils.tatqa_metric import TaTQAEmAndF1, normalize_multi_span
from utils.tatqa_utils import normalize_answer, is_number, to_number, scale_to_num
import re
import os
import argparse
from tqdm import tqdm
from typing import Dict, List, Union, Any, Tuple

# def extract_answer_from_response(response: str) -> str:
#     """Extract final answer from model response"""
#     # Look for "Answer: " pattern at the end of the text
#     match = re.search(r'Answer:\s*(.*?)$', response, re.DOTALL)
#     if match:
#         return match.group(1).strip()
#     return ""


def extract_final_answer(response):
    """Extract final answer from model response"""
    if not response:
        return ""

    answer_tag_pattern = re.search(r"<Answer>(.*?)</Answer>", response, re.DOTALL)
    if answer_tag_pattern:
        answer_content = answer_tag_pattern.group(1).strip()

        if "Answer:" in answer_content:
            return answer_content.split("Answer:")[-1].strip()
        return answer_content

    elif "Answer:" in response:
        return response.split("Answer:")[-1].strip()

    return response


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
        # model_answer = item.get("model_answer", "")
        full_response = item.get("full_response", "").replace("**", "")

        model_answer = extract_final_answer(full_response)
        gold_answer = item.get("gold_answer", [])
        answer_type = item.get("answer_type", "")
        scale = item.get("scale", "")

        processed_results[question_id] = {
            "model_answer": model_answer,
            "prompt": item.get("prompt", ""),
            "full_response": full_response,
            "gold_answer": gold_answer,
            "answer_type": answer_type,
            "scale": scale,
        }

    return processed_results


def normalize_numeric_answer(answer):
    if isinstance(answer, list):
        return [
            str(item) if isinstance(item, (int, float)) else item for item in answer
        ]
    elif isinstance(answer, (int, float)):
        return str(answer)
    else:
        return answer


def evaluate(results_dict: Dict[str, Dict]) -> Tuple[float, float]:
    """
    Evaluate model answers against gold answers
    """
    em_and_f1 = TaTQAEmAndF1()

    for qid, item in tqdm(results_dict.items(), desc="Evaluating"):
        model_answer = item["model_answer"]
        gold_answer = item["gold_answer"]
        answer_type = item["answer_type"]
        scale = item["scale"]

        if answer_type == "multi-span" and not isinstance(model_answer, list):
            model_answer = normalize_multi_span(model_answer)
        elif answer_type == "count":

            if isinstance(model_answer, list) and len(model_answer) > 0:
                model_answer = model_answer[0]
            try:
                model_answer = str(int(float(str(model_answer))))
            except:
                model_answer = str(model_answer)

            if isinstance(gold_answer, list) and len(gold_answer) > 0:
                gold_answer = gold_answer[0]
            try:
                gold_answer = str(int(float(str(gold_answer))))
            except:
                gold_answer = str(gold_answer)
        elif answer_type == "arithmetic" and not isinstance(model_answer, list):
            model_answer = [str(model_answer)]

        if answer_type != "count":
            if not isinstance(model_answer, list):
                model_answer = [model_answer]
            if not isinstance(gold_answer, list):
                gold_answer = [gold_answer]

        ground_truth = {
            "answer": gold_answer,
            "answer_type": answer_type,
            "scale": scale,
            "uid": qid,
        }

        try:
            em_and_f1(
                ground_truth=ground_truth, prediction=model_answer, pred_scale=scale
            )
        except Exception as e:
            print(f"Error evaluating answer for question {qid}: {e}")
            print(f"Gold answer: {gold_answer}")
            print(f"Model answer: {model_answer}")
            continue

    exact_match, f1_score, scale_score, op_score = em_and_f1.get_overall_metric()

    details = em_and_f1.get_raw()

    return exact_match, f1_score, details


def save_results(
    exact_match: float,
    f1_score: float,
    details: List[Dict],
    output_file: str,
    results_file: str,
):
    """
    Save evaluation results to a file
    """
    with open(results_file, "r", encoding="utf-8") as f:
        original_results = json.load(f)

    original_data_map = {}
    for item in original_results:
        if "id" in item:
            original_data_map[item["id"]] = item

    enhanced_details = []
    for detail in details:
        detail_copy = detail.copy()

        uid = detail.get("uid", "")
        if uid in original_data_map:
            original_item = original_data_map[uid]
            detail_copy["prompt"] = original_item.get("prompt", "")
            detail_copy["full_response"] = original_item.get(
                "full_response", ""
            ).replace("**", "")

        enhanced_details.append(detail_copy)

    results = {
        "exact_match": exact_match * 100,
        "f1_score": f1_score * 100,
        "details": enhanced_details,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TatQA results")
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
    exact_match, f1_score, details = evaluate(results_dict)

    # Print summary results
    print("--------------------------------------")
    print(f"Exact-match accuracy: {exact_match * 100:.2f}")
    print(f"F1 score: {f1_score * 100:.2f}")
    print(f"{exact_match * 100:.2f} & {f1_score * 100:.2f}")
    print("--------------------------------------")

    # Save detailed results
    save_results(exact_match, f1_score, details, args.output_file, args.results_file)


if __name__ == "__main__":
    main()
