import json
import re
import collections
import string
import sys
import argparse
import os


def clean_answer_tags(answer_text):
    """Clean answer tags and extract the most relevant answer text"""
    if not answer_text:
        return ""

    # Remove <answer> and </answer> tags
    cleaned_text = re.sub(r"</?Answer>", "", answer_text)

    # Check if it contains a thinking block
    think_pattern = r"<Analyze>(.*?)</Analyze>\s*"
    think_match = re.search(think_pattern, cleaned_text, re.DOTALL)

    # If thinking block is found
    if think_match:
        # Get the text inside the thinking block
        think_content = think_match.group(1)
        # Extract content after the thinking block
        after_think_content = cleaned_text[think_match.end() :].strip()

        # If there's content after the thinking block, use that
        if after_think_content:
            cleaned_text = after_think_content
        else:
            # If there's no content after the thinking block, look for an answer inside it
            answer_pattern = r"(?:the\s+answer\s+is\s*:?\s*)(.*?)(?:$|\.|\n)"
            answer_match = re.search(answer_pattern, think_content, re.IGNORECASE)

            if answer_match:
                # Extract answer from inside the thinking block
                extracted_answer = answer_match.group(1).strip()
                # Handle possible quotes and periods
                extracted_answer = re.sub(
                    r'^["\'](.*?)["\']\.?$', r"\1", extracted_answer
                )
                return extracted_answer

            # If there's no explicit "the answer is" in the thinking block, try matching "Therefore, ..." or "Thus, ..."
            conclusion_pattern = r"(?:therefore|thus|so|hence|consequently|in conclusion)[,:]?\s+(.*?)(?:$|\.|\n)"
            conclusion_match = re.search(
                conclusion_pattern, think_content, re.IGNORECASE
            )

            if conclusion_match:
                return conclusion_match.group(1).strip()

    # Look for "the answer is" pattern in the full text
    answer_pattern = r"(?:the\s+answer\s+is\s*:?\s*)(.*?)(?:$|\.|\n)"
    answer_match = re.search(answer_pattern, cleaned_text, re.IGNORECASE)

    if answer_match:
        # Extract the matching content
        extracted_answer = answer_match.group(1).strip()
        # Handle possible quotes
        extracted_answer = re.sub(r'^["\'](.*)["\']$', r"\1", extracted_answer)
        return extracted_answer

    # If no specific pattern was found, return the cleaned text
    return cleaned_text.strip()


def extract_answer_from_response(model_answer):
    """Extract final answer from the model response, supporting multiple formats"""
    if not model_answer:
        return ""

    # Case 1: <answer>...</answer> tag
    answer_tag_pattern = re.search(r"<Answer>(.*?)</Answer>", model_answer, re.DOTALL)
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
    """Normalize answer text: lowercase, remove punctuation, articles, and extra whitespace"""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    """Get normalized tokens from text"""
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    """Calculate exact match score"""
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    """Calculate F1 score"""
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If answer is empty, F1 is 1 (if both are empty), else 0
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def prepare_model_results(model_results_file):
    """Prepare model results data, extract the extracted answers"""
    with open(model_results_file, "r", encoding="utf-8") as f:
        model_results = json.load(f)

    # Build mapping from question ID to predicted answer
    results = []
    for item in model_results:
        # extracted_answer = item.get("extracted_answer", "")
        extracted_answer = extract_answer_from_response(item.get("model_answer", ""))
        cleaned_answer = clean_answer_tags(extracted_answer)

        result = {
            "question_id": item["question_id"],
            "question": item.get("question", ""),
            "pred": cleaned_answer,  # Use extracted answer as prediction
        }
        results.append(result)

    return results


def prepare_reference_from_results(model_results_file):
    """Prepare reference answer data from model results file"""
    with open(model_results_file, "r", encoding="utf-8") as f:
        model_results = json.load(f)

    # Build mapping from question ID to gold answer
    reference = {"reference": {}, "table": [], "passage": []}

    for item in model_results:
        question_id = item["question_id"]
        reference["reference"][question_id] = item.get(
            "ground_answer", ""
        )  # Get ground truth from results

        # If there's target type information, use it to classify table or passage questions
        if "target" in item and item.get("target") and len(item["target"]) > 2:
            if item["target"][2] is not None:  # If there's a link, it's a passage
                reference["passage"].append(question_id)
            else:
                reference["table"].append(question_id)
        # If there's no target type information, infer from other fields in the results
        elif "relevant_links" in item and item.get("relevant_links"):
            # If there are relevant links, this is a mixed/passage question
            reference["passage"].append(question_id)
        else:
            # Default to table type
            reference["table"].append(question_id)

    return reference


def prepare_test_answers(test_answers_file):
    """Prepare test answer data"""
    with open(test_answers_file, "r", encoding="utf-8") as f:
        test_answers = json.load(f)

    # Build mapping from question ID to gold answer
    reference = {"reference": {}, "table": [], "passage": []}

    for item in test_answers:
        question_id = item["question_id"]
        reference["reference"][question_id] = item["pred"]  # Use the correct answer

        # Determine if it's a table or passage based on target type
        if "target" in item and item["target"] and len(item["target"]) > 2:
            if item["target"][2] is not None:  # If there's a link, it's a passage
                reference["passage"].append(question_id)
            else:
                reference["table"].append(question_id)

    # Ensure each question is assigned to either table or passage
    for qid in reference["reference"].keys():
        if qid not in reference["table"] and qid not in reference["passage"]:
            reference["table"].append(qid)  # Default to table type

    return reference


def get_raw_scores(model_results, reference):
    """Calculate exact match and F1 scores"""
    # Keep original scoring logic
    exact_scores = {}
    f1_scores = {}

    # Find question IDs that actually exist in the model results
    evaluated_qids = set(example["question_id"] for example in model_results)

    # Filter reference answers by question IDs that are actually evaluated
    filtered_reference = {
        "reference": {
            k: v for k, v in reference["reference"].items() if k in evaluated_qids
        },
        "table": [k for k in reference["table"] if k in evaluated_qids],
        "passage": [k for k in reference["passage"] if k in evaluated_qids],
    }

    # Calculate scores
    for example in model_results:
        qas_id = example["question_id"]
        if qas_id in filtered_reference["reference"]:
            gold_answer = filtered_reference["reference"][qas_id]
            prediction = example["pred"]

            exact_scores[qas_id] = compute_exact(gold_answer, prediction)
            f1_scores[qas_id] = compute_f1(gold_answer, prediction)

    # Use filtered question ID list
    qid_list = list(filtered_reference["reference"].keys())
    total = len(qid_list)

    table_list = filtered_reference["table"]
    passage_list = filtered_reference["passage"]

    # Prevent division by zero
    table_exact = 0
    table_f1 = 0
    if table_list:
        table_exact = (
            100.0 * sum(exact_scores.get(k, 0) for k in table_list) / len(table_list)
        )
        table_f1 = (
            100.0 * sum(f1_scores.get(k, 0) for k in table_list) / len(table_list)
        )

    passage_exact = 0
    passage_f1 = 0
    if passage_list:
        passage_exact = (
            100.0
            * sum(exact_scores.get(k, 0) for k in passage_list)
            / len(passage_list)
        )
        passage_f1 = (
            100.0 * sum(f1_scores.get(k, 0) for k in passage_list) / len(passage_list)
        )

    total_exact = (
        100.0 * sum(exact_scores.get(k, 0) for k in qid_list) / total
        if total > 0
        else 0
    )
    total_f1 = (
        100.0 * sum(f1_scores.get(k, 0) for k in qid_list) / total if total > 0 else 0
    )

    # Add additional information
    return collections.OrderedDict(
        [
            ("table exact", table_exact),
            ("table f1", table_f1),
            ("passage exact", passage_exact),
            ("passage f1", passage_f1),
            ("total em", total_exact),
            ("total f1", total_f1),
            ("total", total),  # Total number of questions actually evaluated
            ("table_count", len(table_list)),  # Number of table questions
            ("passage_count", len(passage_list)),  # Number of passage questions
            (
                "total_reference",
                len(reference["reference"]),
            ),  # Total number of questions in the reference
        ]
    )


def create_eval_format(model_results_file, output_file):
    """Create output file in evaluation format"""
    # Keep the original implementation
    with open(model_results_file, "r", encoding="utf-8") as f:
        model_results = json.load(f)

    eval_format = []
    for item in model_results:
        eval_item = {
            "question_id": item["question_id"],
            "pred": item.get("extracted_answer", ""),
        }
        eval_format.append(eval_item)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_format, f, ensure_ascii=False, indent=2)

    print(f"Evaluation format file saved to: {output_file}")


def evaluate_model(model_results_file, evaluate_save_dir, create_eval_format_flag=True):
    """Evaluate model results"""

    # If we need to create an evaluation format file
    if create_eval_format_flag:
        output_file = os.path.splitext(model_results_file)[0] + "_eval_format.json"
        create_eval_format(model_results_file, output_file)

    # Prepare data
    model_results = prepare_model_results(model_results_file)
    reference = prepare_reference_from_results(
        model_results_file
    )  # Now getting reference from model results

    # Calculate scores
    scores = get_raw_scores(model_results, reference)

    # Output results
    print("\n==========OTTQA Evaluation Results ==========")
    print(f"Model results file: {model_results_file}")
    print(f"Evaluation save directory: {evaluate_save_dir}")
    print(
        f"Questions actually evaluated: {scores['total']} / {scores['total_reference']}"
    )  # Show actual evaluated/total questions
    print("-" * 40)
    print(f"Number of table questions: {scores['table_count']}")
    print(f"Number of passage questions: {scores['passage_count']}")
    print(f"Table questions exact match: {scores['table exact']:.2f}%")
    print(f"Table questions F1 score: {scores['table f1']:.2f}%")
    print(f"Passage questions exact match: {scores['passage exact']:.2f}%")
    print(f"Passage questions F1 score: {scores['passage f1']:.2f}%")
    print(f"Overall exact match: {scores['total em']:.2f}%")
    print(f"Overall F1 score: {scores['total f1']:.2f}%")
    print("======================================\n")

    # Create detailed evaluation results, including ground truth and model answers
    detailed_results = []

    # Load the full model results to get detailed model responses
    with open(model_results_file, "r", encoding="utf-8") as f:
        full_model_results = json.load(f)

    # Create a dictionary of model responses for quick lookup
    model_results_dict = {item["question_id"]: item for item in full_model_results}

    # Merge data
    for qid in reference["reference"].keys():
        if qid in model_results_dict:
            model_item = model_results_dict[qid]

            # Determine if the answer is correct
            gold_answer = reference["reference"][qid]
            model_answer = extract_answer_from_response(
                model_item.get("model_answer", "")
            )
            cleaned_answer = clean_answer_tags(model_answer)

            is_correct = compute_exact(gold_answer, cleaned_answer)

            # Question type
            question_type = "passage" if qid in reference["passage"] else "table"

            # Create detailed result item
            detail_item = {
                "question_id": qid,
                "question": model_item.get("question", ""),
                "question_type": question_type,
                "gold_answer": gold_answer,
                "model_answer": cleaned_answer,
                "prompt": model_item.get("prompt", ""),
                "full_response": model_item.get("model_answer", ""),
                "is_correct": is_correct,
                "exact_match": is_correct,
                "f1_score": compute_f1(gold_answer, cleaned_answer),
            }

            # Add target information (if available)
            if "target" in model_item:
                detail_item["target"] = model_item["target"]

            detailed_results.append(detail_item)

    # Create final evaluation result, including summary and detailed results
    final_result = {"summary": scores, "detailed_results": detailed_results}

    # Ensure the parent directory for the evaluation save directory exists
    os.makedirs(os.path.dirname(evaluate_save_dir), exist_ok=True)

    # Save results to file
    with open(evaluate_save_dir, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print(f"Detailed evaluation results saved to: {evaluate_save_dir}")

    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate OTTQA predictions")
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to model prediction results file",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to save evaluation results"
    )
    parser.add_argument("--base_path", type=str, help="Base path for the project")

    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        print(f"Error: Model results file does not exist: {args.results_file}")
        sys.exit(1)

    # Evaluate model results
    evaluate_model(args.results_file, args.output_file)


if __name__ == "__main__":
    main()
