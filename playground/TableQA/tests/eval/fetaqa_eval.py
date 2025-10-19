import json
import re
import argparse
import os
import numpy as np
from typing import Dict, List, Tuple
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def process_result_file(results_file: str) -> List[Dict]:
    """
    Process results file and return a list of examples
    """
    with open(results_file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)

            # Handle different possible JSON structures
            if isinstance(data, dict):
                if "results" in data:
                    results = data["results"]
                else:
                    results = list(data.values())
            elif isinstance(data, list):
                results = data
            else:
                raise ValueError(f"Unexpected data format in {results_file}")

        except json.JSONDecodeError:
            # Try reading as JSONL if JSON parsing fails
            f.seek(0)  # Go back to the beginning of the file
            results = []
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

    examples = []
    for item in results:
        if "reference_answer" not in item or "generated_answer" not in item:
            continue

        examples.append(
            {
                "id": item.get("feta_id", ""),
                "question": item.get("question", ""),
                "reference_answer": item.get("reference_answer", ""),
                "generated_answer": item.get("generated_answer", ""),
                "prompt": item.get("prompt", ""),
                "model_full_response": item.get("model_full_response", ""),
            }
        )

    return examples


def calculate_bleu(reference: str, candidate: str) -> float:
    """
    Calculate BLEU score between reference and candidate strings
    """
    if not reference or not candidate:
        return 0.0

    # Tokenize the sentences
    reference_tokens = word_tokenize(reference.lower())
    candidate_tokens = word_tokenize(candidate.lower())

    # Use smoothing to avoid zero scores when there are no matching n-grams
    smoothie = SmoothingFunction().method1

    # Calculate BLEU using 1-4 gram precision
    try:
        return sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothie,
        )
    except:
        # Fallback to unigram precision if higher n-grams fail
        return sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            weights=(1, 0, 0, 0),
            smoothing_function=smoothie,
        )


def evaluate(examples: List[Dict]) -> Tuple[float, List[Dict]]:
    """
    Evaluate examples using BLEU score
    """
    total_bleu = 0
    details = []

    for ex in tqdm(examples, desc="Evaluating"):
        reference = ex["reference_answer"]
        candidate = ex["generated_answer"]

        # Calculate BLEU score
        bleu_score = calculate_bleu(reference, candidate)
        total_bleu += bleu_score

        # Store detailed results
        detail = {
            "id": ex["id"],
            "question": ex["question"],
            "reference_answer": reference,
            "generated_answer": candidate,
            "bleu_score": bleu_score,
            "prompt": ex.get("prompt", ""),
            "full_response": ex.get("model_full_response", ""),
        }
        details.append(detail)

    avg_bleu = total_bleu / len(examples) if examples else 0
    return avg_bleu, details


def save_results(
    bleu_score: float, details: List[Dict], output_file: str, results_file: str
):
    """
    Save evaluation results to a file
    """
    with open(results_file, "r", encoding="utf-8") as f:
        original_results = json.load(f)

    original_data_map = {}
    for item in original_results:
        if "feta_id" in item:
            original_data_map[item["feta_id"]] = item

    # Calculate additional statistics
    bleu_scores = [detail["bleu_score"] for detail in details]

    statistics = {
        "avg_bleu": bleu_score,
        "min_bleu": min(bleu_scores) if bleu_scores else 0,
        "max_bleu": max(bleu_scores) if bleu_scores else 0,
        "median_bleu": np.median(bleu_scores) if bleu_scores else 0,
        "std_bleu": np.std(bleu_scores) if bleu_scores else 0,
        "total_examples": len(details),
    }

    results = {
        "bleu_score": bleu_score * 100,  # Convert to percentage
        "statistics": statistics,
        "details": details,
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FeTaQA results using BLEU score"
    )
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

    # try:
    #     # Import NLTK components
    #     import nltk
    #     nltk.download('punkt', quiet=True)
    # except ImportError:
    #     print("NLTK not found, installing...")
    #     os.system('pip install nltk')
    #     import nltk
    #     nltk.download('punkt', quiet=True)

    # Process the results file
    examples = process_result_file(args.results_file)
    print(f"Loaded {len(examples)} examples for evaluation")

    # Evaluate
    avg_bleu, details = evaluate(examples)

    # Print summary results
    print("=" * 40)
    print(f"Average BLEU Score: {avg_bleu * 100:.2f}%")
    print("=" * 40)

    # Save detailed results
    save_results(avg_bleu, details, args.output_file, args.results_file)


if __name__ == "__main__":
    main()
