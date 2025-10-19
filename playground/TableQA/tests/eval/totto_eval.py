import json
import re
import argparse
import os
import numpy as np
from typing import Dict, List, Tuple, Union, Any
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def process_result_file(results_file: str) -> List[Dict]:
    """
    Process results file and return a list of examples
    Handles the case where each example has multiple reference sentences
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
        # Skip items without necessary fields
        if "reference_sentences" not in item or "generated_sentence" not in item:
            continue

        # Handle case where reference_sentences is not a list
        reference_sentences = item.get("reference_sentences", [])
        if not isinstance(reference_sentences, list):
            reference_sentences = [reference_sentences]

        examples.append(
            {
                "id": item.get("example_id", ""),
                "question": item.get("prompt", ""),
                "reference_sentences": reference_sentences,
                "generated_sentence": item.get("generated_sentence", ""),
                "prompt": item.get("prompt", ""),
                "model_full_response": item.get("model_full_response", ""),
            }
        )

    return examples


def calculate_bleu(reference: Union[str, List[str]], candidate: str) -> float:
    """
    Calculate BLEU score between reference and candidate strings
    Reference can be a single string or a list of strings
    """
    if not reference or not candidate:
        return 0.0

    # If reference is a string, convert to list with one element
    if isinstance(reference, str):
        reference = [reference]

    # Tokenize the candidate sentence
    candidate_tokens = word_tokenize(candidate.lower())

    # Tokenize all reference sentences
    reference_tokens_list = [word_tokenize(ref.lower()) for ref in reference]

    # Use smoothing to avoid zero scores when there are no matching n-grams
    smoothie = SmoothingFunction().method1

    # Calculate BLEU scores for each reference
    bleu_scores = []
    for ref_tokens in reference_tokens_list:
        try:
            # Calculate BLEU using 1-4 gram precision
            score = sentence_bleu(
                [ref_tokens],
                candidate_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothie,
            )
            bleu_scores.append(score)
        except:
            # Fallback to unigram precision if higher n-grams fail
            try:
                score = sentence_bleu(
                    [ref_tokens],
                    candidate_tokens,
                    weights=(1, 0, 0, 0),
                    smoothing_function=smoothie,
                )
                bleu_scores.append(score)
            except:
                # If all fails, use 0
                bleu_scores.append(0)

    # Return the maximum BLEU score among all references
    return max(bleu_scores) if bleu_scores else 0


def evaluate(examples: List[Dict]) -> Tuple[float, List[Dict]]:
    """
    Evaluate examples using BLEU score
    For each example, take the maximum BLEU score against all reference sentences
    """
    total_bleu = 0
    details = []

    for ex in tqdm(examples, desc="Evaluating"):
        reference_sentences = ex["reference_sentences"]
        candidate = ex["generated_sentence"]

        # Calculate BLEU score against all references and take the maximum
        bleu_score = calculate_bleu(reference_sentences, candidate)
        total_bleu += bleu_score

        # Calculate individual BLEU scores for each reference
        individual_scores = []
        for i, ref in enumerate(reference_sentences):
            individual_score = calculate_bleu(ref, candidate)
            individual_scores.append(individual_score)

        # Store detailed results
        detail = {
            "id": ex["id"],
            "question": ex["question"],
            "reference_sentences": reference_sentences,
            "generated_sentence": candidate,
            "max_bleu_score": bleu_score,
            "individual_bleu_scores": individual_scores,
            "prompt": ex.get("prompt", ""),
            "full_response": ex.get("model_full_response", ""),
        }
        details.append(detail)

    avg_bleu = total_bleu / len(examples) if examples else 0
    return avg_bleu, details


def save_results(bleu_score: float, details: List[Dict], output_file: str):
    """
    Save evaluation results to a file
    """
    # Calculate additional statistics
    bleu_scores = [detail["max_bleu_score"] for detail in details]

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
        description="Evaluate ToTTo results using BLEU score"
    )
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
    save_results(avg_bleu, details, args.output_file)


if __name__ == "__main__":
    main()
