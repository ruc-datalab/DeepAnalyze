import json
import re
import os
import time
import logging
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from prompt import WIKITQ_EVAL

import os.path

# Get current file directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get tests directory by going up one level
project_root = os.path.dirname(current_dir)

sys.path.append(project_root)
sys.path.append(project_root)
from utils.llm import call_api_with_retry, initialize_client
from eval.wikitq_eval import (
    evaluate_answers,
    normalize_answer,
    exact_match_enhanced,
    extract_predicted_answer,
)


# Setup logging
def setup_logger(log_file):
    """Set up the logger"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("wikitq_evaluator")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class BaseEvaluator:
    """Base evaluator class that defines shared methods"""

    def __init__(self, model_path, logger=None):
        self.model_path = model_path
        self.logger = logger or logging.getLogger("wikitq_evaluator")

    def create_evaluation_prompt(self, question, reference_answer, predicted_answer):
        """
        Create evaluation prompt
        """
        prompt = WIKITQ_EVAL.format(
            question=question,
            candidate_answer=predicted_answer,
            correct_answer=reference_answer,
        )
        return prompt


class VLLMEvaluator(BaseEvaluator):
    """
    Use VLLM for local model evaluation
    """

    def __init__(
        self, model_path, max_model_len=8192, tensor_parallel_size=1, logger=None
    ):
        super().__init__(model_path, logger)

        # Default EOS tokens list
        self.EOS = ["<|im_end|>", "</s>"]

        self.logger.info(f"Initializing VLLM evaluator with model: {model_path}")

        try:
            self.model = LLM(
                model=model_path,
                max_model_len=max_model_len,
                trust_remote_code=True,
                distributed_executor_backend="ray",
                tensor_parallel_size=tensor_parallel_size,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.logger.info("VLLM evaluator initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize VLLM: {e}")
            raise

    def batch_evaluate(self, evaluation_items, batch_size=16):
        """
        Use VLLM to evaluate samples in batches
        """
        all_results = []

        # Create batches
        for i in range(0, len(evaluation_items), batch_size):
            batch = evaluation_items[i : i + batch_size]

            # Create prompts for this batch
            prompts = []
            for item in batch:
                prompt = self.create_evaluation_prompt(
                    item["question"], item["expected_answer"], item["predicted_answer"]
                )
                prompts.append(prompt)

            # Convert prompts to chat format
            chat_prompts = []
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                chat_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                chat_prompts.append(chat_prompt)

            # Generate evaluations in batch
            try:
                batch_start_time = time.time()
                self.logger.info(f"Evaluating batch of {len(batch)} items using VLLM")

                responses = self.model.generate(
                    prompts=chat_prompts,
                    sampling_params=SamplingParams(
                        max_tokens=1024,
                        temperature=0.0,
                        top_p=1.0,
                        stop=self.EOS,
                    ),
                    use_tqdm=True,
                )

                # Process responses
                for j, (item, response) in enumerate(zip(batch, responses)):
                    evaluation_text = response.outputs[0].text.strip()

                    is_correct, answer_text = extract_evaluation_result(evaluation_text)

                    result = {
                        "id": item.get("id", f"item-{i+j}"),
                        "question": item["question"],
                        "expected_answer": item["expected_answer"],
                        "predicted_answer": item["predicted_answer"],
                        "is_correct": is_correct,
                        "explanation": evaluation_text,
                    }
                    all_results.append(result)

                    # Add detailed logs for each sample
                    self.logger.info(f"Item {i+j+1}: ID={result['id']}")
                    self.logger.info(f"  Question: {result['question']}")
                    self.logger.info(f"  Expected: {result['expected_answer']}")
                    self.logger.info(f"  Predicted: {result['predicted_answer']}")
                    self.logger.info(f"  LLM Decision: {answer_text}")
                    self.logger.info(
                        f"  Marked as: {'Correct' if is_correct else 'Incorrect'}"
                    )
                    self.logger.info("  " + "-" * 50)

                batch_time = time.time() - batch_start_time
                self.logger.info(
                    f"Batch evaluated in {batch_time:.2f}s ({batch_time/len(batch):.2f}s per item)"
                )

            except Exception as e:
                self.logger.error(f"Error in batch evaluation: {e}")

                # Create empty results for failed items
                for item in batch:
                    result = {
                        "id": item.get("id", "unknown"),
                        "question": item["question"],
                        "expected_answer": item["expected_answer"],
                        "predicted_answer": item["predicted_answer"],
                        "is_correct": False,
                        "explanation": f"Evaluation failed: {str(e)}",
                    }
                    all_results.append(result)

        return all_results


class APIEvaluator(BaseEvaluator):
    """
    Use API calls to evaluate with closed-source models, supporting parallel processing
    """

    def __init__(self, model_path, logger=None, max_retries=5):
        """
        Initialize API evaluator

        Args:
            model_path: API model path or name
            logger: Logger
            max_retries: Maximum number of retry attempts
        """
        super().__init__(model_path, logger)
        self.max_retries = max_retries
        self.logger.info(
            f"Initializing API evaluator for model: {model_path} with max_retries={max_retries}"
        )

        try:
            self.client_info = initialize_client({"model_path": model_path})
            self.logger.info(f"API client initialized for {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize API client: {e}")
            raise

    def evaluate_single_item(self, item, item_index):
        """
        Evaluate a single sample for parallel processing

        Args:
            item: Sample to evaluate
            item_index: Sample index

        Returns:
            Evaluation result dictionary
        """
        item_start_time = time.time()

        try:
            prompt = self.create_evaluation_prompt(
                item["question"], item["expected_answer"], item["predicted_answer"]
            )

            # Prepare API call
            messages = [{"role": "user", "content": prompt}]

            # Call API using the class-defined maximum retry count
            success, response = call_api_with_retry(
                self.client_info,
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
                top_p=1.0,
                max_retries=self.max_retries,  # Use instance variable to control max retries
            )

            if success:
                # Process response based on model type
                if self.client_info["model_type"] == "openai":
                    evaluation_text = response.choices[0].message.content
                else:
                    # Closed-source models directly return text content
                    evaluation_text = response

                is_correct, answer_text = extract_evaluation_result(evaluation_text)

                result = {
                    "id": item.get("id", f"item-{item_index}"),
                    "question": item["question"],
                    "expected_answer": item["expected_answer"],
                    "predicted_answer": item["predicted_answer"],
                    "is_correct": is_correct,
                    "explanation": evaluation_text,
                    "answer_text": answer_text,  # Save for logging
                    "success": True,
                    "processing_time": time.time() - item_start_time,
                }
            else:
                # API call failed
                self.logger.error(
                    f"API call failed for item {item.get('id', f'item-{item_index}')} after {self.max_retries} retries: {response}"
                )
                result = {
                    "id": item.get("id", f"item-{item_index}"),
                    "question": item["question"],
                    "expected_answer": item["expected_answer"],
                    "predicted_answer": item["predicted_answer"],
                    "is_correct": False,
                    "explanation": f"API call failed after {self.max_retries} retries: {response}",
                    "success": False,
                    "processing_time": time.time() - item_start_time,
                }

        except Exception as e:
            self.logger.error(f"Error evaluating item {item_index}: {e}")
            result = {
                "id": item.get("id", f"item-{item_index}"),
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "predicted_answer": item["predicted_answer"],
                "is_correct": False,
                "explanation": f"Evaluation error: {str(e)}",
                "success": False,
                "processing_time": time.time() - item_start_time,
            }

        return result

    def batch_evaluate(self, evaluation_items, batch_size=16):
        """
        Use parallel API calls to evaluate samples in batches

        Args:
            evaluation_items: List of samples to evaluate
            batch_size: Concurrent processing batch size

        Returns:
            List of evaluation results
        """
        import concurrent.futures

        all_results = []
        total_items = len(evaluation_items)

        self.logger.info(
            f"Starting parallel API evaluation of {total_items} items with concurrency={batch_size}"
        )

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Create future task list
            future_to_item = {
                executor.submit(self.evaluate_single_item, item, i): (i, item)
                for i, item in enumerate(evaluation_items)
            }

            # Process completed tasks
            progress_counter = 0
            batch_start_time = time.time()

            for future in concurrent.futures.as_completed(future_to_item):
                i, item = future_to_item[future]
                progress_counter += 1

                try:
                    result = future.result()
                    all_results.append(result)

                    # Log evaluation results
                    self.logger.info(
                        f"Item {progress_counter}/{total_items}: ID={result['id']}"
                    )
                    self.logger.info(f"  Question: {result['question']}")
                    self.logger.info(f"  Expected: {result['expected_answer']}")
                    self.logger.info(f"  Predicted: {result['predicted_answer']}")

                    if result["success"]:
                        self.logger.info(
                            f"  LLM Decision: {result.get('answer_text', 'Unknown')}"
                        )
                    else:
                        self.logger.info(f"  LLM Decision: Failed")

                    self.logger.info(
                        f"  Marked as: {'Correct' if result['is_correct'] else 'Incorrect'}"
                    )
                    self.logger.info(
                        f"  Processing time: {result['processing_time']:.2f}s"
                    )
                    self.logger.info("  " + "-" * 50)

                    # Output progress every 10 samples or at the last sample
                    if progress_counter % 10 == 0 or progress_counter == total_items:
                        elapsed_time = time.time() - batch_start_time
                        self.logger.info(
                            f"Progress: {progress_counter}/{total_items} items processed, "
                            f"elapsed time: {elapsed_time:.2f}s, "
                            f"average: {elapsed_time/progress_counter:.2f}s per item"
                        )

                except Exception as e:
                    self.logger.error(
                        f"Unexpected error processing result for item {i}: {e}"
                    )

        elapsed_time = time.time() - batch_start_time
        self.logger.info(
            f"Completed API evaluation of {total_items} items in {elapsed_time:.2f}s "
            f"({elapsed_time/total_items:.2f}s per item)"
        )

        # Sort results by original order
        all_results.sort(
            key=lambda x: int(x["id"].split("-")[-1]) if "-" in x["id"] else 0
        )

        return all_results


def extract_evaluation_result(evaluation_text):
    """
    Extract Yes/No results from LLM evaluation text

    Args:
        evaluation_text: Evaluation text generated by LLM

    Returns:
        bool: Extracted result, True means correct answer, False means wrong answer
        str: Original answer text (for logging)
    """
    yes_match = re.search(r"\bYes\b", evaluation_text)
    no_match = re.search(r"\bNo\b", evaluation_text)

    # If we find a clear Yes, return True
    if yes_match and not no_match:
        return True, "Yes"

    # If we find a clear No, return False
    if no_match and not yes_match:
        return False, "No"

    # If both Yes and No or neither, check the last line or last non-empty string
    lines = [line.strip() for line in evaluation_text.split("\n") if line.strip()]
    if lines:
        last_line = lines[-1].strip()
        if last_line.lower() == "yes":
            return True, "Yes (from last line)"
        elif last_line.lower() == "no":
            return False, "No (from last line)"

    # Check frequency of "Yes" or "No" in the text to avoid interference
    yes_count = len(re.findall(r"\byes\b", evaluation_text.lower()))
    no_count = len(re.findall(r"\bno\b", evaluation_text.lower()))

    if yes_count > no_count:
        return True, f"Yes (inferred from {yes_count} occurrences vs {no_count} no's)"
    elif no_count > yes_count:
        return False, f"No (inferred from {no_count} occurrences vs {yes_count} yes's)"

    # When result cannot be determined, default to False
    return False, "Unknown (defaulting to No)"


def is_api_model(model_path):
    """
    Determine if a model needs to be called through API
    """
    # Check if model path is a local path or API model name
    api_model_prefixes = ["gemini", "claude", "claude-3-7-sonnet", "deepseek-r1"]

    # If it's a local path (contains slash or backslash), use VLLM
    if "/" in model_path or "\\" in model_path:
        return False

    # Check if it starts with known API model prefixes
    for prefix in api_model_prefixes:
        if model_path.startswith(prefix):
            return True

    # Default to VLLM
    return False


def combined_evaluate(
    input_file,
    output_file,
    model_path,
    log_file=None,
    batch_size=8,
    tensor_parallel_size=1,
    max_model_len=8192,
    max_retries=5,
    verbose=True,
):
    """
    Combined evaluation method using exact match and LLM evaluation:
    1. First evaluate all samples using exact match method
    2. For samples that fail exact match, use LLM for secondary evaluation
    3. Merge results from both evaluation methods
    """
    if log_file is None:
        dir_name = os.path.dirname(output_file)
        os.makedirs(dir_name, exist_ok=True)
        log_file = os.path.join(
            dir_name, f"combined_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

    logger = setup_logger(log_file)
    logger.info(
        f"Starting combined evaluation (exact match + LLM) using model: {model_path}"
    )
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")

    # Record start time
    start_time = time.time()

    # Step 1: Use exact match method for evaluation
    logger.info("Step 1: Performing exact match evaluation...")

    # First create temporary output file path for exact match evaluation
    temp_exact_match_file = os.path.join(
        os.path.dirname(output_file),
        f"temp_exact_match_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    exact_match_results = evaluate_answers(
        input_file, temp_exact_match_file, verbose=False
    )

    if exact_match_results is None:
        logger.error("Exact match evaluation failed")
        return None

    # Collect statistics from exact match evaluation results
    em_stats = exact_match_results["summary"]
    em_results_list = exact_match_results["results"]

    # Create index for quick lookup
    em_results_by_id = {item["id"]: item for item in em_results_list}

    logger.info(
        f"Exact match evaluation completed: {em_stats['exact_matches']} / {em_stats['answered_samples']} correct"
    )

    # Step 2: Filter samples that failed exact match evaluation for LLM evaluation
    items_for_llm = []
    for item in em_results_list:
        if item.get("is_exact_match") is False:  # Exact match judged as incorrect
            items_for_llm.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "expected_answer": item["expected_answer"],
                    "predicted_answer": item["predicted_answer"],
                }
            )

    logger.info(
        f"Step 2: Selected {len(items_for_llm)} samples that failed exact match for LLM evaluation"
    )

    # If no samples need LLM evaluation, directly return exact match results
    if not items_for_llm:
        logger.info("No samples require LLM evaluation, using exact match results only")

        # Write exact match results directly to the specified output file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(exact_match_results, f, indent=4, ensure_ascii=False)

        # Delete temporary files
        try:
            os.remove(temp_exact_match_file)
        except:
            pass

        if verbose:
            logger.info("\n===== Combined Evaluation Summary =====")
            logger.info(f"Total samples: {em_stats['total_samples']}")
            logger.info(f"Exact match correct: {em_stats['exact_matches']}")
            logger.info(f"Exact match rate: {em_stats['exact_match_rate'] * 100:.2f}%")
        return exact_match_results

    # Step 3: Save samples needing LLM evaluation to temporary file
    temp_file = os.path.join(
        os.path.dirname(output_file),
        f"temp_for_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(items_for_llm, f, indent=4, ensure_ascii=False)

    # Step 4: Use LLM to evaluate these samples
    logger.info(
        "Step 3: Performing LLM evaluation on samples that failed exact match..."
    )
    llm_output_file = os.path.join(
        os.path.dirname(output_file),
        f"llm_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    # Determine if using API model
    use_api = is_api_model(model_path)

    # Initialize appropriate evaluator
    try:
        if use_api:
            logger.info(
                f"Using API evaluator for model: {model_path} with max_retries={max_retries}"
            )
            evaluator = APIEvaluator(
                model_path=model_path, logger=logger, max_retries=max_retries
            )
        else:
            logger.info(f"Using local VLLM evaluator for model: {model_path}")
            evaluator = VLLMEvaluator(
                model_path=model_path,
                max_model_len=max_model_len,
                tensor_parallel_size=tensor_parallel_size,
                logger=logger,
            )
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        return None

    # Perform LLM evaluation
    logger.info(f"Evaluating {len(items_for_llm)} items with LLM")
    llm_evaluation_results = evaluator.batch_evaluate(
        items_for_llm, batch_size=batch_size
    )

    # Step 5: Merge evaluation results
    logger.info("Step 4: Merging exact match and LLM evaluation results...")

    # Organize LLM evaluation results by ID for easy lookup
    llm_results_by_id = {item["id"]: item for item in llm_evaluation_results}

    # Create final combined results list
    combined_results_list = []

    # Statistics
    combined_correct = 0
    total_evaluated = 0

    # Process each sample
    for item in em_results_list:
        item_id = item["id"]
        combined_item = dict(item)  # Copy original item

        # Check if this sample has exact match result
        if item.get("is_exact_match") is True:
            # Exact match is correct, no need for LLM evaluation
            combined_item["evaluation_method"] = "exact_match"
            combined_item["is_correct"] = True
            combined_correct += 1
        elif item_id in llm_results_by_id:
            # Exact match is incorrect, use LLM evaluation result
            llm_result = llm_results_by_id[item_id]
            combined_item["evaluation_method"] = "llm"
            combined_item["is_correct"] = llm_result["is_correct"]
            combined_item["llm_explanation"] = llm_result["explanation"]
            if llm_result["is_correct"]:
                combined_correct += 1
        else:
            # This sample has no LLM evaluation result (possibly due to missing expected answer)
            combined_item["evaluation_method"] = "exact_match"
            combined_item["is_correct"] = False

        total_evaluated += 1
        combined_results_list.append(combined_item)

    # Calculate accuracy
    accuracy = combined_correct / total_evaluated if total_evaluated > 0 else 0

    # Create final result
    combined_stats = {
        "total_samples": em_stats["total_samples"],
        "evaluated_samples": total_evaluated,
        "exact_match_correct": em_stats["exact_matches"],
        "llm_additional_correct": combined_correct - em_stats["exact_matches"],
        "total_correct": combined_correct,
        "accuracy": accuracy,
        "evaluation_model": model_path,
        "evaluation_time": time.time() - start_time,
    }

    final_result = {"summary": combined_stats, "results": combined_results_list}

    # Save results
    logger.info(f"Saving combined evaluation results to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)

    # Clean up temporary files
    try:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if os.path.exists(llm_output_file):
            os.remove(llm_output_file)
        if os.path.exists(temp_exact_match_file):
            os.remove(temp_exact_match_file)
    except Exception as e:
        logger.warning(f"Could not clean up temporary files: {e}")

    # Output summary
    if verbose:
        logger.info("\n===== Combined Evaluation Summary =====")
        logger.info(f"Total samples: {combined_stats['total_samples']}")
        logger.info(f"Exact match correct: {combined_stats['exact_match_correct']}")
        logger.info(
            f"Additional LLM correct: {combined_stats['llm_additional_correct']}"
        )
        logger.info(f"Total correct: {combined_stats['total_correct']}")
        logger.info(f"Final accuracy: {combined_stats['accuracy'] * 100:.2f}%")
        logger.info(
            f"Total evaluation time: {combined_stats['evaluation_time']:.2f} seconds"
        )

    return final_result


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Combined exact match + LLM evaluation for WikiTQ"
    )

    parser.add_argument(
        "--results_file", required=True, help="Input file path with predictions"
    )
    parser.add_argument(
        "--output_file", required=True, help="Output file path for evaluation results"
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to the LLM model for evaluation"
    )
    parser.add_argument(
        "--log_file", type=str, default=None, help="Path for log file (optional)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for LLM evaluation"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1, help="Tensor parallelism size"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Maximum retry attempts for API calls",
    )
    parser.add_argument(
        "--evaluation_mode",
        type=str,
        default="combined",
        choices=["exact", "llm", "combined"],
        help="Evaluation mode: exact match only, LLM only, or combined",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default=None,
        help="Base path for the project (optional)",
    )

    return parser.parse_args()


def main():
    """Main function that handles command line arguments"""
    args = parse_arguments()

    # Process base_path if provided
    if args.base_path:
        if not os.path.exists(args.base_path):
            print(f"Warning: Provided base_path {args.base_path} does not exist")

    # Choose evaluation method based on evaluation mode
    if args.evaluation_mode == "exact":
        # Use exact match evaluation only
        result = evaluate_answers(
            input_file=args.results_file, output_file=args.output_file, verbose=True
        )
    elif args.evaluation_mode == "llm":
        # Use LLM evaluation only
        from wikitq_eval import evaluate_with_llm

        result = evaluate_with_llm(
            input_file=args.results_file,
            output_file=args.output_file,
            model_path=args.model_path,
            log_file=args.log_file,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            max_retries=args.max_retries,
            verbose=True,
        )
    else:  # combined (default)
        # Use combined evaluation
        result = combined_evaluate(
            input_file=args.results_file,
            output_file=args.output_file,
            model_path=args.model_path,
            log_file=args.log_file,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            max_retries=args.max_retries,
            verbose=True,
        )

    if result is None:
        print("Evaluation failed")
        exit(1)


if __name__ == "__main__":
    main()
