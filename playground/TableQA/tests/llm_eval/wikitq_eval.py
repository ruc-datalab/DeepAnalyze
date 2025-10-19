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

    # Check frequency of Yes/No in text to avoid interference
    yes_count = len(re.findall(r"\byes\b", evaluation_text.lower()))
    no_count = len(re.findall(r"\bno\b", evaluation_text.lower()))

    if yes_count > no_count:
        return True, f"Yes (inferred from {yes_count} occurrences vs {no_count} no's)"
    elif no_count > yes_count:
        return False, f"No (inferred from {no_count} occurrences vs {yes_count} yes's)"

    # When result cannot be determined, default to False
    return False, "Unknown (defaulting to No)"


def extract_predicted_answer(model_answer):
    """Extract predicted answer from model's response"""
    if not model_answer:
        return None

    # Try to match content wrapped in <answer> tags
    answer_tag_pattern = r"<answer>(.*?)</answer>"
    answer_tag_match = re.search(answer_tag_pattern, model_answer, re.DOTALL)
    if answer_tag_match:
        model_answer = answer_tag_match.group(1).strip()

    # # Try to match content after "Answer:" or similar
    # match = re.search(r'Answer:\s*(.+?)(?:\n|$|\.|")', model_answer, re.IGNORECASE)
    # if match:
    #     return match.group(1).strip()

    # If no specific answer format is found, return the whole response
    return model_answer.strip() if model_answer else None


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


def evaluate_with_llm(
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
    Use LLM to evaluate WikiTableQuestions, supporting both local VLLM and remote API calls
    """
    if log_file is None:
        # Still keep auto-generation logic but ensure directory exists
        dir_name = os.path.dirname(output_file)
        os.makedirs(dir_name, exist_ok=True)
        log_file = os.path.join(
            dir_name, f"llm_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    else:
        # If log file path is provided, ensure parent directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = setup_logger(log_file)
    logger.info(f"Starting LLM-based evaluation using model: {model_path}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")

    # Record start time
    start_time = time.time()

    # Determine whether to use local VLLM or API calls
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

    # Load data
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        logger.info(f"Loaded {len(data)} samples for evaluation")
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        return None

    # Prepare items for evaluation - code remains unchanged
    evaluation_items = []
    for i, item in enumerate(data):
        # Find expected answer
        expected_answer = None
        if "answer" in item:
            expected_answer = item["answer"]
        elif "truth_answer" in item:
            expected_answer = item["truth_answer"]
        elif "true_answer" in item:
            expected_answer = item["true_answer"]
        elif "expected_answer" in item:
            expected_answer = item["expected_answer"]

        # If answer still not found, use default value
        if expected_answer is None:
            expected_answer = "Unknown"
            logger.warning(
                f"Item {item.get('id', f'item-{i}')} missing golden answer, using default"
            )

        # First check if extracted answer already exists
        if "extracted_answer" in item:
            predicted_answer = item["extracted_answer"]
        else:
            # Use enhanced extraction function
            model_answer = item.get("model_answer", "")
            predicted_answer = extract_predicted_answer(model_answer)

        # Continue evaluation even if answer is empty
        evaluation_items.append(
            {
                "id": item.get("id", f"item-{i}"),
                "question": item.get("question", ""),
                "expected_answer": expected_answer or "Unknown",
                "predicted_answer": predicted_answer or "No answer",
                "full_response": item.get("model_answer", ""),
            }
        )

    logger.info(f"Prepared {len(evaluation_items)} items for LLM evaluation")

    # For API evaluator, batch_size means concurrent request count
    if use_api:
        logger.info(f"Using API evaluator with concurrency={batch_size}")
    else:
        logger.info(f"Using VLLM evaluator with batch_size={batch_size}")

    evaluation_results = evaluator.batch_evaluate(
        evaluation_items, batch_size=batch_size
    )

    # Post-processing and statistics remain unchanged...
    total_evaluated = len(evaluation_results)
    correct_count = sum(1 for result in evaluation_results if result["is_correct"])
    accuracy = correct_count / total_evaluated if total_evaluated > 0 else 0

    stats = {
        "total_samples": len(data),
        "evaluated_samples": total_evaluated,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "model_used": model_path,
        "evaluation_time": time.time() - start_time,
    }

    final_result = {"summary": stats, "results": evaluation_results}

    # Save results
    logger.info(f"Evaluation complete. Saving results to {output_file}...")

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=4, ensure_ascii=False)
        logger.info(f"Evaluation results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")

    # Output summary
    if verbose:
        logger.info("\n===== LLM Evaluation Summary =====")
        logger.info(f"Total samples: {stats['total_samples']}")
        logger.info(f"Evaluated samples: {stats['evaluated_samples']}")
        logger.info(f"Correct answers: {stats['correct_count']}")
        logger.info(f"Accuracy: {stats['accuracy'] * 100:.2f}%")
        logger.info(f"Total evaluation time: {stats['evaluation_time']:.2f} seconds")
        logger.info(
            f"Evaluation model: {model_path} (via {'API' if use_api else 'VLLM'})"
        )

    return final_result


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="LLM-based WikiTQ prediction evaluation"
    )

    parser.add_argument("--results_file", help="Input file path with predictions")
    parser.add_argument("--output_file", help="Output file path for evaluation results")
    parser.add_argument("--model_path", help="Path to the LLM model for evaluation")
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
        "--base_path",
        type=str,
        default=None,
        help="Base path for the project (optional)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=10,
        help="Maximum retry attempts for API calls (default: 5)",
    )

    return parser.parse_args()


def main():
    """Main function that handles command line arguments"""
    args = parse_arguments()

    # Process base_path if provided
    if args.base_path:
        if not os.path.exists(args.base_path):
            print(f"Warning: Provided base_path {args.base_path} does not exist")

    # Run LLM-based evaluation
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

    if result is None:
        print("Evaluation failed")
        exit(1)


if __name__ == "__main__":
    main()
