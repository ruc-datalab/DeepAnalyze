import json
import os
import logging
import time
import re
import sys
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from prompt import COT_PROMPT_FETAQA_TEMPLATE


# Setup logging
def setup_logger(log_file):
    """Set up the logger"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("fetaqa_processor")
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class VLLMGenerator:
    """
    A class for generating text using vLLM with support for different models.
    """

    def __init__(self, model_path, max_model_len=16384, tensor_parallel_size=1):
        """
        Initialize the VLLMGenerator with model and tokenizer.
        """
        # Default EOS tokens list - can be overridden based on model
        self.EOS = ["<|im_end|>", "</s>"]

        self.model = LLM(
            model=model_path,
            max_model_len=max_model_len,
            trust_remote_code=True,
            distributed_executor_backend="mp",
            tensor_parallel_size=tensor_parallel_size,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def generate(
        self, prompts, max_new_tokens=2048, temperature=0.0, top_p=1.0, verbose=False
    ):
        try:
            # Apply chat template to prompts
            chat_prompts = []
            for prompt in prompts:
                # Convert to chat format with a single user message
                messages = [{"role": "user", "content": prompt}]
                chat_prompts.append(
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )

            if verbose and len(chat_prompts) > 0:
                print("Example chat prompt:")
                print(chat_prompts[0])

            # Batch generation with vLLM
            vllm_outputs = self.model.generate(
                prompts=chat_prompts,
                sampling_params=SamplingParams(
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=self.EOS,
                ),
                use_tqdm=True,
            )

            # Process generated outputs
            raw_generations = [x.outputs[0].text for x in vllm_outputs]
            return raw_generations

        except Exception as e:
            print(f"Error in vLLM generation: {str(e)}")
            raise


def format_table_with_highlights(table_data, highlighted_cell_ids):
    """Format table data as a string, marking highlighted cells"""
    if not table_data:
        return ""

    result = ""
    for row_idx, row in enumerate(table_data):
        row_str = []
        for col_idx, cell in enumerate(row):
            # Check if this is a highlighted cell
            is_highlighted = [row_idx, col_idx] in highlighted_cell_ids
            if is_highlighted:
                row_str.append(f"<hl>{cell}</hl>")
            else:
                row_str.append(str(cell))

        result += " | ".join(row_str) + "\n"

    return result.strip()


def create_prompt_from_fetaqa(item: Dict[str, Any]) -> str:
    """Create prompt for FeTaQA item"""
    # Get table, context, question and highlighted cells
    table_data = item.get("table_array", [])
    highlighted_cell_ids = item.get("highlighted_cell_ids", [])
    page_title = item.get("table_page_title", "")
    section_title = item.get("table_section_title", "")
    question = item.get("question", "")

    # Format table with highlighted cells
    table_str = format_table_with_highlights(table_data, highlighted_cell_ids)

    # Create prompt
    prompt = COT_PROMPT_FETAQA_TEMPLATE.format(
        table=table_str,
        page_title=page_title,
        section_title=section_title,
        question=question,
    )

    return prompt


def extract_answer_from_response(model_answer):
    """Extract the final answer from the model response, supporting multiple formats"""
    if not model_answer:
        return ""

    # Try to extract answer from <answer> tags
    answer_tag_pattern = re.search(r"<Answer>(.*?)</Answer>", model_answer, re.DOTALL)
    if answer_tag_pattern:
        answer_content = answer_tag_pattern.group(1).strip()

        # Check if there's an "Answer:" prefix inside the tags
        if "Answer:" in answer_content:
            return answer_content.split("Answer:", 1)[1].strip()
        return answer_content

    # Try to match "Answer: xxx" pattern, capturing the entire sentence
    # This pattern specifically looks for "Answer:" and captures everything after it
    answer_pattern = re.search(
        r"Answer:\s*(.*?)(?:\Z)", model_answer, re.IGNORECASE | re.DOTALL
    )
    if answer_pattern:
        complete_answer = answer_pattern.group(1).strip()
        # Remove any trailing code blocks or irrelevant content
        if "```" in complete_answer:
            complete_answer = complete_answer.split("```")[0].strip()
        return complete_answer

    # If no explicit marker, try to use the last paragraph as the answer
    paragraphs = model_answer.strip().split("\n\n")
    for paragraph in reversed(paragraphs):
        cleaned_paragraph = paragraph.strip()
        if (
            cleaned_paragraph
            and not cleaned_paragraph.startswith("```")
            and not cleaned_paragraph.endswith("```")
        ):
            return cleaned_paragraph

    # If all extraction methods fail, return the original response
    return model_answer


def process_fetaqa_data_batch(
    input_file,
    output_file,
    model_path,
    log_file,
    max_tokens=2048,
    temperature=0.0,
    tensor_parallel_size=1,
    batch_size=16,
    start_from=0,
):
    """Process FeTaQA dataset with batched VLLM inference"""
    logger = setup_logger(log_file)

    # Record start time
    start_time = time.time()
    logger.info(
        f"Started processing FeTaQA data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Using model: {model_path}")
    logger.info(f"Batch size: {batch_size}")

    # Initialize VLLM generator
    try:
        generator = VLLMGenerator(
            model_path=model_path,
            max_model_len=16384,
            tensor_parallel_size=tensor_parallel_size,
        )
        logger.info(f"VLLM generator initialized successfully")
    except Exception as e:
        logger.error(f"VLLM initialization failed: {e}")
        return

    # Read data items
    data_items = []
    try:
        # Check file extension to determine format
        if input_file.endswith(".jsonl"):
            # JSONL format - read line by line
            with open(input_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        item = json.loads(line.strip())
                        data_items.append(item)
            logger.info(f"Loaded {len(data_items)} examples from JSONL file")
        else:
            # Standard JSON format
            with open(input_file, "r", encoding="utf-8") as f:
                data_items = json.load(f)
            logger.info(f"Loaded {len(data_items)} examples from JSON file")
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return

    # Check if intermediate results exist
    results = []
    processed_ids = set()
    if os.path.exists(f"{output_file}.temp"):
        try:
            with open(f"{output_file}.temp", "r", encoding="utf-8") as f:
                results = json.load(f)

            # Get IDs of already processed items
            for result in results:
                processed_ids.add(result.get("feta_id", ""))

            logger.info(f"Loaded intermediate results with {len(results)} records")
            logger.info(f"Found {len(processed_ids)} already processed items")
        except Exception as e:
            logger.error(
                f"Failed to load intermediate results: {e}, starting from beginning"
            )
            results = []
            processed_ids = set()

    # Filter out already processed items
    remaining_items = []
    for item in data_items[start_from:]:
        item_id = item.get("feta_id", "")
        if item_id not in processed_ids:
            remaining_items.append(item)

    logger.info(f"Remaining items to process: {len(remaining_items)}/{len(data_items)}")

    success_count = len(results)
    error_count = 0

    # Process data in batches
    for batch_start in range(0, len(remaining_items), batch_size):
        batch_end = min(batch_start + batch_size, len(remaining_items))
        current_batch = remaining_items[batch_start:batch_end]

        # Create prompts for current batch
        prompts = []
        for item in current_batch:
            prompt = create_prompt_from_fetaqa(item)
            prompts.append(prompt)

        # Generate responses for the batch
        try:
            responses = generator.generate(
                prompts=prompts,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
                verbose=(batch_start == 0),  # Only show example on first batch
            )

            # Process each response
            for i, (item, response) in enumerate(zip(current_batch, responses)):
                item_start_time = time.time()

                item_id = item.get("feta_id", f"item-{i+batch_start}")
                question = item.get("question", "")
                expected_answer = item.get("answer", "")

                global_item_index = batch_start + i + 1
                logger.info(
                    f"Processing example {global_item_index}/{len(remaining_items)}... [ID: {item_id}]"
                )

                # Extract final answer
                generated_answer = extract_answer_from_response(response)

                # 计算处理时间
                item_time = time.time() - item_start_time

                # Log detailed information
                logger.info(f"Question: {question}")
                logger.info(f"Reference answer: {expected_answer}")
                logger.info(f"Generated answer: {generated_answer}")
                logger.info(f"Processing time: {item_time:.6f} seconds")
                logger.info("-" * 50)

                # Build result object
                result = {
                    "feta_id": item_id,
                    "question": question,
                    "prompt": prompts[i],
                    "reference_answer": expected_answer,
                    "model_full_response": response,
                    "generated_answer": generated_answer,
                    "processing_time": item_time,
                }

                results.append(result)
                success_count += 1

            with open(f"{output_file}.temp", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            error_count += 1
            logger.error(f"Error processing batch: {e}")
            # Save what we have so far
            with open(f"{output_file}.temp", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # Save final results to JSON file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results file: {e}")

    # Log summary information
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Processing completed! Total time: {total_time:.2f} seconds")
    logger.info(f"Successfully processed: {success_count}/{len(data_items)}")
    logger.info(f"Processing failures: {error_count}/{len(data_items)}")
    average_time_per_item = (
        total_time / len(remaining_items) if len(remaining_items) > 0 else 0
    )
    logger.info(
        f"Average processing time per item: {average_time_per_item:.2f} seconds"
    )
    logger.info("=" * 60)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process FeTaQA dataset with VLLM batch inference"
    )

    parser.add_argument("--output_file", type=str, help="Path to save results")
    parser.add_argument("--model_path", type=str, help="Model path or identifier")
    parser.add_argument("--log_file", type=str, help="Path to log file")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for model generation",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4096, help="Maximum tokens for model output"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=2, help="Tensor parallelism size"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for inference"
    )
    parser.add_argument(
        "--start_from", type=int, default=0, help="Start processing from this index"
    )
    parser.add_argument("--base_path", type=str, help="Base path for the project")

    return parser.parse_args()


def main():
    args = parse_arguments()

    base_path = None
    if args.base_path and os.path.exists(args.base_path):
        base_path = args.base_path
    else:
        # Try to find base path automatically
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(current_dir) == "tests":
            base_path = os.path.dirname(current_dir)

    if not base_path:
        print("Error: Unable to find project root directory")
        exit(1)

    print(f"Using root path: {base_path}")

    input_file = os.path.join(base_path, "data/fetaqa/test.jsonl")  # FeTaQA test set

    output_file = args.output_file
    model_path = args.model_path
    log_file = args.log_file

    max_tokens = args.max_tokens

    start_from = args.start_from

    temperature = args.temperature

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    process_fetaqa_data_batch(
        input_file=input_file,
        output_file=output_file,
        model_path=model_path,
        log_file=log_file,
        max_tokens=max_tokens,
        temperature=temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size,
        start_from=start_from,
    )


if __name__ == "__main__":
    main()
