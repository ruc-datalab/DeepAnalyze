import json
import os
import logging
import time
import re
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from prompt import COT_PROMPT_OTTQA_TEMPLATE


# Setup logging
def setup_logger(log_file):
    """Set up the logger"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("ottqa_processor")
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


class VLLMGenerator:
    """
    A class for generating text using vLLM with support for different models.
    """

    def __init__(self, model_path, max_model_len=8192, tensor_parallel_size=1):
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


def format_table_for_prompt(table_data):
    """Format table data into a string representation for prompts"""
    result = []

    # Add introduction information
    if "intro" in table_data and table_data["intro"]:
        result.append(f"Introduction: {table_data['intro']}")

    # Add section title and text
    if "section_title" in table_data and table_data["section_title"]:
        section_text = table_data.get("section_text", "")
        result.append(f"Section Description: {section_text}")

    # Add table content
    if "header" in table_data and table_data["header"] and "data" in table_data:
        # Get the header
        if table_data["header"][0]:
            headers = table_data["header"][0]
            # Format header
            header_row = " | ".join(headers)
            result.append(header_row)

        # Format data rows
        for row in table_data["data"]:
            result.append(" | ".join(row))

    return "\n".join(result)


def format_text_for_prompt(link_data):
    """Format linked text data into a string representation for prompts"""
    result = []

    for link, content in link_data.items():
        # Extract the last part of the link as the prefix
        link_parts = link.split("/")
        prefix = link_parts[-1] if link_parts else link

        # Add text with prefix
        result.append(f"{prefix}: {content}")

    return "\n\n".join(result)


def create_prompt_from_hybridqa(item):
    """Create prompt for a HybridQA item"""
    # Format table
    table_str = format_table_for_prompt(item["table"])

    # Format text
    text_str = format_text_for_prompt(item["relevant_links"])

    # Generate prompt
    prompt = COT_PROMPT_OTTQA_TEMPLATE.format(
        table=table_str, text=text_str, question=item["question"]
    )

    return prompt


def extract_answer_from_response(model_answer):
    """
    Extract final answer from the model response, supporting multiple formats:
    1. "Answer: xxx" format
    2. "<answer>xxx</answer>" tag format
    3. "<answer>Answer: xxx</answer>" combined format
    """
    # If response is empty, return empty string
    if not model_answer:
        return ""

    # Check if there's an <answer> tag
    answer_tag_pattern = re.search(r"<Answer>(.*?)</Answer>", model_answer, re.DOTALL)
    if answer_tag_pattern:
        # Extract content from <answer> tag
        answer_content = answer_tag_pattern.group(1).strip()

        # Check if there's an "Answer:" mark inside the tag
        if "Answer:" in answer_content:
            return answer_content.split("Answer:")[-1].strip()
        # Otherwise return the full content inside the tag
        return answer_content

    # If no <answer> tag but there's an "Answer:" mark
    elif "Answer:" in model_answer:
        return model_answer.split("Answer:")[-1].strip()

    # If no markers, return the entire response
    return model_answer.strip()


def process_hybridqa_data_batch(
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
    """Process HybridQA dataset with batched VLLM inference"""
    logger = setup_logger(log_file)

    # Record start time
    start_time = time.time()
    logger.info(
        f"Started processing HybridQA data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Using model: {model_path}")
    logger.info(f"Batch size: {batch_size}")

    # Initialize VLLM generator
    try:
        generator = VLLMGenerator(
            model_path=model_path,
            max_model_len=8192,
            tensor_parallel_size=tensor_parallel_size,
        )
        logger.info(f"VLLM generator initialized successfully")
    except Exception as e:
        logger.error(f"VLLM initialization failed: {e}")
        return

    # Read data items
    data_items = []
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data_items = json.load(f)
        logger.info(f"Loaded {len(data_items)} questions from input file")
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
                processed_ids.add(result.get("question_id", ""))

            logger.info(f"Loaded intermediate results with {len(results)} records")
            logger.info(f"Found {len(processed_ids)} already processed items")
        except Exception as e:
            logger.error(
                f"Failed to load intermediate results: {e}, starting from beginning"
            )
            results = []
            processed_ids = set()

    # Filter already processed items and apply start_from
    remaining_items = []
    for idx, item in enumerate(data_items):
        if idx < start_from:
            continue
        item_id = item.get("question_id", f"item_{idx}")
        if item_id not in processed_ids:
            remaining_items.append(item)

    logger.info(f"Remaining items to process: {len(remaining_items)}/{len(data_items)}")

    success_count = len(results)
    error_count = 0

    # Prepare evaluation format
    predictions = {
        result.get("question_id", ""): result.get("extracted_answer", "")
        for result in results
    }

    # Process data in batches
    for batch_start in range(0, len(remaining_items), batch_size):
        batch_end = min(batch_start + batch_size, len(remaining_items))
        current_batch = remaining_items[batch_start:batch_end]

        # Create prompts for current batch
        prompts = []
        for item in current_batch:
            prompt = create_prompt_from_hybridqa(item)
            prompts.append(prompt)

        # Generate responses for the batch
        try:
            batch_start_time = time.time()
            responses = generator.generate(
                prompts=prompts,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.8,
                verbose=(batch_start == 0),  # Only show example on first batch
            )
            batch_time = time.time() - batch_start_time
            logger.info(
                f"Batch inference completed in {batch_time:.2f} seconds ({batch_time/len(current_batch):.2f} seconds per item)"
            )

            # Process each response
            for i, (item, response) in enumerate(zip(current_batch, responses)):
                # Record start time for processing this item
                item_start_time = time.time()

                # Extract item information
                item_id = item.get("question_id", f"item-{i+batch_start}")
                question = item.get("question", "")
                grouth_answer = item.get("answer", "")

                # Calculate current item index
                global_item_index = batch_start + i + 1

                logger.info(
                    f"Processing item {global_item_index}/{len(remaining_items)}... [ID: {item_id}]"
                )

                # Extract final answer
                extracted_answer = extract_answer_from_response(response)

                # Calculate processing time
                item_time = time.time() - item_start_time

                # Log detailed information
                logger.info(f"Question: {question}")
                logger.info(f"Model answer: {extracted_answer}")
                logger.info(f"Processing time: {item_time:.6f} seconds")
                logger.info("-" * 50)

                # Build result object
                result = {
                    "question_id": item_id,
                    "question": question,
                    "prompt": prompts[i],
                    "model_answer": response,
                    "extracted_answer": extracted_answer,
                    "ground_answer": grouth_answer,
                    "processing_time": item_time,
                }

                # Add to results and update tracking
                results.append(result)
                success_count += 1

                # Add to predictions dict for evaluation format
                predictions[item_id] = extracted_answer

            # Save batch results
            with open(f"{output_file}.temp", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            error_count += 1
            logger.error(f"Error processing batch: {e}")
            # Save what we have so far
            with open(f"{output_file}.temp", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # Save detailed results to JSON file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save detailed results file: {e}")

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
        description="Process OTTQA dataset with VLLM batch inference"
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

    input_file = os.path.join(base_path, "data/ottqa/dev_top_1.json")

    output_file = args.output_file
    model_path = args.model_path
    log_file = args.log_file

    max_tokens = args.max_tokens

    start_from = args.start_from

    temperature = args.temperature

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    process_hybridqa_data_batch(
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
