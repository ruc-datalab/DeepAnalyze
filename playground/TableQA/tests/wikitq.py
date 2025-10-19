import json
import os
import logging
import time
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm


# Setup logging
def setup_logger(log_file):
    """Set up the logger"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("table_qa_processor")
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


def create_prompt_from_wikitq(
    item, is_think=False, table_format="markdown", perturbation="none"
):
    """
    Create prompt for WikiTableQuestion item with different table formats and perturbations

    Args:
        item: The input item containing table and question
        is_think: Whether to use thinking prompt template
        table_format: Format for table representation - "markdown", "csv", or "dataframe"
        perturbation: Type of table perturbation - "none", "row_shuffle", "col_shuffle", "both_shuffle"

    Returns:
        Formatted prompt string
    """
    table_str = ""
    if "table" in item:
        # Deep copy table data to avoid modifying original data
        import copy
        import random

        table = copy.deepcopy(item["table"])

        # Get headers and data rows
        header = table[0] if len(table) > 0 else []
        data_rows = table[1:] if len(table) > 1 else []

        # Apply perturbation
        row_shuffle = perturbation in ["row_shuffle", "both_shuffle"]
        col_shuffle = perturbation in ["col_shuffle", "both_shuffle"]

        # Apply column perturbation
        if col_shuffle and len(header) > 1:
            # Generate new column order
            col_indices = list(range(len(header)))
            random.shuffle(col_indices)

            # Reorder headers and data rows
            header = [header[i] for i in col_indices]
            data_rows = [
                [row[i] if i < len(row) else "" for i in col_indices]
                for row in data_rows
            ]

        # Apply row perturbation
        if row_shuffle and len(data_rows) > 1:
            # Keep headers unchanged, randomly shuffle data rows
            random.shuffle(data_rows)

        # Recombine table
        shuffled_table = [header] + data_rows

        # Generate table string based on selected format
        if table_format == "markdown":
            # Create Markdown format table string representation
            for row in shuffled_table:
                table_str += " | ".join([str(cell) for cell in row]) + "\n"

        elif table_format == "csv":
            # Create CSV format table string representation
            for row in shuffled_table:
                table_str += ",".join([f'"{str(cell)}"' for cell in row]) + "\n"

        elif table_format == "dataframe":
            # Import pandas library
            import pandas as pd

            # Create DataFrame-like table representation
            if len(shuffled_table) > 0:
                # Get headers and data
                headers = shuffled_table[0]
                data = shuffled_table[1:] if len(shuffled_table) > 1 else []

                # Create pandas DataFrame
                df = pd.DataFrame(data, columns=headers)

                # Configure pandas display options to show complete table
                pd.set_option("display.expand_frame_repr", False)
                pd.set_option("display.width", 500000)
                pd.set_option("display.max_rows", None)
                pd.set_option("display.max_columns", None)
                pd.set_option("display.max_colwidth", None)

                # Convert DataFrame to string
                table_str = df.to_string(index=True)

                # Add DataFrame information line
                table_str += f"\n\n[{len(data)} rows x {len(headers)} columns]\n"

                pd.reset_option("display.expand_frame_repr")
                pd.reset_option("display.width")
                pd.reset_option("display.max_rows")
                pd.reset_option("display.max_columns")
                pd.reset_option("display.max_colwidth")

        else:
            # Default use Markdown format
            for row in shuffled_table:
                table_str += " | ".join([str(cell) for cell in row]) + "\n"

        # If perturbation was applied, add explanatory note
        if perturbation != "none" and not is_think:  # Add note in non-thinking mode
            notes = []
            if row_shuffle:
                notes.append("Row order has been randomly shuffled (except header)")
            if col_shuffle:
                notes.append("Column order has been randomly shuffled")

            if notes:
                table_str += "\nNote: " + ", ".join(notes) + "\n"

    # Use different templates based on thinking requirement
    if is_think:
        from prompt_think import COT_PROMPT_TEMPLATE

        prompt = COT_PROMPT_TEMPLATE.format(table=table_str, question=item["question"])
    else:
        from prompt import COT_PROMPT_TEMPLATE

        prompt = COT_PROMPT_TEMPLATE.format(table=table_str, question=item["question"])

    return prompt


def process_table_qa_data_batch(
    input_file,
    output_file,
    model_path,
    log_file,
    max_tokens=2048,
    temperature=0.0,
    tensor_parallel_size=1,
    batch_size=16,
    start_from=0,
    is_think=False,
    table_format="markdown",
    perturbation="none",
):
    """Process WikiTableQuestions dataset with batched VLLM inference"""
    logger = setup_logger(log_file)

    # Record start time
    start_time = time.time()
    logger.info(
        f"Started processing table QA data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
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
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data_items.append(json.loads(line))
        logger.info(f"Loaded {len(data_items)} data items")
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return

    # Check for intermediate results
    results = []
    processed_ids = set()
    if os.path.exists(f"{output_file}.temp"):
        try:
            with open(f"{output_file}.temp", "r", encoding="utf-8") as f:
                results = json.load(f)

            # Get IDs of already processed items
            for result in results:
                processed_ids.add(result.get("id", ""))

            logger.info(f"Loaded intermediate results with {len(results)} records")
            logger.info(f"Found {len(processed_ids)} already processed items")
        except Exception as e:
            logger.error(
                f"Failed to load intermediate results: {e}, starting from beginning"
            )
            results = []
            processed_ids = set()

    # Filter out already processed items and apply start_from
    remaining_items = []
    for idx, item in enumerate(data_items):
        if idx < start_from:
            continue
        item_id = item.get("id", f"item_{idx}")
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
            prompt = create_prompt_from_wikitq(
                item, is_think, table_format=table_format, perturbation=perturbation
            )
            prompts.append(prompt)
        # Generate responses for batch
        try:
            batch_start_time = time.time()
            responses = generator.generate(
                prompts=prompts,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
                verbose=(batch_start == 0),  # Only show example in first batch
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
                item_id = item.get("id", f"item-{i+batch_start}")
                question = item.get("question", "")

                # Calculate current item index
                global_item_index = batch_start + i + 1

                logger.info(
                    f"Processing item {global_item_index}/{len(remaining_items)}... [ID: {item_id}]"
                )

                # Calculate processing time
                item_time = time.time() - item_start_time

                # Log detailed information
                logger.info(f"Question: {question}")
                logger.info(f"Golden answer: {item['answer']}")
                logger.info(f"Model answer: {response}")
                logger.info(f"Processing time: {item_time:.6f} seconds")
                logger.info("-" * 50)

                # Build result object
                result = {
                    "id": item_id,
                    "source": item.get("source", {}),
                    "prompt": prompts[i],
                    "question": question,
                    "answer": item["answer"],
                    "model_answer": response,
                    "processing_time": item_time,
                }

                results.append(result)
                success_count += 1

                # # Save intermediate results every 5 items
                # if (global_item_index % 5 == 0):
                #     with open(f"{output_file}.temp", 'w', encoding='utf-8') as f:
                #         json.dump(results, f, ensure_ascii=False, indent=2)
                #     logger.info(f"Saved intermediate results - {len(results)}/{len(data_items)} items processed")

            # Save batch results
            with open(f"{output_file}.temp", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            error_count += 1
            logger.error(f"Error processing batch: {e}")
            # Save progress so far
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
    if len(data_items) > 0:
        logger.info(f"Success rate: {success_count/len(data_items)*100:.2f}%")
    average_time_per_item = (
        total_time / len(remaining_items) if len(remaining_items) > 0 else 0
    )
    logger.info(
        f"Average processing time per item: {average_time_per_item:.2f} seconds"
    )
    logger.info("=" * 60)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process WikiTableQuestions dataset with VLLM batch inference"
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
    parser.add_argument(
        "--think", action="store_true", help="increase output verbosity"
    )
    # Table format parameters
    parser.add_argument(
        "--table_format",
        type=str,
        default="markdown",
        choices=["markdown", "csv", "dataframe"],
        help="Format for table representation in prompts",
    )

    # Table perturbation parameters
    parser.add_argument(
        "--perturbation",
        type=str,
        default="none",
        choices=["none", "row_shuffle", "col_shuffle", "both_shuffle"],
        help="Type of table perturbation to apply",
    )

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Handle base_path
    base_path = None
    if args.base_path and os.path.exists(args.base_path):
        base_path = args.base_path
    else:
        # Try to automatically find base_path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(current_dir) == "tests":
            base_path = os.path.dirname(current_dir)

    if not base_path:
        print("Error: Unable to find project root directory")
        exit(1)

    print(f"Using root path: {base_path}")

    # Set file paths
    input_file = os.path.join(base_path, "data/wikitq/test.jsonl")

    # Use command line parameters if provided, otherwise use default values
    output_file = args.output_file
    model_path = args.model_path
    log_file = args.log_file

    # Use maximum tokens from command line parameters
    max_tokens = args.max_tokens

    # Use starting index from command line parameters
    start_from = args.start_from

    # Use temperature value from command line parameters
    temperature = args.temperature

    # Ensure output and log directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Process data
    process_table_qa_data_batch(
        input_file=input_file,
        output_file=output_file,
        model_path=model_path,
        log_file=log_file,
        max_tokens=max_tokens,
        temperature=temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size,
        start_from=start_from,
        is_think=args.think,
        table_format=args.table_format,
        perturbation=args.perturbation,
    )


if __name__ == "__main__":
    main()
