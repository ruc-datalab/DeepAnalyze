import json
import os
import logging
import time
import argparse
import re
from datetime import datetime
from typing import List, Dict, Any
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm


# Setup logging
def setup_logger(log_file):
    """Set up the logger"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("tatqa_processor")
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


def create_prompt_from_tatqa(item, question_item, is_think):
    """Create prompt from TaTQA data item"""
    # Process table
    table_data = item["table"]["table"]
    table_str = ""
    for row in table_data:
        table_str += " | ".join([str(cell) for cell in row]) + "\n"

    # Process text paragraphs with paragraph numbers
    text_parts = []
    for i, para in enumerate(item["paragraphs"], 1):
        # Add paragraph number before each paragraph
        text_parts.append(f"Paragraph {i}: {para['text']}")
    text_str = "\n".join(text_parts)

    if is_think:
        from prompt_think import COT_PROMPT_TATQA_TEMPLATE

        # Generate prompt
        prompt = COT_PROMPT_TATQA_TEMPLATE.format(
            table=table_str, text=text_str, question=question_item["question"]
        )
    else:
        from prompt import COT_PROMPT_TATQA_TEMPLATE

        # Generate prompt
        prompt = COT_PROMPT_TATQA_TEMPLATE.format(
            table=table_str, text=text_str, question=question_item["question"]
        )

    return prompt


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


def find_gold_answer_with_info(test_data, question_text, table_uid):
    """Find gold answer and other info for a given question from the gold dataset"""
    for item in test_data:
        for question in item.get("questions", []):
            if question.get("question", "").strip() == question_text.strip():
                return {
                    "answer": question.get("answer", "No answer found"),
                    "answer_type": question.get("answer_type", "span"),
                    "answer_from": question.get("answer_from", "text"),
                    "scale": question.get("scale", ""),
                    "derivation": question.get("derivation", ""),
                    "rel_paragraphs": question.get("rel_paragraphs", []),
                    "req_comparison": question.get("req_comparison", False),
                    "facts": question.get("facts", []),
                    "consts": question.get("consts", []),
                    "mappings": question.get("mappings", []),
                }

    return {
        "answer": "No gold answer found",
        "answer_type": "",
        "answer_from": "",
        "scale": "",
        "derivation": "",
        "rel_paragraphs": [],
        "req_comparison": False,
        "facts": [],
        "consts": [],
        "mappings": [],
    }


def process_tatqa_data_batch(
    input_file,
    gold_file,
    output_file,
    model_path,
    log_file,
    max_tokens=2048,
    temperature=0.0,
    tensor_parallel_size=1,
    batch_size=16,
    is_think=False,
):
    """Process TaTQA dataset with batched VLLM inference"""
    logger = setup_logger(log_file)

    # Record start time
    start_time = time.time()
    logger.info(
        f"Started processing TaTQA data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info(f"Input file: {input_file}")
    logger.info(f"Gold answer file: {gold_file}")
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

    # Read test data JSON file
    test_data = []
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        logger.info(f"Loaded {len(test_data)} table data items from test file")
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return

    # Read gold answers JSON file
    gold_data = []
    try:
        with open(gold_file, "r", encoding="utf-8") as f:
            gold_data = json.load(f)
        logger.info(f"Loaded gold answers with {len(gold_data)} items")
    except Exception as e:
        logger.error(f"Failed to read gold answer file: {e}")
        # Continue without gold answers
        logger.warning("Will proceed without gold answers")

    # Expand one-table-many-questions into multiple question items
    expanded_items = []
    question_count = 0

    for table_idx, table_item in enumerate(test_data):
        table_id = table_item["table"]["uid"]

        # Skip empty tables or those without questions
        if not table_item.get("questions"):
            continue

        for q_idx, question_item in enumerate(table_item["questions"]):
            question_count += 1
            # Create a more structured ID: table_index-question_index (e.g., T001-Q005)
            item_id = f"T{table_idx+1:03d}-Q{q_idx+1:03d}"

            expanded_items.append(
                {
                    "table_item": table_item,
                    "question_item": question_item,
                    "id": item_id,
                    "original_ids": {
                        "table_uid": table_id,
                        "question_uid": question_item.get("uid", ""),
                    },
                }
            )

    logger.info(
        f"Expanded into {len(expanded_items)} questions from {question_count} total questions"
    )

    # Check if intermediate results exist
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

    # Filter out already processed items
    remaining_items = [
        item for item in expanded_items if item["id"] not in processed_ids
    ]
    logger.info(
        f"Remaining items to process: {len(remaining_items)}/{len(expanded_items)}"
    )

    # Process data in batches
    for batch_start in range(0, len(remaining_items), batch_size):
        batch_end = min(batch_start + batch_size, len(remaining_items))
        current_batch = remaining_items[batch_start:batch_end]

        # Create prompts for current batch
        prompts = []
        for item in current_batch:
            table_item = item["table_item"]
            question_item = item["question_item"]
            prompt = create_prompt_from_tatqa(table_item, question_item, is_think)
            prompts.append(prompt)

        # Generate responses for the batch
        try:
            # batch_start_time = time.time()
            responses = generator.generate(
                prompts=prompts,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.8,
                verbose=(batch_start == 0),  # Only show example on first batch
            )
            # Process each response
            for i, (item, response) in enumerate(zip(current_batch, responses)):
                # Extract item metadata
                item_start_time = time.time()
                item_id = item["id"]
                table_item = item["table_item"]
                question_item = item["question_item"]
                table_uid = item["original_ids"]["table_uid"]
                question_text = question_item["question"]

                # 计算当前项是整个数据集中的第几项
                global_item_index = batch_start + i + 1

                # 记录详细的处理信息
                logger.info(
                    f"Processing item {global_item_index}/{len(expanded_items)}... [ID: {item_id}]"
                )

                # Find gold answer info
                gold_answer_info = find_gold_answer_with_info(
                    gold_data, question_text, table_uid
                )
                gold_answer = gold_answer_info["answer"]

                # Extract final answer
                final_answer = extract_final_answer(response)

                # 计算处理时间
                item_time = time.time() - item_start_time
                # 记录详细日志
                logger.info(f"Question: {question_text}")
                logger.info(f"Gold answer: {gold_answer}")
                logger.info(f"Answer type: {gold_answer_info['answer_type']}")
                logger.info(f"Model answer: {final_answer}")
                logger.info(f"Processing time: {item_time:.6f} seconds")

                logger.info("-" * 50)

                # Create result object
                result = {
                    "id": item_id,
                    "original_ids": item["original_ids"],
                    "prompt": prompts[i],
                    "question": question_text,
                    "model_answer": final_answer,
                    "full_response": response,
                    "gold_answer": gold_answer,
                    "answer_type": gold_answer_info["answer_type"],
                    "answer_from": gold_answer_info["answer_from"],
                    "scale": gold_answer_info["scale"],
                    "derivation": gold_answer_info["derivation"],
                    "rel_paragraphs": gold_answer_info["rel_paragraphs"],
                    "req_comparison": gold_answer_info["req_comparison"],
                    "facts": gold_answer_info["facts"],
                    "consts": gold_answer_info["consts"],
                    "mappings": gold_answer_info["mappings"],
                    "processing_time": item_time,
                }

                results.append(result)

            with open(f"{output_file}.temp", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Still save what we have so far
            with open(f"{output_file}.temp", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # Save final results to JSON file
    try:
        # Sort results by ID before saving
        results.sort(key=lambda x: x["id"])

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results file: {e}")

    # Log summary information
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Processing completed! Total time: {total_time:.2f} seconds")
    logger.info(f"Successfully processed: {len(results)}/{len(expanded_items)} items")
    average_time_per_item = (
        total_time / (len(expanded_items) - len(processed_ids))
        if (len(expanded_items) - len(processed_ids)) > 0
        else 0
    )
    logger.info(
        f"Average processing time per item: {average_time_per_item:.2f} seconds"
    )
    logger.info("=" * 60)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process TaTQA dataset with VLLM batch inference"
    )

    parser.add_argument(
        "--output_file", type=str, default=None, help="Path to save results"
    )  # required=True,
    parser.add_argument(
        "--model_path", type=str, help="Model path or identifier"
    )  # required=True,
    parser.add_argument("--data_path", type=str, help="Path to input data file")
    parser.add_argument("--gold_file", type=str, help="Path to gold answer file")
    parser.add_argument(
        "--log_file",
        type=str,
        default="./logs/tatqa_inference.log",
        help="Path to log file",
    )  # required=True,
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
        "--batch_size", type=int, default=64, help="Batch size for inference"
    )
    parser.add_argument("--base_path", type=str, help="Base path for the project")
    parser.add_argument(
        "--think", action="store_true", help="increase output verbosity"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Determine base path
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

    # Setup paths
    input_file = args.data_path or os.path.join(
        base_path, "data/tatqa/tatqa_dataset_test.json"
    )
    gold_file = args.gold_file or os.path.join(
        base_path, "data/tatqa/tatqa_dataset_test_gold.json"
    )
    output_file = args.output_file
    model_path = args.model_path
    log_file = args.log_file

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Process data with batched VLLM inference
    process_tatqa_data_batch(
        input_file=input_file,
        gold_file=gold_file,
        output_file=output_file,
        model_path=model_path,
        log_file=log_file,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size,
        is_think=args.think,
    )


if __name__ == "__main__":
    main()
