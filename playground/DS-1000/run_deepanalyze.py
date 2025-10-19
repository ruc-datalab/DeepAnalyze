import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# from datasets import load_dataset
from tqdm import tqdm
import gzip
import concurrent.futures as cfuts
import argparse
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
import re


SYS_PROMPT = """
Write a short code following the given format and indentation. Only provide the code completion needed. Don't repeat the context code.
Your response should end with:
Answer:
```python
<your_code_here>
```"""


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
                        enable_thinking=True,
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

            if verbose and len(raw_generations) > 0:
                print("Example chat prompt:")
                print(raw_generations[0])

            return raw_generations

        except Exception as e:
            print(f"Error in vLLM generation: {str(e)}")
            raise


def extract_python_block(text):
    """
    从字符串中提取最后一个包含标记的```python ...```代码块
    """
    pattern = r"```python.*?```"  # 包括前后的```python 和 ```
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="codex002",
    help="which results to run",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="codex002",
    help="which results to run",
)
parser.add_argument(
    "--resume",
    action="store_true",
    default=False,
    help="where to resume inference",
)
args = parser.parse_args()
# model_name = os.path.basename(args.model)

model_name = args.model_name

cached_cnt = 0
if os.path.exists(f"data/{model_name.replace('/', '-')}-answers.jsonl"):
    if args.resume:
        cached_cnt = len(
            open(f"data/{model_name.replace('/', '-')}-answers.jsonl", "r").readlines()
        )
    else:
        exit(0)


generator = VLLMGenerator(
    model_path=args.model, max_model_len=16384, tensor_parallel_size=1
)


ds1000 = [json.loads(l) for l in gzip.open("data/ds1000.jsonl.gz", "rt").readlines()]

remaining_items = ds1000[cached_cnt:]

batch_size = 128

# Process data in batches
for batch_start in range(0, len(remaining_items), batch_size):
    batch_end = min(batch_start + batch_size, len(remaining_items))
    current_batch = remaining_items[batch_start:batch_end]

    # Create prompts for current batch
    prompts = []
    for item in current_batch:
        prompt = (
            item["prompt"]
            .replace("BEGIN SOLUTION\n<code>", "")
            .replace("<code>", "```python")
            .replace("</code>", "```")
            + SYS_PROMPT
        )
        prompts.append(prompt)

    responses = generator.generate(
        prompts=prompts,
        max_new_tokens=16384,
        temperature=0.0,
        top_p=0.8,
        verbose=(batch_start == 0),  # Only show example on first batch
    )

    with open(f"data/{model_name.replace('/', '-')}-answers.jsonl", "a+") as f:
        for code in responses:
            res = code
            stop_words = ["</code>", "# SOLUTION END"]
            for stop_word in stop_words:
                code = code.split(stop_word)[0]

            code = extract_python_block(code)
            r = {
                "id": cached_cnt,
                "code": code,
                "response": res,
                "metadata": ds1000[cached_cnt]["metadata"],
            }
            f.write(json.dumps(r) + "\n")
            cached_cnt += 1
