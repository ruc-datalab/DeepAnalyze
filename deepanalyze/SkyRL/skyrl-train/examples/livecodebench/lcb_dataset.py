"""Script to prepare code datasets for training and testing.

This script processes code problem datasets into a standardized format for training
and testing models. It loads problems from various code datasets (APPS, CodeForces,
LiveCodeBench etc.), adds appropriate instruction prompts, and saves the processed
data as parquet files.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

"""Utility functions for loading and processing datasets."""

LCB_SYSTEM_MESSAGE_GENERIC = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."

LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

LCB_FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."


# Dataset constants
LIVECODEBENCH = "livecodebench"


def load_dataset(
    dataset_name: str, split: str, base_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Load a dataset from a JSON file.

    Args:
        dataset_name: Name of the dataset (e.g., "livecodebench")
        split: Either "train" or "test"
        base_dir: Base directory containing the dataset files

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the dataset items.

    Raises:
        ValueError: If the dataset file cannot be found or contains invalid JSON.
    """
    if base_dir is None:
        current_dir = os.path.dirname(os.path.realpath(__file__))
    else:
        current_dir = base_dir

    file_path = os.path.join(current_dir, split, "code", f"{dataset_name}.json")

    if not os.path.exists(file_path):
        raise ValueError(f"Dataset file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {file_path}")
    except Exception as e:
        raise ValueError(f"Error loading dataset: {str(e)}")


def fetch_live_code_bench_system_prompt(prompt: str, starter_code: str = None):
    # https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
    prompt = LCB_SYSTEM_MESSAGE_GENERIC + "\n\n" + prompt
    if starter_code:
        prompt += f"### Format: {LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def process_example(
    example: Dict[str, Any], idx: int, dataset_name: str, split: str
) -> Optional[Dict[str, Any]]:
    """Process a single dataset example.

    Args:
        example: Raw dataset example
        idx: Index of the example
        dataset_name: Name of the dataset
        split: Dataset split ('train' or 'test')

    Returns:
        Processed example dictionary or None if processing fails
    """
    question = example.pop("problem")
    tests = example.pop("tests")

    if example.get("metadata", {}):
        assert (
            "func_name" in example["metadata"]
        ), f"Function name is not found, check if your LCB data is preprocessed correctly: {example['metadata']}"
        if isinstance(tests, dict):
            tests["metadata"] = example["metadata"]
        else:
            for test in tests:
                assert isinstance(test, dict), "Test is not a dict"
                test["metadata"] = example["metadata"]

    tests = json.dumps(tests)

    if dataset_name == LIVECODEBENCH:
        starter_code = example.get("starter_code", None)
        question = fetch_live_code_bench_system_prompt(question, starter_code)
    if isinstance(question, dict):
        question = json.dumps(question)
    data = {
        "data_source": dataset_name,
        "prompt": [{"role": "user", "content": question}],
        "ability": "code",
        "reward_spec": {"method": "rule", "ground_truth": tests},
        "extra_info": {
            "split": split,
            "index": idx,
            "reference": example.get("completion", None),  # For leetcode
        },
    }
    return data


def process_dataset(
    dataset_name: str,
    split: str,
    dataset_dir: str,
    local_dir: str,
    max_rows: Optional[int] = None,
):
    """Process a dataset for a given split.

    Args:
        dataset_name: Name of the dataset
        split: Either "train" or "test"
        dataset_dir: Directory containing raw datasets
        local_dir: Directory to save processed datasets
        max_rows: Maximum number of rows to process
    """
    # Load dataset
    raw_data = load_dataset(dataset_name, split, base_dir=dataset_dir)
    print(f"{split.capitalize()} dataset {dataset_name}: {len(raw_data)} examples")

    # Process examples
    processed_data = []
    for idx, example in enumerate(raw_data):
        processed_example = process_example(example, idx, dataset_name, split)
        if processed_example is not None:
            processed_data.append(processed_example)

    # Apply max_rows limit if specified
    if max_rows:
        processed_data = processed_data[:max_rows]

    # Save individual dataset files
    df = pd.DataFrame(processed_data)
    parquet_path = os.path.join(local_dir, f"{split}_{dataset_name}.parquet")
    json_path = os.path.join(local_dir, f"{split}_{dataset_name}.json")

    print(f"Writing {len(df)} rows to {split}_{dataset_name}.parquet")
    df.to_parquet(parquet_path)

    if split == "test":
        print(f"Writing {len(df)} rows to {split}_{dataset_name}.json")
        df.to_json(json_path, orient="records")

    return processed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process datasets for DeepScaler training"
    )
    parser.add_argument(
        "--local_dir",
        default=os.path.expanduser("~/data/lcb"),
        help="Local directory to save processed datasets",
    )
    parser.add_argument(
        "--dataset_dir",
        default=os.path.expanduser("~/data/lcb"),
        help="Directory containing raw input datasets",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Maximum number of rows to include in output files (truncate if set).",
    )
    args = parser.parse_args()

    local_dir = args.local_dir
    print(f"Local_dir:{local_dir}")

    # Make local directory if it doesn't exist
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)

    # Process train dataset
    train_data = process_dataset(
        LIVECODEBENCH, "train", args.dataset_dir, local_dir, args.max_rows
    )

    # Process test dataset
    val_data = process_dataset(
        LIVECODEBENCH, "test", args.dataset_dir, local_dir, args.max_rows
    )

    # Save combined train dataset
    all_train_df = pd.DataFrame(train_data)
    all_train_df.to_parquet(os.path.join(local_dir, "deepcoder_train.parquet"))
    all_train_df.to_json(
        os.path.join(local_dir, "deepcoder_train.json"), orient="records"
    )
    print(
        f"Writing {len(all_train_df)} rows to deepcoder_train.parquet and deepcoder_train.json"
    )
