"""
Preprocess the dataset for the 'multiply' environment in parquet format.
"""

import argparse
import random
import os
from datasets import Dataset


def generate_multiplication_problem(num_digits):
    """Generate a random multiplication problem with n-digit numbers."""
    # Generate random n-digit numbers
    min_val = 10 ** (num_digits - 1)
    max_val = 10**num_digits - 1

    num1 = random.randint(min_val, max_val)
    num2 = random.randint(min_val, max_val)

    question = f"{num1} * {num2}"
    answer = num1 * num2

    return question, str(answer)


def create_dataset(num_examples, num_digits, split_name):
    """Create a dataset of multiplication problems."""
    examples = []

    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant that solves multiplication problems. Please solve the given multiplication problem step by step. Put your final answer in \\boxed{answer} format.",
    }

    for idx in range(num_examples):
        question, answer = generate_multiplication_problem(num_digits)

        data = {
            "data_source": "synthetic_multiply",
            "prompt": [
                system_prompt,
                {
                    "role": "user",
                    "content": question,
                },
            ],
            "env_class": "multiply",
            "reward_spec": {
                "method": "rule",
                "ground_truth": answer,
            },
            "extra_info": {
                "num_digits": num_digits,
                "split": split_name,
            },
        }
        examples.append(data)

    return Dataset.from_list(examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/multiply")
    parser.add_argument(
        "--num_digits",
        type=int,
        default=2,
        help="Number of digits for each number in multiplication",
    )
    parser.add_argument(
        "--train_size", type=int, default=10000, help="Number of training examples"
    )
    parser.add_argument(
        "--test_size", type=int, default=200, help="Number of test examples"
    )

    args = parser.parse_args()

    # Generate datasets
    train_dataset = create_dataset(args.train_size, args.num_digits, "train")
    val_dataset = create_dataset(args.test_size, args.num_digits, "test")

    # Save datasets
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))

    print(
        f"Generated {args.train_size} training examples and {args.test_size} test examples"
    )
    print(f"Using {args.num_digits}-digit numbers")
    print(f"Saved to {output_dir}")
