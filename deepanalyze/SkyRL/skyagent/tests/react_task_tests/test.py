from skyagent import AutoAgentRunner
from transformers import AutoTokenizer
import datasets
import asyncio
import os
import argparse
import pandas as pd


def main(yaml_path, dataset_path, split, num_samples, model_name):
    os.environ["OPENAI_API_KEY"] = ""

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if os.path.exists(dataset_path):
        # Local parquet file
        try:
            df = pd.read_parquet(dataset_path)
        except Exception:
            # Fallback for nested columns
            import polars as pl

            df = pl.read_parquet(dataset_path).to_pandas()
        dataset = datasets.Dataset.from_pandas(df)
    else:
        dataset = datasets.load_dataset(dataset_path, split=split)

    if num_samples > 0:
        dataset = dataset.select(range(num_samples))
    dataset = dataset.select(range(16))  # Get first 16 items
    agent_generator = AutoAgentRunner.from_task(
        yaml_path, infer_engine=None, tokenizer=tokenizer
    )
    output = asyncio.run(agent_generator.run(dataset))
    rewards = output["rewards"]

    mean_reward = sum(rewards) / len(rewards)
    print(f"Mean reward with yaml {yaml_path}: {mean_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True, help="Path to yaml configuration file")
    parser.add_argument("--dataset", required=True, help="Path to dataset parquet file")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to process from the dataset",
    )
    parser.add_argument(
        "--split", default="train", help="Dataset split to use (default: train)"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name")

    args = parser.parse_args()
    main(args.yaml, args.dataset, args.split, args.num_samples, args.model)
