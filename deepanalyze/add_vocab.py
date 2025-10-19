#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extend a local Hugging Face model tokenizer by adding custom tokens.

Usage:
    python extend_tokenizer.py \
        --model_path /path/to/original/model \
        --save_path /path/to/save/extended/model \
        --add_tags

Example:
    python extend_tokenizer.py \
        --model_path ./DeepSeek-R1-0528-Qwen3-8B \
        --save_path ./DeepSeek-R1-0528-Qwen3-8B-AddTags2
"""

import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_and_extend_model(model_path, new_tokens=None):
    """
    Load a local Hugging Face model and extend its tokenizer vocabulary.
    Initialize <Analyze> and </Analyze> embeddings using <think> and </think>.

    Args:
        model_path (str): Path to the local model directory
        new_tokens (list): List of new tokens to add to the tokenizer
    """
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    original_vocab_size = tokenizer.vocab_size
    print(f"Original vocabulary size: {original_vocab_size}")

    if new_tokens:
        num_added_tokens = tokenizer.add_tokens(new_tokens)
        print(f"Added {num_added_tokens} new tokens")

        if num_added_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))
            print(f"Resized embeddings to new vocab size: {len(tokenizer)}")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Extend tokenizer and model embeddings with new tokens."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the local base model (e.g., ./DeepSeek-R1-0528-Qwen3-8B)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the extended model/tokenizer (e.g., ./DeepSeek-R1-0528-Qwen3-8B-AddTags2)",
    )
    parser.add_argument(
        "--add_tags",
        action="store_true",
        help="Add default tag tokens like <Analyze>, <Understand>, <Code>, etc.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")

    new_tokens = None
    if args.add_tags:
        new_tokens = [
            "<Analyze>",
            "</Analyze>",
            "<Understand>",
            "</Understand>",
            "<Code>",
            "</Code>",
            "<Execute>",
            "</Execute>",
            "<Answer>",
            "</Answer>",
        ]
        print(f"🔹 Adding default tokens: {new_tokens}")

    model, tokenizer = load_and_extend_model(args.model_path, new_tokens)

    sample_text = (
        "<Analyze>\nTo determine the count of customers supported by Steve Johnson, we need to.\n</Analyze>\n"
        "<Code>\n```python\n```\n</Code>\n"
        "<Execute>\n=== Employees Data ===\n\n</Execute>\n"
        "<Understand>\nFrom the execution results, we can observe.\n</Understand>\n"
        "<Answer>\nSteve Johnson supports 18 customers.\n</Answer>"
    )
    encoded = tokenizer.encode(sample_text, return_tensors="pt")
    decoded = [tokenizer.decode(x) for x in encoded[0]]

    print("\nample text encoding test:")
    print(f"Input: {sample_text}")
    print(f"Encoded tensor shape: {encoded.shape}")
    print(f"Decoded tokens: {decoded[:20]} ...")  # show first 20 tokens only

    os.makedirs(args.save_path, exist_ok=True)
    tokenizer.save_pretrained(args.save_path)
    model.save_pretrained(args.save_path)
    print(f"\nExtended model & tokenizer saved to: {args.save_path}")


if __name__ == "__main__":
    main()
