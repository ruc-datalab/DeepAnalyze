"""
uv run --isolated --extra dev pytest tests/cpu/dataset/test_preprocess.py
"""

import pytest
import torch
from omegaconf import OmegaConf
from skyrl_train.dataset.preprocess import (
    convert_prompts_responses_to_batch_tensors,
)

from unittest.mock import MagicMock


@pytest.fixture
def cfg():
    return OmegaConf.create({"trainer": {"max_prompt_length": 10}, "generator": {"max_generate_length": 5}})


# NOTE (sumanthrh): the tests in this file are hardcoded to use the below character-level tokenizer
@pytest.fixture
def tokenizer():
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 2

    # encode("abc") -> [97, 98, 99]
    def fake_encode(text):
        if isinstance(text, list):
            return [fake_encode(t) for t in text]
        return [ord(c) for c in text]

    mock_tokenizer.encode.side_effect = fake_encode

    # tokenizer("abc") -> {"input_ids": [...], "attention_mask": [...]}
    def fake_tokenizer_call(text, **kwargs):
        if isinstance(text, list):
            dicts = [fake_tokenizer_call(t, **kwargs) for t in text]
            return {
                "input_ids": [d["input_ids"] for d in dicts],
                "attention_mask": [d["attention_mask"] for d in dicts],
            }
        ids = [ord(c) for c in text]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }

    mock_tokenizer.side_effect = fake_tokenizer_call

    def fake_tokenizer_decode(ids, **kwargs):
        return "".join([chr(i) for i in ids])

    mock_tokenizer.decode.side_effect = fake_tokenizer_decode

    def fake_tokenizer_decode_list(ids, **kwargs):
        return [fake_tokenizer_decode(i) for i in ids]

    mock_tokenizer.batch_decode.side_effect = fake_tokenizer_decode_list

    return mock_tokenizer


def test_convert_prompts_responses_to_batch_tensors_exact(tokenizer, cfg):
    prompts = ["abc", "12345"]
    outputs = ["def", "67890"]
    prompts = tokenizer(prompts)["input_ids"]
    outputs = tokenizer(outputs)["input_ids"]

    loss_masks = [[1, 1, 0], [1, 1, 1, 0, 0]]
    custom_rewards = [torch.tensor([0, 1, 0]), torch.tensor([1, 0, 0, 0, 0])]

    sequences, attention_mask, action_mask, ret_custom_rewards, ret_loss_masks, ret_log_probs = (
        convert_prompts_responses_to_batch_tensors(
            tokenizer,
            prompts,
            outputs,
            custom_rewards,
            loss_masks,
        )
    )

    # loss mask should be the same length as the action mask (padded to the longest input)
    assert sequences.shape[0] == len(prompts)
    assert action_mask.shape == ret_loss_masks.shape
    assert torch.equal(ret_loss_masks[0], torch.tensor([1, 1, 0, 0, 0]))
    assert torch.equal(ret_loss_masks[1], torch.tensor([1, 1, 1, 0, 0]))
    assert torch.equal(ret_custom_rewards[0], torch.tensor([0, 1, 0, 0, 0]))
    assert torch.equal(ret_custom_rewards[1], torch.tensor([1, 0, 0, 0, 0]))


def test_convert_prompts_responses_to_batch_tensors_different_lengths(cfg, tokenizer):
    # Test with inputs of different lengths
    prompts = ["Short", "This is a longer prompt"]
    outputs = ["Long response here", "Short"]
    prompts = tokenizer(prompts)["input_ids"]
    outputs = tokenizer(outputs)["input_ids"]
    custom_rewards = [torch.tensor([1.0, 0.5, 0.3]), torch.tensor([0.8])]
    loss_masks = [[1, 1, 1], [1]]

    sequences, attention_mask, action_mask, ret_custom_rewards, ret_loss_masks, ret_log_probs = (
        convert_prompts_responses_to_batch_tensors(
            tokenizer,
            prompts,
            outputs,
            custom_rewards,
            loss_masks,
        )
    )

    max_response_len = max([len(output) for output in outputs])

    # Check shapes
    assert sequences.shape[0] == 2  # batch size
    assert attention_mask.shape == sequences.shape
    # Tensor.shape can be directly compared with tuples
    assert action_mask.shape == (2, max_response_len)
    assert ret_custom_rewards.shape == (2, max_response_len)
    assert ret_loss_masks.shape == (2, max_response_len)

    # Verify padding is applied correctly
    # First input is shorter than second input. the input is left padded
    assert sequences[0, 0] == tokenizer.pad_token_id
    # second output is shorter than first output. the output is right padded
    assert sequences[1, -1] == tokenizer.pad_token_id


def test_convert_prompts_responses_to_batch_tensors_empty_input(cfg, tokenizer):
    # Test with empty input
    prompts = []
    outputs = []
    custom_rewards = []
    loss_masks = []

    with pytest.raises(AssertionError):
        convert_prompts_responses_to_batch_tensors(
            tokenizer,
            prompts,
            outputs,
            custom_rewards,
            loss_masks,
        )


def test_convert_prompts_responses_to_batch_tensors_mismatched_lengths(cfg, tokenizer):
    # Test with mismatched input lengths
    prompts = ["Hello", "World"]
    outputs = ["Response"]
    prompts = tokenizer(prompts)["input_ids"]
    outputs = tokenizer(outputs)["input_ids"]
    custom_rewards = [torch.tensor([1.0])]
    loss_masks = [[1]]

    with pytest.raises(AssertionError):
        convert_prompts_responses_to_batch_tensors(
            tokenizer,
            prompts,
            outputs,
            custom_rewards,
            loss_masks,
        )
