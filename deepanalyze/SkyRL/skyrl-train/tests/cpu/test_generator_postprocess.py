"""
Test for token-level rewards support in RayPPOTrainer.postprocess_generator_output method.

Run with:
uv run --isolated --extra dev pytest tests/cpu/test_generator_postprocess.py
"""

from unittest.mock import MagicMock
from omegaconf import OmegaConf

from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.generators.base import GeneratorOutput


class DummyDataset:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return "dummy"

    def collate_fn(self, batch):
        return batch


def create_config(batch_size):
    return OmegaConf.create(
        {
            "trainer": {
                "train_batch_size": batch_size,
                "eval_batch_size": batch_size,
                "resume_mode": "none",
                "seed": 42,
                "epochs": 1,
            },
            "generator": {
                "n_samples_per_prompt": 1,
            },
        }
    )


def test_response_level_rewards():
    """Test postprocess_generator_output with response-level rewards (List[float])."""

    # Test length=1
    config = create_config(1)
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )

    generator_output: GeneratorOutput = {
        "prompt_token_ids": [[1, 2]],
        "response_ids": [[3, 4, 5]],
        "rewards": [1.0],  # Response-level reward
        "loss_masks": [[1, 1, 1]],
        "stop_reasons": ["stop"],
        "rollout_metrics": None,
    }

    result = trainer.postprocess_generator_output(generator_output, ["uid1"])

    # Verify conversion to per-token rewards
    assert result["rewards"] == [[0.0, 0.0, 1.0]]

    # Test length=2
    config = create_config(2)
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )

    generator_output: GeneratorOutput = {
        "prompt_token_ids": [[1, 2], [3, 4]],
        "response_ids": [[5, 6], [7, 8, 9]],
        "rewards": [1.0, 0.5],  # Response-level rewards
        "loss_masks": [[1, 1], [1, 1, 1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_metrics": None,
    }

    result = trainer.postprocess_generator_output(generator_output, ["uid1", "uid2"])

    # Verify conversion to per-token rewards
    assert result["rewards"] == [[0.0, 1.0], [0.0, 0.0, 0.5]]


def test_token_level_rewards():
    """Test postprocess_generator_output with token-level rewards (List[List[float]])."""

    # Test length=1
    config = create_config(1)
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )

    per_token_rewards = [[0.1, 0.2, 0.3]]
    generator_output: GeneratorOutput = {
        "prompt_token_ids": [[1, 2]],
        "response_ids": [[3, 4, 5]],
        "rewards": per_token_rewards,  # Token-level rewards
        "loss_masks": [[1, 1, 1]],
        "stop_reasons": ["stop"],
        "rollout_metrics": None,
    }

    result = trainer.postprocess_generator_output(generator_output, ["uid1"])

    # Verify token-level rewards are unchanged
    assert result["rewards"] == per_token_rewards

    # Test length=2
    config = create_config(2)
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )

    per_token_rewards = [[0.1, 0.3], [0.2, 0.1, 0.1]]
    generator_output: GeneratorOutput = {
        "prompt_token_ids": [[1, 2], [3, 4]],
        "response_ids": [[5, 6], [7, 8, 9]],
        "rewards": per_token_rewards,  # Token-level rewards
        "loss_masks": [[1, 1], [1, 1, 1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_metrics": None,
    }

    result = trainer.postprocess_generator_output(generator_output, ["uid1", "uid2"])

    # Verify token-level rewards are unchanged
    assert result["rewards"] == per_token_rewards
