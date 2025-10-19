"""
Tests for ppo_train method in worker classes.

Run with:
uv run --isolated --extra dev -- pytest tests/gpu/test_ppo_train.py
"""

import pytest
import ray
from omegaconf import DictConfig

from tests.gpu.utils import (
    init_worker_with_type,
    make_dummy_training_batch,
    get_test_actor_config,
)


@pytest.fixture
def cfg() -> DictConfig:
    """Get test configuration with minimal settings for fast testing."""
    cfg = get_test_actor_config()

    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.policy_mini_batch_size = 2
    cfg.generator.n_samples_per_prompt = 1
    cfg.trainer.placement.policy_num_gpus_per_node = 2

    return cfg


def test_ppo_train_basic_execution(cfg):
    """
    Test that ppo_train runs and returns correct structure.

    This test validates:
    - ppo_train method executes successfully
    - Returns TrainingOutputBatch with correct metadata structure
    - Contains expected training metrics
    """
    try:
        cfg.trainer.strategy = "deepspeed"  # Strategy logic is not tested here.

        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        train_data = make_dummy_training_batch(batch_size=2, seq_len=10, num_actions=4)
        train_data.metadata["global_step"] = 0

        # Run ppo_train
        results = ray.get(
            actor_group.async_run_ray_method("pass_through", "ppo_train", train_data)
        )
        assert (
            len(results) == cfg.trainer.placement.policy_num_gpus_per_node
        ), "Should get result from each GPU"

        result = results[0]  # Check first worker result
        assert hasattr(result, "metadata"), "Result should have metadata attribute"
        assert "train_status" in result.metadata, "Should have train_status in metadata"

        train_status = result.metadata["train_status"]

        # Validate expected training metrics are present
        expected_metrics = [
            "policy_loss",
            "policy_update_steps",
            "policy_lr",
            "ppo_clip_ratio",
            "policy_entropy",
        ]

        for metric in expected_metrics:
            assert metric in train_status, f"Should have {metric} in train_status"
            assert isinstance(
                train_status[metric], (int, float)
            ), f"{metric} should be numeric"

        # Simple check for metric values
        assert (
            train_status["policy_update_steps"] > 0
        ), "Should have completed at least one update step"
        assert train_status["policy_lr"] > 0, "Should have positive learning rate"

    finally:
        ray.shutdown()


def test_ppo_train_critic_worker(cfg):
    """
    Test that ppo_train works for critic worker as well.
    """
    try:
        cfg.trainer.strategy = "deepspeed"  # Strategy logic is not tested here.

        actor_group = init_worker_with_type(
            "critic",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        # Create training batch directly
        train_data = make_dummy_training_batch(batch_size=2, seq_len=10, num_actions=4)
        train_data.metadata["global_step"] = 0

        # Run ppo_train
        results = ray.get(
            actor_group.async_run_ray_method("pass_through", "ppo_train", train_data)
        )

        result = results[0]
        assert hasattr(result, "metadata"), "Result should have metadata attribute"
        assert "train_status" in result.metadata, "Should have train_status in metadata"

        train_status = result.metadata["train_status"]

        # Validate critic-specific metrics
        expected_critic_metrics = [
            "critic_loss",
            "critic_update_steps",
            "values_mean",
            "critic_lr",
        ]

        for metric in expected_critic_metrics:
            assert (
                metric in train_status
            ), f"Should have {metric} in critic train_status"
            assert isinstance(
                train_status[metric], (int, float)
            ), f"{metric} should be numeric"

        assert (
            train_status["critic_update_steps"] > 0
        ), "Should have completed at least one critic update step"

        print(f"Critic ppo_train completed successfully with metrics: {train_status}")
    finally:
        ray.shutdown()


@pytest.mark.parametrize(
    "test_id, micro_train_batch_size_per_gpu, policy_mini_batch_size, n_samples_per_prompt, update_epochs_per_batch, batch_size, expected_optimizer_steps",
    [
        ("accumulation_calculation", 2, 8, 2, 1, 8, 1),
        ("optimizer_stepping", 1, 8, 1, 1, 12, 3),
        ("multiple_epochs", 1, 4, 1, 3, 6, 9),
    ],
    ids=["accumulation_calculation", "optimizer_stepping", "multiple_epochs"],
)
def test_gradient_accumulation_scenarios(
    test_id,
    micro_train_batch_size_per_gpu,
    policy_mini_batch_size,
    n_samples_per_prompt,
    update_epochs_per_batch,
    batch_size,
    expected_optimizer_steps,
):
    """
    Test that gradient accumulation and optimizer stepping work correctly across various scenarios.

    Validates:
    - Correct calculation of accumulation steps based on configuration.
    - Optimizer stepping at correct intervals for single and multiple epochs.
    - Consistent behavior across different batch and minibatch sizes.
    """
    try:
        cfg = get_test_actor_config()
        cfg.trainer.strategy = "deepspeed"  # Strategy logic is not tested here.
        cfg.trainer.placement.policy_num_gpus_per_node = 2

        # Set scenario-specific config
        cfg.trainer.micro_train_batch_size_per_gpu = micro_train_batch_size_per_gpu
        cfg.trainer.policy_mini_batch_size = policy_mini_batch_size
        cfg.generator.n_samples_per_prompt = n_samples_per_prompt
        cfg.trainer.update_epochs_per_batch = update_epochs_per_batch

        # For logging and assertions, calculate expected accumulation steps
        dp_size = cfg.trainer.placement.policy_num_gpus_per_node
        policy_mini_batch_size_per_gpu = (
            policy_mini_batch_size * n_samples_per_prompt
        ) // dp_size
        # If micro_train_batch_size_per_gpu is 0, this indicates an issue in configuration, but for safety:
        accumulation_steps = (
            policy_mini_batch_size_per_gpu // micro_train_batch_size_per_gpu
            if micro_train_batch_size_per_gpu > 0
            else 1
        )
        if accumulation_steps == 0:
            accumulation_steps = 1  # Should not be 0, must step at least once.

        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        train_data = make_dummy_training_batch(
            batch_size=batch_size, seq_len=10, num_actions=4
        )
        train_data.metadata["global_step"] = 0

        result = ray.get(
            actor_group.async_run_ray_method("pass_through", "ppo_train", train_data)
        )[0]

        train_status = result.metadata["train_status"]
        actual_optimizer_steps = train_status["policy_update_steps"]

        assert actual_optimizer_steps == expected_optimizer_steps, (
            f"Test '{test_id}' failed: Expected {expected_optimizer_steps} optimizer steps, got {actual_optimizer_steps}. "
            f"Config: micro_batch={micro_train_batch_size_per_gpu}, mini_batch={policy_mini_batch_size}, "
            f"n_samples={n_samples_per_prompt}, epochs={update_epochs_per_batch}, "
            f"data_batch_size={batch_size}, accumulation_steps={accumulation_steps}"
        )

        print(f"Gradient accumulation scenario '{test_id}' PASSED:")
        print(
            f"   - Config: micro_batch={micro_train_batch_size_per_gpu}, mini_batch={policy_mini_batch_size}, "
            f"n_samples={n_samples_per_prompt}, epochs={update_epochs_per_batch}"
        )
        print(f"   - Data batch size: {batch_size}")
        print(f"   - Expected accumulation steps: {accumulation_steps}")
        print(f"   - Expected optimizer steps: {expected_optimizer_steps}")
        print(f"   - Actual optimizer steps: {actual_optimizer_steps}")
    finally:
        ray.shutdown()
