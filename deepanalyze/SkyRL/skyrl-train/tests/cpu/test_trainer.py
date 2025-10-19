"""
uv  run --isolated --extra dev pytest tests/cpu/test_trainer.py
"""

import torch
import pytest
from jaxtyping import Float, Integer
from omegaconf import OmegaConf
from pytest import approx
from unittest.mock import MagicMock, patch


from skyrl_train.distributed.dispatch import MeshRank
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch
import numpy as np
from skyrl_train.workers.worker import PolicyWorkerBase, CriticWorkerBase
from skyrl_train.workers.worker_utils import BatchIterator
from skyrl_train.utils.utils import validate_batch_sizes


class DummyDataset:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return "dummy"

    def collate_fn(self, batch):
        return batch


@pytest.fixture
def dummy_tokenizer():
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 2

    # encode("abc") -> [97, 98, 99]
    mock_tokenizer.encode.side_effect = lambda x: [ord(c) for c in x]

    # tokenizer("abc") -> {"input_ids": [...], "attention_mask": [...]}
    def fake_tokenizer_call(text, **kwargs):
        ids = [ord(c) for c in text]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }

    mock_tokenizer.side_effect = fake_tokenizer_call

    return mock_tokenizer


@pytest.fixture
def dummy_config():
    return OmegaConf.create(
        {
            "trainer": {
                "project_name": "unit-test",
                "run_name": "test-run",
                "logger": "tensorboard",
                "micro_train_batch_size_per_gpu": 2,
                "train_batch_size": 2,
                "eval_batch_size": 2,
                "update_epochs_per_batch": 1,
                "epochs": 1,
                "max_prompt_length": 20,
                "gamma": 0.99,
                "lambd": 0.95,
                "use_sample_packing": False,
                "seed": 42,
                "algorithm": {
                    "advantage_estimator": "grpo",
                    "use_kl_estimator_k3": False,
                    "use_abs_kl": False,
                    "kl_estimator_type": "k1",
                    "reward_clip_range": 5.0,
                    "use_kl_loss": True,
                    "kl_loss_coef": 0.0,
                    "lambd": 1.0,
                    "gamma": 1.0,
                    "eps_clip_low": 0.2,
                    "eps_clip_high": 0.2,
                    "clip_ratio_c": 3.0,
                    "value_clip": 0.2,
                    "normalize_reward": True,
                    "policy_loss_type": "regular",
                    "loss_reduction": "token_mean",
                    "grpo_norm_by_std": True,
                },
                "resume_mode": "none",
            },
            "generator": {
                "max_generate_length": 20,
                "n_samples_per_prompt": 1,
                "batched": False,
                "env_class": "gsm8k",
                "max_turns": 1,
            },
        }
    )


@pytest.fixture
def dummy_generator():
    return MagicMock()


def _get_test_data(trainer: RayPPOTrainer):
    trainer.critic_model = MagicMock()  # pretend we're using a critic

    batch_size = 2
    total_seq_len = 5
    action_len = 3

    # Create test data
    ret_sequences: Float[torch.Tensor, "batch_size total_seq_len"] = torch.randint(
        0, 1000, (batch_size, total_seq_len)
    )
    ret_attention_masks: Float[torch.Tensor, "batch_size total_seq_len"] = torch.ones(
        (batch_size, total_seq_len)
    )
    ret_loss_masks: Integer[torch.Tensor, "batch_size total_seq_len"] = torch.stack(
        [
            torch.tensor([1, 1, 0, 0, 0], dtype=torch.int32),
            torch.tensor([1, 1, 1, 0, 0], dtype=torch.int32),
        ],
        dim=0,
    )
    base_log_probs: Float[torch.Tensor, "batch_size total_seq_len"] = torch.log(
        torch.tensor([[0.1, 0.2, 0.3, 0.2, 0.2], [0.25, 0.25, 0.25, 0.15, 0.10]])
    )
    action_log_probs: Float[torch.Tensor, "batch_size total_seq_len"] = torch.log(
        torch.tensor([[0.1, 0.3, 0.2, 0.2, 0.2], [0.3, 0.3, 0.2, 0.1, 0.1]])
    )
    action_masks: Integer[torch.Tensor, "batch_size total_seq_len"] = torch.stack(
        [
            torch.tensor([1, 1, 1, 0, 0], dtype=torch.int32),
            torch.tensor([1, 1, 1, 1, 1], dtype=torch.int32),
        ],
        dim=0,
    )
    actual_response_lengths: Float[torch.Tensor, "batch_size"] = action_masks.sum(
        dim=-1
    ).to(float)
    custom_rewards_all: Float[torch.Tensor, "batch_size total_seq_len"] = torch.stack(
        [
            torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]),
        ],
        dim=0,
    )
    values: Float[torch.Tensor, "batch_size action_len"] = torch.randn(
        batch_size, action_len
    )
    r: Float[torch.Tensor, "batch_size action_len"] = torch.randn(
        batch_size, action_len
    )
    uids: np.ndarray[str] = np.array(["0", "0"])

    # Run method
    data = TrainingInputBatch(
        {
            "sequences": ret_sequences,
            "attention_mask": ret_attention_masks,
            "loss_mask": ret_loss_masks,
            "base_action_log_probs": base_log_probs,
            "action_log_probs": action_log_probs,
            "response_mask": action_masks,
            "custom_rewards": custom_rewards_all,
            "values": values,
            "rm_rewards": r,
        },
    )
    data.metadata = {
        "uids": uids,
        "response_length": action_len,
        "avg_response_length": actual_response_lengths.mean().item(),
    }
    data = trainer.apply_reward_kl_penalty(data)

    return data


def test_calculate_kl_create_experience_batched(dummy_config):
    trainer = RayPPOTrainer(
        cfg=dummy_config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=DummyDataset(),
        inference_engine_client=None,
        generator=dummy_generator,
    )
    data = _get_test_data(trainer)
    # Assertions
    metrics = data.metadata["metrics"]
    assert metrics["avg_kl_max"] == approx(0.3143, abs=1e-4)
    # Note; the raw KL mean is 0.054, but then the masked mean is different.
    assert metrics["avg_kl"] == approx(0.1249, abs=1e-4)


@patch(
    "skyrl_train.utils.ppo_utils.compute_advantages_and_returns", new_callable=MagicMock
)
def test_calc_advantages_and_returns(mock_compute_adv_and_ret, dummy_config):
    trainer = RayPPOTrainer(
        cfg=dummy_config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=DummyDataset(),
        inference_engine_client=None,
        generator=dummy_generator,
    )
    data = _get_test_data(trainer)

    # Mocked return values
    mock_advantages = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]]
    )
    mock_returns = torch.tensor([[0.6, 0.7, 0.8, 0.9, 1.0], [1.1, 1.2, 1.3, 1.4, 1.5]])

    # Set up mocks
    mock_compute_adv_and_ret.return_value = (mock_advantages, mock_returns)

    # Run the method
    data = trainer.compute_advantages_and_returns(data)
    metrics = data.metadata["metrics"]

    # Assertions
    assert torch.allclose(data["advantages"], mock_advantages)
    assert torch.allclose(data["returns"], mock_returns)
    assert isinstance(metrics, dict)
    assert "avg_rewards" in metrics
    assert "avg_response_length" in metrics
    assert "avg_advantages_abs" in metrics
    assert metrics["avg_advantages"] == approx(
        torch.masked_select(mock_advantages, data["response_mask"].bool())
        .mean()
        .item(),
        rel=1e-5,
    )


def test_normalize_mini_batch_size():
    """Test the _normalize_mini_batch_size method with various configurations."""

    # Create minimal worker instances for testing
    class TestPolicyWorker(PolicyWorkerBase):
        def init_model(self, *args, **kwargs):
            pass

        def offload_to_cpu(self, pin_memory=True, non_blocking=True):
            pass

        def backload_to_gpu(self, non_blocking=True):
            pass

        def _forward_micro_batch(self, micro_batch):
            pass

    class TestCriticWorker(CriticWorkerBase):
        def init_model(self, *args, **kwargs):
            pass

        def offload_to_cpu(self, pin_memory=True, non_blocking=True):
            pass

        def backload_to_gpu(self, non_blocking=True):
            pass

        def _forward_micro_batch(self, micro_batch):
            pass

    def create_policy_worker_with_config(
        train_batch_size,
        policy_mini_batch_size,
        micro_train_batch_size_per_gpu,
        n_samples_per_prompt,
        dp_size,
    ):
        """Helper to create policy worker with specific config."""
        cfg = OmegaConf.create(
            {
                "trainer": {
                    "train_batch_size": train_batch_size,
                    "policy_mini_batch_size": policy_mini_batch_size,
                    "micro_train_batch_size_per_gpu": micro_train_batch_size_per_gpu,
                    "algorithm": {
                        "policy_loss_type": "regular",
                    },
                },
                "generator": {
                    "n_samples_per_prompt": n_samples_per_prompt,
                },
            }
        )

        worker = TestPolicyWorker(
            cfg=cfg,
            world_size=dp_size,
            rank=0,
            local_rank=0,
            master_addr="localhost",
            master_port=12345,
            sequence_parallel_size=1,
        )

        # Mock mesh_rank
        worker.mesh_rank = MeshRank(
            dp=0, sp=0, tp=0, pp=0, world_size=dp_size, dp_size=dp_size
        )

        return worker

    def create_critic_worker_with_config(
        train_batch_size,
        critic_mini_batch_size,
        micro_train_batch_size_per_gpu,
        n_samples_per_prompt,
        dp_size,
    ):
        """Helper to create critic worker with specific config."""
        cfg = OmegaConf.create(
            {
                "trainer": {
                    "train_batch_size": train_batch_size,
                    "critic_mini_batch_size": critic_mini_batch_size,
                    "micro_train_batch_size_per_gpu": micro_train_batch_size_per_gpu,
                },
                "generator": {
                    "n_samples_per_prompt": n_samples_per_prompt,
                },
            }
        )

        worker = TestCriticWorker(
            cfg=cfg,
            world_size=dp_size,
            rank=0,
            local_rank=0,
            master_addr="localhost",
            master_port=12345,
            sequence_parallel_size=1,
        )

        # Mock mesh_rank
        worker.mesh_rank = MeshRank(
            dp=0, sp=0, tp=0, pp=0, world_size=dp_size, dp_size=dp_size
        )

        return worker

    # Test Case 1: Basic valid configuration for PolicyWorker
    policy_worker = create_policy_worker_with_config(
        train_batch_size=128,
        policy_mini_batch_size=16,
        micro_train_batch_size_per_gpu=2,
        n_samples_per_prompt=2,
        dp_size=4,
    )
    policy_worker._normalize_mini_batch_size()

    expected_policy_mini_batch_size_per_gpu = (16 * 2) // 4  # 8
    assert (
        policy_worker.policy_mini_batch_size_per_gpu
        == expected_policy_mini_batch_size_per_gpu
    )

    # Test Case 2: Basic valid configuration for CriticWorker
    critic_worker = create_critic_worker_with_config(
        train_batch_size=128,
        critic_mini_batch_size=8,
        micro_train_batch_size_per_gpu=2,
        n_samples_per_prompt=2,
        dp_size=4,
    )
    critic_worker._normalize_mini_batch_size()

    expected_critic_mini_batch_size_per_gpu = (8 * 2) // 4  # 4
    assert (
        critic_worker.critic_mini_batch_size_per_gpu
        == expected_critic_mini_batch_size_per_gpu
    )

    # Test Case 3: Single GPU (dp_size=1) for PolicyWorker
    policy_worker = create_policy_worker_with_config(
        train_batch_size=32,
        policy_mini_batch_size=8,
        micro_train_batch_size_per_gpu=4,
        n_samples_per_prompt=1,
        dp_size=1,
    )
    policy_worker._normalize_mini_batch_size()

    expected_policy_mini_batch_size_per_gpu = (8 * 1) // 1  # 8
    assert (
        policy_worker.policy_mini_batch_size_per_gpu
        == expected_policy_mini_batch_size_per_gpu
    )

    # Test Case 4: High n_samples_per_prompt for CriticWorker
    critic_worker = create_critic_worker_with_config(
        train_batch_size=256,
        critic_mini_batch_size=32,
        micro_train_batch_size_per_gpu=8,
        n_samples_per_prompt=4,
        dp_size=2,
    )
    critic_worker._normalize_mini_batch_size()

    expected_critic_mini_batch_size_per_gpu = (32 * 4) // 2  # 64
    assert (
        critic_worker.critic_mini_batch_size_per_gpu
        == expected_critic_mini_batch_size_per_gpu
    )

    # Test Case 5: Error case - mesh_rank not initialized
    policy_worker_no_mesh = create_policy_worker_with_config(
        train_batch_size=128,
        policy_mini_batch_size=16,
        micro_train_batch_size_per_gpu=2,
        n_samples_per_prompt=1,
        dp_size=4,
    )
    policy_worker_no_mesh.mesh_rank = None

    with pytest.raises(RuntimeError, match="mesh_rank must be initialized"):
        policy_worker_no_mesh._normalize_mini_batch_size()


def test_validate_batch_sizes():
    """Test the validate_batch_sizes function with various configurations to trigger all error cases."""

    def create_test_config(
        train_batch_size=128,
        policy_mini_batch_size=16,
        critic_mini_batch_size=8,
        micro_train_batch_size_per_gpu=2,
        micro_forward_batch_size_per_gpu=4,
        n_samples_per_prompt=2,
        policy_num_nodes=1,
        policy_num_gpus_per_node=4,
        critic_num_nodes=1,
        critic_num_gpus_per_node=4,
        policy_sequence_parallel_size=1,
        critic_sequence_parallel_size=1,
        critic_model_path=None,
    ):
        """Helper to create config for validation testing."""
        return OmegaConf.create(
            {
                "trainer": {
                    "train_batch_size": train_batch_size,
                    "policy_mini_batch_size": policy_mini_batch_size,
                    "critic_mini_batch_size": critic_mini_batch_size,
                    "micro_train_batch_size_per_gpu": micro_train_batch_size_per_gpu,
                    "micro_forward_batch_size_per_gpu": micro_forward_batch_size_per_gpu,
                    "placement": {
                        "policy_num_nodes": policy_num_nodes,
                        "policy_num_gpus_per_node": policy_num_gpus_per_node,
                        "critic_num_nodes": critic_num_nodes,
                        "critic_num_gpus_per_node": critic_num_gpus_per_node,
                    },
                    "policy": {
                        "sequence_parallel_size": policy_sequence_parallel_size,
                    },
                    "critic": {
                        "model": {
                            "path": critic_model_path,
                        },
                        "sequence_parallel_size": critic_sequence_parallel_size,
                    },
                },
                "generator": {
                    "n_samples_per_prompt": n_samples_per_prompt,
                },
            }
        )

    # Test Case 1: Valid configuration
    cfg = create_test_config()
    validate_batch_sizes(cfg)  # Should not raise any exceptions

    # Test Case 2: Error case - train_batch_size < policy_mini_batch_size
    cfg = create_test_config(train_batch_size=8, policy_mini_batch_size=16)
    with pytest.raises(AssertionError):
        validate_batch_sizes(cfg)

    # Test Case 3: Error case - train_batch_size < critic_mini_batch_size
    cfg = create_test_config(train_batch_size=4, critic_mini_batch_size=8)
    with pytest.raises(AssertionError):
        validate_batch_sizes(cfg)

    # Test Case 4: Error case - policy_mini_batch_size = 0
    cfg = create_test_config(policy_mini_batch_size=0)
    with pytest.raises(
        AssertionError, match="policy_mini_batch_size must be greater than 0"
    ):
        validate_batch_sizes(cfg)

    # Test Case 5: Error case - critic_mini_batch_size = 0
    cfg = create_test_config(critic_mini_batch_size=0, critic_model_path="test")
    with pytest.raises(
        AssertionError, match="critic_mini_batch_size must be greater than 0"
    ):
        validate_batch_sizes(cfg)

    # Test Case 6: Error case - micro_train_batch_size_per_gpu = 0
    cfg = create_test_config(micro_train_batch_size_per_gpu=0)
    with pytest.raises(
        AssertionError, match="micro_train_batch_size_per_gpu must be greater than 0"
    ):
        validate_batch_sizes(cfg)

    # Test Case 7: Error case - micro_forward_batch_size_per_gpu = 0
    cfg = create_test_config(micro_forward_batch_size_per_gpu=0)
    with pytest.raises(
        AssertionError, match="micro_forward_batch_size_per_gpu must be greater than 0"
    ):
        validate_batch_sizes(cfg)

    # Test Case 8: Error case - train_batch_size not divisible by (policy_mini_batch_size * policy_dp_size)
    cfg = create_test_config(
        train_batch_size=100, policy_mini_batch_size=16, policy_num_gpus_per_node=4
    )
    # Should fail because train_batch_size is not evenly divisible by policy batch requirements
    with pytest.raises(
        AssertionError,
        match="train_batch_size .* should be divisible by policy_mini_batch_size",
    ):
        validate_batch_sizes(cfg)

    # Test Case 9: Error case - train_batch_size not divisible by (critic_mini_batch_size * critic_dp_size)
    cfg = create_test_config(
        train_batch_size=100,
        policy_mini_batch_size=5,
        critic_mini_batch_size=16,
        critic_num_gpus_per_node=4,
        critic_model_path="test",
    )
    # Should fail because train_batch_size is not evenly divisible by critic batch requirements
    with pytest.raises(
        AssertionError,
        match="train_batch_size .* should be divisible by critic_mini_batch_size",
    ):
        validate_batch_sizes(cfg)

    # Test Case 10: Error case - policy_mini_batch_size_per_gpu not divisible by micro_train_batch_size_per_gpu
    cfg = create_test_config(
        policy_mini_batch_size=8,
        n_samples_per_prompt=1,
        policy_num_gpus_per_node=1,
        micro_train_batch_size_per_gpu=3,
    )
    # Should fail because policy mini batch per GPU is not evenly divisible by micro batch size
    with pytest.raises(
        AssertionError,
        match="normalized policy_mini_batch_size_per_gpu .* should be divisible by micro_train_batch_size_per_gpu",
    ):
        validate_batch_sizes(cfg)

    # Test Case 11: Error case - critic_mini_batch_size_per_gpu not divisible by micro_train_batch_size_per_gpu
    cfg = create_test_config(
        train_batch_size=144,
        policy_mini_batch_size=12,  # Policy validation passes
        critic_mini_batch_size=8,  # Critic micro batch divisibility fails
        n_samples_per_prompt=1,
        critic_num_gpus_per_node=1,
        micro_train_batch_size_per_gpu=3,
        critic_model_path="test",
    )
    # Should fail because critic mini batch per GPU is not evenly divisible by micro batch size
    with pytest.raises(
        AssertionError,
        match="normalized critic_mini_batch_size_per_gpu .* should be divisible by micro_train_batch_size_per_gpu",
    ):
        validate_batch_sizes(cfg)

    # Test Case 12: Valid configuration with sequence parallelism
    cfg = create_test_config(
        policy_sequence_parallel_size=2,
        critic_sequence_parallel_size=2,
        policy_num_gpus_per_node=8,
        critic_num_gpus_per_node=8,
    )
    validate_batch_sizes(cfg)  # Should not raise any exceptions

    # Test Case 13: Valid configuration - train_batch_size not divisible by (critic_mini_batch_size * critic_dp_size), but critic model path is None
    cfg = create_test_config(
        train_batch_size=100,
        policy_mini_batch_size=5,
        critic_mini_batch_size=16,
        critic_num_gpus_per_node=4,
        critic_model_path=None,
    )
    validate_batch_sizes(cfg)

    # Test Case 14: Valid configuration - critic_mini_batch_size is invalid but critic model is not specified
    cfg = create_test_config(critic_mini_batch_size=0, critic_model_path=None)
    validate_batch_sizes(cfg)

    # Test Case 15: Error case - train_batch_size_per_gpu not divisible by policy_mini_batch_size_per_gpu
    cfg = create_test_config(
        train_batch_size=10,
        policy_mini_batch_size=5,
        policy_num_gpus_per_node=2,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=1,
    )
    with pytest.raises(
        AssertionError,
        match="policy_train_batch_size_per_gpu .* should be divisible by policy_mini_batch_size_per_gpu",
    ):
        validate_batch_sizes(cfg)

    # Test Case 16: Error case - train_batch_size_per_gpu not divisible by critic_mini_batch_size_per_gpu
    cfg = create_test_config(
        train_batch_size=10,
        policy_mini_batch_size=10,
        policy_num_gpus_per_node=1,
        critic_mini_batch_size=5,
        critic_num_gpus_per_node=2,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=1,
        critic_model_path="test",
    )
    with pytest.raises(
        AssertionError,
        match="critic_train_batch_size_per_gpu .* should be divisible by critic_mini_batch_size_per_gpu",
    ):
        validate_batch_sizes(cfg)


def test_ppo_train_batch_calculations():
    """Test the key batch calculations and control flow in ppo_train methods."""

    # Create test configuration
    cfg = OmegaConf.create(
        {
            "trainer": {
                "micro_train_batch_size_per_gpu": 2,
                "update_epochs_per_batch": 1,
                "algorithm": {
                    "policy_loss_type": "regular",
                },
            },
            "generator": {
                "sampling_params": {
                    "temperature": 1.0,
                },
            },
        }
    )

    # Create dummy databatch with known size
    batch_size = (
        12  # This will create 6 micro batches with micro_train_batch_size_per_gpu=2
    )
    response_length = 4  # number of actions
    dummy_databatch = TrainingInputBatch(
        {
            "sequences": torch.randint(
                0, 100, (batch_size, 10)
            ),  # dummy token sequences
            "attention_mask": torch.ones(batch_size, 10),
            "action_log_probs": torch.randn(batch_size, response_length),
            "base_action_log_probs": torch.randn(batch_size, response_length),
            "values": torch.randn(batch_size, response_length),
            "returns": torch.randn(batch_size, response_length),
            "advantages": torch.randn(batch_size, response_length),
            "loss_mask": torch.ones(batch_size, response_length),
            "response_mask": torch.ones(batch_size, response_length),
            "rollout_logprobs": None,
        },
    )
    dummy_databatch.metadata = {"global_step": 0, "response_length": response_length}

    # Helper function to create worker with minimal setup
    def create_test_worker(worker_class):
        worker = worker_class(
            cfg=cfg,
            world_size=1,
            rank=0,
            local_rank=0,
            master_addr="localhost",
            master_port=12345,
            sequence_parallel_size=1,
        )
        # Set appropriate mini batch size per gpu based on worker type
        if worker_class == PolicyWorkerBase:
            worker.policy_mini_batch_size_per_gpu = (
                6  # Should result in 3 micro batches per mini batch
            )
        elif worker_class == CriticWorkerBase:
            worker.critic_mini_batch_size_per_gpu = (
                6  # Should result in 3 micro batches per mini batch
            )

        # Mock dependencies
        worker.strategy = MagicMock()
        worker.strategy.is_rank_0.return_value = False  # Disable progress bars
        worker.strategy.all_reduce.return_value = {"loss": 0.5, "lr": 1e-4}

        # Always set model for all worker types (policy/critic need this for ppo_train)
        worker.model = MagicMock()

        return worker

    # Test PolicyWorkerBase
    policy_worker = create_test_worker(PolicyWorkerBase)

    # Mock training_step to track calls and verify accumulation behavior
    policy_training_calls = []

    def mock_policy_training_step(
        experience, global_step, local_step, accumulation_steps
    ):
        policy_training_calls.append(
            {"local_step": local_step, "accumulation_steps": accumulation_steps}
        )
        return {"policy_loss": 0.5, "policy_lr": 1e-4, "entropy": 2.0}

    policy_worker.training_step = mock_policy_training_step

    # Calculate expected values based on new accumulation logic
    dataloader = BatchIterator(
        dummy_databatch,
        sample_batch_size=cfg.trainer.micro_train_batch_size_per_gpu,
        drop_last=False,
    )
    total_micro_batches = len(dataloader)  # Should be 6
    micro_batches_per_mini_batch = (
        policy_worker.policy_mini_batch_size_per_gpu
        // cfg.trainer.micro_train_batch_size_per_gpu
    )  # 6 // 2 = 3
    # New logic: accumulation_steps = micro_batches_per_mini_batch (accumulate within mini-batch)
    expected_accumulation_steps = micro_batches_per_mini_batch  # Should be 3

    # Run policy ppo_train with minimal mocking
    with (
        patch("torch.distributed.barrier"),
        patch("tqdm.tqdm", side_effect=lambda x, **kwargs: x),
    ):  # Disable progress bar
        result = policy_worker.ppo_train(dummy_databatch)

    # Verify Policy Worker Results
    assert (
        len(policy_training_calls) == total_micro_batches
    ), f"PolicyWorker: Expected {total_micro_batches} training_step calls, got {len(policy_training_calls)}"

    # Verify accumulation_steps are consistent (should equal micro_batches_per_mini_batch)
    for call in policy_training_calls:
        assert (
            call["accumulation_steps"] == expected_accumulation_steps
        ), f"PolicyWorker: Expected accumulation_steps={expected_accumulation_steps}, got {call['accumulation_steps']}"

    # Verify no early termination (all micro batches processed)
    expected_local_steps = list(range(total_micro_batches))
    actual_local_steps = [call["local_step"] for call in policy_training_calls]
    assert (
        actual_local_steps == expected_local_steps
    ), f"PolicyWorker: Expected local_steps {expected_local_steps}, got {actual_local_steps}"

    # Verify result structure
    assert "train_status" in result.metadata
    train_status = result.metadata["train_status"]
    assert "policy_update_steps" in train_status

    # Verify policy_update_steps calculation (should be total_calls / accumulation_steps)
    expected_policy_update_steps_normalized = (
        len(policy_training_calls) / expected_accumulation_steps
    )
    assert (
        train_status["policy_update_steps"] == expected_policy_update_steps_normalized
    )

    # Test CriticWorkerBase with same accumulation logic
    critic_worker = create_test_worker(CriticWorkerBase)

    critic_training_calls = []

    def mock_critic_training_step(
        experience, global_step, local_step, accumulation_steps
    ):
        critic_training_calls.append(
            {"local_step": local_step, "accumulation_steps": accumulation_steps}
        )
        return {"critic_loss": 0.3, "values": 1.0, "critic_lr": 1e-4}

    critic_worker.training_step = mock_critic_training_step

    # Run critic ppo_train
    with (
        patch("torch.distributed.barrier"),
        patch("tqdm.tqdm", side_effect=lambda x, **kwargs: x),
        patch("torch.cuda.empty_cache"),
    ):
        result = critic_worker.ppo_train(dummy_databatch)

    # Verify Critic Worker Results
    assert (
        len(critic_training_calls) == total_micro_batches
    ), f"CriticWorker: Expected {total_micro_batches} training_step calls, got {len(critic_training_calls)}"

    # Verify accumulation_steps are consistent for critic (should equal micro_batches_per_mini_batch)
    for call in critic_training_calls:
        assert (
            call["accumulation_steps"] == expected_accumulation_steps
        ), f"CriticWorker: Expected accumulation_steps={expected_accumulation_steps}, got {call['accumulation_steps']}"

    # Verify no early termination for critic
    actual_local_steps = [call["local_step"] for call in critic_training_calls]
    assert (
        actual_local_steps == expected_local_steps
    ), f"CriticWorker: Expected local_steps {expected_local_steps}, got {actual_local_steps}"

    # Verify result structure for critic
    assert "train_status" in result.metadata
    train_status = result.metadata["train_status"]
    assert "critic_update_steps" in train_status
    assert (
        train_status["critic_update_steps"]
        == len(critic_training_calls) / expected_accumulation_steps
    )


def test_build_dataloader_seeding(dummy_config):
    """Test that build_dataloader correctly seeds the dataloader for reproducible shuffling."""

    # Create a dataset with multiple distinct items to test shuffling
    class MultiItemDataset:
        def __init__(self, size=10):
            self.data = [f"item_{i}" for i in range(size)]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

        def collate_fn(self, batch):
            return batch

    dataset = MultiItemDataset(size=20)

    # Test 1: Same seed should produce same shuffling
    config1 = dummy_config.copy()
    config1.trainer.seed = 42
    config1.trainer.train_batch_size = 5

    config2 = dummy_config.copy()
    config2.trainer.seed = 42  # Same seed
    config2.trainer.train_batch_size = 5

    # Create trainers with same seed
    trainer1 = RayPPOTrainer(
        cfg=config1,
        tracker=None,
        tokenizer=None,
        train_dataset=dataset,
        eval_dataset=None,
        inference_engine_client=None,
        generator=None,
    )

    trainer2 = RayPPOTrainer(
        cfg=config2,
        tracker=None,
        tokenizer=None,
        train_dataset=dataset,
        eval_dataset=None,
        inference_engine_client=None,
        generator=None,
    )

    # Build dataloaders
    dataloader1 = trainer1.build_dataloader(dataset, is_train=True)
    dataloader2 = trainer2.build_dataloader(dataset, is_train=True)

    # Get first batch from each dataloader
    first_batch1 = next(iter(dataloader1))
    first_batch2 = next(iter(dataloader2))

    # With same seed, first batches should be identical
    assert (
        first_batch1 == first_batch2
    ), f"Same seed should produce same first batch, got {first_batch1} vs {first_batch2}"

    # Test 2: Different seeds should produce different shuffling
    config3 = dummy_config.copy()
    config3.trainer.seed = 123  # Different seed
    config3.trainer.train_batch_size = 5

    trainer3 = RayPPOTrainer(
        cfg=config3,
        tracker=None,
        tokenizer=None,
        train_dataset=dataset,
        eval_dataset=None,
        inference_engine_client=None,
        generator=None,
    )

    dataloader3 = trainer3.build_dataloader(dataset, is_train=True)
    first_batch3 = next(iter(dataloader3))

    # With different seed, first batch should be different
    # Note: There's a tiny chance they could be the same by random chance, but very unlikely with 20 items
    assert (
        first_batch1 != first_batch3
    ), f"Different seeds should produce different first batches, but both gave {first_batch1}"
