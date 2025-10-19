"""
Integration test for full trainer checkpointing functionality.

This test validates that the RayPPOTrainer can save and restore ALL training state,
ensuring that training can resume exactly where it left off.

Run with:
uv run --isolated --extra dev --with deepspeed -- pytest tests/gpu/test_trainer_full_checkpointing.py
"""

import ray
import pytest
import hydra
import torch
import os
import shutil
import tempfile
from omegaconf import DictConfig
from torch.utils.data import Dataset
from unittest.mock import MagicMock
from transformers import AutoTokenizer

from skyrl_train.utils.tracking import Tracking
from skyrl_train.trainer import RayPPOTrainer
from tests.gpu.utils import import_worker, ray_init_for_tests
from skyrl_train.entrypoints.main_base import config_dir

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


class DummyDataset(Dataset):
    """Minimal dataset for testing"""

    def __init__(self, size=10):
        self.data = [
            ([{"role": "user", "content": f"Question {i}"}], None) for i in range(size)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        return batch


def get_test_trainer_config(
    strategy: str, fsdp2_cpu_offload: bool = False
) -> DictConfig:
    """Create minimal trainer config for testing"""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.critic.model.path = MODEL_NAME  # Enable critic for testing
    cfg.trainer.strategy = strategy
    if strategy == "fsdp2":
        cfg.trainer.policy.fsdp_config.cpu_offload = fsdp2_cpu_offload

    # Use minimal settings for faster testing
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.placement.critic_num_gpus_per_node = 2
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.critic_num_nodes = 1
    cfg.trainer.algorithm.use_kl_loss = (
        False  # disable ref model so we just have policy and critic (4 GPUs)
    )
    cfg.trainer.placement.colocate_all = False  # Disable colocation for simpler testing
    cfg.trainer.train_batch_size = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.epochs = 1
    cfg.trainer.logger = "console"
    cfg.generator.n_samples_per_prompt = 1
    cfg.generator.num_inference_engines = 1
    cfg.generator.inference_engine_tensor_parallel_size = 2

    # Use temporary directories
    cfg.trainer.export_path = tempfile.mkdtemp(prefix="trainer_ckpt_test_")
    cfg.trainer.ckpt_path = cfg.trainer.export_path

    # Enable checkpointing with correct config names
    cfg.trainer.ckpt_interval = 1  # Save every step
    cfg.trainer.resume_mode = "none"  # Initially false, will be set to True for resume

    return cfg


def create_minimal_trainer(cfg: DictConfig):
    """Create a minimal trainer setup for testing"""
    # Create minimal tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Create dummy dataset
    train_dataset = DummyDataset(size=4)  # Small dataset for quick testing

    # Create mock generator for testing
    mock_generator = MagicMock()

    # Create tracker
    tracker = Tracking(
        project_name=cfg.trainer.project_name,
        experiment_name=cfg.trainer.run_name,
        default_backend=cfg.trainer.logger,
        config=cfg,
    )

    # Create trainer (no inference engine needed for checkpointing tests)
    trainer = RayPPOTrainer(
        cfg=cfg,
        tracker=tracker,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        inference_engine_client=None,
        generator=mock_generator,
    )

    return trainer


def capture_training_state(trainer):
    """Capture comprehensive training state for comparison"""
    state = {}

    # Capture trainer attributes
    state["global_step"] = trainer.global_step

    return state


@pytest.mark.parametrize(
    ("strategy, fsdp2_cpu_offload"),
    [
        ("deepspeed", False),
        ("fsdp", False),
        ("fsdp2", False),
        ("fsdp2", True),
    ],
)
def test_trainer_full_checkpointing(ray_init_fixture, strategy, fsdp2_cpu_offload):
    """
    Test full trainer checkpointing by:
    1. Creating trainer and setting it up
    2. Saving checkpoint
    3. Capturing training state
    4. Destroying trainer
    5. Creating new trainer with resume enabled
    6. Loading checkpoint
    7. Verifying all state matches
    8. Continuing training to ensure it works
    """
    cfg = get_test_trainer_config(strategy, fsdp2_cpu_offload)

    checkpoint_dir = None
    try:
        # ============= PHASE 1: Initial Training and Save =============
        print("Phase 1: Initial training and checkpoint save")

        trainer1 = create_minimal_trainer(cfg)

        # Get worker classes
        PolicyWorker = import_worker(strategy, "policy")
        CriticWorker = import_worker(strategy, "critic")
        RefWorker = import_worker(strategy, "ref")
        RewardWorker = import_worker(strategy, "reward")

        # Build models
        trainer1.build_models(PolicyWorker, CriticWorker, RefWorker, RewardWorker)

        # Set initial global step and simulate training state
        trainer1.global_step = 2  # Simulate having done 2 steps

        # Save checkpoint
        trainer1.save_checkpoints()

        # Capture state before teardown
        state_before = capture_training_state(trainer1)
        checkpoint_dir = os.path.join(
            cfg.trainer.export_path, f"global_step_{trainer1.global_step}"
        )

        # Verify checkpoint structure was created
        expected_files = [
            os.path.join(checkpoint_dir, "policy"),
            os.path.join(checkpoint_dir, "critic"),
            os.path.join(checkpoint_dir, "trainer_state.pt"),
            os.path.join(checkpoint_dir, "data.pt"),
        ]
        for expected_file in expected_files:
            assert os.path.exists(
                expected_file
            ), f"Expected checkpoint file/dir not found: {expected_file}"

        # Verify atomic tracking file
        latest_ckpt_file = os.path.join(
            cfg.trainer.ckpt_path, "latest_ckpt_global_step.txt"
        )
        assert os.path.exists(latest_ckpt_file)
        with open(latest_ckpt_file, "r") as f:
            latest_step = int(f.read())
        assert (
            latest_step == trainer1.global_step
        ), "Atomic tracking file has incorrect step after first save"

        # Verify trainer state content
        print("Verifying checkpoint content...")
        loaded_trainer_state = torch.load(
            os.path.join(checkpoint_dir, "trainer_state.pt"),
            map_location="cpu",
            weights_only=False,
        )

        # Check key configuration values are preserved
        assert (
            loaded_trainer_state["config"]["trainer"]["train_batch_size"]
            == cfg.trainer.train_batch_size
        ), "train_batch_size not preserved in checkpoint"
        assert (
            loaded_trainer_state["config"]["trainer"]["strategy"] == strategy
        ), "strategy not preserved in checkpoint"
        assert (
            loaded_trainer_state["global_step"] == trainer1.global_step
        ), "global_step not preserved in checkpoint"

        # Cleanup first trainer
        del trainer1
        ray.shutdown()

        # ============= PHASE 2: Resume from Checkpoint =============
        print("Phase 2: Resume from checkpoint")
        ray_init_for_tests()
        # Create new config with resume enabled
        cfg_resume = get_test_trainer_config(strategy, fsdp2_cpu_offload)
        cfg_resume.trainer.resume_mode = "from_path"  # Enable resume
        cfg_resume.trainer.resume_path = checkpoint_dir  # Set resume path
        cfg_resume.trainer.export_path = cfg.trainer.export_path  # Use same export path
        cfg_resume.trainer.ckpt_path = cfg.trainer.ckpt_path

        trainer2 = create_minimal_trainer(cfg_resume)

        # Build models again
        trainer2.build_models(PolicyWorker, CriticWorker, RefWorker, RewardWorker)

        # Load checkpoints
        loaded_global_step = trainer2.load_checkpoints()
        assert (
            loaded_global_step == 2
        ), f"Expected global_step=2, got {loaded_global_step}"

        # Capture state after loading
        state_after = capture_training_state(trainer2)

        # ============= PHASE 3: Verify State Matches =============
        print("Phase 3: Verify state consistency")

        # Compare captured states
        for key in state_before:
            assert (
                state_after[key] == state_before[key]
            ), f"State mismatch for {key}: before={state_before[key]}, after={state_after[key]}"

        # ============= PHASE 4: Continue Training =============
        print("Phase 4: Second checkpoint save")

        # Try to save another checkpoint to test cleanup logic
        trainer2.global_step = 3
        trainer2.save_checkpoints()

        next_checkpoint_dir = os.path.join(
            cfg.trainer.export_path, f"global_step_{trainer2.global_step}"
        )
        assert os.path.exists(
            next_checkpoint_dir
        ), "Could not save checkpoint after resume"

        # Verify atomic tracking file is updated
        latest_ckpt_file = os.path.join(
            cfg.trainer.ckpt_path, "latest_ckpt_global_step.txt"
        )
        assert os.path.exists(latest_ckpt_file)
        with open(latest_ckpt_file, "r") as f:
            latest_step = int(f.read())
        assert (
            latest_step == trainer2.global_step
        ), "Atomic tracking file was not updated after second save"

    finally:
        if checkpoint_dir and os.path.exists(os.path.dirname(checkpoint_dir)):
            print(
                f"Cleaning up checkpoint directory: {os.path.dirname(checkpoint_dir)}"
            )
            shutil.rmtree(os.path.dirname(checkpoint_dir))
