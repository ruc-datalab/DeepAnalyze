"""
Test save_hf_model and load_hf_model functionality for different strategies.

Run with:
uv run --isolated --extra dev --with deepspeed -- pytest tests/gpu/test_save_load_model.py
"""

import ray
import pytest
import hydra
import torch
import os
import shutil
import tempfile
from omegaconf import DictConfig
import json

from tests.gpu.utils import (
    init_worker_with_type,
    make_dummy_experience,
    get_model_logits_from_actor,
    ray_init_for_tests,
)
from skyrl_train.entrypoints.main_base import config_dir

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_ARCH = "Qwen2ForCausalLM"


def get_test_actor_config(strategy: str) -> DictConfig:
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.strategy = strategy

    # Use temporary directories for testing
    cfg.trainer.ckpt_path = tempfile.mkdtemp(prefix="model_test_ckpt_")
    cfg.trainer.export_path = tempfile.mkdtemp(prefix="model_test_save_")

    return cfg


@pytest.mark.parametrize(
    "strategy",
    [
        "deepspeed",
        "fsdp",
        "fsdp2",
    ],
)
def test_save_load_hf_model(ray_init_fixture, strategy):
    """
    Test save_hf_model functionality by:
    1. Loading a pretrained model into an ActorGroup
    2. Running a forward pass to get some outputs
    3. Saving model in HuggingFace format using save_hf_model
    4. Loading model from saved HuggingFace format and comparing outputs
    """
    cfg = get_test_actor_config(strategy)

    model_save_dir = None
    try:
        # ============= PHASE 1: Train and Save =============
        actor_group_1 = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        # Create dummy experience for training step
        dummy_experience = make_dummy_experience()
        global_step, local_step, accumulation_steps = 0, 0, 1

        # Step 1: Do one training step
        ray.get(
            actor_group_1.async_run_ray_method(
                "pass_through",
                "training_step",
                dummy_experience,
                global_step,
                local_step,
                accumulation_steps,
            )
        )

        # Step 2: Create test input and compute logits from trained model
        dp_size = actor_group_1.actor_infos[0].rank.dp_size
        test_input = torch.randint(
            0, 1000, (dp_size, 20), device="cpu"
        )  # batch_size=dp_size, seq_len=20
        attention_mask = torch.ones_like(test_input)

        logits_from_trained_model = get_model_logits_from_actor(
            actor_group_1, test_input, attention_mask
        )

        # Step 3: Save model in HuggingFace format
        export_dir = os.path.join(cfg.trainer.export_path, "global_step_1", "policy")
        ray.get(
            actor_group_1.async_run_ray_method(
                "pass_through", "save_hf_model", export_dir=export_dir, tokenizer=None
            )
        )

        # Verify that model files were saved
        model_save_dir = export_dir
        expected_files = [
            "config.json",
            "model.safetensors",
        ]  # Basic HuggingFace model files
        for expected_file in expected_files:
            file_path = os.path.join(model_save_dir, expected_file)
            assert os.path.exists(
                file_path
            ), f"Expected model file not found: {file_path}"

        with open(os.path.join(model_save_dir, "config.json"), "r") as f:
            config = json.load(f)
        assert config["architectures"] == [
            MODEL_ARCH
        ], "Architecture should be Qwen2ForCausalLM"

        # Step 4: Destroy first worker to ensure fresh weights.
        ray.shutdown()

        # ============= PHASE 2: Fresh Worker Loading from Saved Path =============
        ray_init_for_tests()
        # Create a new config that points to the saved model instead of the original model
        cfg_fresh = get_test_actor_config(strategy)
        # IMPT: Point to the saved model directory instead of original model
        cfg_fresh.trainer.policy.model.path = model_save_dir

        actor_group_2 = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg_fresh.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg_fresh,
        )

        # Step 5: Compute logits from worker that loaded the saved model
        logits_from_loaded_saved_model = get_model_logits_from_actor(
            actor_group_2, test_input, attention_mask
        )

        # Step 6: Compare logits - they should match the original trained model exactly
        torch.testing.assert_close(
            logits_from_trained_model,
            logits_from_loaded_saved_model,
            atol=1e-8,
            rtol=1e-8,
        )

    finally:
        # Clean up temporary directories
        for temp_dir in [
            cfg.trainer.ckpt_path,
            cfg.trainer.export_path,
            model_save_dir,
        ]:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
