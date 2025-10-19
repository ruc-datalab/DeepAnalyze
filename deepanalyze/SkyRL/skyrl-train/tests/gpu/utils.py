import os
import ray
import torch
import time
import requests
import importlib
from loguru import logger
from ray.util.placement_group import placement_group
from omegaconf import DictConfig
import hydra
from typing import List
from transformers import AutoTokenizer
from functools import lru_cache

from skyrl_train.dataset.replay_buffer import Experience
from skyrl_train.workers.worker import PPORayActorGroup
from skyrl_train.dataset import PromptDataset
from skyrl_train.training_batch import (
    TensorBatch,
    TrainingInputBatch,
    TrainingOutputBatch,
)
from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.utils import get_ray_pg_ready_with_timeout
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.generators.base import GeneratorInput, ConversationType
from skyrl_train.utils.utils import peer_access_supported


TEST_DATA_PATH = os.path.expanduser("~/data/gsm8k/validation.parquet")


def get_test_actor_config() -> DictConfig:
    """Get base config with test-specific overrides."""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

        cfg.trainer.policy.model.path = "Qwen/Qwen2.5-0.5B-Instruct"

        return cfg


def make_dummy_tensorbatch(seq_len=10, num_actions=4) -> TensorBatch:
    B, T = 2, seq_len
    data = TensorBatch(
        sequences=torch.ones(B, T, dtype=int, device="cpu"),
        attention_mask=torch.ones(B, T, dtype=int, device="cpu"),
    )
    data.metadata = {"response_length": num_actions}
    return data


def make_dummy_training_batch(
    batch_size=2, seq_len=10, num_actions=4
) -> TrainingInputBatch:
    """Create a dummy TrainingInputBatch"""

    torch.manual_seed(42)

    # Add all the required fields for training
    data = TrainingInputBatch(
        {
            "sequences": torch.randint(0, 100, (batch_size, seq_len), device="cpu"),
            "attention_mask": torch.ones(
                (batch_size, seq_len), dtype=int, device="cpu"
            ),
            "action_log_probs": 0.4
            * torch.ones((batch_size, num_actions), device="cpu"),
            "base_action_log_probs": 0.3
            * torch.ones((batch_size, num_actions), device="cpu"),
            "values": 0.5 * torch.ones((batch_size, num_actions), device="cpu"),
            "returns": 0.5 * torch.ones((batch_size, num_actions), device="cpu"),
            "advantages": 0.6 * torch.ones((batch_size, num_actions), device="cpu"),
            "loss_mask": torch.ones((batch_size, num_actions), dtype=int, device="cpu"),
            "response_mask": torch.ones(
                (batch_size, num_actions), dtype=int, device="cpu"
            ),
        }
    )
    data.metadata = {"response_length": num_actions}
    return data


def make_dummy_experience(seq_len=10, num_actions=4) -> Experience:
    torch.manual_seed(42)
    B, T = 2, seq_len
    num_actions = num_actions

    return Experience(
        sequences=torch.randint(0, 100, (B, T), device="cpu"),
        action_log_probs=0.4 * torch.ones((B, num_actions), device="cpu"),
        base_action_log_probs=0.3 * torch.ones((B, num_actions), device="cpu"),
        values=0.5 * torch.ones((B, num_actions), device="cpu"),
        returns=0.5 * torch.ones((B, num_actions), device="cpu"),
        advantages=0.6 * torch.ones((B, num_actions), device="cpu"),
        attention_mask=torch.ones((B, T), dtype=int, device="cpu"),
        loss_mask=torch.ones((B, num_actions), dtype=int, device="cpu"),
        action_mask=torch.ones((B, num_actions), dtype=int, device="cpu"),
        num_actions=num_actions,
        info={},
    )


def get_test_deepspeed_strategy(cfg):
    from skyrl_train.distributed.deepspeed_strategy import DeepspeedStrategy

    return DeepspeedStrategy(
        seed=42,
        micro_train_batch_size_per_gpu=1,
        train_batch_size=128,
        zero_stage=3,
        bf16=True,
        cfg=cfg,
    )


def get_test_fsdp_strategy(cfg):
    from skyrl_train.distributed.fsdp_strategy import FSDPStrategy

    return FSDPStrategy(
        seed=42,
        max_norm=1.0,
        micro_train_batch_size_per_gpu=1,
        train_batch_size=128,
        cfg=cfg,
    )


def import_worker(strategy: str, worker_type: str):
    if strategy == "deepspeed":
        module_path = "skyrl_train.workers.deepspeed.deepspeed_worker"
    elif strategy in ("fsdp", "fsdp2"):
        module_path = "skyrl_train.workers.fsdp.fsdp_worker"
    else:
        raise ValueError(f"Unknown strategy type for {worker_type}: {strategy}")

    module = importlib.import_module(module_path)
    return getattr(module, f"{worker_type.capitalize()}Worker")


def init_worker_with_type(
    worker_type: str, shared_pg=None, colocate_all=False, num_gpus_per_node=1, cfg=None
) -> PPORayActorGroup:
    if cfg is None:
        cfg = get_test_actor_config()

    if shared_pg is not None:
        pg = shared_pg
        num_gpus_per_actor = 0.2
    else:
        bundles = [{"GPU": num_gpus_per_node, "CPU": num_gpus_per_node}]
        pg = placement_group(bundles, strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=30)
        num_gpus_per_actor = 0.75

    worker_cls = import_worker(cfg.trainer.strategy, worker_type)
    model = PPORayActorGroup(
        cfg,
        num_nodes=1,  # single node for testing
        num_gpus_per_node=num_gpus_per_node,
        ray_actor_type=worker_cls,
        pg=pg,
        num_gpus_per_actor=num_gpus_per_actor,
        colocate_all=colocate_all,
        sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
        record_memory=cfg.trainer.policy.record_memory,
    )
    # we use policy model path for all tests (regardless of actor type)
    ray.get(model.async_init_model(cfg.trainer.policy.model.path))
    return model


class Timer:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.opt(depth=1).info(
            f"{self.message}, time cost: {time.time() - self.start_time:.2f}s"
        )


def get_available_gpus():
    """Get list of available GPU IDs from CUDA_VISIBLE_DEVICES or all available GPUs"""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        # Parse CUDA_VISIBLE_DEVICES (can be comma-separated list)
        gpu_ids = [
            int(x.strip()) for x in cuda_visible.split(",") if x.strip().isdigit()
        ]
        return gpu_ids
    else:
        # If not set, warn user but proceed with all GPUs
        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_ids = list(range(gpu_count))
                print(
                    f"CUDA_VISIBLE_DEVICES not set. Using all {gpu_count} GPUs: {gpu_ids}"
                )
                print(
                    "This might conflict with other processes. Consider setting CUDA_VISIBLE_DEVICES explicitly."
                )
                return gpu_ids
            else:
                return []
        except Exception as e:
            print(f"Error getting available GPUs: {e}")
            return []


def wait_for_server(
    url: str, health_path: str, timeout: int = 60, interval: float = 1.0
):
    start_time = time.time()
    while True:
        try:
            response = requests.get(f"http://{url}/{health_path}")
            if response.ok:
                return
        except requests.exceptions.ConnectionError:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Server at {url} did not come online within {timeout} seconds"
                )
            time.sleep(interval)


def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    # Initialize matrix of zeros
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # Initialize first column and first row of the matrix
    for i in range(m + 1):
        dp[i][0] = i  # Deletion from s1 to empty string
    for j in range(n + 1):
        dp[0][j] = j  # Insertion to s1 from empty string
    # Compute the Levenshtein distance matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1  # No cost if characters match
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )
    return dp[m][n]


def are_responses_similar(
    responses_a: List[str], responses_b: List[str], tolerance: float = 0.01
) -> float:
    if len(responses_a) != len(responses_b):
        return False

    total_length = 0
    total_diff = 0

    for s1, s2 in zip(responses_a, responses_b):
        max_len = max(len(s1), len(s2))
        total_length += max_len
        diff = levenshtein(s1, s2)
        total_diff += diff

    difference = float(total_diff / total_length)
    return difference <= tolerance


def get_test_prompts(model: str, num_samples: int = 20) -> List[ConversationType]:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    # Ensure pad_token is set correctly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = PromptDataset(
        [TEST_DATA_PATH],
        tokenizer,
        max_prompt_length=512,
    )

    # Extract the actual prompts from the dataset
    prompts = []
    for i in range(min(num_samples, len(dataset))):
        prompt_data, _, _ = dataset[i]  # dataset returns (messages, env_class, extra)
        prompts.append(prompt_data)

    return prompts


def get_test_generator_input(
    model: str,
    num_prompts: int = 20,
    n_samples_per_prompt: int = 1,
    max_prompt_length: int = 512,
    data_path: str = TEST_DATA_PATH,
    env_class: str = "gsm8k",
):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    # Ensure pad_token is set correctly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = PromptDataset(
        [data_path],
        tokenizer,
        max_prompt_length=max_prompt_length,
    )

    prompts = []
    env_extras = []
    for i in range(min(num_prompts, len(dataset))):
        prompt_data, _, env_extra = dataset[
            i
        ]  # dataset returns (messages, env_class, extra)
        prompts.extend([prompt_data] * n_samples_per_prompt)
        env_extras.extend([env_extra] * n_samples_per_prompt)

    env_classes = [env_class] * len(prompts)

    input_batch: GeneratorInput = {
        "prompts": prompts,
        "env_classes": env_classes,
        "env_extras": env_extras,
    }

    return input_batch


def get_model_logits_from_actor(
    actor_group: PPORayActorGroup, input_sequences, attention_mask
):
    """Helper function to get model logits for comparison"""

    seq_len = input_sequences.shape[1]
    num_actions_val = seq_len - 5  # Leave some tokens for response

    data = TrainingInputBatch(
        {
            "sequences": input_sequences,
            "attention_mask": attention_mask,
        }
    )
    data.metadata = {"response_length": num_actions_val}

    results_refs = actor_group.async_run_ray_method("mesh", "forward", data)
    results = ray.get(results_refs)
    ret_databatch: TrainingOutputBatch = concatenate_outputs_after_mesh_dispatch(
        actor_group.actor_infos, results
    )
    logits = ret_databatch["output"]

    return logits


@lru_cache(5)
def log_once(msg):
    logger.info(msg)
    return None


def ray_init_for_tests():
    env_vars = {}
    if not peer_access_supported(max_num_gpus_per_node=4):
        log_once("Disabling NCCL P2P for test environment")
        env_vars = {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1"}
    ray.init(runtime_env={"env_vars": env_vars})
