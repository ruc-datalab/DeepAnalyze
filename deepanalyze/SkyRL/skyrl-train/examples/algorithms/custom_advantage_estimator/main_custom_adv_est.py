"""
uv run --isolated --extra vllm -m examples.algorithm.custom_advantage_estimator.main_custom_adv_est
"""

import ray
import hydra
import torch
import numpy as np
from omegaconf import DictConfig
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.utils.ppo_utils import AdvantageEstimatorRegistry


# Example of custom advantage estimator: "simple_baseline"
def compute_simple_baseline_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, **kwargs
):
    """
    A simple custom advantage estimator that uses response-level rewards
    and computes advantages against a simple baseline.

    This is just an example - replace with your own logic.
    """
    with torch.no_grad():
        response_rewards = (token_level_rewards * response_mask).sum(dim=-1, keepdim=True)

        # Simple baseline: use the mean reward across the batch
        baseline = response_rewards.mean()
        advantages = (response_rewards - baseline) * response_mask
        returns = advantages.clone()

        return advantages, returns


# Register the custom advantage estimator
AdvantageEstimatorRegistry.register("simple_baseline", compute_simple_baseline_advantage)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    exp = BasePPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
