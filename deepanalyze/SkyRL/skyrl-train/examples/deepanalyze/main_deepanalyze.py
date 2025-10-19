"""
uv run --isolated --extra vllm -m examples.llm_as_a_judge.main_llm_judge
"""

import ray
import hydra
from omegaconf import DictConfig
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_gym.envs import register
import skyrl_gym


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # Register the llm_as_a_judge environment inside the entrypoint task (no need to modify the skyrl-gym package).
    register(
        id="deepanalyze",
        entry_point="examples.deepanalyze.deepanalyze_env:DeepAnalyzeEnv",
    )
    skyrl_gym.pprint_registry()
    # import pdb;pdb.set_trace()

    # make sure that the training loop is not run on the head node.
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
