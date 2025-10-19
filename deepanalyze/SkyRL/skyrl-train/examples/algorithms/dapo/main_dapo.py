"""
uv run --isolated --extra vllm -m examples.algorithms.dapo.main_dapo
"""

import ray
import hydra
import torch
from typing import List
from omegaconf import DictConfig
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg

from skyrl_train.generators.base import GeneratorOutput


class DAPOTrainer(RayPPOTrainer):
    """
    Custom trainer for DAPO.

    Overrides the postprocess_generator_output method to additionally apply soft overlong punishment to rewards.
    """

    @torch.no_grad()
    def postprocess_generator_output(self, generator_output: GeneratorOutput, uids: List[str]) -> GeneratorOutput:
        """
        Overrides the postprocess_generator_output method to additionally apply DAPO specific soft overlong punishment to rewards.

        Args:
            generator_output: GeneratorOutput
            uids: List[str]

        Returns:
            GeneratorOutput
        """
        overlong_buffer_len = self.cfg.trainer.algorithm.overlong_buffer.len
        overlong_buffer_penalty_factor = self.cfg.trainer.algorithm.overlong_buffer.penalty_factor
        # modify rewards here
        prompt_token_ids = generator_output["prompt_token_ids"]
        response_ids = generator_output["response_ids"]
        rewards = generator_output["rewards"]

        assert not isinstance(rewards[0], list), "we assume verifiable sequence level rewards here"

        # get the prompt length
        prompt_lengths = [len(prompt) for prompt in prompt_token_ids]

        # get the response length
        response_lengths = [len(response) for response in response_ids]

        # get the max context length
        max_context_length = (
            self.cfg.generator.max_input_length + self.cfg.generator.sampling_params.max_generate_length
        )

        # apply soft overlong punishment
        for i, (prompt_length, response_length) in enumerate(zip(prompt_lengths, response_lengths)):
            # max_exceed_length is the beginning of the overlong buffer
            max_exceed_length = max_context_length - overlong_buffer_len - prompt_length
            # if the response is within the overlong buffer, apply the penalty
            if response_length > max_exceed_length and response_length <= max_context_length - prompt_length:
                exceed_length = response_length - max_exceed_length
                penalty = exceed_length / overlong_buffer_len * overlong_buffer_penalty_factor

                rewards[i] -= penalty
            # if the response is outside the overlong buffer, set the reward to 0
            elif response_length > max_context_length - prompt_length:
                # if self.cfg.generator.apply_overlong_filtering is true, loss masks are already set to 0 for these responses
                rewards[i] = 0.0

        generator_output["rewards"] = rewards

        # use base class impl for metrics and per-token reward conversion
        return super().postprocess_generator_output(generator_output, uids)


class DAPOExp(BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return DAPOTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    exp = DAPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
