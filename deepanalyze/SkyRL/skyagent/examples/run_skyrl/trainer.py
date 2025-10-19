from typing import List, Dict, Any, Optional, Tuple

from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl_train.generators.base import (
    GeneratorInput,
    GeneratorOutput,
    GeneratorInterface,
)
from skyrl_train.dataset.preprocess import (
    convert_prompts_responses_to_batch_tensors,
)
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
import torch

import uuid


class SkyAgentPPOTrainer(RayPPOTrainer):
    @torch.no_grad()
    async def generate(
        self,
        input_batch: GeneratorInput,
    ) -> GeneratorOutput:
        """
        Generate rollouts.

        If colocate_all is enabled:
        - before calling this method, the policy model should be on CPU and inference engine should
            be awake (i.e. on GPU).
        - after calling this method, the same model placement still holds.
        """
        generator_output: GeneratorOutput = await self.generator.generate(input_batch)

        # add rollout metrics to self.all_metrics
        if generator_output["rollout_metrics"] is not None:
            self.all_metrics.update(generator_output["rollout_metrics"])

        if len(generator_output["response_ids"]) <= 0:
            raise RuntimeError("No outputs generated")

        # remove the assert
        # assert len(input_batch["prompts"] * self.cfg.generator.n_samples_per_prompt) == len(
        #     generator_output["response_ids"]
        # ), f"generate objects number must be equal to all inputs number, got {len(input_batch['prompts'])} and {len(generator_output['response_ids'])}"

        return generator_output

    def _prepare_generator_input(
        self,
        n_samples_per_prompt: int,
        rand_prompts: List[Any],
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[GeneratorInput, List[str]]:
        """
        Replicate prompts if needed and generate uids.
        """
        # uids for each sample - NOTE: we assume that generate returns samples in the same order as passed in
        uids = sum(
            [[str(uuid.uuid4())] * n_samples_per_prompt for _ in rand_prompts], []
        )
        n_samples_per_prompt = 1

        all_prompts = sum(
            [[prompt["prompt"]] * n_samples_per_prompt for prompt in rand_prompts], []
        )

        all_envs = sum(
            [
                [
                    (
                        prompt["env_class"]
                        if prompt["env_class"] is not None
                        else self.cfg.environment.env_class
                    )
                ]
                * self.cfg.generator.n_samples_per_prompt
                for prompt in rand_prompts
            ],
            [],
        )

        # all the other columns are env_extras
        env_extras = sum(
            [[prompt["env_extras"]] * n_samples_per_prompt for prompt in rand_prompts],
            [],
        )
        request_sampling_params = (
            get_sampling_params_for_backend(self.cfg.generator.backend, sampling_params)
            if sampling_params is not None
            else None
        )
        generator_input: GeneratorInput = {
            "prompts": all_prompts,
            "env_classes": all_envs,
            "env_extras": env_extras,
            "sampling_params": request_sampling_params,
        }

        return generator_input, uids
