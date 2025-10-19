"""
uv run --isolated --extra vllm --extra dev -- pytest -s -vvv tests/gpu/test_grpo_sp_sanity.py
"""

import os
from hydra import initialize, compose
from omegaconf import DictConfig
from pathlib import Path
from loguru import logger
import numpy as np
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir
from skyrl_train.trainer import RayPPOTrainer
import ray
from tqdm import tqdm
from skyrl_train.utils import Timer
from skyrl_train.utils.ppo_utils import normalize_advantages_dict


import asyncio


class TestExp(BasePPOExp):
    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        return RayPPOTestTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


class RayPPOTestTrainer(RayPPOTrainer):
    def train(self):
        """
        Main training loop for PPO
        """
        self.all_metrics = {}
        self.all_timings = {}

        # create rank0 policy model and inference_engines groups, then broadcast weights to inference_engines
        with Timer("setup_policy_and_generator"):
            self.setup_policy_and_generator()

        # main training loop
        consumed_samples = 0
        num_rollouts_per_episodes = (
            self.num_update_steps_per_episodes
            // self.cfg.trainer.update_epochs_per_batch
            // self.cfg.generator.n_samples_per_prompt
        )

        self.global_step = consumed_samples // self.cfg.trainer.train_batch_size
        start_episode = (
            consumed_samples
            // self.cfg.trainer.train_batch_size
            // num_rollouts_per_episodes
        )
        consumed_samples = consumed_samples % (
            num_rollouts_per_episodes * self.cfg.trainer.train_batch_size
        )

        # Run just one iteration for testing
        for episode in range(start_episode, start_episode + 1):
            pbar = tqdm(
                range(self.train_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{self.cfg.trainer.num_episodes}]",
            )
            for iter, rand_prompts in enumerate(self.train_dataloader):
                with Timer("step", self.all_timings):
                    all_prompts = sum(
                        [
                            [prompt[0]] * self.cfg.generator.n_samples_per_prompt
                            for prompt in rand_prompts
                        ],
                        [],
                    )
                    all_extras = sum(
                        [
                            [prompt[1]] * self.cfg.generator.n_samples_per_prompt
                            for prompt in rand_prompts
                        ],
                        [],
                    )

                    # 1.1 generation phase
                    with Timer("generate", self.all_timings):
                        data = asyncio.run(self.generate(all_prompts, all_extras))

                    # 1.2 compute rewards
                    with Timer("compute_rewards", self.all_timings):
                        data = self.compute_rewards(data)
                        # keep only the keys needed later on
                        data = data.pop(
                            non_tensor_batch_keys=[
                                "response_ids",
                                "prompt_ids",
                                "loss_mask",
                                "custom_rewards",
                            ]
                        )

                    # 2. print example just for debugging
                    vis = self.tokenizer.decode(
                        data.non_tensor_batch["response_ids"][0]
                    )
                    print("example: ", vis)

                    with Timer("convert_to_batch_regular", self.all_timings):
                        data = self.convert_to_batch_regular(data)

                    # sequences are the full input ids for the model.
                    data = data.select(
                        batch_keys=[
                            "sequences",
                            "attention_mask",
                            "custom_rewards",
                            "loss_mask",
                            "response_mask",
                        ],
                        non_tensor_batch_keys=["response_ids"],
                    )

                    # 1.4 inference and calculate values, log probs, rewards, kl divergence
                    with Timer("fwd_logprobs_values_reward", self.all_timings):
                        data = self.fwd_logprobs_values_reward(data)

                    # 1.5 calculate kl divergence and create experiences
                    with Timer("calc_values_logprobs_rewards_kl", self.all_timings):
                        data = self.calculate_kl(data)
                        logger.info(f"Number of sequences: {len(data['sequences'])}")

                    # 3. calculate advantages and returns / along with tensorboard logging
                    with Timer("calc_advantages_and_returns", self.all_timings):
                        data = self.compute_advantages_and_returns(data)
                        # remove some unwanted keys
                        data.pop(batch_keys=["custom_rewards", "rm_rewards"])

                        if self.cfg.trainer.algorithm.advantage_batch_normalize:
                            data = normalize_advantages_dict(data)

                    # 4. train policy/critic model
                    with Timer("train_critic_and_policy", self.all_timings):
                        status = self.train_critic_and_policy(data)

                # 5. set logs
                logger.info(status)
                pbar.update()
                # log epoch info
                self.all_metrics.update(
                    {"trainer/epoch": episode, "step": self.global_step}
                )
                self.tracker.log(self.all_metrics, step=self.global_step)
                self.tracker.log(
                    {"timing/" + k: v for k, v in self.all_timings.items()},
                    step=self.global_step,
                )
                self.global_step += 1

                # Return metrics after one iteration
                return self.all_metrics


def run_exp_and_get_metrics(exp: BasePPOExp, cfg: DictConfig):
    metrics = exp.run()
    # ray shutdown will clear all state for the ray session
    ray.shutdown()
    return metrics


def run_with_hydra(func, config_name: str):
    current_directory = Path(__file__).parent.absolute()
    abs_config_dir = Path(config_dir).absolute()
    relative_config_dir = os.path.relpath(abs_config_dir, current_directory)
    print("relative_config_dir: ", relative_config_dir)
    with initialize(version_base=None, config_path=relative_config_dir):
        cfg = compose(config_name=config_name)
        func(cfg)


def ppo_run(cfg: DictConfig) -> None:
    # Configure test settings
    cfg.trainer.train_batch_size = 8
    cfg.trainer.num_episodes = 1
    cfg.trainer.policy_mini_batch_size = 8
    # use zero temperature for consistency.
    # We will anyways only check for log probability values so this is fine.
    cfg.generator.sampling_params.temperature = 0.0
    cfg.trainer.placement.policy_num_gpus_per_node = 4
    cfg.trainer.placement.critic_num_gpus_per_node = 4
    cfg.trainer.placement.ref_num_gpus_per_node = 4
    cfg.generator.num_inference_engines = 2
    cfg.generator.inference_engine_tensor_parallel_size = 2
    cfg.generator.gpu_memory_utilization = 0.7

    # Run baseline (no sequence parallel)
    cfg.trainer.policy.sequence_parallel_size = 1
    exp_baseline = TestExp(cfg)
    metrics_baseline = run_exp_and_get_metrics(exp_baseline, cfg)
    print("Baseline metrics: ", metrics_baseline)

    # Run with sequence parallel
    cfg.trainer.policy.sequence_parallel_size = 2
    exp_sp = TestExp(cfg)
    metrics_sp = run_exp_and_get_metrics(exp_sp, cfg)
    print("Metrics with sequence parallel: ", metrics_sp)

    # Compare policy entropy values
    # NOTE: typical values are ~ 0.225 and ~ 0.228
    # Some difference can be due to ignoring attention mask with seq parallelism
    np.testing.assert_allclose(
        metrics_sp["policy/policy_entropy"],
        metrics_baseline["policy/policy_entropy"],
        atol=5e-3,
    )


def test_ppo_run():
    run_with_hydra(ppo_run, "ppo_base_config")


if __name__ == "__main__":
    test_ppo_run()
