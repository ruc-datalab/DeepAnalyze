from skyrl_train.trainer import RayPPOTrainer
from loguru import logger
import random
from skyrl_train.utils.utils import Timer


class FullCtxTrainer(RayPPOTrainer):
    """A dummy trainer that tests configurations with max sequence length.

    This trainer is meant to help users validate their configuration setup by:
    1. Creating max length sequences directly
    2. Running a few training steps

    This helps catch OOM issues early before running full training.
    """

    def train(self):
        """Run a few training steps with max sequence length."""
        logger.info("Starting dummy training with max sequence length...")

        self.global_step = 0

        # Setup policy and generator
        with Timer("setup_policy_and_generator", self.all_timings):
            self.setup_policy_and_generator()

        # Run a few training steps
        self.global_step += 1  # start from 1
        for step in range(self.cfg.trainer.num_dummy_steps):
            logger.info(
                f"Running dummy training step {step + 1}/{self.cfg.trainer.num_dummy_steps}"
            )

            # Run a single training step
            with Timer("step", self.all_timings):
                # Create training input directly with max length sequences
                num_samples = (
                    self.cfg.trainer.train_batch_size
                    * self.cfg.generator.n_samples_per_prompt
                )
                uids = [str(i) for i in range(self.cfg.trainer.train_batch_size)]
                prompt_token_ids = [
                    [random.randint(0, self.tokenizer.vocab_size - 1)]
                    * self.cfg.generator.max_input_length
                ] * self.cfg.trainer.train_batch_size
                prompt_token_ids = sum(
                    [
                        [prompt_token_id] * self.cfg.generator.n_samples_per_prompt
                        for prompt_token_id in prompt_token_ids
                    ],
                    [],
                )
                response_ids = [
                    [random.randint(0, self.tokenizer.vocab_size - 1)]
                    * self.cfg.generator.sampling_params.max_generate_length
                ] * num_samples
                uids = sum(
                    [[uid] * self.cfg.generator.n_samples_per_prompt for uid in uids],
                    [],
                )

                dummy_generator_output = {
                    "prompt_token_ids": prompt_token_ids,
                    "response_ids": response_ids,
                    "rewards": [
                        [0]
                        * (self.cfg.generator.sampling_params.max_generate_length - 1)
                        + [random.randint(0, 1)]
                    ]
                    * num_samples,
                    "loss_masks": [
                        [1] * self.cfg.generator.sampling_params.max_generate_length
                    ]
                    * num_samples,
                }
                training_input = self.convert_to_training_input(
                    dummy_generator_output, uids
                )

                # Run forward pass
                training_input = self.fwd_logprobs_values_reward(training_input)

                # 1.5 apply kl divergence penalty to rewards
                if self.cfg.trainer.algorithm.use_kl_in_reward:
                    with Timer("apply_reward_kl_penalty", self.all_timings):
                        training_input = self.apply_reward_kl_penalty(training_input)

                # 3. calculate advantages and returns
                with Timer("compute_advantages_and_returns", self.all_timings):
                    training_input = self.compute_advantages_and_returns(training_input)
                    # remove some unwanted keys
                    for key in ["custom_rewards", "rm_rewards"]:
                        training_input.pop(key)
                    training_input.metadata.pop("uids")

                # 4. train policy/critic model
                with Timer("train_critic_and_policy", self.all_timings):
                    status = self.train_critic_and_policy(training_input)

                self.tracker.log(self.all_metrics, step=self.global_step)
                self.all_metrics = {}
                self.tracker.log(
                    {"timing/" + k: v for k, v in self.all_timings.items()},
                    step=self.global_step,
                )
                self.all_timings = {}
                self.global_step += 1

                logger.info(f"Step {step + 1} completed. Status: {status}")

        logger.info("Dummy training completed successfully!")
