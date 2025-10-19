import asyncio
import traceback
import sys
from loguru import logger
from skyrl_train.trainer import RayPPOTrainer
from tqdm import tqdm
from skyrl_train.utils import Timer
from skyrl_train.utils.ppo_utils import normalize_advantages_dict
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.generators.base import GeneratorOutput
from skyrl_train.utils.trainer_utils import ResumeMode
from skyrl_train.weights_manager import InferenceWeightsManager


class AsyncRayPPOTrainer(RayPPOTrainer):

    async def train(self):
        """
        Main training loop for PPO
        """
        assert (
            not self.cfg.trainer.placement.colocate_all
        ), "colocate_all is not supported for async training"

        self.weights_manager = InferenceWeightsManager(
            self.policy_model, self.inference_engine_client, False
        )
        # With non-colocated training, the eval weights manager is no-op, but we keep it for consistency
        self.eval_weights_manager = InferenceWeightsManager(
            self.policy_model, self.inference_engine_client, False, no_sync=True
        )

        self.global_step = 0

        # Load checkpoint state if resumption is enabled
        if self.resume_mode != ResumeMode.NONE:
            with Timer("load_checkpoints"):
                self.load_checkpoints()
                logger.info(f"Resumed training from global_step {self.global_step}")

        # create rank0 policy model and inference_engines groups, then broadcast weights to inference_engines
        with Timer("setup_policy_and_generator"):
            self.setup_policy_and_generator()

        # Eval before training
        if self.cfg.trainer.eval_interval > 0 and self.cfg.trainer.eval_before_train:
            with self.eval_weights_manager:
                with Timer("eval", self.all_timings):
                    eval_metrics = await self.eval()
                    self.tracker.log(eval_metrics, step=self.global_step)

        # main training loop
        pbar = tqdm(
            total=self.total_training_steps,
            initial=self.global_step,
            desc="Training Step Progress",
        )
        # Start from step 1
        self.global_step += 1
        for epoch in range(self.cfg.trainer.epochs):
            # while this is just off by one, you can image a more general queue based approach
            # where the generation buffer holds a list of objects that the trainer can read from
            # bit by bit.
            generation_buffer = asyncio.Queue(maxsize=1)
            self.sync_finished = asyncio.Event()
            self.generation_ack = asyncio.Event()

            # start generator task
            generator_task = asyncio.create_task(
                self._run_generate_loop(generation_buffer)
            )

            for idx in range(len(self.train_dataloader)):
                with Timer("step", self.all_timings):
                    status = await self._run_training(generation_buffer)

                    # request the generation loop that we should sync sometime soon.
                    if idx != len(self.train_dataloader) - 1:
                        await self.generation_ack.wait()

                    # sync weights
                    async with Timer("sync_weights", self.all_timings):
                        await self.weights_manager.async_sync_policy_weights_to_inference_engines()

                    self.sync_finished.set()
                    self.generation_ack.clear()

                # 5. set logs
                logger.info(status)
                # log epoch info
                self.all_metrics.update(
                    {"trainer/epoch": epoch, "trainer/global_step": self.global_step}
                )
                self.tracker.log(self.all_metrics, step=self.global_step)
                self.all_metrics = {}
                pbar.update(1)

                if self.cfg.trainer.eval_interval > 0 and (
                    self.global_step % self.cfg.trainer.eval_interval == 0
                    or self.global_step == self.total_training_steps
                ):
                    with self.eval_weights_manager:
                        with Timer("eval", self.all_timings):
                            eval_metrics = await self.eval()
                            self.all_metrics.update(eval_metrics)
                if (
                    self.cfg.trainer.ckpt_interval > 0
                    and self.global_step % self.cfg.trainer.ckpt_interval == 0
                ):
                    with Timer("save_checkpoints", self.all_timings):
                        self.save_checkpoints()
                if (
                    self.cfg.trainer.hf_save_interval > 0
                    and self.global_step % self.cfg.trainer.hf_save_interval == 0
                ):
                    with Timer("save_hf_model", self.all_timings):
                        self.save_models()
                self.tracker.log(
                    {"timing/" + k: v for k, v in self.all_timings.items()},
                    step=self.global_step,
                )
                self.all_timings = {}
                self.global_step += 1

            if self.cfg.trainer.update_ref_every_epoch and self.ref_model is not None:
                with Timer("update_ref_with_policy", self.all_timings):
                    await asyncio.to_thread(self.update_ref_with_policy)

            # cancel generation task for this epoch
            generator_task.cancel()

        pbar.close()
        if self.cfg.trainer.ckpt_interval > 0:
            with Timer("save_checkpoints", self.all_timings):
                self.save_checkpoints()
                logger.info("Saved final checkpoint.")
        if self.cfg.trainer.hf_save_interval > 0:
            with Timer("save_hf_model", self.all_timings):
                self.save_models()
                logger.info("Saved final model.")
        logger.info("Training done!")

    async def _run_training(self, generation_buffer):
        # Get a generation future and await on the object
        generator_output, uids = (
            await generation_buffer.get()
        )  # GeneratorOutput, List[str]

        # print example just for debugging
        vis = self.tokenizer.decode(generator_output["response_ids"][0])
        print("example: ", vis)

        with Timer("convert_to_training_input", self.all_timings):
            training_input: TrainingInputBatch = self.convert_to_training_input(
                generator_output, uids
            )

        # inference and calculate values, log probs, rewards, kl divergence
        with Timer("fwd_logprobs_values_reward", self.all_timings):
            training_input = await asyncio.to_thread(
                self.fwd_logprobs_values_reward, training_input
            )

        # calculate kl divergence and create experiences
        if self.cfg.trainer.algorithm.use_kl_in_reward:
            with Timer("apply_reward_kl_penalty", self.all_timings):
                training_input = self.apply_reward_kl_penalty(training_input)

        # calculate advantages and returns / along with tensorboard logging
        with Timer("compute_advantages_and_returns", self.all_timings):
            training_input = self.compute_advantages_and_returns(training_input)
            # remove some unwanted keys
            for key in ["custom_rewards", "rm_rewards"]:
                training_input.pop(key)
            training_input.metadata.pop("uids")

            if self.cfg.trainer.algorithm.advantage_batch_normalize:
                training_input = normalize_advantages_dict(training_input)

        if self.cfg.trainer.dump_data_batch:
            # dump data to file
            with Timer("dump_data_batch"):
                self.dump_data(
                    training_input,
                    file_name=f"global_step_{self.global_step}_training_input",
                )

        # train policy/critic model
        with Timer("train_critic_and_policy", self.all_timings):
            status = await asyncio.to_thread(
                self.train_critic_and_policy, training_input
            )

        return status

    async def _run_generate_loop(self, generation_buffer: asyncio.Queue):
        try:
            for i, rand_prompts in enumerate(self.train_dataloader):
                # truncate data to have even shards
                rand_prompts = self._remove_tail_data(rand_prompts)
                generator_input, uids = self._prepare_generator_input(
                    self.cfg.generator.n_samples_per_prompt, rand_prompts
                )

                # generation phase
                async with Timer("generate", self.all_timings):
                    generator_output: GeneratorOutput = await self.generate(
                        generator_input
                    )
                    generator_output = self.postprocess_generator_output(
                        generator_output, uids
                    )

                # Add to generation buffer
                await generation_buffer.put((generator_output, uids))

                # If the buffer is full, start weight sync
                # Don't weight sync in the first step, because we let the generator run one step ahead
                if generation_buffer.full() and i != 0:
                    # Signal that generation is done, ready for weight sync
                    self.generation_ack.set()
                    await self.sync_finished.wait()
                    # Clear the sync request for next sync
                    self.sync_finished.clear()
        # We have an explicit try-catch here because asyncio doesn't propagate exceptions to the main thread.
        except Exception as e:
            logger.error(f"Generator errored out with exception: {e}")
            logger.error(f"Traceback: \n{traceback.format_exc()}")
            sys.exit(1)
