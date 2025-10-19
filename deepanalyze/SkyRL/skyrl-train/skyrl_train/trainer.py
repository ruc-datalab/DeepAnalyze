import asyncio
import math
import os
import shutil
from typing import Any, List, Optional, Dict, Tuple, Union
from jaxtyping import Float
from pathlib import Path
import ray
import uuid
import torch
from loguru import logger
from omegaconf import DictConfig
from ray.util.placement_group import PlacementGroup, placement_group
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from skyrl_train.dataset import PromptDataset
from skyrl_train.utils.tracking import Tracking
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl_train.generators.base import (
    GeneratorInput,
    GeneratorOutput,
    GeneratorInterface,
)
from skyrl_train.generators.utils import (
    concatenate_generator_outputs,
    get_metrics_from_generator_output,
)
from skyrl_train.dataset.preprocess import (
    convert_prompts_responses_to_batch_tensors,
)
from skyrl_train.utils import ppo_utils
from skyrl_train.utils import trainer_utils
from skyrl_train.utils import Timer, get_ray_pg_ready_with_timeout
from skyrl_train.utils.ppo_utils import (
    compute_approx_kl,
    masked_mean,
    get_kl_controller,
    FixedKLController,
    AdaptiveKLController,
    normalize_advantages_dict,
)
from skyrl_train.distributed.dispatch import (
    MeshRank,
    concatenate_outputs_after_mesh_dispatch,
    ActorInfo,
)
from skyrl_train.workers.worker import PPORayActorGroup
from skyrl_train.weights_manager import (
    InferenceWeightsManager,
    ConditionalWeightsManager,
)
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.utils.trainer_utils import (
    cleanup_old_checkpoints,
    run_on_each_node,
    get_node_ids,
    extract_step_from_path,
    validate_consistency_for_latest_checkpoint,
    calculate_per_dataset_metrics,
    dump_per_dataset_eval_results,
    validate_generator_output,
    GLOBAL_STEP_PREFIX,
    ResumeMode,
    DynamicSamplingState,
)


class RayPPOTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        tracker: Tracking,
        tokenizer: AutoTokenizer,
        train_dataset: PromptDataset,
        inference_engine_client: InferenceEngineClient,
        generator: GeneratorInterface,
        colocate_pg: Optional[PlacementGroup] = None,
        eval_dataset: Optional[PromptDataset] = None,
    ):
        self.cfg = cfg
        self.tracker = tracker
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.inference_engine_client = inference_engine_client
        self.generator = generator
        self.train_dataloader = self.build_dataloader(train_dataset, is_train=True)
        self.eval_dataloader = (
            self.build_dataloader(eval_dataset, is_train=False)
            if eval_dataset is not None
            else None
        )
        self.colocate_pg = colocate_pg

        self.resume_mode = ResumeMode(cfg.trainer.resume_mode)

        self.all_metrics = {}
        self.all_timings = {}

        # initialized in `build_models`
        self.policy_model: PPORayActorGroup = None
        self.critic_model: Optional[PPORayActorGroup] = None
        self.ref_model: Optional[PPORayActorGroup] = None
        self.reward_model: Optional[PPORayActorGroup] = None
        # used for checkpoint cleanup
        self._node_ids: Optional[List[str]] = None

        self.weights_manager: InferenceWeightsManager = None
        self.eval_weights_manager: InferenceWeightsManager = None

        self.dynamic_sampling_state: Optional[DynamicSamplingState] = None

        self.reward_kl_controller: Optional[
            Union[FixedKLController, AdaptiveKLController]
        ] = None

    def build_dataloader(self, dataset: PromptDataset, is_train=True):
        """
        Build the dataloader for the training or evaluation dataset
        """
        # prepare dataloader
        batch_size = (
            self.cfg.trainer.train_batch_size
            if is_train
            else self.cfg.trainer.eval_batch_size
        )

        # Seed the dataloader for reproducibility.
        seeded_generator = torch.Generator()
        seeded_generator.manual_seed(self.cfg.trainer.seed)

        dataloader = StatefulDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True if is_train else False,
            collate_fn=dataset.collate_fn,
            num_workers=8,
            drop_last=True if is_train else False,
            generator=seeded_generator,
        )
        if is_train:
            self.total_training_steps = len(dataloader) * self.cfg.trainer.epochs
            print(f"Total steps: {self.total_training_steps}")
        else:
            print(f"Validation set size: {len(dataloader)}")

        return dataloader

    @torch.no_grad()
    async def eval(self) -> Dict[str, float]:
        """
        Run generation and scoring on the evaluation dataset.

        The eval metrics are recorded after having finished training `self.global_step` steps.
        Metrics recorded in global_step 0 corresponds to evaluations before training.

        Returns:
            A dictionary of evaluation metrics.
        """
        # 0. Make a copy of self.all_metrics (will restore at the end)
        # eval() might accidentally mutate `self.all_metrics` since it is mutated in
        # methods like `self.generate()`.
        all_metrics_copy = self.all_metrics.copy()

        # 1. Get all generator outputs
        generator_outputs: List[GeneratorOutput] = []
        concat_all_envs: List[str] = []
        concat_env_extras: List[Dict[str, Any]] = []
        concat_uids: List[str] = []
        sampling_params = self.cfg.generator.eval_sampling_params
        pbar = tqdm(
            total=len(self.eval_dataloader), initial=0, desc="Evaluation Progress"
        )
        for _, prompts in enumerate(self.eval_dataloader):
            pbar.update(1)
            generator_input, uids = self._prepare_generator_input(
                self.cfg.generator.eval_n_samples_per_prompt, prompts, sampling_params
            )
            generator_output: GeneratorOutput = await self.generate(generator_input)
            generator_outputs.append(generator_output)
            concat_all_envs.extend(generator_input["env_classes"])
            concat_env_extras.extend(generator_input["env_extras"])
            concat_uids.extend(uids)
        concat_generator_outputs: GeneratorOutput = concatenate_generator_outputs(
            generator_outputs
        )

        # Extract data_sources from env_extras
        concat_data_sources = [
            env_extra.get("data_source") for env_extra in concat_env_extras
        ]
        vis = self.tokenizer.decode(generator_output["response_ids"][0])
        print("Eval output example: ", vis)

        # 2. Group data by data source and calculate per-dataset metrics
        eval_metrics = calculate_per_dataset_metrics(
            concat_generator_outputs,
            concat_uids,
            concat_data_sources,
            self.cfg.generator.eval_n_samples_per_prompt,
        )

        # 3. Calculate overall metrics across all datasets
        overall_avg_score, overall_pass_at_n = get_metrics_from_generator_output(
            concat_generator_outputs, concat_uids
        )
        eval_metrics.update(
            {
                "eval/all/avg_score": overall_avg_score,
                f"eval/all/pass_at_{self.cfg.generator.eval_n_samples_per_prompt}": overall_pass_at_n,
            }
        )

        # 4. Prepare dumping data
        if self.cfg.trainer.dump_eval_results:
            with Timer("dump_eval_results"):
                data_save_dir = (
                    Path(self.cfg.trainer.export_path)
                    / "dumped_evals"
                    / f"global_step_{self.global_step}_evals"
                )
                data_save_dir.mkdir(parents=True, exist_ok=True)
                dump_per_dataset_eval_results(
                    data_save_dir,
                    self.tokenizer,
                    concat_generator_outputs,
                    concat_data_sources,
                    concat_all_envs,
                    concat_env_extras,
                    eval_metrics,
                )

        # 5. Restore self.all_metrics
        self.all_metrics = all_metrics_copy

        return eval_metrics

    def train(self):
        """
        Main training loop for PPO
        """

        self.global_step = 0
        self.weights_manager = InferenceWeightsManager(
            self.policy_model,
            self.inference_engine_client,
            self.cfg.trainer.placement.colocate_all,
        )
        # NOTE(Charlie): sglang's engine needs to sync weights after wake up. see https://github.com/sgl-project/sglang/issues/7939
        # Change it to True after sglang fixes the issue.
        sync_before_eval = (
            self.cfg.trainer.placement.colocate_all
            and self.cfg.generator.backend == "sglang"
        )
        self.eval_weights_manager = InferenceWeightsManager(
            self.policy_model,
            self.inference_engine_client,
            self.cfg.trainer.placement.colocate_all,
            no_sync=not sync_before_eval,
        )

        # Load checkpoint state if resumption is enabled
        if self.resume_mode != ResumeMode.NONE:
            with Timer("load_checkpoints"):
                self.load_checkpoints()
                logger.info(f"Resumed training from global_step {self.global_step}")

        # create rank0 policy model and inference_engines groups, then broadcast weights to inference_engines
        with Timer("setup_policy_and_generator"):
            self.setup_policy_and_generator()
            if self.cfg.trainer.placement.colocate_all:
                self.policy_model.backload_to_gpu()

        # Eval before training
        if self.cfg.trainer.eval_interval > 0 and self.cfg.trainer.eval_before_train:
            with self.eval_weights_manager:
                with Timer("eval", self.all_timings):
                    eval_metrics = asyncio.run(self.eval())
                    self.tracker.log(eval_metrics, step=self.global_step)
            # Policy model is backloaded to GPU after eval
            if self.cfg.trainer.placement.colocate_all:
                self.policy_model.backload_to_gpu()

        # setup for dynamic sampling
        keep_sampling = False

        # initialize kl controller
        if self.cfg.trainer.algorithm.use_kl_in_reward:
            self.reward_kl_controller = get_kl_controller(self.cfg.trainer.algorithm)

        # main training loop
        pbar = tqdm(
            total=self.total_training_steps,
            initial=self.global_step,
            desc="Training Batches Processed",
        )
        self.global_step += 1  # start training at global_step 1
        for epoch in range(self.cfg.trainer.epochs):
            for iter, rand_prompts in enumerate(self.train_dataloader):
                with Timer("step", self.all_timings):

                    # 0. truncate data to have even shards
                    rand_prompts = self._remove_tail_data(rand_prompts)
                    generator_input, uids = self._prepare_generator_input(
                        self.cfg.generator.n_samples_per_prompt, rand_prompts
                    )

                    # if we are continuing sampling, we don't want to trigger weight management
                    weights_manager = ConditionalWeightsManager(
                        self.weights_manager, not keep_sampling
                    )

                    # NOTE: Policy model is on GPU at the beginning of each training step
                    # After exiting the context manager, policy model is on CPU with `colocate_all` enabled.
                    # Policy model stays on cpu because the training loop will carefully backload different models depending on colocation strategy
                    with weights_manager:
                        # 1.1 generation phase
                        with Timer("generate", self.all_timings):
                            generator_output: GeneratorOutput = asyncio.run(
                                self.generate(generator_input)
                            )

                        # dynamic sampling
                        if self.cfg.trainer.algorithm.dynamic_sampling.type is not None:
                            generator_output, uids, keep_sampling = (
                                self.handle_dynamic_sampling(generator_output, uids)
                            )
                            # update weights manager condition to ensure we trigger sleep only when we are not continuing sampling
                            weights_manager.update_condition(not keep_sampling)
                            if keep_sampling:  # continue sampling
                                # update progress bar for current batch (but not global step)
                                pbar.update(1)
                                continue

                    # 1.2 postprocess rewards
                    with Timer("postprocess_generator_output", self.all_timings):
                        generator_output = self.postprocess_generator_output(
                            generator_output, uids
                        )

                    # 2. print example just for debugging
                    vis = self.tokenizer.decode(generator_output["response_ids"][0])
                    print("example: ", vis)

                    torch.cuda.empty_cache()

                    with Timer("convert_to_training_input", self.all_timings):
                        training_input: TrainingInputBatch = (
                            self.convert_to_training_input(generator_output, uids)
                        )
                        logger.info(
                            f"Number of sequences: {len(training_input['sequences'])}"
                        )

                    # 1.4 inference and calculate values, log probs, rewards, kl divergence
                    with Timer("fwd_logprobs_values_reward", self.all_timings):
                        training_input = self.fwd_logprobs_values_reward(training_input)

                    # 1.5 apply kl divergence penalty to rewards
                    if self.cfg.trainer.algorithm.use_kl_in_reward:
                        with Timer("apply_reward_kl_penalty", self.all_timings):
                            training_input = self.apply_reward_kl_penalty(
                                training_input
                            )

                    # 3. calculate advantages and returns
                    with Timer("compute_advantages_and_returns", self.all_timings):
                        training_input = self.compute_advantages_and_returns(
                            training_input
                        )
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

                    # 4. train policy/critic model
                    # Policy model is backloaded to GPU during training
                    with Timer("train_critic_and_policy", self.all_timings):
                        status = self.train_critic_and_policy(training_input)

                # 5. set logs
                logger.info(status)
                # log epoch info
                self.all_metrics.update(
                    {"trainer/epoch": epoch, "trainer/global_step": self.global_step}
                )
                if self.cfg.trainer.eval_interval > 0 and (
                    self.global_step % self.cfg.trainer.eval_interval == 0
                    or self.global_step == self.total_training_steps
                ):
                    with self.eval_weights_manager:
                        with Timer("eval", self.all_timings):
                            eval_metrics = asyncio.run(self.eval())
                            self.all_metrics.update(eval_metrics)
                    # Policy model is backloaded to GPU after eval
                    if self.cfg.trainer.placement.colocate_all:
                        self.policy_model.backload_to_gpu()

                self.tracker.log(self.all_metrics, step=self.global_step)
                self.all_metrics = {}

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

                # update progress bar after logging
                pbar.update(1)

                self.global_step += 1

                del training_input, generator_output

            if self.cfg.trainer.update_ref_every_epoch and self.ref_model is not None:
                with Timer("update_ref_with_policy", self.all_timings):
                    self.update_ref_with_policy()

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

    def _remove_tail_data(self, entries: List[Any]) -> List[Any]:
        """Remove tail data to have even shards"""
        dp_size = self.policy_model.actor_infos[0].rank.dp_size
        if self.critic_model is not None:
            dp_size = math.lcm(dp_size, self.critic_model.actor_infos[0].rank.dp_size)
        if self.ref_model is not None:
            dp_size = math.lcm(dp_size, self.ref_model.actor_infos[0].rank.dp_size)
        if self.reward_model is not None:
            dp_size = math.lcm(dp_size, self.reward_model.actor_infos[0].rank.dp_size)
        return entries[: (len(entries) // dp_size) * dp_size]

    def _prepare_generator_input(
        self,
        n_samples_per_prompt: int,
        rand_prompts: List[Any],
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[GeneratorInput, List[str]]:
        """
        Replicate prompts if needed and generate uids.
        """
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

        # uids for each sample - NOTE: we assume that generate returns samples in the same order as passed in
        uids = sum(
            [[str(uuid.uuid4())] * n_samples_per_prompt for _ in rand_prompts], []
        )
        return generator_input, uids

    def build_models(self, PolicyWorker, CriticWorker, RefWorker, RewardWorker=None):
        """
        Initialize the actors for training, and handle colocation logic
        """
        cfg = self.cfg
        pg = None

        if RewardWorker is not None and cfg.trainer.reward.model.path:
            raise NotImplementedError("reward models are not supported yet")

        use_ref_model = (
            cfg.trainer.algorithm.use_kl_loss or cfg.trainer.algorithm.use_kl_in_reward
        )

        if cfg.trainer.placement.colocate_all:
            num_policy_gpus = (
                cfg.trainer.placement.policy_num_gpus_per_node
                * cfg.trainer.placement.policy_num_nodes
            )
            num_critic_gpus = (
                cfg.trainer.placement.critic_num_gpus_per_node
                * cfg.trainer.placement.critic_num_nodes
            )
            num_ref_gpus = (
                cfg.trainer.placement.ref_num_gpus_per_node
                * cfg.trainer.placement.ref_num_nodes
            )
            num_rollout_gpus = (
                cfg.generator.num_inference_engines
                * cfg.generator.inference_engine_tensor_parallel_size
            )
            assert (
                num_policy_gpus == num_rollout_gpus
            ), "num_policy_gpus and num_rollout_gpus must be the same when colocating all models"
            pg = self.colocate_pg

            policy_model = PPORayActorGroup(
                cfg,
                cfg.trainer.placement.policy_num_nodes,
                cfg.trainer.placement.policy_num_gpus_per_node,
                PolicyWorker,
                pg=pg,
                num_gpus_per_actor=0.2 if pg else 1,
                colocate_all=True,
                sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
                record_memory=cfg.trainer.policy.record_memory,
            )
            if use_ref_model:
                assert (
                    num_policy_gpus == num_ref_gpus
                ), "num_policy_gpus and num_ref_gpus must be the same when colocating policy and ref model"
                ref_model = PPORayActorGroup(
                    cfg,
                    cfg.trainer.placement.ref_num_nodes,
                    cfg.trainer.placement.ref_num_gpus_per_node,
                    RefWorker,
                    pg=pg,
                    num_gpus_per_actor=0.2 if pg else 1,
                    colocate_all=True,
                    sequence_parallel_size=cfg.trainer.ref.sequence_parallel_size,
                )
            else:
                ref_model = None

            if cfg.trainer.critic.model.path:
                assert (
                    num_policy_gpus == num_critic_gpus
                ), "num_policy_gpus and num_critic_gpus must be the same when colocating policy and critic model"
                critic_model = PPORayActorGroup(
                    cfg,
                    cfg.trainer.placement.critic_num_nodes,
                    cfg.trainer.placement.critic_num_gpus_per_node,
                    CriticWorker,
                    pg=pg,
                    num_gpus_per_actor=0.2,
                    colocate_all=True,
                    sequence_parallel_size=cfg.trainer.critic.sequence_parallel_size,
                )
            else:
                critic_model = None

            # reward model
            if RewardWorker is not None and cfg.trainer.reward.model.path:
                reward_model = PPORayActorGroup(
                    cfg,
                    cfg.trainer.placement.reward_num_nodes,
                    cfg.trainer.placement.reward_num_gpus_per_node,
                    RewardWorker,
                    pg=pg,
                    num_gpus_per_actor=0.2,
                    colocate_all=True,
                    sequence_parallel_size=cfg.trainer.reward.sequence_parallel_size,
                )
            else:
                reward_model = None

        else:
            if cfg.trainer.placement.colocate_policy_ref and use_ref_model:
                assert (
                    cfg.trainer.placement.policy_num_nodes
                    == cfg.trainer.placement.ref_num_nodes
                    and cfg.trainer.placement.policy_num_gpus_per_node
                    == cfg.trainer.placement.ref_num_gpus_per_node
                ), "num_nodes and num_gpus_per_node must be the same when colocate policy and ref model."

                bundles = [
                    {
                        "GPU": cfg.trainer.placement.policy_num_gpus_per_node,
                        "CPU": cfg.trainer.placement.policy_num_gpus_per_node,
                    }
                    for _ in range(cfg.trainer.placement.policy_num_nodes)
                ]
                pg = placement_group(bundles, strategy="PACK")
                get_ray_pg_ready_with_timeout(pg, timeout=30)

            policy_model = PPORayActorGroup(
                cfg,
                cfg.trainer.placement.policy_num_nodes,
                cfg.trainer.placement.policy_num_gpus_per_node,
                PolicyWorker,
                pg=pg,
                num_gpus_per_actor=0.75 if pg else 1,
                colocate_all=False,
                sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
            )
            if use_ref_model:
                ref_model = PPORayActorGroup(
                    cfg,
                    cfg.trainer.placement.ref_num_nodes,
                    cfg.trainer.placement.ref_num_gpus_per_node,
                    RefWorker,
                    pg=pg,
                    num_gpus_per_actor=0.25 if pg else 1,
                    colocate_all=False,
                    sequence_parallel_size=cfg.trainer.ref.sequence_parallel_size,
                )
            else:
                ref_model = None

            # if colocated, create placement group for critic and reward model explicitly.
            pg = None
            if cfg.trainer.placement.colocate_critic_reward:
                assert (
                    cfg.trainer.placement.critic_num_nodes
                    == cfg.trainer.placement.reward_num_nodes
                    and cfg.trainer.placement.critic_num_gpus_per_node
                    == cfg.trainer.placement.reward_num_gpus_per_node
                ), "num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."

                bundles = [
                    {
                        "GPU": cfg.trainer.placement.critic_num_gpus_per_node,
                        "CPU": cfg.trainer.placement.critic_num_gpus_per_node,
                    }
                    for _ in range(cfg.trainer.placement.critic_num_nodes)
                ]
                pg = placement_group(bundles, strategy="PACK")
                get_ray_pg_ready_with_timeout(pg, timeout=30)

            if cfg.trainer.critic.model.path:
                critic_model = PPORayActorGroup(
                    cfg,
                    cfg.trainer.placement.critic_num_nodes,
                    cfg.trainer.placement.critic_num_gpus_per_node,
                    CriticWorker,
                    pg=pg,
                    num_gpus_per_actor=0.75 if pg else 1,
                    colocate_all=False,
                    sequence_parallel_size=cfg.trainer.critic.sequence_parallel_size,
                )
            else:
                critic_model = None

            # reward model
            if RewardWorker is not None and cfg.trainer.reward.model.path:
                reward_model = PPORayActorGroup(
                    cfg,
                    cfg.trainer.placement.reward_num_nodes,
                    cfg.trainer.placement.reward_num_gpus_per_node,
                    RewardWorker,
                    pg=pg,
                    num_gpus_per_actor=0.25 if pg else 1,
                    colocate_all=False,
                    sequence_parallel_size=cfg.trainer.reward.sequence_parallel_size,
                )
            else:
                reward_model = None

        if not cfg.trainer.placement.colocate_all:
            refs = []
            if ref_model is not None:
                refs.extend(ref_model.async_init_model(cfg.trainer.policy.model.path))
            refs.extend(
                policy_model.async_init_model(
                    cfg.trainer.policy.model.path,
                    num_training_steps=self.total_training_steps,
                )
            )
            if cfg.trainer.critic.model.path:
                refs.extend(
                    critic_model.async_init_model(
                        cfg.trainer.critic.model.path,
                        num_training_steps=self.total_training_steps,
                    )
                )
            if cfg.trainer.reward.model.path:
                refs.extend(
                    reward_model.async_init_model(cfg.trainer.reward.model.path)
                )
            ray.get(refs)
            ray.get(
                policy_model.async_run_ray_method(
                    "pass_through", "_set_pad_token_id", self.tokenizer.pad_token_id
                )
            )
        else:
            if ref_model is not None:
                ray.get(ref_model.async_init_model(cfg.trainer.policy.model.path))
                ref_model.offload_to_cpu()
            ray.get(
                policy_model.async_init_model(
                    cfg.trainer.policy.model.path,
                    num_training_steps=self.total_training_steps,
                )
            )
            ray.get(
                policy_model.async_run_ray_method(
                    "pass_through", "_set_pad_token_id", self.tokenizer.pad_token_id
                )
            )
            policy_model.offload_to_cpu()
            if cfg.trainer.critic.model.path:
                ray.get(
                    critic_model.async_init_model(
                        cfg.trainer.critic.model.path,
                        num_training_steps=self.total_training_steps,
                    )
                )
                critic_model.offload_to_cpu()
            if cfg.trainer.reward.model.path:
                ray.get(reward_model.async_init_model(cfg.trainer.reward.model.path))
                reward_model.offload_to_cpu()

        self.policy_model: PPORayActorGroup = policy_model
        self.critic_model: Optional[PPORayActorGroup] = critic_model
        self.ref_model: Optional[PPORayActorGroup] = ref_model
        self.reward_model: Optional[PPORayActorGroup] = reward_model

        logger.info("init policy/ref/critic/reward models done")

    def setup_policy_and_generator(self):
        """
        Setup the connection between policy model and inference engine for weight syncing.
        """
        ray.get(
            self.policy_model.async_run_ray_method(
                "pass_through", "init_weight_sync_state", self.inference_engine_client
            )
        )
        logger.info(
            "Initialized weight sync state for policy model and inference engines."
        )

    def convert_to_training_input(
        self, generator_output: GeneratorOutput, uids: List[str]
    ) -> TrainingInputBatch:
        """Converts lists to a padded batch of tensors for training"""
        prompt_ids: List[List[int]] = generator_output["prompt_token_ids"]
        response_ids: List[List[int]] = generator_output["response_ids"]
        custom_rewards: List[List[float]] = generator_output["rewards"]
        loss_masks: List[List[int]] = generator_output["loss_masks"]

        logprobs: Optional[List[List[float]]] = generator_output.get(
            "rollout_logprobs", None
        )

        (
            sequences_tensor,
            attention_masks_tensor,
            response_masks_tensor,
            custom_rewards_tensor,
            loss_masks_tensor,
            rollout_logprobs_tensor,
        ) = convert_prompts_responses_to_batch_tensors(
            self.tokenizer,
            prompt_ids,
            response_ids,
            custom_rewards,
            loss_masks,
            logprobs,
        )
        # sanity check for tis
        if self.cfg.trainer.algorithm.use_tis:
            assert (
                rollout_logprobs_tensor is not None
            ), "expected non-null rollout logprobs tensor with  `trainer.algorithm.use_tis` as `True`"
            assert (
                rollout_logprobs_tensor.shape == loss_masks_tensor.shape
            ), "Logprobs should look like responses"
        training_input = TrainingInputBatch(
            {
                "sequences": sequences_tensor,  # Full trajectories (padded and concatenated prompts and responses)
                "attention_mask": attention_masks_tensor,
                "response_mask": response_masks_tensor,
                "custom_rewards": custom_rewards_tensor,
                "loss_mask": loss_masks_tensor,
                "rollout_logprobs": rollout_logprobs_tensor,
            },
        )
        training_input.metadata = {
            "uids": uids,
        }
        # padded response length
        training_input.metadata["response_length"] = response_masks_tensor.shape[1]
        training_input.metadata["avg_response_length"] = sum(
            len(sample_response_ids) for sample_response_ids in response_ids
        ) / len(response_ids)
        return training_input

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

        validate_generator_output(input_batch, generator_output)

        return generator_output

    @torch.no_grad()
    def postprocess_generator_output(
        self, generator_output: GeneratorOutput, uids: List[str]
    ) -> GeneratorOutput:
        """
        Converts to per token rewards and computes pass@N.

        In the future algorithm specific reward or loss mask post processing should be done here.
        """
        mean_raw_reward, pass_at_n = get_metrics_from_generator_output(
            generator_output,
            uids,
        )

        rewards: Union[List[float], List[List[float]]] = generator_output["rewards"]
        responses: List[List[int]] = generator_output["response_ids"]
        per_token_rewards: List[List[float]] = []

        # Check if rewards are already token-level (List[List[float]]) or response-level (List[float])
        if rewards and isinstance(rewards[0], list):
            # Token-level rewards: rewards is List[List[float]]
            per_token_rewards = rewards
        else:
            # Response-level rewards: rewards is List[float], convert to per-token rewards
            for reward, response in zip(rewards, responses):
                per_token_reward = [0.0] * len(response)
                per_token_reward[-1] = float(reward)
                per_token_rewards.append(per_token_reward)

        n_samples_per_prompt = self.cfg.generator.n_samples_per_prompt

        reward_metrics = {
            f"reward/avg_pass_at_{n_samples_per_prompt}": pass_at_n,
            "reward/avg_raw_reward": mean_raw_reward,
        }
        self.all_metrics.update(reward_metrics)
        logger.info(
            f"reward/avg_pass_at_{n_samples_per_prompt}: {pass_at_n}, reward/avg_raw_reward: {mean_raw_reward}"
        )

        # re-assign reward but now it's per token rewards
        generator_output["rewards"] = per_token_rewards
        return generator_output

    @torch.no_grad()
    def compute_advantages_and_returns(
        self, data: TrainingInputBatch
    ) -> TrainingInputBatch:
        """Calculate advantages and returns for the data batch.

        Expects:
            - `["sequences"]`: Integer[torch.Tensor, "batch_size seqlen"]
            - `["response_mask"]`: Integer[torch.Tensor, "batch_size seqlen"]
            - `["loss_mask"]`: Integer[torch.Tensor, "batch_size seqlen"]
            - `["values"]`: Float[torch.Tensor, "batch_size"]
            - `["rm_rewards"]`: Float[torch.Tensor, "batch_size"]
            - `["custom_rewards"]`: Float[torch.Tensor, "batch_size seqlen"]
            - `.metadata["uids"]`: List[str]

        Adds:
            - `["advantages"]`: Float[torch.Tensor, "batch_size seqlen"]
            - `["returns"]`: Float[torch.Tensor, "batch_size seqlen"]
        """
        # TODO (erictang000): we are just supporting custom rewards for now
        token_level_rewards = data["custom_rewards"]

        advantages, returns = ppo_utils.compute_advantages_and_returns(
            token_level_rewards=token_level_rewards,
            response_mask=data["response_mask"],
            index=data.metadata["uids"],
            adv_estimator=self.cfg.trainer.algorithm.advantage_estimator,
            config=self.cfg.trainer.algorithm,
            values=data["values"],
            gamma=self.cfg.trainer.algorithm.gamma,
            lambd=self.cfg.trainer.algorithm.lambd,
            grpo_norm_by_std=self.cfg.trainer.algorithm.grpo_norm_by_std,
        )
        data["returns"] = returns
        data["advantages"] = advantages

        return_sums = token_level_rewards.sum(dim=-1)
        avg_rewards: float = return_sums.mean().item()

        avg_response_length = data.metadata["avg_response_length"]
        data = data.to("cpu")

        valid_advantages = torch.masked_select(
            data["advantages"], data["response_mask"].bool()
        )
        avg_advantages: float = valid_advantages.mean().item()
        avg_advantages_abs: float = valid_advantages.abs().mean().item()

        if "metrics" not in data.metadata:
            data.metadata["metrics"] = {}
        data.metadata["metrics"].update(
            {
                "avg_rewards": avg_rewards,
                "avg_response_length": avg_response_length,
                "avg_advantages": avg_advantages,
                "avg_advantages_abs": avg_advantages_abs,
            }
        )

        logger.info(
            f"avg_raw_rewards: {avg_rewards}, avg_response_length: {avg_response_length}"
        )
        self.all_metrics.update(
            {
                "loss/avg_raw_rewards": avg_rewards,
                "loss/avg_raw_advantages": avg_advantages,
                "loss/avg_raw_advantages_abs": avg_advantages_abs,
            }
        )
        return data

    def dump_data(self, data: TrainingInputBatch, file_name: str):
        """
        Dump data to pickle file
        """
        data_save_dir = Path(self.cfg.trainer.export_path) / "dumped_data"
        data_save_dir.mkdir(parents=True, exist_ok=True)
        data.save(data_save_dir / f"{file_name}.pkl")

    @torch.no_grad()
    def fwd_logprobs_values_reward(
        self,
        training_input: TrainingInputBatch,
    ):
        """
        Calculate values from the critic, log probs from the policy and ref model, and rewards from the reward model
        and then calculate the kl divergence between the action log probs and the base action log probs.

        Expects:
            - `["sequences"]`: Integer[torch.Tensor, "batch_size seqlen"]
            - `["attention_mask"]`: Integer[torch.Tensor, "batch_size seqlen"]
            - `.metadata["response_length"]`: Int

        Adds:
            - `["base_action_log_probs"]`: Float[torch.Tensor, "batch_size seqlen"]
            - `["action_log_probs"]`: Float[torch.Tensor, "batch_size seqlen"]
            - `["values"]`: Float[torch.Tensor, "batch_size"]
            - `["rm_rewards"]`: Float[torch.Tensor, "batch_size"]
        """
        data_fwd_pass = training_input.select(
            keys=["sequences", "attention_mask"], metadata_keys=["response_length"]
        )

        def collect_results(actor_infos, results, key):
            ret_outputs: TrainingOutputBatch = concatenate_outputs_after_mesh_dispatch(
                actor_infos, results
            )
            return ret_outputs[key]

        base_log_probs = None
        action_log_probs = None
        values = None

        # calculate critic values
        if self.cfg.trainer.placement.colocate_all and self.critic_model is not None:
            self.critic_model.backload_to_gpu()

        if self.critic_model is not None:
            value_refs = self.critic_model.async_run_ray_method(
                "mesh", "forward", data=data_fwd_pass
            )
            if self.cfg.trainer.placement.colocate_all:
                all_rank_values = ray.get(value_refs)
                values = collect_results(
                    self.critic_model.actor_infos, all_rank_values, key="output"
                )
                self.critic_model.offload_to_cpu()

        # calculate ref log probs
        if self.ref_model is not None:
            if (
                self.cfg.trainer.placement.colocate_policy_ref
                or self.cfg.trainer.placement.colocate_all
            ):
                self.ref_model.backload_to_gpu()

            base_action_log_probs_refs = self.ref_model.async_run_ray_method(
                "mesh", "forward", data=data_fwd_pass
            )

        # handle colocate critic and reward model
        if (
            self.cfg.trainer.placement.colocate_critic_reward
            and not self.cfg.trainer.placement.colocate_all
            and self.critic_model is not None
        ):
            all_rank_values = ray.get(value_refs)
            values = collect_results(
                self.critic_model.actor_infos, all_rank_values, key="output"
            )
            ray.get(
                self.critic_model.async_run_ray_method("pass_through", "empty_cache")
            )

        if self.ref_model is not None:
            # handle colocate policy and ref model
            if (
                self.cfg.trainer.placement.colocate_policy_ref
                or self.cfg.trainer.placement.colocate_all
            ):
                all_rank_base_log_probs: List[TrainingOutputBatch] = ray.get(
                    base_action_log_probs_refs
                )
                base_log_probs = collect_results(
                    self.ref_model.actor_infos, all_rank_base_log_probs, key="output"
                )
                self.ref_model.offload_to_cpu()
                ray.get(
                    self.ref_model.async_run_ray_method("pass_through", "empty_cache")
                )
        else:
            base_log_probs = None

        # calculate rewards
        rewards = None
        if self.cfg.trainer.use_orm_score and self.reward_model:
            reward_refs = self.reward_model.async_run_ray_method("mesh", "forward")

            if self.cfg.trainer.placement.colocate_all:
                all_rank_rewards = ray.get(reward_refs)
                rewards = collect_results(
                    self.reward_model.actor_infos, all_rank_rewards, key="output"
                )

        # calculate action log probs
        if self.cfg.trainer.placement.colocate_all:
            self.policy_model.backload_to_gpu()

        action_log_probs_refs = self.policy_model.async_run_ray_method(
            "mesh", "forward", data=data_fwd_pass
        )
        if self.cfg.trainer.placement.colocate_all:
            all_rank_action_log_probs: List[TrainingOutputBatch] = ray.get(
                action_log_probs_refs
            )
            action_log_probs = collect_results(
                self.policy_model.actor_infos, all_rank_action_log_probs, key="output"
            )
            self.policy_model.offload_to_cpu()

        # wait all models done
        # if not colocate_policy_ref, then need to gather base_log_probs
        # if not colocate_critic_reward and self.critic_model is not None, then need to gather value
        # reward_refs is always handled at last
        if not self.cfg.trainer.placement.colocate_all:
            if not self.cfg.trainer.placement.colocate_policy_ref:
                if (
                    not self.cfg.trainer.placement.colocate_critic_reward
                    and self.critic_model is not None
                ):
                    all_rank_values = ray.get(value_refs)
                    values = collect_results(
                        self.critic_model.actor_infos, all_rank_values, key="output"
                    )

                if self.ref_model is not None:
                    all_rank_base_log_probs: List[TrainingOutputBatch] = ray.get(
                        base_action_log_probs_refs
                    )
                    base_log_probs = collect_results(
                        self.ref_model.actor_infos,
                        all_rank_base_log_probs,
                        key="output",
                    )
                else:
                    base_log_probs = None

            elif (
                not self.cfg.trainer.placement.colocate_critic_reward
                and self.critic_model is not None
            ):
                all_rank_values = ray.get(value_refs)
                values = collect_results(
                    self.critic_model.actor_infos, all_rank_values, key="output"
                )

            all_rank_action_log_probs: List[TrainingOutputBatch] = ray.get(
                action_log_probs_refs
            )
            action_log_probs = collect_results(
                self.policy_model.actor_infos, all_rank_action_log_probs, key="output"
            )

            if self.cfg.trainer.use_orm_score and self.reward_model:
                all_rank_rewards: List[TrainingOutputBatch] = ray.get(reward_refs)
                rewards = collect_results(
                    self.reward_model.actor_infos, all_rank_rewards, key="output"
                )

        if not self.cfg.trainer.placement.colocate_all:
            empty_cache_refs = self.policy_model.async_run_ray_method(
                "pass_through", "empty_cache"
            )
            if self.ref_model is not None:
                empty_cache_refs.extend(
                    self.ref_model.async_run_ray_method("pass_through", "empty_cache")
                )
            if self.critic_model is not None:
                empty_cache_refs.extend(
                    self.critic_model.async_run_ray_method(
                        "pass_through", "empty_cache"
                    )
                )
            if self.reward_model is not None:
                empty_cache_refs.extend(
                    self.reward_model.async_run_ray_method(
                        "pass_through", "empty_cache"
                    )
                )
            ray.get(empty_cache_refs)

        sequences_all: torch.Tensor = training_input["sequences"]
        # NOTE (sumanthrh): The slicing is needed to make sure that the batch dimension doesn't change for the tensordict.
        rewards = rewards[: len(sequences_all)] if rewards is not None else None
        base_log_probs = (
            base_log_probs[: len(sequences_all)] if base_log_probs is not None else None
        )
        action_log_probs = action_log_probs[: len(sequences_all)]
        values = values[: len(sequences_all)] if values is not None else None

        training_input["base_action_log_probs"] = base_log_probs
        training_input["action_log_probs"] = action_log_probs
        training_input["values"] = values
        # rewards from the reward model
        training_input["rm_rewards"] = rewards  # `None` or torch.Tensor

        if self.cfg.generator.sampling_params.logprobs is not None:
            # calculates the difference in probs between inference and trainer components
            prob_diff = (
                training_input["rollout_logprobs"].exp() - action_log_probs.exp()
            ).abs()
            prob_diff_mean = prob_diff.mean().item()
            prob_diff_std = prob_diff.std().item()
            self.all_metrics.update(
                {
                    "policy/rollout_train_prob_diff_mean": prob_diff_mean,
                    "policy/rollout_train_prob_diff_std": prob_diff_std,
                }
            )
        return training_input

    def apply_reward_kl_penalty(
        self,
        data: TrainingInputBatch,
    ) -> TrainingInputBatch:
        """Applies a penalty for KL divergence between the policy log probs and the base model log probs to the rewards."""
        loss_masks_all: torch.Tensor = data["loss_mask"]
        custom_rewards: torch.Tensor = data["custom_rewards"]
        base_action_log_probs: torch.Tensor = data["base_action_log_probs"]
        action_log_probs: torch.Tensor = data["action_log_probs"]

        # single batched computation
        kl: Float[torch.Tensor, "batch_size seqlen"] = compute_approx_kl(  # type: ignore
            action_log_probs,
            base_action_log_probs,
            loss_mask=loss_masks_all,
            kl_estimator_type=self.cfg.trainer.algorithm.kl_estimator_type,
        )
        kl_max: Float[torch.Tensor, "batch_size"] = torch.max(kl.abs(), dim=-1)[
            0
        ]  # noqa: F821
        kl_mean: Float[torch.Tensor, "batch_size"] = masked_mean(
            kl, loss_masks_all, dim=-1
        )  # noqa: F821

        # NOTE (erictang000): only supporting custom rewards currently
        kl_loss_coef = (
            self.reward_kl_controller.value
            if self.reward_kl_controller is not None
            else self.cfg.trainer.algorithm.kl_loss_coef
        )
        custom_rewards = custom_rewards - kl * max(0, kl_loss_coef)
        data["custom_rewards"] = custom_rewards

        avg_kl: float = kl_mean.mean().item()
        avg_kl_max: float = kl_max.mean().item()

        # update the kl controller
        if self.reward_kl_controller is not None:
            self.reward_kl_controller.update(
                current=avg_kl, n_steps=kl.shape[0]
            )  # n_steps is just the batch size
        if "metrics" not in data.metadata:
            data.metadata["metrics"] = {}

        data.metadata["metrics"].update(
            {
                "avg_kl": avg_kl,
                "avg_kl_max": avg_kl_max,
                "kl_loss_coef": kl_loss_coef,
            }
        )

        self.all_metrics.update(
            {
                "loss/avg_kl": avg_kl,
                "loss/avg_kl_max": avg_kl_max,
                "loss/kl_loss_coef": kl_loss_coef,
            }
        )

        return data

    def train_critic_and_policy(self, data: TrainingInputBatch):
        """
        Run the training step for the policy and critic models (this is overlapped if colocate_all is False).
        """
        data.metadata["global_step"] = self.global_step
        if self.cfg.trainer.placement.colocate_all:
            if self.critic_model is not None:
                with Timer("critic_train", self.all_timings):
                    self.critic_model.backload_to_gpu()
                    critic_statuses = ray.get(
                        self.critic_model.async_run_ray_method(
                            "mesh", "ppo_train", data
                        )
                    )
                    self.critic_model.offload_to_cpu()
            with Timer("policy_train", self.all_timings):
                self.policy_model.backload_to_gpu()
                policy_statuses = ray.get(
                    self.policy_model.async_run_ray_method("mesh", "ppo_train", data)
                )
        else:
            if self.critic_model is not None:
                with Timer("policy_critic_overlap_train", self.all_timings):
                    policy_refs = self.policy_model.async_run_ray_method(
                        "mesh", "ppo_train", data
                    )
                    critic_refs = self.critic_model.async_run_ray_method(
                        "mesh", "ppo_train", data
                    )
                    policy_statuses = ray.get(policy_refs)
                    critic_statuses = ray.get(critic_refs)
            else:
                with Timer("policy_train", self.all_timings):
                    policy_statuses = ray.get(
                        self.policy_model.async_run_ray_method(
                            "mesh", "ppo_train", data
                        )
                    )

        empty_cache_refs = []
        if self.critic_model is not None:
            critic_status = critic_statuses[0].metadata["train_status"]
            for k, v in critic_status.items():
                self.all_metrics.update({f"critic/{k}": v})
            empty_cache_refs += self.critic_model.async_run_ray_method(
                "pass_through", "empty_cache"
            )

        policy_status = policy_statuses[0].metadata["train_status"]
        for k, v in policy_status.items():
            self.all_metrics.update({f"policy/{k}": v})
        empty_cache_refs += self.policy_model.async_run_ray_method(
            "pass_through", "empty_cache"
        )
        ray.get(empty_cache_refs)

        return policy_status

    def handle_dynamic_sampling(
        self, generator_output: GeneratorOutput, uids: List[str]
    ) -> Tuple[GeneratorOutput, List[str], bool]:
        """
        Handle dynamic sampling for the current batch.

        Accumulates the generator output and UIDs across batches if we are sampling repeatedly
        and applies the dynamic sampling strategy (i.e. filter, replace) to the current batch.
        If we hit the limit of max sample batches, we raise an error.

        Args:
            generator_output: Current batch generator output
            uids: Current batch UIDs

        Returns:
            processed_output: Filtered generator output
            processed_uids: Filtered UIDs
            keep_sampling: Whether to keep sampling
        """
        # Prepare sampling configuration
        max_sample_batches = (
            self.cfg.trainer.algorithm.dynamic_sampling.max_sample_batches
        )
        dynamic_sampling_config = {
            "type": self.cfg.trainer.algorithm.dynamic_sampling.type,
            "max_sample_batches": max_sample_batches,
            "min_replace_ratio": self.cfg.trainer.algorithm.dynamic_sampling.min_replace_ratio,
            "train_batch_size": self.cfg.trainer.train_batch_size,
            "n_samples_per_prompt": self.cfg.generator.n_samples_per_prompt,
        }

        if self.dynamic_sampling_state is None:
            self.dynamic_sampling_state: DynamicSamplingState = {
                "sample_batch_count": 1,
            }
        else:
            self.dynamic_sampling_state["sample_batch_count"] += 1

        # Handle dynamic sampling using utilities
        processed_output, processed_uids, keep_sampling, updated_state = (
            trainer_utils.handle_dynamic_sampling(
                generator_output,
                uids,
                dynamic_sampling_config,
                self.dynamic_sampling_state,
            )
        )

        # Check max resample limit, and if we hit it, raise an error
        if (
            keep_sampling
            and max_sample_batches > 0
            and self.dynamic_sampling_state["sample_batch_count"] >= max_sample_batches
        ):
            raise RuntimeError(
                f"Exiting training loop due to hitting dynamic sampling limit for "
                f"{self.cfg.trainer.algorithm.dynamic_sampling.type} strategy with "
                f"{self.cfg.trainer.algorithm.dynamic_sampling.max_sample_batches} max sample batches. "
                f"Please check your data difficulty distribution."
            )
        # Update state
        self.dynamic_sampling_state = updated_state

        if not keep_sampling:
            # Reset state when sampling is complete
            self.dynamic_sampling_state = None

        return processed_output, processed_uids, keep_sampling

    def _get_dp_group_models(self, rank: int, model_type: str = ""):
        model = getattr(self, model_type)
        if model_type == "reward_model":
            model = model[0]
        return model._actor_handlers[rank]

    def _get_mesh_rank(self, rank: int, model_type: str = "") -> MeshRank:
        model: PPORayActorGroup = getattr(self, model_type)
        actor_info: ActorInfo = model.actor_infos[rank]
        return actor_info.rank

    def save_checkpoints(self):
        """
        Save the model, optimizer, and training states to disk.
        """
        # Create global step folder structure
        global_step_folder = os.path.join(
            self.cfg.trainer.ckpt_path, f"global_step_{self.global_step}"
        )
        policy_save_dir = os.path.join(global_step_folder, "policy")
        critic_save_dir = os.path.join(global_step_folder, "critic")
        # TODO(tgriggs): Add reward model checkpointing.

        os.makedirs(global_step_folder, exist_ok=True)

        # Save policy checkpoint
        ray.get(
            self.policy_model.async_run_ray_method(
                "pass_through",
                "save_ckpt",
                global_step=self.global_step,
                ckpt_dir=policy_save_dir,
                tokenizer=self.tokenizer,
            )
        )

        # Save critic checkpoint (if it exists)
        if self.critic_model is not None:
            if self.cfg.trainer.placement.colocate_all:
                self.policy_model.offload_to_cpu()
                self.critic_model.backload_to_gpu()

            ray.get(
                self.critic_model.async_run_ray_method(
                    "pass_through",
                    "save_ckpt",
                    global_step=self.global_step,
                    ckpt_dir=critic_save_dir,
                    tokenizer=self.tokenizer,
                )
            )

            if self.cfg.trainer.placement.colocate_all:
                self.critic_model.offload_to_cpu()
                self.policy_model.backload_to_gpu()

        # Save dataloader state
        dataloader_save_path = os.path.join(global_step_folder, "data.pt")
        try:
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_save_path)
            logger.info(f"Saved dataloader state to {dataloader_save_path}")
        except Exception as e:
            logger.warning(f"Failed to save dataloader state: {e}")

        # Save additional trainer state
        trainer_state = {
            "global_step": self.global_step,
            "config": self.cfg,
        }
        trainer_state_path = os.path.join(global_step_folder, "trainer_state.pt")
        torch.save(trainer_state, trainer_state_path)
        logger.info(f"Saved trainer state to {trainer_state_path}")

        # Atomic tracking - write this last after all saves succeed
        latest_checkpoint_file = os.path.join(
            self.cfg.trainer.ckpt_path, "latest_ckpt_global_step.txt"
        )
        with open(latest_checkpoint_file, "w") as f:
            f.write(str(self.global_step))

        logger.info(
            f"Successfully saved checkpoint for global_step_{self.global_step} to: {global_step_folder}"
        )

        # Clean up old checkpoints after successful save
        with Timer("cleanup_old_checkpoints", self.all_timings):
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        if not self._node_ids:
            self._node_ids = get_node_ids(
                self.policy_model, self.critic_model, self.ref_model
            )
        run_on_each_node(
            self._node_ids,
            cleanup_old_checkpoints,
            self.cfg.trainer.ckpt_path,
            self.cfg.trainer.max_ckpts_to_keep,
            self.global_step,
        )
        # run on driver as well
        # NOTE (sumanthrh): the function will get called twice on the node with driver process, but it's ok because it's idempotent
        cleanup_old_checkpoints(
            self.cfg.trainer.ckpt_path,
            self.cfg.trainer.max_ckpts_to_keep,
            self.global_step,
        )

    def load_checkpoints(self) -> int:
        """
        Load complete checkpoint state and return the global_step to resume from.
        Returns 0 if no checkpoint is loaded.
        """
        checkpoint_path = None
        # Check if resumption is enabled
        if self.resume_mode == ResumeMode.NONE:
            logger.info(
                "Checkpoint resumption disabled, starting training from scratch"
            )
            return 0
        # first, let's get resume_path
        elif self.resume_mode == ResumeMode.LATEST:
            latest_checkpoint_file = (
                Path(self.cfg.trainer.ckpt_path) / "latest_ckpt_global_step.txt"
            )
            if not latest_checkpoint_file.exists():
                logger.info("No checkpoint found, starting training from scratch")
                return 0
            with open(latest_checkpoint_file, "r") as f:
                ckpt_iteration = int(f.read())
            checkpoint_path = (
                Path(self.cfg.trainer.ckpt_path)
                / f"{GLOBAL_STEP_PREFIX}{ckpt_iteration}"
            )
            # Run validation: Make sure ckpt folder is consistent with latest_ckpt_global_step.txt
            validate_consistency_for_latest_checkpoint(
                self.cfg.trainer.ckpt_path,
                ckpt_iteration,
                checkpoint_path,
                latest_checkpoint_file,
                self.cfg.trainer.ckpt_interval,
            )
        else:
            # Get and validate resume path
            checkpoint_path = Path(self.cfg.trainer.resume_path)
            if not checkpoint_path:
                raise ValueError(
                    "`trainer.resume_path` must be specified when resume_mode is 'from_path'"
                )

            # Validate that it's a global_step directory
            if GLOBAL_STEP_PREFIX not in checkpoint_path.name:
                raise ValueError(
                    f"`trainer.resume_path` must point to a directory whose name starting with {GLOBAL_STEP_PREFIX}, got: {checkpoint_path}"
                )

        # Validate that the path exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from: {checkpoint_path}")

        # Make path absolute if it's relative
        checkpoint_path = checkpoint_path.resolve()

        # Extract global step from checkpoint path
        global_step = extract_step_from_path(checkpoint_path)
        if global_step == -1:
            raise ValueError(
                f"Checkpoint path {checkpoint_path} is not a valid checkpoint path"
            )
        self.global_step = global_step
        logger.info(f"Resuming from global_step: {global_step}")

        # Define paths for different checkpoint components
        policy_ckpt_dir = checkpoint_path / "policy"
        critic_ckpt_dir = checkpoint_path / "critic"
        trainer_state_path = checkpoint_path / "trainer_state.pt"
        dataloader_state_path = checkpoint_path / "data.pt"

        # Validate that required checkpoint files exist
        if not trainer_state_path.exists():
            raise FileNotFoundError(
                f"Trainer state file not found: {trainer_state_path}"
            )

        # 1. Load and validate trainer state
        trainer_state = torch.load(
            trainer_state_path, map_location="cpu", weights_only=False
        )
        saved_global_step = trainer_state.get("global_step", global_step)
        logger.info("Successfully loaded trainer state")
        if saved_global_step != global_step:
            logger.warning(
                f"Global step mismatch: path={global_step}, saved={saved_global_step}. Using path value."
            )

        # 2. Load dataloader state if available
        if dataloader_state_path.exists():
            try:
                dataloader_state = torch.load(
                    dataloader_state_path, map_location="cpu", weights_only=False
                )
                self.train_dataloader.load_state_dict(dataloader_state)
                logger.info("Successfully loaded dataloader state")
            except Exception as e:
                logger.warning(
                    f"Failed to load dataloader state: {e}. Dataloader will start from beginning."
                )
        else:
            logger.warning(
                f"No dataloader state found at {dataloader_state_path}. Dataloader will start from beginning."
            )

        # 3. Load policy checkpoint
        logger.info(f"Loading policy checkpoint from {policy_ckpt_dir}")
        _ = ray.get(
            self.policy_model.async_run_ray_method(
                "pass_through",
                "load_ckpt",
                ckpt_dir=policy_ckpt_dir,
                load_optimizer_states=True,
                load_lr_scheduler_states=True,
            )
        )
        logger.info("Successfully loaded policy checkpoint")

        # 4. Load critic checkpoint if it exists and we have a critic model
        if self.critic_model is not None:
            logger.info(f"Loading critic checkpoint from {critic_ckpt_dir}")
            _ = ray.get(
                self.critic_model.async_run_ray_method(
                    "pass_through",
                    "load_ckpt",
                    ckpt_dir=critic_ckpt_dir,
                    load_optimizer_states=True,
                    load_lr_scheduler_states=True,
                )
            )
            logger.info("Successfully loaded critic checkpoint")

        logger.info(
            f"Successfully loaded complete checkpoint state from global_step_{global_step}"
        )
        return global_step

    def save_models(self):
        """
        Save the model parameters in HF format at `cfg.trainer.export_path`.
        """
        policy_export_dir = os.path.join(
            self.cfg.trainer.export_path, f"global_step_{self.global_step}", "policy"
        )
        ray.get(
            self.policy_model.async_run_ray_method(
                "pass_through", "save_hf_model", policy_export_dir, self.tokenizer
            )
        )
        if self.critic_model is not None:
            critic_export_dir = os.path.join(
                self.cfg.trainer.export_path,
                f"global_step_{self.global_step}",
                "critic",
            )
            ray.get(
                self.critic_model.async_run_ray_method(
                    "pass_through", "save_hf_model", critic_export_dir, self.tokenizer
                )
            )
        logger.info("Successfully saved model weights.")

    def update_ref_with_policy(self):
        """
        Update the reference model with the policy model weights (required by some algorithms)

        If colocate_all is enabled:
        - before calling this method, the policy model should be on GPU, and inference engine should be asleep / on CPU.
        - after calling this method, the same model placement still holds.
        """
        # TODO(tgriggs): Make policy-to-ref sync faster.
        policy_export_dir = os.path.join(
            self.cfg.trainer.export_path, f"global_step_{self.global_step}", "policy"
        )
        ray.get(
            self.policy_model.async_run_ray_method(
                "pass_through", "save_hf_model", policy_export_dir, self.tokenizer
            )
        )
        # NOTE (sumanthrh): This is for the memory efficient case where we can't keep policy and ref model state on GPU together
        # We thus offload the policy model to CPU and then load the ref model from the policy model checkpoint, and then backload the policy model to GPU
        if self.cfg.trainer.placement.colocate_all:
            self.policy_model.offload_to_cpu()
        ray.get(self.ref_model.async_init_model(policy_export_dir))
        if self.cfg.trainer.placement.colocate_all:
            self.ref_model.offload_to_cpu()
            self.policy_model.backload_to_gpu()

        # Clean up temporary saved model files
        try:
            shutil.rmtree(policy_export_dir)
            logger.info(
                f"Cleaned up temporary policy export directory: {policy_export_dir}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to clean up temporary policy export directory {policy_export_dir}: {e}"
            )

        logger.info(
            "Successfully update ref model with policy model, training continue."
        )
