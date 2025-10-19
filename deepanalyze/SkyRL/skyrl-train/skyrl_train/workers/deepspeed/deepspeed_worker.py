import asyncio

import deepspeed
import ray
import torch
import torch.distributed
from loguru import logger
from transformers import AutoModel
from transformers.trainer import get_scheduler


from skyrl_train.models import get_llm_for_sequence_regression, Actor
from skyrl_train.distributed.deepspeed_strategy import DeepspeedStrategy
from skyrl_train.utils import get_physical_gpu_id
from skyrl_train.utils.utils import str_to_torch_dtype
from skyrl_train.workers.worker import (
    PolicyWorkerBase,
    CriticWorkerBase,
    RewardWorkerBase,
    RefWorkerBase,
)


class DeepSpeedPolicyWorkerBase(PolicyWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True):
        # TODO (erictang000): this is where this was getting called previously - do we need to do this every time?
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(self.model, pin_memory, non_blocking)

    def backload_to_gpu(self, non_blocking=True):
        self.strategy.backload_to_gpu(self.model, non_blocking)

    def init_model(self, model_id_or_path, num_training_steps: int = None):
        assert self.cfg.trainer.strategy in ("deepspeed")
        self.zero_stage = self.cfg.trainer.policy.deepspeed_config.zero_optimization.stage
        if self.cfg.trainer.policy.optimizer_config.max_grad_norm > 0:
            self.cfg.trainer.policy.deepspeed_config.gradient_clipping = (
                self.cfg.trainer.policy.optimizer_config.max_grad_norm
            )
        strategy = DeepspeedStrategy(
            self.cfg.trainer.policy.deepspeed_config,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            train_batch_size=self.cfg.trainer.train_batch_size,
            zero_stage=self.zero_stage,
            bf16=self.cfg.trainer.bf16,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        # Update per-gpu mini batch size based on device mesh
        self._normalize_mini_batch_size()

        ds_config = strategy.get_ds_train_config()
        actor = Actor(
            model_id_or_path,
            use_flash_attention_2=self.cfg.trainer.flash_attn,
            bf16=self.cfg.trainer.bf16,
            target_modules=self.cfg.trainer.target_modules,
            ds_config=ds_config,
            sequence_parallel_size=self.sequence_parallel_size,
            use_sample_packing=self.cfg.trainer.use_sample_packing,
            use_torch_compile=self.cfg.trainer.policy.use_torch_compile,
        )

        # configure optimizer
        actor_optim = strategy.create_optimizer(
            actor,
            lr=self.cfg.trainer.policy.optimizer_config.lr,
            betas=self.cfg.trainer.policy.optimizer_config.adam_betas,
            weight_decay=self.cfg.trainer.policy.optimizer_config.weight_decay,
            offload_after_step=self.cfg.trainer.policy.optimizer_config.offload_after_step,
        )

        actor_scheduler = get_scheduler(
            self.cfg.trainer.policy.optimizer_config.scheduler,
            actor_optim,
            num_warmup_steps=self.cfg.trainer.policy.optimizer_config.num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        if self.cfg.trainer.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": self.cfg.trainer.gradient_checkpointing_use_reentrant}
            )

        self._seq_parallel_monkey_patch(model=actor.model)

        # prepare models/optimizers...
        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
        )

        self.use_cuda_ipc = False
        if self.cfg.generator.weight_sync_backend == "nccl" and self.cfg.trainer.placement.colocate_all:
            self.use_cuda_ipc = True

        self._model_update_group_name = None

    def process_sequences(self, sequences, input_len, eos_token_id, pad_token_id):
        return self.model.process_sequences(sequences, input_len, eos_token_id, pad_token_id)

    def _set_pad_token_id(self, pad_token_id):
        # NOTE (sumanthrh): self.model -> Actor; self.model -> DeepSpeedEngine, self.model.module -> AutoModelForCausalLM
        self.model.model.module.config.pad_token_id = pad_token_id

    def _handle_termination(self):
        logger.info("Received termination signal. Destroying weights update group.")
        if torch.distributed.get_rank() == 0:
            try:
                loop = asyncio.get_running_loop()
                loop.run_until_complete(self.inference_engine_client.teardown())
            except Exception as e:
                logger.error(f"Error destroying weights update group: {e}")

    async def broadcast_to_inference_engines(self, inference_engine_client):
        use_prefix_cache = self.cfg.generator.enable_prefix_caching
        generator_dtype = str_to_torch_dtype(self.cfg.generator.model_dtype)
        cache_reset_task = None
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            cache_reset_task = inference_engine_client.reset_prefix_cache()

        torch.cuda.empty_cache()
        model = self.model.model.module
        if not self.use_cuda_ipc:
            for name, param in model.named_parameters():
                if torch.distributed.get_rank() == 0:
                    shape = param.shape if self.zero_stage != 3 else param.ds_shape

                    update_weight_task = asyncio.create_task(
                        inference_engine_client.update_named_weights(
                            {
                                "names": [name],
                                "dtypes": [self.cfg.generator.model_dtype],
                                "shapes": [shape],
                            }
                        )
                    )

                # broadcast
                def gather_and_broadcast(param):
                    # For ZeRO-3, allgather sharded parameter and broadcast to all InferenceEngines by rank 0
                    with deepspeed.zero.GatheredParameters([param], enabled=self.zero_stage == 3):
                        if torch.distributed.get_rank() == 0:
                            param = param.to(generator_dtype)
                            torch.distributed.broadcast(param.data, 0, group=self._model_update_group)

                await asyncio.to_thread(gather_and_broadcast, param)
                if torch.distributed.get_rank() == 0:
                    await update_weight_task
            torch.distributed.barrier()
        # CUDA IPC
        else:
            from torch.multiprocessing.reductions import reduce_tensor

            weights_update_request = {"names": [], "dtypes": [], "shapes": [], "extras": []}
            current_size = 0

            for name, param in model.named_parameters():
                # For ZeRO-3, allgather sharded parameter and broadcast to all InferenceEngines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=self.zero_stage == 3):
                    weight = param.data.clone()
                    weight = weight.to(generator_dtype)
                    ipc_handle = reduce_tensor(weight)

                    ipc_handle = {get_physical_gpu_id(): ipc_handle}
                    ipc_handle_list = [None] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

                    if torch.distributed.get_rank() == 0:
                        ipc_handles = {}
                        for d in ipc_handle_list:
                            ipc_handles.update(d)

                        shape = param.shape if self.zero_stage != 3 else param.ds_shape

                        weights_update_request["names"].append(name)
                        weights_update_request["dtypes"].append(self.cfg.generator.model_dtype)
                        weights_update_request["shapes"].append(shape)
                        weights_update_request["extras"].append(
                            {
                                "ipc_handles": ipc_handles,
                            }
                        )
                        current_size += weight.nbytes
                        # We send in batches as an optimization
                        # sync if threshold is reached
                        if current_size / (1024**3) > self.cfg.generator.weight_transfer_threshold_cuda_ipc_GB:
                            await inference_engine_client.update_named_weights(weights_update_request)
                            current_size = 0
                            weights_update_request = {"names": [], "dtypes": [], "shapes": [], "extras": []}
                            # force collect any sent tensors if possible to be memory efficient
                            torch.cuda.ipc_collect()

                    torch.distributed.barrier()
                    torch.cuda.synchronize()

            # sync any remaining weights
            if torch.distributed.get_rank() == 0 and len(weights_update_request["names"]) > 0:
                await asyncio.create_task(inference_engine_client.update_named_weights(weights_update_request))
                current_size = 0
                weights_update_request = {"names": [], "dtypes": [], "shapes": [], "extras": []}
            torch.distributed.barrier()

        if cache_reset_task is not None:
            await cache_reset_task
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        stats = {}
        model = self.model.model.module
        for name, param in model.named_parameters():
            with deepspeed.zero.GatheredParameters([param], enabled=self.zero_stage == 3):
                tensor_stats = {
                    "mean": param.data.mean().item(),
                    "std": param.data.std().item(),
                    "norm": param.data.norm().item(),
                    "shape": tuple(param.shape),
                    "max": param.data.max().item(),
                    "min": param.data.min().item(),
                }
                stats[name] = tensor_stats

        return stats


class DeepSpeedCriticWorkerBase(CriticWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(self.model, pin_memory, non_blocking)

    def backload_to_gpu(self, non_blocking=True):
        self.strategy.backload_to_gpu(self.model, non_blocking)

    def init_model(self, model_id_or_path, num_training_steps: int = None):
        assert self.cfg.trainer.strategy in ("deepspeed")
        self.zero_stage = self.cfg.trainer.critic.deepspeed_config.zero_optimization.stage
        strategy = DeepspeedStrategy(
            self.cfg.trainer.critic.deepspeed_config,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            train_batch_size=self.cfg.trainer.train_batch_size,
            zero_stage=self.zero_stage,
            bf16=self.cfg.trainer.bf16,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        # Update per-gpu mini batch size based on device mesh
        self._normalize_mini_batch_size()

        ds_config = strategy.get_ds_train_config()
        # with torch.device("meta"):
        #     AutoModel.from_pretrained(pretrain, trust_remote_code=True)
        critic = get_llm_for_sequence_regression(
            model_id_or_path,
            "critic",
            normalize_reward=False,
            use_flash_attention_2=self.cfg.trainer.flash_attn,
            bf16=self.cfg.trainer.bf16,
            target_modules=self.cfg.trainer.target_modules,
            ds_config=ds_config,
            value_head_prefix=self.cfg.trainer.algorithm.value_head_prefix,
            init_value_head=self.cfg.trainer.policy.model.path == self.cfg.trainer.critic.model.path,
            sequence_parallel_size=self.sequence_parallel_size,
            use_sample_packing=self.cfg.trainer.use_sample_packing,
        )
        # configure optimizer
        critic_optim = strategy.create_optimizer(
            critic,
            lr=self.cfg.trainer.critic.optimizer_config.lr,
            betas=self.cfg.trainer.critic.optimizer_config.adam_betas,
            weight_decay=self.cfg.trainer.critic.optimizer_config.weight_decay,
            offload_after_step=self.cfg.trainer.critic.optimizer_config.offload_after_step,
        )

        # configure scheduler
        critic_scheduler = get_scheduler(
            self.cfg.trainer.critic.optimizer_config.scheduler,
            critic_optim,
            num_warmup_steps=self.cfg.trainer.critic.optimizer_config.num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        if self.cfg.trainer.gradient_checkpointing:
            critic.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": self.cfg.trainer.gradient_checkpointing_use_reentrant}
            )
        # We set `use_parent_class` because critic model is of type `CriticModel` which is a subclass of the AutoModel class of interest
        self._seq_parallel_monkey_patch(model=critic, use_parent_class=True)

        # prepare models/optimizers...
        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (critic, critic_optim, critic_scheduler),
        )


class DeepSpeedRewardWorkerBase(RewardWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True):
        # deepspeed automatically offloads all model parameters to cpu
        # after forward if param_offload is true, and the reward model has no optimizer state
        # so we don't need to call offload_to_cpu here
        pass

    def backload_to_gpu(self, non_blocking=True):
        pass

    def init_model(self, model_id_or_path):
        assert self.cfg.trainer.strategy in ("deepspeed")
        self.zero_stage = self.cfg.trainer.reward.deepspeed_config.zero_optimization.stage
        strategy = DeepspeedStrategy(
            self.cfg.trainer.reward.deepspeed_config,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            train_batch_size=self.cfg.trainer.train_batch_size,
            zero_stage=self.zero_stage,
            bf16=self.cfg.trainer.bf16,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        with torch.device("meta"):
            AutoModel.from_pretrained(model_id_or_path, trust_remote_code=True)
        model = get_llm_for_sequence_regression(
            model_id_or_path,
            "reward",
            normalize_reward=self.cfg.trainer.algorithm.normalize_reward,
            use_flash_attention_2=self.cfg.trainer.flash_attn,
            bf16=self.cfg.trainer.bf16,
            ds_config=strategy.get_ds_eval_config(),
            value_head_prefix=self.cfg.trainer.algorithm.value_head_prefix,
            sequence_parallel_size=self.sequence_parallel_size,
            use_sample_packing=self.cfg.trainer.use_sample_packing,
        )

        self._seq_parallel_monkey_patch(model=model, use_parent_class=True)

        self.model = self.strategy.prepare(model)
        self.model.eval()


class DeepSpeedRefWorkerBase(RefWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True):
        # deepspeed automatically offloads all model parameters to cpu
        # after forward if param_offload is true, and the ref model has no optimizer state
        # so we don't need to call offload_to_cpu here
        pass

    def backload_to_gpu(self, non_blocking=True):
        pass

    def init_model(self, model_path):
        assert self.cfg.trainer.strategy in ("deepspeed")
        self.zero_stage = self.cfg.trainer.ref.deepspeed_config.zero_optimization.stage
        strategy = DeepspeedStrategy(
            self.cfg.trainer.ref.deepspeed_config,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            train_batch_size=self.cfg.trainer.train_batch_size,
            zero_stage=self.zero_stage,
            bf16=self.cfg.trainer.bf16,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        model = Actor(
            model_path,
            use_flash_attention_2=self.cfg.trainer.flash_attn,
            bf16=self.cfg.trainer.bf16,
            ds_config=strategy.get_ds_eval_config(),
            sequence_parallel_size=self.sequence_parallel_size,
            use_sample_packing=self.cfg.trainer.use_sample_packing,
        )
        self._seq_parallel_monkey_patch(model=model.model)

        self.model = self.strategy.prepare(model)
        self.model.eval()


PolicyWorker = ray.remote(num_gpus=1)(DeepSpeedPolicyWorkerBase)
CriticWorker = ray.remote(num_gpus=1)(DeepSpeedCriticWorkerBase)
RewardWorker = ray.remote(num_gpus=1)(DeepSpeedRewardWorkerBase)
RefWorker = ray.remote(num_gpus=1)(DeepSpeedRefWorkerBase)
