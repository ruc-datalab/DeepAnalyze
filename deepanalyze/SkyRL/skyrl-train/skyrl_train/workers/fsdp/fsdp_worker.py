import asyncio


import ray
import torch
import torch.distributed
from transformers import AutoModel, AutoConfig
from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from skyrl_train.models import Actor, get_llm_for_sequence_regression
from skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl_train.utils import get_physical_gpu_id, str_to_torch_dtype
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl_train.distributed.fsdp_utils import fsdp_version, get_init_weight_context_manager
from skyrl_train.workers.worker import (
    PolicyWorkerBase,
    CriticWorkerBase,
    RewardWorkerBase,
    RefWorkerBase,
)


class FSDPPolicyRayActorBase(PolicyWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(self.model, self.optimizer, pin_memory, non_blocking)

    def backload_to_gpu(self, non_blocking=True):
        self.strategy.backload_to_gpu(self.model, self.optimizer, non_blocking)

    def init_model(self, model_path, num_training_steps: int = None):
        assert self.cfg.trainer.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.trainer.policy.fsdp_config,
            optimizer_config=self.cfg.trainer.policy.optimizer_config,
            fsdp_strategy=self.cfg.trainer.strategy,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            train_batch_size=self.cfg.trainer.train_batch_size,
            num_training_steps=num_training_steps,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        # Update per-gpu mini batch size based on device mesh
        self._normalize_mini_batch_size()

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.strategy.device_mesh
        )
        with init_context():

            actor = Actor(
                model_path,
                use_flash_attention_2=self.cfg.trainer.flash_attn,
                # NOTE (sumanthrh): Model initialization should always be in fp32
                # during training
                bf16=False,
                target_modules=self.cfg.trainer.target_modules,
                sequence_parallel_size=self.cfg.trainer.policy.sequence_parallel_size,
                use_sample_packing=self.cfg.trainer.use_sample_packing,
                use_torch_compile=self.cfg.trainer.policy.use_torch_compile,
            )
            # in-place patch
            self._seq_parallel_monkey_patch(model=actor.model)

            if self.cfg.trainer.gradient_checkpointing:
                actor.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": self.cfg.trainer.gradient_checkpointing_use_reentrant
                    }
                )

        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (actor, None, None),
        )
        assert (
            self.optimizer is not None and self.scheduler is not None
        ), "FSDP preparation should create optimizer and scheduler"

        self.use_cuda_ipc = False
        if self.cfg.generator.weight_sync_backend == "nccl" and self.cfg.trainer.placement.colocate_all:
            self.use_cuda_ipc = True

    async def broadcast_to_inference_engines(self, inference_engine_client):
        use_prefix_cache = self.cfg.generator.enable_prefix_caching
        generator_dtype = str_to_torch_dtype(self.cfg.generator.model_dtype)
        cache_reset_task = None
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            cache_reset_task = inference_engine_client.reset_prefix_cache()

        torch.cuda.empty_cache()
        if fsdp_version(self.model.model) == 1:
            FSDP.set_state_dict_type(
                self.model.model,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )
        params = self.model.model.state_dict()

        if not self.use_cuda_ipc:
            for name, param in params.items():
                if torch.distributed.get_rank() == 0:
                    shape = param.shape

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
                    # For FSDP, gather parameter and broadcast to all InferenceEngines by rank 0
                    device = torch.cuda.current_device()
                    param = param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param
                    # cast to generator dtype
                    param = param.to(generator_dtype)
                    if torch.distributed.get_rank() == 0:
                        torch.distributed.broadcast(param.data, 0, group=self._model_update_group)

                await asyncio.to_thread(gather_and_broadcast, param)
                if torch.distributed.get_rank() == 0:
                    await update_weight_task
                torch.distributed.barrier()
        # CUDA IPC
        else:
            weights_update_request = {"names": [], "dtypes": [], "shapes": [], "extras": []}
            current_size = 0

            for name, param in params.items():
                from torch.multiprocessing.reductions import reduce_tensor

                device = torch.cuda.current_device()
                param = param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param
                param = param.to(generator_dtype)
                weight = param.detach().contiguous()
                ipc_handle = reduce_tensor(weight)

                ipc_handle = {get_physical_gpu_id(): ipc_handle}
                ipc_handle_list = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

                if torch.distributed.get_rank() == 0:
                    ipc_handles = {}
                    for d in ipc_handle_list:
                        ipc_handles.update(d)

                    current_size += weight.nbytes
                    weights_update_request["names"].append(name)
                    weights_update_request["dtypes"].append(self.cfg.generator.model_dtype)
                    weights_update_request["shapes"].append(param.shape)
                    weights_update_request["extras"].append({"ipc_handles": ipc_handles})
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
            if len(weights_update_request["names"]) > 0 and torch.distributed.get_rank() == 0:
                await asyncio.create_task(inference_engine_client.update_named_weights(weights_update_request))
                current_size = 0
                weights_update_request = {"names": [], "dtypes": [], "shapes": [], "extras": []}
            torch.distributed.barrier()
            torch.cuda.synchronize()

        if cache_reset_task is not None:
            await cache_reset_task
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        raise NotImplementedError()

    def _set_pad_token_id(self, pad_token_id):
        # NOTE (sumanthrh): self.model -> Actor; self.model -> DeepSpeedEngine, self.model.module -> AutoModelForCausalLM
        self.model.model.config.pad_token_id = pad_token_id

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on data in inference mode.

        Reshard the model after forward pass to redistribute memory and allow for offloading to cpu.
        """
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        return output


class FSDPCriticRayActorBase(CriticWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(self.model, self.optimizer, pin_memory, non_blocking)

    def backload_to_gpu(self, non_blocking=True):
        self.strategy.backload_to_gpu(self.model, self.optimizer, non_blocking)

    def init_model(self, model_path, num_training_steps: int = None):
        assert self.cfg.trainer.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.trainer.critic.fsdp_config,
            optimizer_config=self.cfg.trainer.critic.optimizer_config,
            fsdp_strategy=self.cfg.trainer.strategy,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            train_batch_size=self.cfg.trainer.train_batch_size,
            num_training_steps=num_training_steps,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        # Update per-gpu mini batch size based on device mesh
        self._normalize_mini_batch_size()

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.strategy.device_mesh
        )
        with init_context():
            critic = get_llm_for_sequence_regression(
                model_path,
                "critic",
                # only for reward model
                normalize_reward=False,
                use_flash_attention_2=self.cfg.trainer.flash_attn,
                # NOTE (sumanthrh): Model initialization should always be in fp32
                # during training
                bf16=False,
                target_modules=self.cfg.trainer.target_modules,
                value_head_prefix=self.cfg.trainer.algorithm.value_head_prefix,
                init_value_head=self.cfg.trainer.policy.model.path == self.cfg.trainer.critic.model.path,
                sequence_parallel_size=self.cfg.trainer.critic.sequence_parallel_size,
                use_sample_packing=self.cfg.trainer.use_sample_packing,
            )
            self._seq_parallel_monkey_patch(model=critic, use_parent_class=True)

            if self.cfg.trainer.gradient_checkpointing:
                critic.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": self.cfg.trainer.gradient_checkpointing_use_reentrant
                    }
                )

        # prepare models/optimizers...
        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (critic, None, None),
        )
        assert self.optimizer is not None

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on data in inference mode.

        Reshard the model after forward pass to redistribute memory and allow for offloading to cpu.
        """
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        return output


class FSDPRewardRayActorBase(RewardWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(self.model, None, pin_memory, non_blocking)

    def backload_to_gpu(self, non_blocking=True):
        self.strategy.backload_to_gpu(self.model, None, non_blocking)

    def init_model(self, model_path):
        assert self.cfg.trainer.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.trainer.reward.fsdp_config,
            fsdp_strategy=self.cfg.trainer.strategy,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            train_batch_size=self.cfg.trainer.train_batch_size,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        with torch.device("meta"):
            AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model = get_llm_for_sequence_regression(
            model_path,
            "reward",
            normalize_reward=self.cfg.trainer.algorithm.normalize_reward,
            use_flash_attention_2=self.cfg.trainer.flash_attn,
            bf16=self.cfg.trainer.bf16,
            value_head_prefix=self.cfg.trainer.algorithm.value_head_prefix,
            sequence_parallel_size=self.cfg.trainer.reward.sequence_parallel_size,
            use_sample_packing=self.cfg.trainer.use_sample_packing,
        )
        self._seq_parallel_monkey_patch(model=model, use_parent_class=True)

        self.model = strategy.prepare(model)
        self.model.eval()

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on data in inference mode.

        Reshard the model after forward pass to redistribute memory and allow for offloading to cpu.
        """
        output = super().forward(data)

        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        return output


class FSDPRefRayActorBase(RefWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(self.model, None, pin_memory, non_blocking)

    def backload_to_gpu(self, non_blocking=True):
        self.strategy.backload_to_gpu(self.model, None, non_blocking)

    def init_model(self, model_path):
        assert self.cfg.trainer.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.trainer.ref.fsdp_config,
            fsdp_strategy=self.cfg.trainer.strategy,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            train_batch_size=self.cfg.trainer.train_batch_size,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.strategy.device_mesh
        )

        with init_context():
            model = Actor(
                model_path,
                use_flash_attention_2=self.cfg.trainer.flash_attn,
                bf16=self.cfg.trainer.bf16,
                sequence_parallel_size=self.cfg.trainer.ref.sequence_parallel_size,
                use_sample_packing=self.cfg.trainer.use_sample_packing,
            )
            self._seq_parallel_monkey_patch(model=model.model)

        self.model = strategy.prepare(model)
        self.model.eval()

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on data in inference mode.

        Reshard the model after forward pass to redistribute memory and allow for offloading to cpu.
        """
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        return output


# Ray remote actors
PolicyWorker = ray.remote(num_gpus=1)(FSDPPolicyRayActorBase)
CriticWorker = ray.remote(num_gpus=1)(FSDPCriticRayActorBase)
RewardWorker = ray.remote(num_gpus=1)(FSDPRewardRayActorBase)
RefWorker = ray.remote(num_gpus=1)(FSDPRefRayActorBase)
