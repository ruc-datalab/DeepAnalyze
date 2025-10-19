import asyncio
import logging
import os
import socket
from typing import Dict, Optional, Type, List, Any, Callable
from ctypes import CDLL, POINTER, Structure, c_char_p, c_int, c_ulong, c_void_p
from tqdm import tqdm
from collections import defaultdict

import ray
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
import torch.distributed
from ray import ObjectRef
from ray.util.placement_group import (
    PlacementGroup,
    PlacementGroupSchedulingStrategy,
    placement_group,
)

from skyrl_train.utils import ray_noset_visible_devices, get_ray_pg_ready_with_timeout
from skyrl_train.utils.ppo_utils import masked_mean
from skyrl_train.distributed.dispatch import (
    MeshRank,
    ActorInfo,
    DispatchRegistry,
    Dispatch,
)
from skyrl_train.distributed.strategy import DistributedStrategy
from transformers import PreTrainedModel
from loguru import logger
from skyrl_train.distributed.ulysses import (
    set_ulysses_sequence_parallel_group,
    apply_monkey_patch,
)
from skyrl_train.distributed.utils import init_custom_process_group
from skyrl_train.utils.torch_utils import chunked_entropy_from_logits
from skyrl_train.utils.ppo_utils import (
    PolicyLossRegistry,
    ppo_critic_loss,
    compute_approx_kl,
)
from skyrl_train.workers.worker_utils import BatchIterator, reduce_metrics
from skyrl_train.dataset.replay_buffer import Experience
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from omegaconf import DictConfig
from pathlib import Path

_SET_AFFINITY = False


# Adapted from OpenRLHF: https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ray/launcher.py#L17
class DistributedTorchRayActor:
    def __init__(
        self,
        world_size,
        rank,
        local_rank,
        master_addr,
        master_port,
        sequence_parallel_size,
        record_memory=False,
    ):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._world_size = world_size
        self._rank = rank
        self._local_rank = local_rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # NOTE: Ray will automatically set the CUDA_VISIBLE_DEVICES
        # environment variable for each actor, so always set device to 0
        # os.environ["LOCAL_RANK"] = str(self._local_rank)
        os.environ["LOCAL_RANK"] = (
            str(ray.get_gpu_ids()[0]) if ray_noset_visible_devices() else "0"
        )
        self.sequence_parallel_size: int = sequence_parallel_size

        self.record_memory = record_memory
        if record_memory:
            torch.cuda.memory._record_memory_history()

    def get_node_local_rank(self):
        return self._local_rank

    def init_worker_process_group(self):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        # setup device mesh
        # TODO: Support TP / PP for DeepSpeed
        # NOTE (sumanthrh): Device mesh and mesh rank are rank specific attributes. For the current way the strategy is defined, it is only meant to interact with worker state; not hold worker state. Thus, this should live outside the strategy object.
        # This device mesh can be common across all the strategies we use
        dp_size = self._world_size // self.sequence_parallel_size
        device_mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda",
            mesh_shape=(dp_size, self.sequence_parallel_size),
            mesh_dim_names=("dp", "sp"),
        )
        self.device_mesh = device_mesh
        self.mesh_rank = MeshRank(
            dp=self.device_mesh.get_local_rank(mesh_dim="dp"),
            sp=self.device_mesh.get_local_rank(mesh_dim="sp"),
            tp=0,
            pp=0,
            world_size=self._world_size,
            dp_size=self.device_mesh.size(0),
        )

    def _seq_parallel_monkey_patch(
        self, model: PreTrainedModel, use_parent_class: bool = False
    ):
        # NOTE (sumanthrh): This sets a global variable that is used during the forward pass for sequence parallelism
        # This works because each worker is it's own process and thus different worker types are isolated
        # TODO (sumanthrh): We should re-visit this and see if we should adopt a context-manager pattern for sequence parallelism
        if self.sequence_parallel_size > 1:
            set_ulysses_sequence_parallel_group(self.device_mesh["sp"].get_group())
            apply_monkey_patch(
                model=model,
                ulysses_sp_size=self.sequence_parallel_size,
                use_parent_class=use_parent_class,
            )

    def get_mesh_rank(self):
        return self.mesh_rank

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

    def get_ray_node_id(self):
        return ray.get_runtime_context().get_node_id()

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port

    # TODO(tgriggs): For numa affinity, pass in the Worker._local_rank for the second arg here. Distinguish 'rank' and 'local_rank' differ here.
    def _set_numa_affinity(self, rank):
        def local_rank_to_real_gpu_id(local_rank):
            cuda_visible_devices = [
                int(x)
                for x in os.environ.get(
                    "CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"
                ).split(",")
            ]
            return cuda_visible_devices[local_rank]

        rank = local_rank_to_real_gpu_id(rank)

        global _SET_AFFINITY
        if _SET_AFFINITY:
            return

        from ctypes.util import find_library

        class bitmask_t(Structure):
            _fields_ = [
                ("size", c_ulong),
                ("maskp", POINTER(c_ulong)),
            ]

        LIBNUMA = CDLL(find_library("numa"))
        LIBNUMA.numa_parse_nodestring.argtypes = [c_char_p]
        LIBNUMA.numa_parse_nodestring.restype = POINTER(bitmask_t)
        LIBNUMA.numa_run_on_node_mask.argtypes = [POINTER(bitmask_t)]
        LIBNUMA.numa_run_on_node_mask.restype = c_int
        LIBNUMA.numa_set_membind.argtypes = [POINTER(bitmask_t)]
        LIBNUMA.numa_set_membind.restype = c_void_p
        LIBNUMA.numa_num_configured_nodes.argtypes = []
        LIBNUMA.numa_num_configured_nodes.restype = c_int

        def numa_bind(nid: int):
            bitmask = LIBNUMA.numa_parse_nodestring(bytes(str(nid), "ascii"))
            LIBNUMA.numa_run_on_node_mask(bitmask)
            LIBNUMA.numa_set_membind(bitmask)

        numa_nodes = LIBNUMA.numa_num_configured_nodes()
        num_gpu_pre_numa_node = 8 // numa_nodes
        numa_bind(self._local_rank // num_gpu_pre_numa_node)
        _SET_AFFINITY = True


class Worker(DistributedTorchRayActor):
    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def init_model(self, *args, **kwargs):
        """Initialize worker state (model, and optimizer if applicable) on worker."""
        raise NotImplementedError()

    def empty_cache(self) -> None:
        """Empty GPU memory cache on Worker's CUDA device"""
        torch.cuda.empty_cache()

    def offload_to_cpu(self, pin_memory=True, non_blocking=True):
        """Offload all worker state to CPU.

        After this function runs, only temporary reserved memory and torch's pre-loaded cuda kernels (~ GB) will remain

        Args:
            pin_memory: Whether to use pinned/ paged-locked memory on CPU
            non_blocking: Whether the operation is non-blocking
        """
        raise NotImplementedError()

    def backload_to_gpu(self, non_blocking=True):
        """Backload worker state to GPU

        Args:
            non_blocking: Whether the operation is non-blocking
        """
        raise NotImplementedError()

    def get_cuda_memory(self) -> Dict[str, Any]:
        """Get CUDA memory usage on worker's CUDA device."""
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "free": free,
            "total": total,
        }

    def save_memory_snapshot(self, global_step=None, local_step=None):
        """Save a snapshot of memory usage on the Worker's CUDA device.

        .. note::
            This function should be called on all the ranks in the worker group simultaneously.
        """
        rank = torch.distributed.get_rank()
        save_path = os.path.join(self.cfg.trainer.ckpt_path, "memory_snapshots")
        if self._local_rank == 0 and not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        torch.distributed.barrier()
        if global_step is None or local_step is None:
            file_name = f"policy_rank_{rank}.pickle"
        else:
            file_name = (
                f"policy_rank_{rank}_training_step_{global_step}_{local_step}.pickle"
            )
        record_memory_path = os.path.join(save_path, file_name)
        if os.path.exists(record_memory_path):
            # seeing issues if we don't remove the file first
            os.remove(record_memory_path)
        torch.cuda.memory._dump_snapshot(record_memory_path)

    async def init_weight_sync_state(
        self, inference_engine_client: InferenceEngineClient
    ):
        """Initialize state for weight syncing with Inference Engine Client

        Initializes a custom process group with the rank 0 Worker and all the inference engine ranks
        for weight syncing.

        .. note::
            This function should be called on all the ranks in the worker group simultaneously.
        """
        assert inference_engine_client is not None

        if torch.distributed.get_rank() == 0:
            master_addr = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            num_inference_engines, tensor_parallel_size = (
                self.cfg.generator.num_inference_engines,
                self.cfg.generator.inference_engine_tensor_parallel_size,
            )
            world_size = num_inference_engines * tensor_parallel_size + 1

            backend = self.cfg.generator.weight_sync_backend

            override_existing = (
                False
                if self.cfg.generator.override_existing_update_group == "disable"
                else True
            )
            group_name = "skyrl"
            self._model_update_group_name = group_name

            tasks = []
            tasks.append(
                inference_engine_client.init_weight_update_communicator(
                    master_addr=master_addr,
                    master_port=master_port,
                    rank_offset=1,
                    world_size=world_size,
                    group_name=group_name,
                    backend=backend,
                    override_existing=override_existing,
                )
            )

            tasks.append(
                asyncio.to_thread(
                    init_custom_process_group,
                    backend=backend,
                    init_method=f"tcp://{master_addr}:{master_port}",
                    world_size=world_size,
                    rank=0,
                    group_name=group_name,
                )
            )
            results = await asyncio.gather(*tasks)
            self._model_update_group = results[-1]

            # # Register signal handlers for termination only on rank 0
            # NOTE (sumanthrh): This doesn't work yet, and is thus commented out.
            # The better way is to just have this specified in __del__, but there is
            # no guarattee that __del__ will be called in general. Ray also doesn't
            # explictly call __del__ when the actor shuts down.
            # It's commented out so that we can fix this in the future.
            # atexit.register(self._handle_termination)

        torch.distributed.barrier()

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on the input batch in inference mode.

        This is a wrapper around `_forward_micro_batch` that runs in micro batches of `cfg.trainer.micro_forward_batch_size_per_gpu`.
        """
        # run in micro batches of cfg.trainer.micro_forward_batch_size_per_gpu
        # TODO (sumanthrh): this can be in the policy/critic impl if the micro batch size can be specific to policy, critic, etc.
        micro_batches = data.chunk(self.cfg.trainer.micro_forward_batch_size_per_gpu)

        outputs = []
        for micro_batch in micro_batches:
            outputs.append(self._forward_micro_batch(micro_batch))
        output = TrainingOutputBatch.cat(outputs)
        if output.device is not None and output.device != torch.device("cpu"):
            output = output.to("cpu")
        return output

    def _forward_micro_batch(
        self, micro_batch: TrainingInputBatch
    ) -> TrainingOutputBatch:
        raise NotImplementedError()


# adapted from OpenReasonerZero: https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/orz/ppo/actors.py
class PPORayActorGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        cfg: config object for workers
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        ray_actor_type (Type[Worker]): PPO model type that this actor group serve on.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        cfg,
        num_nodes,
        num_gpus_per_node,
        ray_actor_type: Type[Worker],
        pg: Optional[PlacementGroup] = None,
        num_gpus_per_actor: float = 1.0,
        resources: Optional[Dict[str, float]] = None,
        num_resources_per_node: Optional[int] = None,
        colocate_all: bool = False,
        sequence_parallel_size: int = 1,
        record_memory: bool = False,
    ) -> None:
        self.cfg = cfg
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.ray_actor_type = ray_actor_type

        # custom resources, see https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node

        self.colocate_all = colocate_all
        self.sequence_parallel_size = sequence_parallel_size
        self.record_memory = record_memory
        self._initiate_actors(pg, num_gpus_per_actor)

    def _initiate_actors(self, pg: Optional[PlacementGroup], num_gpus_per_actor: float):
        """Initialize Ray actors in the worker group.

        Args:
            pg: The placement group for the worker group
            num_gpus_per_actor: The number of gpus to allocate per actor.
        """
        world_size = self._num_nodes * self._num_gpus_per_node
        # Use placement group to lock resources for models of same type
        if self._num_gpus_per_node > 1 and pg is None:
            bundles = [
                {"GPU": self._num_gpus_per_node, "CPU": self._num_gpus_per_node}
                for _ in range(self._num_nodes)
            ]
            if self._resources:
                resources_name = list(self._resources.keys())[0]
                for i in range(len(bundles)):
                    bundles[i][resources_name] = self._num_resources_per_node

            pg = placement_group(bundles, strategy="PACK")
            get_ray_pg_ready_with_timeout(pg, timeout=30)
        if pg:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0
                ),
            ).remote(
                cfg=self.cfg,
                world_size=world_size,
                rank=0,
                local_rank=0,
                master_addr=None,
                master_port=None,
                sequence_parallel_size=self.sequence_parallel_size,
                record_memory=self.record_memory,
            )
        else:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
            ).remote(
                cfg=self.cfg,
                world_size=world_size,
                rank=0,
                local_rank=0,
                master_addr=None,
                master_port=None,
                sequence_parallel_size=self.sequence_parallel_size,
                record_memory=self.record_memory,
            )
        self._actor_handlers = [master_actor]
        # Create worker actors
        if world_size > 1:
            master_addr, master_port = ray.get(
                master_actor.get_master_addr_port.remote()
            )
            for rank in range(1, world_size):
                local_rank = rank % self._num_gpus_per_node

                if pg:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=(
                                rank
                                if self.colocate_all
                                else rank // self._num_gpus_per_node
                            ),
                        ),
                    ).remote(
                        cfg=self.cfg,
                        world_size=world_size,
                        rank=rank,
                        local_rank=local_rank,
                        master_addr=master_addr,
                        master_port=master_port,
                        sequence_parallel_size=self.sequence_parallel_size,
                        record_memory=self.record_memory,
                    )
                else:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                    ).remote(
                        cfg=self.cfg,
                        world_size=world_size,
                        rank=rank,
                        local_rank=local_rank,
                        master_addr=master_addr,
                        master_port=master_port,
                        sequence_parallel_size=self.sequence_parallel_size,
                        record_memory=self.record_memory,
                    )
                self._actor_handlers.append(worker_actor)

        # Initialize process group
        logger.info("Initializing process group for RayActorGroup")
        ray.get(
            [actor.init_worker_process_group.remote() for actor in self._actor_handlers]
        )
        logger.info("Initialized process group for RayActorGroup")
        self.actor_infos = [
            ActorInfo(actor, ray.get(actor.get_mesh_rank.remote()))
            for actor in self._actor_handlers
        ]
        logger.info(
            f"Mesh Ranks: {[actor_info.rank for actor_info in self.actor_infos]}"
        )

    def async_init_model(
        self,
        *args,
        **kwargs,
    ) -> List[ObjectRef]:
        """Asynchronously initialize worker state (model, and optimizer if applicable) from model path on all the workers.

        Returns:
            A list of ray object refs.
        """
        return [
            actor.init_model.remote(*args, **kwargs) for actor in self._actor_handlers
        ]

    def offload_to_cpu(self, nonblocking=False):
        """Offload all worker state to CPU.

        Args:
            nonblocking: Whether this operation is synchronous or asynchronous.
            If `nonblocking=True`, then the function returns a list of object refs.
        """
        refs = [actor.offload_to_cpu.remote() for actor in self._actor_handlers]
        if nonblocking:
            return refs
        return ray.get(refs)

    def backload_to_gpu(self, nonblocking=False):
        """Backload worker state to GPU

        Args:
            nonblocking: Whether this operation is synchronous or asynchronous.
            If `nonblocking=True`, then the function returns a list of ObjectRefs.
        """
        refs = [actor.backload_to_gpu.remote() for actor in self._actor_handlers]
        if nonblocking:
            return refs
        return ray.get(refs)

    def run_method(
        self, dispatch_type: str, method_name: str, *args, **kwargs
    ) -> Optional[TrainingOutputBatch]:
        """Run a method on all actors using specified dispatch type synchronously.

        The method should either return `None` or a `TrainingOutputBatch` object.

        Args:
            dispatch_type: Type of dispatch to use ("mesh" or "pass_through")
            method_name: Name of the method to call on actors
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            Collect results from all the actors.
        """
        dispatch_class: Dispatch = DispatchRegistry.get(dispatch_type)
        # validate the dispatch args to be sent to `.dispatch`
        args, kwargs = dispatch_class.validate_dispatch_args(*args, **kwargs)

        # Dispatch the method call
        object_refs = dispatch_class.dispatch(
            self.actor_infos, method_name, *args, **kwargs
        )
        # Collect results from all the actors
        ret = dispatch_class.sync_collect(self.actor_infos, object_refs)
        return ret

    def async_run_ray_method(
        self, dispatch_type: str, method_name: str, *args, **kwargs
    ) -> List[ObjectRef]:
        """Run a method on all actors using specified dispatch type asynchronously.

        Args:
            dispatch_type: Type of dispatch to use ("mesh" or "pass_through")
            method_name: Name of the method to call on actors
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            List of object references
        """
        dispatch_class: Dispatch = DispatchRegistry.get(dispatch_type)
        # validate the dispatch args to be sent to `.dispatch`
        args, kwargs = dispatch_class.validate_dispatch_args(*args, **kwargs)

        # Dispatch the method call
        object_refs = dispatch_class.dispatch(
            self.actor_infos, method_name, *args, **kwargs
        )
        return object_refs

    async def async_run_method(
        self, dispatch_type: str, method_name: str, *args, **kwargs
    ) -> Optional[TrainingOutputBatch]:
        """Run a method on all actors using specified dispatch type in an asyncio-compatible way.

        Args:
            dispatch_type: Type of dispatch to use ("mesh" or "pass_through")
            method_name: Name of the method to call on actors
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            TrainingOutputBatch: concatenated results from all actors
        """
        dispatch_class: Dispatch = DispatchRegistry.get(dispatch_type)
        # validate the dispatch args to be sent to `.dispatch`
        args, kwargs = dispatch_class.validate_dispatch_args(*args, **kwargs)

        # Dispatch the method call
        object_refs = dispatch_class.dispatch(
            self.actor_infos, method_name, *args, **kwargs
        )
        return await dispatch_class.async_collect(self.actor_infos, object_refs)


class PolicyWorkerBase(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: nn.Module = None
        self.scheduler: LRScheduler = None
        self.optimizer: Optimizer = None
        self.strategy: DistributedStrategy = None
        self.record_memory: bool = False
        self.mesh_rank: MeshRank = None
        self.policy_loss_fn: Callable = PolicyLossRegistry.get(
            self.cfg.trainer.algorithm.policy_loss_type
        )

    def _normalize_mini_batch_size(self):
        """
        Normalize mini batch sizes to per-gpu mini batch sizes..
        """
        if not hasattr(self, "mesh_rank") or self.mesh_rank is None:
            raise RuntimeError(
                "mesh_rank must be initialized before calling _normalize_mini_batch_size()"
            )

        dp_size = self.mesh_rank.dp_size
        self.policy_mini_batch_size_per_gpu = (
            self.cfg.trainer.policy_mini_batch_size
            * self.cfg.generator.n_samples_per_prompt
            // dp_size
        )

    def ppo_train(self, train_data: TrainingInputBatch) -> TrainingOutputBatch:
        global_step = train_data.metadata["global_step"]
        dataloader = BatchIterator(
            train_data,
            sample_batch_size=self.cfg.trainer.micro_train_batch_size_per_gpu,
            drop_last=False,
        )

        micro_batches_per_mini_batch = (
            self.policy_mini_batch_size_per_gpu
            // self.cfg.trainer.micro_train_batch_size_per_gpu
        )
        # The number of steps (over micro batches) to accumulate gradients before taking an optimizer step.
        accumulation_steps = micro_batches_per_mini_batch

        status_list = []
        all_metrics = defaultdict(list)
        policy_update_steps = 0

        for epoch in range(self.cfg.trainer.update_epochs_per_batch):
            pbar = tqdm(
                dataloader,
                desc=f"Actor Train epoch [{epoch + 1}/{self.cfg.trainer.update_epochs_per_batch}]",
                disable=not self.strategy.is_rank_0(),
            )
            for local_step, experience in enumerate(pbar):
                status = self.training_step(
                    experience,
                    global_step,
                    local_step,
                    accumulation_steps,
                )
                policy_update_steps += 1

                # for DP
                # TODO (sumanthrh): this assumes all workers are data parallel.
                # We assume that outputs are replicated within tp or sp group, otherwise this is not correct.
                status = self.strategy.all_reduce(status)

                # weighted mean for kl
                # TODO (sumanthrh): this weighted mean is no longer correct since we use the max response length in the batch.
                # we can log this in the driver
                # if "kl" in status:
                #     status["kl"] *= status["response_length"]
                #     status["kl"] /= status["response_length"]

                short_status = {}

                if "policy_loss" in status:
                    short_status = {
                        "pg": status["policy_loss"],
                        "glen": status["response_length"],
                        "policy_lr": status["policy_lr"],
                        "ent": status["policy_entropy"],
                    }
                    if "reward" in status:
                        short_status["rm"] = status["reward"]

                if "critic_loss" in status:
                    short_status["cri"] = status["critic_loss"]
                    short_status["vals"] = status["values"]
                    short_status["cri_lr"] = status["critic_lr"]

                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]

                status_list.append(status)
                for k, v in status.items():
                    all_metrics[k].append(v)
                pbar.set_postfix(short_status)

        torch.distributed.barrier()
        # not needed beyond status logging
        all_metrics.pop("response_length", None)

        status_mean = reduce_metrics(all_metrics)
        status_mean["policy_update_steps"] = policy_update_steps / accumulation_steps

        # should return an `TrainingOutputBatch`
        output = TrainingOutputBatch()
        output.metadata = {"train_status": status_mean}
        return output

    def training_step(
        self, experience: Experience, global_step, local_step, accumulation_steps
    ) -> Dict[str, float]:
        """
        Perform one micro-batch of training, accumulate gradients, and step the optimizer only after `accumulation_steps` micro-batches.
        """
        self.model.train()
        experience.to_device(torch.cuda.current_device())

        sequences = experience.sequences
        old_action_log_probs = experience.action_log_probs
        base_action_log_probs = (
            experience.base_action_log_probs
            if experience.base_action_log_probs is not None
            else None
        )
        advantages = experience.advantages
        num_actions = experience.num_actions
        attention_mask = experience.attention_mask
        loss_mask = experience.loss_mask
        rollout_action_logprobs = experience.rollout_logprobs

        # TODO (sumanthrh): don't think this does anything for deepspeed or fsdp rn because autocast happens internally
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            # actor loss
            action_log_probs, output = self.model(
                sequences,
                num_actions,
                attention_mask=attention_mask,
                temperature=self.cfg.generator.sampling_params.temperature,
                return_output=True,
                compute_entropy=True,
            )
            # loss function
            # TODO: recompute advantages
            policy_loss, clip_ratio = self.policy_loss_fn(
                action_log_probs,
                old_action_log_probs,
                advantages,
                config=self.cfg.trainer.algorithm,
                loss_mask=loss_mask,
                rollout_logprobs=rollout_action_logprobs,
            )
        # entropy
        with torch.no_grad():
            if self.cfg.trainer.use_sample_packing:
                # batch_size, seqlen
                entropy_BS = output["entropy"]
                entropy_BS = entropy_BS[:, -num_actions - 1 : -1]
            else:
                action_logits = output["logits"][:, -num_actions - 1 : -1, :]
                entropy_BS = chunked_entropy_from_logits(
                    action_logits, requires_grad=False
                )

            entropy = entropy_BS.sum().item() / entropy_BS.numel()

        # kl loss
        if self.cfg.trainer.algorithm.use_kl_loss:
            kl_loss = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                loss_mask=loss_mask,
                kl_estimator_type=self.cfg.trainer.algorithm.kl_estimator_type,
            )
            kl_loss = masked_mean(kl_loss, loss_mask, dim=-1).mean()
        else:
            kl_loss = torch.tensor(0.0)

        loss = policy_loss + kl_loss * self.cfg.trainer.algorithm.kl_loss_coef
        loss = loss / accumulation_steps
        self.strategy.backward(loss, self.model, self.optimizer)

        grad_norm = None
        if (local_step + 1) % accumulation_steps == 0:
            grad_norm = self.strategy.optimizer_step(
                self.optimizer, self.model, self.scheduler, name="actor"
            )
            if grad_norm is not None:
                grad_norm = grad_norm.detach().cpu().item()

        if self.record_memory:
            self.save_memory_snapshot(global_step, local_step)

        # status
        status = {
            "policy_loss": policy_loss.item(),
            "policy_lr": self.scheduler.get_last_lr()[0],
            "ppo_clip_ratio": clip_ratio,
            "policy_entropy": entropy,
        }
        if self.cfg.trainer.algorithm.use_kl_loss:
            status["policy_kl"] = kl_loss.item()

        if grad_norm is not None:
            status["raw_grad_norm"] = grad_norm

        for k, v in experience.info.items():
            if k == "kl":
                # just use the same value as loss if available
                status[k] = (
                    kl_loss.item()
                    if isinstance(kl_loss, torch.Tensor)
                    else status["policy_kl"]
                )
            else:
                status[k] = v.mean().item() if isinstance(v, torch.Tensor) else v

        status["response_length"] = num_actions
        return status

    def save_ckpt(self, global_step: int, ckpt_dir: Path, tokenizer=None):
        self.strategy.save_ckpt(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ckpt_dir=ckpt_dir,
            global_step=global_step,
            node_local_rank=self.get_node_local_rank(),
            tokenizer=tokenizer,
        )

    def load_ckpt(
        self,
        ckpt_dir: Path,
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
    ):
        _, states = self.strategy.load_ckpt(
            model=self.model,
            optimizer=self.optimizer if load_optimizer_states else None,
            scheduler=self.scheduler if load_lr_scheduler_states else None,
            ckpt_dir=ckpt_dir,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
        )
        return states

    def save_hf_model(self, export_dir: str, tokenizer):
        # Save model to HuggingFace format
        self.strategy.save_hf_model(
            self.model,
            export_dir,
            tokenizer=tokenizer,
        )

    def _forward_micro_batch(
        self, micro_batch: TrainingInputBatch
    ) -> TrainingOutputBatch:
        device = torch.cuda.current_device()
        micro_batch.to(device)
        self.model.eval()
        sequences = micro_batch["sequences"]
        response_length = micro_batch.metadata["response_length"]
        attention_mask = micro_batch["attention_mask"]
        with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            policy_logprob = self.model(
                sequences,
                response_length,
                attention_mask,
                return_output=False,
                temperature=self.cfg.generator.sampling_params.temperature,
            )
        policy_logprob = policy_logprob.to("cpu")
        output = TrainingOutputBatch(
            {"output": policy_logprob},
        )
        output.metadata = micro_batch.metadata
        return output

    def process_sequences(self, sequences, input_len, eos_token_id, pad_token_id):
        return self.model.process_sequences(
            sequences, input_len, eos_token_id, pad_token_id
        )


class CriticWorkerBase(Worker):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: nn.Module = None
        self.scheduler: LRScheduler = None
        self.optimizer: Optimizer = None
        self.strategy: DistributedStrategy = None
        self.record_memory: bool = False
        self.mesh_rank: MeshRank = None
        self.critic_loss_fn: Callable = ppo_critic_loss

    def _normalize_mini_batch_size(self):
        """
        Normalize batch sizes based on device mesh and generation parameters.
        """
        if not hasattr(self, "mesh_rank") or self.mesh_rank is None:
            raise RuntimeError(
                "mesh_rank must be initialized before calling _normalize_mini_batch_size()"
            )

        dp_size = self.mesh_rank.dp_size
        self.critic_mini_batch_size_per_gpu = (
            self.cfg.trainer.critic_mini_batch_size
            * self.cfg.generator.n_samples_per_prompt
            // dp_size
        )

    def _forward_micro_batch(
        self,
        micro_batch: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Generates critic values."""
        device = torch.cuda.current_device()
        micro_batch.to(device)
        sequences = micro_batch["sequences"]
        response_length = micro_batch.metadata["response_length"]
        attention_mask = micro_batch["attention_mask"]
        self.model.eval()
        with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            value = self.model(
                sequences,
                response_length,
                attention_mask,
            )
        self.model.train()  # reset model state
        value = value.to("cpu")
        output = TrainingOutputBatch(
            {"output": value},
        )
        output.metadata = micro_batch.metadata
        return output

    def save_hf_model(self, export_dir: str, tokenizer):
        # Save model to HuggingFace format
        self.strategy.save_hf_model(
            self.model,
            export_dir,
            tokenizer=tokenizer,
        )

    def ppo_train(self, train_data: TrainingInputBatch) -> TrainingOutputBatch:
        global_step = train_data.metadata["global_step"]
        dataloader = BatchIterator(
            train_data,
            sample_batch_size=self.cfg.trainer.micro_train_batch_size_per_gpu,
            drop_last=False,
        )

        torch.cuda.empty_cache()
        self.model.train()

        micro_batches_per_mini_batch = (
            self.critic_mini_batch_size_per_gpu
            // self.cfg.trainer.micro_train_batch_size_per_gpu
        )
        # The number of steps (over micro batches) to accumulate gradients before taking an optimizer step.
        accumulation_steps = micro_batches_per_mini_batch

        all_metrics = defaultdict(list)
        critic_update_steps = 0
        for epoch in range(self.cfg.trainer.update_epochs_per_batch):
            pbar = tqdm(
                dataloader,
                desc=f"Critic Train epoch [{epoch + 1}/{self.cfg.trainer.update_epochs_per_batch}]",
                disable=not self.strategy.is_rank_0(),
            )
            for local_step, experience in enumerate(pbar):
                status = self.training_step(
                    experience, global_step, local_step, accumulation_steps
                )
                critic_update_steps += 1

                # for DP
                # TODO (sumanthrh): this assumes all workers are data parallel.
                # We should get more accurate metrics with seq parallel or TP.
                # There are metrics like entropy where we get average over local data size
                status = self.strategy.all_reduce(status)

                for k, v in status.items():
                    all_metrics[k].append(v)
                pbar.set_postfix(status)

        torch.distributed.barrier()

        status_mean = reduce_metrics(all_metrics)
        status_mean["critic_update_steps"] = critic_update_steps / accumulation_steps

        output = TrainingOutputBatch()
        output.metadata = {"train_status": status_mean}
        return output

    def training_step(
        self, experience: Experience, global_step, local_step, accumulation_steps
    ) -> Dict[str, float]:
        """
        Perform one micro-batch of training, accumulate gradients, and step the optimizer only after `accumulation_steps` micro-batches.
        """
        experience.to_device(torch.cuda.current_device())

        sequences = experience.sequences
        old_values = experience.values
        returns = experience.returns
        num_actions = experience.num_actions
        attention_mask = experience.attention_mask
        loss_mask = experience.loss_mask

        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            # critic loss
            values, output = self.model(
                sequences,
                num_actions=num_actions,
                attention_mask=attention_mask,
                return_output=True,
            )
            # loss function
            loss, clipfrac = self.critic_loss_fn(
                values,
                old_values,
                returns,
                config=self.cfg.trainer.algorithm,
                loss_mask=loss_mask,
            )
        loss = loss / accumulation_steps
        self.strategy.backward(loss, self.model, self.optimizer)
        grad_norm = None
        if (local_step + 1) % accumulation_steps == 0:
            grad_norm = self.strategy.optimizer_step(
                self.optimizer, self.model, self.scheduler, name="critic"
            )
            if grad_norm is not None:
                grad_norm = grad_norm.detach().cpu().item()

        # status
        status = {
            "critic_loss": loss.item(),
            "values_mean": masked_mean(values, loss_mask).item(),
            "critic_lr": self.scheduler.get_last_lr()[0],
            "values_clipfrac": clipfrac,
        }
        if grad_norm is not None:
            status["raw_grad_norm"] = grad_norm
        return status

    def save_ckpt(self, global_step: int, ckpt_dir: str, tokenizer=None):
        self.strategy.save_ckpt(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ckpt_dir=ckpt_dir,
            global_step=global_step,
            node_local_rank=self.get_node_local_rank(),
            tokenizer=tokenizer,
        )

    def load_ckpt(
        self, ckpt_dir=None, load_optimizer_states=True, load_lr_scheduler_states=True
    ):
        _, states = self.strategy.load_ckpt(
            model=self.model,
            optimizer=self.optimizer if load_optimizer_states else None,
            scheduler=self.scheduler if load_lr_scheduler_states else None,
            ckpt_dir=ckpt_dir,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
        )
        return states


class RewardWorkerBase(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: nn.Module = None

    def _forward_micro_batch(
        self,
        micro_batch: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        device = torch.cuda.current_device()
        micro_batch.to(device)
        sequences = micro_batch["sequences"]
        attention_mask = micro_batch["attention_mask"]
        self.model.eval()
        with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            reward = self.model(sequences, attention_mask)
        reward = reward.to("cpu")
        output = TrainingOutputBatch(
            {"output": reward},
        )
        output.metadata = micro_batch.metadata
        return output


class RefWorkerBase(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: nn.Module = None

    def _forward_micro_batch(
        self, micro_batch: TrainingInputBatch
    ) -> TrainingOutputBatch:
        device = torch.cuda.current_device()
        micro_batch.to(device)
        sequences = micro_batch["sequences"]
        response_length = micro_batch.metadata["response_length"]
        attention_mask = micro_batch["attention_mask"]
        with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            log_probs = self.model(
                sequences, response_length, attention_mask, return_output=False
            )
        log_probs = log_probs.to("cpu")
        output = TrainingOutputBatch(
            {"output": log_probs},
        )
        output.metadata = micro_batch.metadata
        return output
