import os
import copy
import random
from collections import defaultdict
from datetime import timedelta
from typing import List, Union, Optional
from jaxtyping import Float
import gc
import json

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload, MixedPrecision

from skyrl_train.distributed.strategy import DistributedStrategy
from skyrl_train.models import Actor
from skyrl_train.distributed.utils import ModelOrModelOptimPair
from skyrl_train.distributed.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    init_fn,
    get_fsdp_wrap_policy,
    PrecisionType,
    create_device_mesh,
    fsdp2_clip_grad_norm_,
    fsdp2_get_full_state_dict,
    apply_fsdp2,
    get_sharding_strategy,
    offload_fsdp_model_to_cpu,
    load_fsdp_model_to_gpu,
    offload_fsdp_optimizer,
    load_fsdp_optimizer,
    get_fsdp_state_ctx,
    fsdp_version,
    fsdp2_load_full_state_dict,
)
from transformers.trainer import get_scheduler

from packaging import version

if version.parse(torch.__version__) >= version.parse("2.6"):
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
    )
elif version.parse(torch.__version__) >= version.parse("2.4"):
    from torch.distributed._composable.fsdp import (
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
    )
else:
    CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy = None, None, None


class FSDPStrategy(DistributedStrategy):
    """
    The strategy for training with FSDP.
    """

    def __init__(
        self,
        fsdp_config,
        optimizer_config=None,
        fsdp_strategy: str = "fsdp",
        seed: int = 42,
        micro_train_batch_size_per_gpu=1,
        train_batch_size=1,
        num_training_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert fsdp_strategy in (
            "fsdp",
            "fsdp2",
        ), f"Unsupported FSDP strategy: {fsdp_strategy}"
        self.fsdp_config = fsdp_config
        self.optimizer_config = optimizer_config
        self.fsdp_strategy = fsdp_strategy
        self.max_norm = (
            optimizer_config.max_grad_norm if optimizer_config is not None else 1.0
        )
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size_per_gpu = micro_train_batch_size_per_gpu
        self.seed = seed
        self.device_mesh = None
        self.total_training_steps: Optional[int] = num_training_steps

        # if we are using fsdp 1 or cpu offload is off for fsdp2, then we need to manually offload weights/optimizer to cpu
        self.manual_offload = self.fsdp_strategy == "fsdp" or not self.fsdp_config.get(
            "cpu_offload"
        )
        if self.optimizer_config is not None:
            self.manual_offload_optimizer = (
                self.optimizer_config.get("offload_after_step", True)
                and self.manual_offload
            )
        else:
            self.manual_offload_optimizer = False

        self.time_steps = defaultdict(int)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_distributed(self, timeout=timedelta(minutes=30)) -> None:
        self.set_seed(self.seed)

        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)

        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        self.world_size = dist.get_world_size()
        self.accumulated_gradient = (
            self.train_batch_size
            // self.micro_train_batch_size_per_gpu
            // self.world_size
        )

        self.device_mesh = create_device_mesh(
            world_size=self.world_size, fsdp_size=self.fsdp_config.fsdp_size
        )

    def offload_to_cpu(self, model, optimizer, pin_memory=True, non_blocking=True):
        """
        Offload model weights and optimizer to CPU memory.

        For all cases except fsdp2 with cpu_offload=True, we need to manually offload weights/optimizer to cpu.
        """
        if isinstance(model, Actor):
            model = model.model
        else:
            model = model

        if self.manual_offload:
            offload_fsdp_model_to_cpu(model, empty_cache=True)

            if optimizer is not None and self.manual_offload_optimizer:
                offload_fsdp_optimizer(optimizer)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def backload_to_gpu(self, model, optimizer, non_blocking=True):
        """Reload model weights back to GPU."""
        if isinstance(model, Actor):
            model = model.model
        else:
            model = model

        # if we are using fsdp 1 or cpu offload is off for fsdp2, then we need to manually backload weights/optimizer to gpu
        if self.manual_offload:
            load_fsdp_model_to_gpu(model)
            if optimizer is not None and self.manual_offload_optimizer:
                load_fsdp_optimizer(optimizer, torch.cuda.current_device())

        torch.cuda.synchronize()

    def backward(
        self, loss: torch.Tensor, model, optimizer: optim.Optimizer, **kwargs
    ) -> None:
        """Perform backward pass"""
        loss.backward()

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model,
        scheduler,
        name="model",
        **kwargs,
    ) -> Optional[Float[torch.Tensor, "1"]]:
        """Perform optimizer step"""
        grad_norm = None
        if isinstance(model, Actor):
            model = model.model

        if self.max_norm > 0:
            # NOTE (sumanthrh): All `grad_norm`s returned here are the original grad norms before clipping.
            if isinstance(model, FSDP):
                grad_norm = model.clip_grad_norm_(max_norm=self.max_norm)
            elif isinstance(model, FSDPModule):
                grad_norm = fsdp2_clip_grad_norm_(
                    model.parameters(), max_norm=self.max_norm
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=self.max_norm
                )

        # Skip update if gradient norm is not finite
        if grad_norm is not None and not torch.isfinite(grad_norm):
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                print(f"WARN: rank {rank} grad_norm is not finite: {grad_norm}")
            else:
                print(f"WARN: grad_norm is not finite: {grad_norm}")
            optimizer.zero_grad()
            return grad_norm

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        return grad_norm

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        """Prepare models and optimizers with FSDP"""
        ret = []
        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                assert (
                    len(arg) == 3
                ), f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'
                ret.append(self._fsdp_init_train_model(*arg))
            else:
                ret.append(self._fsdp_init_eval_model(arg))

        return ret[0] if len(ret) == 1 else ret

    def _fsdp_init_model(self, model, is_train=True, is_actor=False):
        # Initialize FSDP wrapping policy
        wrap_policy = get_fsdp_wrap_policy(
            module=model, config=self.fsdp_config.get("wrap_policy", None)
        )

        # Setup mixed precision
        mixed_precision_config = self.fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("param_dtype", "bf16")
            )
            reduce_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("reduce_dtype", "fp32")
            )
            buffer_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("buffer_dtype", "fp32")
            )
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )

        cpu_offload = None

        # sharding strategy
        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Wrap model with FSDP
        if self.fsdp_strategy == "fsdp":
            # cpu offloading will always be none for models that train with FSDP due to correctness issues with gradient accumulation -
            # see https://docs.pytorch.org/docs/stable/fsdp.html
            if not is_train and self.fsdp_config.get("cpu_offload", False):
                cpu_offload = CPUOffload(offload_params=True)
            fsdp_module = FSDP(
                model.model if is_actor else model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=wrap_policy,
                device_id=torch.cuda.current_device(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif self.fsdp_strategy == "fsdp2":
            assert (
                CPUOffloadPolicy is not None
            ), "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                cast_forward_inputs=True,
            )
            if self.fsdp_config.get("cpu_offload", False):
                cpu_offload = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": self.fsdp_config.get(
                    "reshard_after_forward", True
                ),
            }
            actor_module = model.model if is_actor else model
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, self.fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, cpu_offload)
            fsdp_module = actor_module
        else:
            raise NotImplementedError(f"{self.fsdp_strategy} not implemented")

        return fsdp_module

    def _fsdp_init_train_model(self, model, optimizer, scheduler):
        """Initialize a model for training with FSDP"""
        is_actor = isinstance(model, Actor)
        fsdp_module = self._fsdp_init_model(model, is_train=True, is_actor=is_actor)

        optim_config = self.optimizer_config
        if optim_config is not None:
            actor_optimizer = optim.AdamW(
                fsdp_module.parameters(),
                lr=optim_config.lr,
                betas=optim_config.adam_betas,
                weight_decay=optim_config.weight_decay,
            )

            actor_lr_scheduler = get_scheduler(
                optim_config.scheduler,
                actor_optimizer,
                num_warmup_steps=optim_config.num_warmup_steps,
                num_training_steps=self.total_training_steps,
            )
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        if is_actor:
            model.model = fsdp_module
        else:
            model = fsdp_module

        return model, actor_optimizer, actor_lr_scheduler

    def _fsdp_init_eval_model(self, model):
        """Initialize a model for evaluation with FSDP"""
        is_actor = isinstance(model, Actor)
        fsdp_module = self._fsdp_init_model(model, is_train=False, is_actor=is_actor)

        if is_actor:
            model.model = fsdp_module
        else:
            model = fsdp_module

        return model

    def all_reduce(self, data, op="mean"):
        """Perform all_reduce across all processes"""
        assert op in ("mean", "max", "sum")
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_reduce(v, op)
            return ret
        else:
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"

            if is_cpu_tensor:
                data = data.to(torch.cuda.current_device())
            if op == "mean":
                data /= self.world_size
            dist.all_reduce(
                data, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM
            )
            if is_cpu_tensor:
                data = data.cpu()
            return data.item() if not is_tensor else data

    def all_gather(self, data):
        """Perform all_gather across all processes"""
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_gather(v)
            return ret
        else:
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
            is_cpu_tensor = data.device.type == "cpu"

            ret = [
                torch.zeros_like(data).to(torch.cuda.current_device())
                for _ in range(self.world_size)
            ]
            dist.all_gather(ret, data.to(torch.cuda.current_device()))
            return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)

    def _unwrap_model(self, model) -> nn.Module:
        """Unwrap model from Actor or FSDP"""
        # Handle Actor wrapper
        if isinstance(model, Actor):
            return self._unwrap_model(model.model)

        # For FSDP2 models, check if the FSDP model itself has the necessary attributes
        model_type = type(model).__name__
        if "FSDP" in model_type:
            has_config = hasattr(model, "config")
            has_lm_head = hasattr(model, "lm_head")
            has_generate = hasattr(model, "generate")
            if has_config and (has_lm_head or has_generate):
                return model

        # Check for FSDP v1 unwrapping
        if hasattr(model, "_fsdp_wrapped_module"):
            return model._fsdp_wrapped_module

        # If no unwrapping needed, return the original model
        return model

    def save_ckpt(
        self,
        model,
        ckpt_dir,
        global_step,
        node_local_rank,
        optimizer=None,
        scheduler=None,
        client_state={},
        tag=None,
        tokenizer=None,
    ):
        """Save model checkpoint for FSDP"""
        import warnings
        from torch.distributed.fsdp import (
            ShardedStateDictConfig,
            ShardedOptimStateDictConfig,
            StateDictType,
        )

        if node_local_rank == 0:
            os.makedirs(ckpt_dir, exist_ok=True)

        # Wait for checkpoint directory to be created.
        dist.barrier()

        # Extract the actual model for saving
        if isinstance(model, Actor):
            save_model = model.model
        else:
            save_model = model

        if self.fsdp_strategy not in ("fsdp", "fsdp2"):
            raise ValueError(f"Unsupported FSDP strategy: {self.fsdp_strategy}")

        # Set up state dict configurations for sharded saving
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)

        # Define paths for saving individual rank files
        rank = self.get_rank()
        world_size = self.world_size
        model_path = os.path.join(
            ckpt_dir, f"model_world_size_{world_size}_rank_{rank}.pt"
        )
        optim_path = os.path.join(
            ckpt_dir, f"optim_world_size_{world_size}_rank_{rank}.pt"
        )
        extra_path = os.path.join(
            ckpt_dir, f"extra_state_world_size_{world_size}_rank_{rank}.pt"
        )

        # Save using appropriate FSDP context
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(
                save_model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg
            ):
                # Get and save model state dict
                model_state_dict = save_model.state_dict()
                self.print(
                    f"[rank-{rank}]: Saving model to {os.path.abspath(model_path)}"
                )
                torch.save(model_state_dict, model_path)

                # Get and save optimizer state dict if optimizer is provided
                optimizer_state_dict = {}
                if optimizer is not None:
                    optimizer_state_dict = optimizer.state_dict()
                self.print(
                    f"[rank-{rank}]: Saving optim to {os.path.abspath(optim_path)}"
                )
                torch.save(optimizer_state_dict, optim_path)

                # Get scheduler state dict if scheduler is provided
                lr_scheduler_state_dict = {}
                if scheduler is not None:
                    lr_scheduler_state_dict = scheduler.state_dict()

                # Create extra state dict with client state and any additional info
                extra_state_dict = {
                    "lr_scheduler": lr_scheduler_state_dict,
                    "client_state": client_state,
                    "tag": tag,
                    "fsdp_strategy": self.fsdp_strategy,
                    "world_size": world_size,
                    "rank": rank,
                    "global_step": global_step,
                    "rng": self.get_rng_state(),  # Add RNG state for reproducibility
                }

                # Save extra state
                self.print(
                    f"[rank-{rank}]: Saving extra_state to {os.path.abspath(extra_path)}"
                )
                torch.save(extra_state_dict, extra_path)

                # Garbage collect temporary buffers from materializing the state dicts
                gc.collect()

        if self.is_rank_0():
            config_save_model = self._unwrap_model(model)
            self.save_hf_configs(config_save_model, ckpt_dir, tokenizer)

            # Also save runtime FSDP config
            fsdp_config_path = os.path.join(ckpt_dir, "fsdp_config.json")
            with open(fsdp_config_path, "w") as f:
                json.dump(
                    {
                        "fsdp_strategy": self.fsdp_strategy,
                        "world_size": self.world_size,
                    },
                    f,
                    indent=4,
                )

        # Final barrier to ensure all operations complete
        dist.barrier()
        torch.cuda.synchronize()
        self.print(f"[rank-{rank}]: Checkpoint saved to {ckpt_dir}")

    def load_ckpt(
        self,
        model,
        ckpt_dir,
        optimizer=None,
        scheduler=None,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
    ):
        """Load model checkpoint for FSDP"""
        import warnings
        from torch.distributed.fsdp import (
            ShardedStateDictConfig,
            ShardedOptimStateDictConfig,
            StateDictType,
        )

        if ckpt_dir is None:
            raise ValueError("ckpt_dir cannot be None")
        elif not os.path.exists(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        # Extract the actual model for loading
        load_model = model
        if isinstance(model, Actor):
            load_model = model.model

        # Define paths for loading individual rank files
        rank = self.get_rank()
        world_size = self.world_size
        model_path = os.path.join(
            ckpt_dir, f"model_world_size_{world_size}_rank_{rank}.pt"
        )
        optim_path = os.path.join(
            ckpt_dir, f"optim_world_size_{world_size}_rank_{rank}.pt"
        )
        extra_path = os.path.join(
            ckpt_dir, f"extra_state_world_size_{world_size}_rank_{rank}.pt"
        )

        # Check if checkpoint files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        if not os.path.exists(extra_path):
            raise FileNotFoundError(f"Extra state checkpoint not found: {extra_path}")

        # Optimizer path is optional since we may not save optimizer states initially
        optim_exists = os.path.exists(optim_path)

        self.print(f"[rank-{rank}]: Loading model from {os.path.abspath(model_path)}")
        self.print(
            f"[rank-{rank}]: Loading extra_state from {os.path.abspath(extra_path)}"
        )
        if optim_exists:
            self.print(
                f"[rank-{rank}]: Loading optim from {os.path.abspath(optim_path)}"
            )

        # Load state dictionaries from disk
        model_state_dict = torch.load(
            model_path, map_location="cpu", weights_only=False
        )
        extra_state_dict = torch.load(
            extra_path, map_location="cpu", weights_only=False
        )

        optimizer_state_dict = {}
        if optim_exists and load_optimizer_states and not load_module_only:
            optimizer_state_dict = torch.load(
                optim_path, map_location="cpu", weights_only=False
            )

        # Extract scheduler state from extra state
        lr_scheduler_state_dict = extra_state_dict.get("lr_scheduler", {})

        # Set up state dict configurations for sharded loading
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)

        # Load using appropriate FSDP context
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(
                load_model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg
            ):
                # Load model state dict
                load_model.load_state_dict(model_state_dict, strict=load_module_strict)
                self.print(f"[rank-{rank}]: Successfully loaded model state dict")

                # Load optimizer state dict if optimizer object is provided and loading is requested
                if (
                    optimizer is not None
                    and load_optimizer_states
                    and not load_module_only
                    and optimizer_state_dict
                ):
                    optimizer.load_state_dict(optimizer_state_dict)
                    self.print(f"[rank-{rank}]: Successfully loaded optimizer state")

                # Load scheduler state dict if scheduler object is provided and loading is requested
                if (
                    scheduler is not None
                    and load_lr_scheduler_states
                    and not load_module_only
                ):
                    scheduler.load_state_dict(lr_scheduler_state_dict)
                    self.print(f"[rank-{rank}]: Successfully loaded scheduler state")

        # Load RNG state for reproducibility
        if "rng" in extra_state_dict:
            self.load_rng_state(extra_state_dict["rng"])

        # Wait for all ranks to finish loading
        dist.barrier()

        # Create states dict with extra information
        client_state = extra_state_dict.get("client_state", {})
        states = {
            "client_state": client_state,
            "tag": extra_state_dict.get("tag", tag),
            "fsdp_strategy": extra_state_dict.get("fsdp_strategy", self.fsdp_strategy),
            "world_size": extra_state_dict.get("world_size", world_size),
            "rank": extra_state_dict.get("rank", rank),
            "global_step": extra_state_dict.get(
                "global_step", 0
            ),  # Include global_step in return
        }

        self.print(f"[rank-{rank}]: Checkpoint loaded successfully from {ckpt_dir}")

        return ckpt_dir, states

    # TODO (erictang000): Test in multi-node setting
    def save_hf_model(
        self, model: Union[Actor, nn.Module], output_dir: str, tokenizer=None, **kwargs
    ) -> None:
        """Save model in HuggingFace safetensors format using FSDP's full state dict gathering"""

        # Step 1: Create output directory (rank 0 only)
        if self.is_rank_0():
            os.makedirs(output_dir, exist_ok=True)
            self.print(f"[rank-0]: Created output directory: {output_dir}")

        # Step 2: Extract models - get both the model for saving metadata and the FSDP model for state dict
        model_to_save = self._unwrap_model(model)  # For saving config/metadata
        fsdp_model = (
            model.model if isinstance(model, Actor) else model
        )  # For state dict collection

        # Validate that we have a proper HuggingFace model
        if not hasattr(model_to_save, "config") or not hasattr(
            model_to_save, "save_pretrained"
        ):
            raise ValueError(
                "Model must be a HuggingFace model with config and save_pretrained method"
            )

        # Step 3: Determine FSDP version and collect full state dict
        fsdp_ver = fsdp_version(fsdp_model)
        self.print(f"[rank-{self.get_rank()}]: Detected FSDP version: {fsdp_ver}")

        if fsdp_ver == 2:
            # Use FSDP2 API - collects on rank 0 only
            output_state_dict = fsdp2_get_full_state_dict(
                fsdp_model, cpu_offload=True, rank0_only=True
            )
        elif fsdp_ver == 1:
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
            )

            options = StateDictOptions(
                full_state_dict=True, cpu_offload=True, broadcast_from_rank0=False
            )
            output_state_dict = get_model_state_dict(fsdp_model, options=options)
            if not self.is_rank_0():
                output_state_dict.clear()
        else:
            raise ValueError(f"Unsupported FSDP version: {fsdp_ver}")

        # Step 4: Save on rank 0 only
        if self.is_rank_0():
            # Save the model in HuggingFace format using safetensors
            model_to_save.save_pretrained(
                output_dir,
                state_dict=output_state_dict,
                safe_serialization=True,
                **kwargs,  # Always use safetensors
            )

            # Determine which config to save
            config_to_save = model_to_save.config

            # Fix architecture name by removing FSDP prefix if present
            if (
                hasattr(config_to_save, "architectures")
                and config_to_save.architectures
            ):
                # Create a copy of the config to avoid modifying the original
                config_to_save = copy.deepcopy(config_to_save)

                # Fix architecture names to remove FSDP prefix
                fixed_architectures = []
                for arch in config_to_save.architectures:
                    fixed_arch = arch
                    if arch.startswith("FSDP"):
                        # Remove "FSDP" prefix (for fsdp2)
                        fixed_arch = arch[len("FSDP") :]
                        self.print(
                            f"[rank-0]: Fixed architecture name: {arch} -> {fixed_arch}"
                        )
                    fixed_architectures.append(fixed_arch)

                config_to_save.architectures = fixed_architectures

            # Save the config
            config_to_save.save_pretrained(output_dir)

            # Save tokenizer if provided
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)

            self.print(f"[rank-0]: Successfully saved model to {output_dir}")

        dist.barrier()
