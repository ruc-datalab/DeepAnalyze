# This code is adapted from OpenRLHF
# https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/deepspeed/deepspeed.py

import os
import random
import shutil
from collections import defaultdict
from datetime import timedelta
from typing import List, Union, Optional
from omegaconf import OmegaConf
from jaxtyping import Float

import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from torch import distributed as dist
from torch.optim import Optimizer
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum

from skyrl_train.distributed.strategy import DistributedStrategy
from skyrl_train.models import Actor
from skyrl_train.distributed.utils import (
    get_optimizer_grouped_parameters,
    ModelOrModelOptimPair,
)

from safetensors.torch import save_file
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint


def _z3_params_to_fetch(param_list):
    return [
        p
        for p in param_list
        if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


class DeepspeedStrategy(DistributedStrategy):
    """
    The strategy for training with Accelerator.
    """

    def __init__(
        self,
        deepspeed_config,
        seed: int = 42,
        micro_train_batch_size_per_gpu=1,
        train_batch_size=1,
        zero_stage=3,
        bf16=True,
    ) -> None:
        super().__init__()

        self.deepspeed_config = deepspeed_config
        self.stage = zero_stage
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size_per_gpu = micro_train_batch_size_per_gpu
        self.bf16 = bf16
        self.seed = seed

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

        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed(timeout=timeout)
        self.world_size = dist.get_world_size()
        self.accumulated_gradient = (
            self.train_batch_size
            // self.micro_train_batch_size_per_gpu
            // self.world_size
        )

    def create_optimizer(self, model, offload_after_step=True, **kwargs) -> Optimizer:
        if isinstance(model, Actor):
            model = model.model
        # TODO (sumanthrh): Support this
        if not offload_after_step:
            raise NotImplementedError(
                "Disabling offload after step is not supported for deepspeed"
            )
        # Optimizer
        cpu_optimizer = (
            self.deepspeed_config.zero_optimization.offload_optimizer.device == "cpu"
        )
        AdamOptimizer = DeepSpeedCPUAdam if cpu_optimizer else FusedAdam
        optim_params = get_optimizer_grouped_parameters(model, kwargs["weight_decay"])
        optim = AdamOptimizer(optim_params, **kwargs)
        return optim

    def offload_to_cpu(self, model, pin_memory=True, non_blocking=True):
        """This function guaratees the memory are all released (only torch context cache <100M will remain)."""
        if isinstance(model, Actor):
            model = model.model

        if model.config["zero_optimization"]["offload_optimizer"]["device"] == "cpu":
            # if doing optimizer offload, no need to offload states
            return
        elif model.zero_optimization_stage() == 3:
            from deepspeed.runtime.zero.offload_config import OffloadStateTypeEnum

            model.optimizer.offload_states(
                include=[
                    OffloadStateTypeEnum.optim_states,
                    OffloadStateTypeEnum.contiguous_grad_buffer,
                    OffloadStateTypeEnum.hp_params,
                    # this will break for deepspeed < 0.16.5, https://github.com/deepspeedai/DeepSpeed/pull/7050
                    OffloadStateTypeEnum.lp_grads,
                    # OffloadStateTypeEnum.lp_params, # dangerous
                ],
                device=OffloadDeviceEnum.cpu,
                pin_memory=pin_memory,
                non_blocking=non_blocking,
            )
            torch.cuda.synchronize()
            return

        raise NotImplementedError("Zero stage < 3 is not supported")

    def backload_to_gpu(self, model, non_blocking=True):
        # NOTE: this function reloads the weights, ensuring the calculation
        if isinstance(model, Actor):
            model = model.model
        else:
            model = model

        if model.config["zero_optimization"]["offload_optimizer"]["device"] == "cpu":
            # if doing optimizer offload, no need to reload states
            return
        if model.zero_optimization_stage() == 3:
            model.reload_states(non_blocking=non_blocking)
            torch.cuda.synchronize()
            return

        raise NotImplementedError("Zero stage < 3 is not supported")

    def backward(
        self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs
    ) -> None:
        if isinstance(model, Actor):
            model = model.model
        model.backward(loss)

    # TODO(sumanthrh): Support logging grad norm here and verify grad clipping.
    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        scheduler,
        name="model",
        **kwargs,
    ) -> Optional[Float[torch.Tensor, "1"]]:
        if isinstance(model, Actor):
            model = model.model
        model.step()

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        ret = []
        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                assert (
                    len(arg) == 3
                ), f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'
                ret.append(self._ds_init_train_model(*arg))
            else:
                ret.append(self._ds_init_eval_model(arg))

        return ret[0] if len(ret) == 1 else ret

    def _ds_init_train_model(self, model, optim, scheduler):
        is_actor = isinstance(model, Actor)
        ds_config = self.get_ds_train_config()

        engine, optim, _, scheduler = deepspeed.initialize(
            model=model.model if is_actor else model,
            optimizer=optim,
            lr_scheduler=scheduler,
            config=ds_config,
            dist_init_required=True,
        )
        if is_actor:
            model.model = engine
        else:
            model = engine

        return model, optim, scheduler

    def _ds_init_eval_model(self, model):
        if not model:
            return model
        is_actor = isinstance(model, Actor)
        ds_config = self.get_ds_eval_config()

        engine, *_ = deepspeed.initialize(
            model=model.model if is_actor else model,
            config=ds_config,
            dist_init_required=True,
        )
        if is_actor:
            model.model = engine
        else:
            model = engine
        return model

    def all_reduce(self, data, op="mean"):
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
        if isinstance(model, Actor):
            return self._unwrap_model(model.model)
        elif hasattr(model, "module"):
            return model.module
        else:
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
        if isinstance(model, Actor):
            model = model.model

        assert isinstance(model, deepspeed.DeepSpeedEngine)

        if node_local_rank == 0:
            os.makedirs(ckpt_dir, exist_ok=True)

        dist.barrier()

        extra_state_dict = {
            "client_state": client_state,
            "deepspeed_config": OmegaConf.to_container(self.deepspeed_config),
            "global_step": global_step,
            "rng": self.get_rng_state(),  # Add RNG state for reproducibility
        }

        model.save_checkpoint(ckpt_dir, tag=tag, client_state=extra_state_dict)

        # Save HuggingFace config and tokenizer
        if self.is_rank_0():
            config_save_model = self._unwrap_model(model)
            self.save_hf_configs(config_save_model, ckpt_dir, tokenizer)

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
        if isinstance(model, Actor):
            model = model.model

        assert isinstance(model, deepspeed.DeepSpeedEngine)
        load_path, states = model.load_checkpoint(
            ckpt_dir,
            tag,
            load_module_strict=load_module_strict,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,  # DeepSpeed handles this automatically
            load_module_only=load_module_only,
        )
        if load_path is None:
            raise Exception(f"[deepspeed] failed to resume from checkpoint {ckpt_dir}")

        # Load RNG state for reproducibility (if present)
        if "rng" in states:
            self.load_rng_state(states["rng"])
            if self.is_rank_0():
                self.print(
                    f"[rank-{self.get_rank()}]: Loaded RNG state from checkpoint"
                )

        return load_path, states

    def save_hf_model(
        self, model: nn.Module, output_dir: str, tokenizer=None, **kwargs
    ) -> None:
        """
        Save only the model weights into a single HuggingFace‐compatible `model.safetensors`
        by doing a temporary DeepSpeed checkpoint → FP32 state_dict → safetensors.
        """
        # Unwrap Actor if necessary
        if isinstance(model, Actor):
            model = model.model
        assert isinstance(
            model, deepspeed.DeepSpeedEngine
        ), "Expected a DeepSpeedEngine"

        # Rank 0 makes directories or writes files
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

        # Create a temporary DS checkpoint folder (only on rank 0)
        temp_ckpt_dir = os.path.join(output_dir, "temp_deepspeed_ckpt")
        if rank == 0:
            os.makedirs(temp_ckpt_dir, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

        # Use DeepSpeed to write a ZeRO checkpoint. The parameter shards will land
        # under temp_ckpt_dir/model_conversion/mp_rank_XX/…
        model.save_checkpoint(temp_ckpt_dir, tag="model_conversion")
        if dist.is_initialized():
            dist.barrier()

        if rank == 0:
            # Gather all shards from that DS checkpoint into one CPU FP32 state_dict
            fp32_state_dict = get_fp32_state_dict_from_zero_checkpoint(
                temp_ckpt_dir, tag="model_conversion"
            )

            # Handle tied embeddings if needed (e.g. Qwen2‐0.5B)
            unwrapped_model = self._unwrap_model(model)
            if (
                getattr(unwrapped_model.config, "tie_word_embeddings", False)
                and "lm_head.weight" in fp32_state_dict
            ):
                fp32_state_dict.pop("lm_head.weight", None)

            # Write the single-file safetensors
            safetensors_path = os.path.join(output_dir, "model.safetensors")
            save_file(fp32_state_dict, safetensors_path)

            # Save the config.json so we can re-create the same architecture later
            unwrapped_model.config.save_pretrained(output_dir)

            # If a tokenizer was passed, save it here too
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)

            # Clean up the temporary checkpoint folder
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)

        dist.barrier()

    def get_ds_train_config(self):
        ds_config = OmegaConf.to_container(self.deepspeed_config)
        disable_trace_cache = ds_config.pop("disable_trace_cache", False)
        if disable_trace_cache:
            ds_config["zero_optimization"]["stage3_prefetch_bucket_size"] = 0
            ds_config["zero_optimization"]["stage3_max_live_parameters"] = 0
            ds_config["zero_optimization"]["stage3_max_reuse_distance"] = 0
        ds_config["steps_per_print"] = 100
        ds_config["bf16"] = {"enabled": self.bf16}

        # these need to be specified for deepspeed setup, but we manually handle
        # gradient accumulation in the training loop
        ds_config["train_micro_batch_size_per_gpu"] = (
            self.micro_train_batch_size_per_gpu
        )
        ds_config["gradient_accumulation_steps"] = 1

        return ds_config

    def get_ds_eval_config(self):
        ds_config = OmegaConf.to_container(self.deepspeed_config)
        ds_config["steps_per_print"] = 100
        ds_config["bf16"] = {"enabled": self.bf16}
        ds_config["train_micro_batch_size_per_gpu"] = (
            self.micro_train_batch_size_per_gpu
        )
        ds_config["gradient_accumulation_steps"] = 1

        return ds_config
