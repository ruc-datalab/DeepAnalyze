# This code is adapted from VERL
# https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
# The original copyright is reproduced below:
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from contextlib import nullcontext
from typing import Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from transformers.trainer_pt_utils import get_module_class_from_name
from torch.distributed.device_mesh import init_device_mesh

from packaging import version

if version.parse(torch.__version__) >= version.parse("2.6"):
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
        fully_shard,
    )
elif version.parse(torch.__version__) >= version.parse("2.4"):
    from torch.distributed._composable.fsdp import (
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
        fully_shard,
    )
else:
    fully_shard, MixedPrecisionPolicy, FSDPModule, CPUOffloadPolicy = (
        None,
        None,
        None,
        None,
    )


def init_fn(x: torch.nn.Module):
    if torch.distributed.get_rank() != 0:
        x = x.to_empty(device=torch.cuda.current_device(), recurse=False)
        torch.cuda.empty_cache()
    return x


def get_init_weight_context_manager(use_meta_tensor=True, mesh: DeviceMesh = None):
    from accelerate import init_empty_weights

    def cpu_init_weights():
        return torch.device("cpu")

    if use_meta_tensor:
        if mesh is None:
            init_context = (
                init_empty_weights
                if torch.distributed.get_rank() != 0
                else cpu_init_weights
            )
        else:
            init_context = (
                init_empty_weights
                if mesh.get_coordinate()[-1] != 0
                else cpu_init_weights
            )
    else:
        init_context = cpu_init_weights
    return init_context


def get_fsdp_wrap_policy(module, config=None, is_lora=False):
    """Get FSDP wrap policy for the module.

    Args:
        module: The module to get wrap policy for
        config: Configuration for wrap policy
        is_lora: Whether to enable lambda policy for LoRA modules
    """
    if config is None:
        config = {}

    def _get_attr(attr_name, default_value=None):
        if hasattr(config, "get"):
            return config.get(attr_name, default_value)
        else:
            return getattr(config, attr_name, default_value)

    if _get_attr("disable", False):
        return None

    default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = _get_attr(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )
    min_num_params = _get_attr("min_num_params", 0)
    auto_wrap_policy = None

    policies = []

    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy

    # Add lambda policy for LoRA modules if is_lora is True
    if is_lora:

        def lambda_policy_fn(module):
            return bool(
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )

        lambda_policy = functools.partial(
            lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn
        )
        policies.append(lambda_policy)

    if min_num_params > 0:
        size_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=min_num_params
        )
        policies.append(size_policy)
    elif fsdp_transformer_layer_cls_to_wrap is not None:
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(module, layer_class)
            if transformer_cls is None:
                raise Exception(
                    "Could not find the transformer layer class to wrap in the model."
                )
            else:
                transformer_cls_to_wrap.add(transformer_cls)

        transformer_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_cls_to_wrap,
        )
        policies.append(transformer_policy)

    if len(policies) > 0:
        auto_wrap_policy = functools.partial(_or_policy, policies=policies)

    return auto_wrap_policy


@torch.no_grad()
def offload_fsdp_model_to_cpu(model: FSDP, empty_cache: bool = True):
    if fsdp_version(model) == 2:
        offload_fsdp2_model_to_cpu(model, empty_cache)
        return

    assert isinstance(model, FSDP)
    # lazy init FSDP model
    _lazy_init(model, model)
    assert model._is_root, "Only support root model offloading to CPU"
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        assert (
            flat_param.data.data_ptr() == flat_param._local_shard.data_ptr()
            and id(flat_param.data) != id(flat_param._local_shard)
            and flat_param.data.size() == flat_param._local_shard.size()
        )
        handle.flat_param_to(torch.device("cpu"), non_blocking=True)
        # the following still keeps id(._local_shard) != id(.data)
        flat_param._local_shard = flat_param.data
        assert id(flat_param._local_shard) != id(flat_param.data)
    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def offload_fsdp2_model_to_cpu(model, empty_cache: bool = True):
    model.to("cpu", non_blocking=True)
    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def load_fsdp_model_to_gpu(model: FSDP):
    if fsdp_version(model) == 2:
        load_fsdp2_model_to_gpu(model)
        return

    assert isinstance(model, FSDP)
    # lazy init FSDP model
    _lazy_init(model, model)
    assert model._is_root, "Only support root model loading to GPU"
    device_id = torch.cuda.current_device()
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        handle.flat_param_to(torch.device(f"cuda:{device_id}"), non_blocking=True)
        # the following still keeps id(._local_shard) != id(.data)
        flat_param._local_shard = flat_param.data


@torch.no_grad()
def load_fsdp2_model_to_gpu(model):
    device = torch.cuda.current_device()
    model.to(device, non_blocking=True)


@torch.no_grad()
def offload_fsdp_optimizer(optimizer):
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)


@torch.no_grad()
def load_fsdp_optimizer(optimizer, device_id):
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)


def fsdp_version(model):
    if isinstance(model, FSDP):
        return 1
    elif FSDPModule is not None and isinstance(model, FSDPModule):
        return 2
    else:
        return 0


def get_fsdp_state_ctx(model, state_type, state_cfg, optim_cfg):
    if fsdp_version(model) == 1:
        return FSDP.state_dict_type(model, state_type, state_cfg, optim_cfg)
    else:
        return nullcontext()


# Fsdp2 load full state dict from `accelerate`
# Reference: https://github.com/huggingface/accelerate/blob/0af621bbecc0e43f5d43766a4945d3d2236bb8a9/src/accelerate/utils/fsdp_utils.py#L455
# NOTE (sumanthrh): The original code from `accelerate` assumes init on meta device - with cpu init only on rank 0, but the code is compatible with cpu init on all ranks.
def fsdp2_load_full_state_dict(model: torch.nn.Module, full_sd: dict, cpu_offload=None):
    """
    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        model (`torch.nn.Module`):
            The model to load the state dict into, expected to be on meta device or a VRAM spike can occur
        full_sd (`dict`): The full state dict to load, can be only on rank 0
    """
    import torch.distributed as dist
    from torch.distributed.tensor import distribute_tensor

    # Model was previously copied to meta device
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}

    # Rank 0 distributes the full state dict to other ranks
    def _infer_parameter_dtype(model, param_name, empty_param):
        try:
            old_param = model.get_parameter_or_buffer(param_name)
        except AttributeError:
            # Need this for LORA, as there some params are not *parameters* of sorts
            base_param_name, local_param_name = param_name.rsplit(".", 1)
            submodule = model.get_submodule(base_param_name)
            old_param = getattr(submodule, local_param_name)

        is_torch_e4m3fn_available = hasattr(torch, "float8_e4m3fn")
        casting_dtype = None
        is_param_float8_e4m3fn = (
            is_torch_e4m3fn_available and empty_param.dtype == torch.float8_e4m3fn
        )

        if empty_param.dtype.is_floating_point and not is_param_float8_e4m3fn:
            casting_dtype = old_param.dtype

        return old_param is not None and old_param.is_contiguous(), casting_dtype

    def _cast_and_contiguous(tensor, to_contiguous, dtype):
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if to_contiguous:
            tensor = tensor.contiguous()
        return tensor

    if dist.get_rank() == 0:
        for (param_name, full_param), sharded_param in zip(
            full_sd.items(), meta_sharded_sd.values()
        ):
            full_param = full_param.detach().cuda()
            mesh = sharded_param.device_mesh
            dist.broadcast(full_param, src=0, group=mesh.get_group())
            sharded_tensor = distribute_tensor(
                full_param, mesh, sharded_param.placements
            )
            to_contiguous, casting_dtype = _infer_parameter_dtype(
                model,
                param_name,
                full_param,
            )
            sharded_tensor = _cast_and_contiguous(
                sharded_tensor, to_contiguous, casting_dtype
            )
            sharded_sd[param_name] = sharded_tensor
    # We need this else to have a matching `broadcast` for all of the ranks, else we deadlock
    else:
        for param_name, sharded_param in meta_sharded_sd.items():
            full_tensor = torch.empty(
                sharded_param.size(), device="cuda", dtype=sharded_param.dtype
            )
            mesh = sharded_param.device_mesh
            dist.broadcast(full_tensor, src=0, group=mesh.get_group())
            sharded_tensor = distribute_tensor(
                full_tensor, mesh, sharded_param.placements
            )
            to_contiguous, casting_dtype = _infer_parameter_dtype(
                model,
                param_name,
                full_tensor,
            )
            sharded_tensor = _cast_and_contiguous(
                sharded_tensor, to_contiguous, casting_dtype
            )
            sharded_sd[param_name] = sharded_tensor

    # we set `assign=True` because our params can be on meta device
    model.load_state_dict(sharded_sd, assign=True)

    # If we don't offload FSDP2 Module to CPU and then back to GPU,
    # it will occupy a large amount of reserved GPU memory，which can not be released using torch.cuda.empty_cache()
    # even if we are using cpu_offload
    # TODO (erictang000): this requires an additional offload + backload, see if this can be avoided
    # Credit: https://github.com/volcengine/verl/pull/1667
    offload_fsdp2_model_to_cpu(model)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    if not cpu_offload:
        load_fsdp2_model_to_gpu(model)
    return model


def fsdp2_get_full_state_dict(
    model: torch.nn.Module, cpu_offload=True, rank0_only=True
):
    """
    Get the full state dict from an FSDP2 model using proper PyTorch FSDP2 APIs.
    This function will gather the complete state dict on rank 0 only by default.

    Args:
        model (`torch.nn.Module`): The FSDP2 model to get state dict from
        cpu_offload (`bool`): Whether to offload to CPU
        rank0_only (`bool`): Whether to gather full state dict only on rank 0

    Returns:
        dict: The full state dict (only on rank 0 if rank0_only=True, empty dict on other ranks)
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    # All ranks must participate in the collective operation
    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=cpu_offload,
        broadcast_from_rank0=False,  # We want to get, not set
    )

    # This must be called on all ranks for the collective operation to work
    state_dict = get_model_state_dict(model, options=options)

    # If rank0_only is True, clear the state_dict on non-rank-0 processes
    if rank0_only and dist.get_rank() != 0:
        # Clear the state dict on non-rank-0 processes to save memory
        state_dict.clear()

    return state_dict


def apply_fsdp2(model, fsdp_kwargs, config):
    """model: AutoModelForCausalLM"""
    assert (
        CPUOffloadPolicy is not None
    ), "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get("wrap_policy", {}).get(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )

    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]

    assert (
        len(fsdp_transformer_layer_cls_to_wrap) > 0
        and fsdp_transformer_layer_cls_to_wrap[0] is not None
    )

    modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (
            isinstance(module, nn.Embedding) and not model.config.tie_word_embeddings
        ):
            modules.append(module)

    for idx, module in enumerate(modules):
        fully_shard(module, **fsdp_kwargs)
    fully_shard(
        model, **fsdp_kwargs
    )  # fsdp2 will not reshard_after_forward for root module


def fsdp2_clip_grad_norm_(
    parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None
):
    """torch.nn.utils.clip_grad_norm_ can't run on cpu parameter DTensor"""
    from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    total_norm = total_norm.to(torch.cuda.current_device(), non_blocking=True)
    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )
    else:
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(world_size // fsdp_size, fsdp_size),
            mesh_dim_names=["ddp", "fsdp"],
        )
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(
            f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2"
        )
    return sharding_strategy


"""
Adapted from Cruise.
"""

HALF_LIST = [16, "16", "fp16", "float16", torch.float16]
FLOAT_LIST = [32, "32", "fp32", "float32", torch.float32]
BFLOAT_LIST = ["bf16", "bfloat16", torch.bfloat16]


class PrecisionType:
    """Type of precision used.

    >>> PrecisionType.HALF == 16
    True
    >>> PrecisionType.HALF in (16, "16")
    True
    """

    HALF = "16"
    FLOAT = "32"
    FULL = "64"
    BFLOAT = "bf16"
    MIXED = "mixed"

    @staticmethod
    def supported_type(precision: Union[str, int]) -> bool:
        return any(x == precision for x in PrecisionType)

    @staticmethod
    def supported_types() -> list[str]:
        return [x.value for x in PrecisionType]

    @staticmethod
    def is_fp16(precision):
        return precision in HALF_LIST

    @staticmethod
    def is_fp32(precision):
        return precision in FLOAT_LIST

    @staticmethod
    def is_bf16(precision):
        return precision in BFLOAT_LIST

    @staticmethod
    def to_dtype(precision):
        if precision in HALF_LIST:
            return torch.float16
        elif precision in FLOAT_LIST:
            return torch.float32
        elif precision in BFLOAT_LIST:
            return torch.bfloat16
        else:
            raise RuntimeError(f"unexpected precision: {precision}")

    @staticmethod
    def to_str(precision):
        if precision == torch.float16:
            return "fp16"
        elif precision == torch.float32:
            return "fp32"
        elif precision == torch.bfloat16:
            return "bf16"
        else:
            raise RuntimeError(f"unexpected precision: {precision}")
