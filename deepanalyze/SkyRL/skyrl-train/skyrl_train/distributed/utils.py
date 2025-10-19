"""Communication utilities.

This file currently provides the following functionalities, with code mainly
sourced from PyTorch:

1. Provides init process group capability without being restricted by default
   process group. PyTorch assumes users always use its default process group.
2. Provides CUDAIPC capability, which allows bypassing torch's multiprocessing
   to use GPU shared memory, for example to communicate with vllm workers using
   shared memory.
"""

from __future__ import annotations

import socket
from datetime import timedelta
from typing import Any, Optional, Union, Tuple

import torch
import torch.distributed
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)
from torch.multiprocessing.reductions import rebuild_cuda_tensor
import torch.nn as nn
from torch.optim import Optimizer

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]


def get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
# https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/distributed_util.py
def init_custom_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    assert (store is None) or (
        init_method is None
    ), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    pg_options_param_name = (
        "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    )
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    no_decay_name_list=[
        "bias",
        "layer_norm.weight",
        "layernorm.weight",
        "norm.weight",
        "ln_f.weight",
    ],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


class CUDAIPCHandle:
    def __init__(
        self,
        tensor_type: type,
        size: tuple,
        stride: tuple,
        offset: int,
        storage_type: type,
        dtype: torch.dtype,
        device: torch.device,
        handle: bytes,
        storage_size_bytes: bytes,
        storage_offset_bytes: bytes,
        requires_grad: bool,
        ref_counter_handle: bytes,
        ref_counter_offset: bytes,
        event_handle: bytes,
        event_sync_required: bool,
    ):
        self.tensor_type = tensor_type
        self.size = size
        self.stride = stride
        self.offset = offset
        self.storage_type = storage_type
        self.dtype = dtype
        self.device = device
        self.handle = handle
        self.storage_size_bytes = storage_size_bytes
        self.storage_offset_bytes = storage_offset_bytes
        self.requires_grad = requires_grad
        self.ref_counter_handle = ref_counter_handle
        self.ref_counter_offset = ref_counter_offset
        self.event_handle = event_handle
        self.event_sync_required = event_sync_required

    def rebuild(self) -> torch.Tensor:
        # NOTE: Rebuild within the same process is not thread-safe and will
        #       likely crash (segfault core dump).
        return rebuild_cuda_tensor(
            self.tensor_type,
            self.size,
            self.stride,
            self.offset,
            self.storage_type,
            self.dtype,
            self.device,
            self.handle,
            self.storage_size_bytes,
            self.storage_offset_bytes,
            self.requires_grad,
            self.ref_counter_handle,
            self.ref_counter_offset,
            self.event_handle,
            self.event_sync_required,
        )

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> CUDAIPCHandle:
        if not tensor.is_cuda:
            raise ValueError("Tensor must be on CUDA device to use CUDAIPC")
        tensor = tensor.share_memory_()
        storage = tensor._typed_storage()
        (
            device,
            handle,
            storage_size_bytes,
            storage_offset_bytes,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        ) = storage._share_cuda_()
        tensor_offset = tensor.storage_offset()
        return CUDAIPCHandle(
            tensor_type=type(tensor),
            size=tensor.size(),
            stride=tensor.stride(),
            # tensor offset in its storage
            offset=tensor_offset,
            storage_type=type(storage),
            dtype=tensor.dtype,
            device=device,
            # identifier which CUDA allocation is the storage in.
            handle=handle,
            # size(in bytes) of the storage
            storage_size_bytes=storage_size_bytes,
            # offset(in bytes) of the storage in the CUDA allocation
            storage_offset_bytes=storage_offset_bytes,
            requires_grad=tensor.requires_grad,
            ref_counter_handle=ref_counter_handle,
            ref_counter_offset=ref_counter_offset,
            event_handle=event_handle,
            event_sync_required=event_sync_required,
        )
