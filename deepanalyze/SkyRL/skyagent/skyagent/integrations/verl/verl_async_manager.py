import asyncio
import heapq
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from tensordict import TensorDict
from transformers import AutoTokenizer

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr, rollout_trace_op
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager
from verl.workers.rollout.async_server import AsyncServerBase


from verl.experimental.agent_loop.agent_loop import AgentLoopWorker, AgentLoopManager

from skyagent import AutoAgentRunner



def async_server_class(
    rollout_backend: str, rollout_backend_module: Optional[str] = None, rollout_backend_class: Optional[str] = None
) -> type[AsyncServerBase]:
    """Get async server class.

    Args:
        rollout_backend: str, rollout backend type (alias), should be "vllm" or "sglang".
        rollout_backend_module: Optional[str], import path of the rollout backend.
        rollout_backend_class: Optional[str], class name of the rollout backend.

    Returns:
        Type[AsyncServerBase]: async server class.
    """
    if rollout_backend_class is None and rollout_backend_module is None:
        # If both are None, use the default backend class
        # Do not change the original import behavior
        # importlib.import_module and from ... import ... have subtle differences in ray

        if rollout_backend == "vllm":
            from .skagent_async_vllm_server import SkyAgentAsyncvLLMServer

            return SkyAgentAsyncvLLMServer
        elif rollout_backend == "sglang":
            from .skyagent_async_sglang_server import SkyAgentAsyncSglangServer

            raise NotImplementedError("Sglang backend for verl with skyagent is not implemented right now")

            return AsyncSglangServer
        else:
            raise NotImplementedError(f"rollout backend {rollout_backend} is not supported")

    if rollout_backend_module is None or rollout_backend_class is None:
        raise ValueError("rollout_backend_module and rollout_backend_class must be both provided for customization")

    from verl.utils.import_utils import load_extern_type

    return load_extern_type(rollout_backend_module, rollout_backend_class)

class SkyAgentLoopManager(AgentLoopManager):
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group.
        """
        self.config = config
        self.worker_group = worker_group

        self._initialize_llm_servers()
        # self._init_agent_loop_workers()
        
        # init tokenizer
        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

        # init generator
        self.server_manager = AsyncLLMServerManager(config, self.async_llm_servers)
        self.skyagent_generator = AutoAgentRunner.from_task(task_yaml=config.skyagent.task_yaml, infer_engine=self.server_manager,
            tokenizer=self.tokenizer)

        # Initially we're in sleep mode.
        self.sleep()
    
    # initialize here with the custom `async_server_class` implementation
    def _initialize_llm_servers(self):
        self.rollout_tp_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size

        register_center = ray.get_actor(f"{self.worker_group.name_prefix}_register_center")
        workers_info = ray.get(register_center.get_worker_info.remote())
        assert len(workers_info) == self.worker_group.world_size

        self.async_llm_servers = [None] * self.rollout_dp_size
        self.server_addresses = [None] * self.rollout_dp_size

        if self.config.actor_rollout_ref.rollout.agent.custom_async_server:
            server_class = async_server_class(
                rollout_backend=self.config.actor_rollout_ref.rollout.name,
                rollout_backend_module=self.config.actor_rollout_ref.rollout.agent.custom_async_server.path,
                rollout_backend_class=self.config.actor_rollout_ref.rollout.agent.custom_async_server.name,
            )
        else:
            server_class = async_server_class(rollout_backend=self.config.actor_rollout_ref.rollout.name)

        # Start all server instances, restart if address already in use.
        unready_dp_ranks = set(range(self.rollout_dp_size))
        while len(unready_dp_ranks) > 0:
            servers = {
                rollout_dp_rank: server_class.options(
                    # make sure AsyncvLLMServer colocates with its corresponding workers
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                        soft=False,
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(self.config, self.rollout_dp_size, rollout_dp_rank, self.worker_group.name_prefix)
                for rollout_dp_rank in unready_dp_ranks
            }

            for rollout_dp_rank, server in servers.items():
                try:
                    address = ray.get(server.get_server_address.remote())
                    self.server_addresses[rollout_dp_rank] = address
                    self.async_llm_servers[rollout_dp_rank] = server
                    unready_dp_ranks.remove(rollout_dp_rank)
                except Exception:
                    ray.kill(server)
                    print(f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting...")

        # All server instances are ready, init AsyncLLM engine.
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])

    def _postprocess(self, inputs: list[Dict[str, Any]]) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # inputs
        self.tokenizer.padding_side = "left"
        max_prompt_length = max(max([len(input_ids) for input_ids in inputs["prompt_token_ids"]]), self.config.actor_rollout_ref.rollout.prompt_length)
        outputs = self.tokenizer.pad(
            [{"input_ids": input_ids} for input_ids in inputs["prompt_token_ids"]],
            padding="max_length",
            max_length=max_prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        prompt_ids, prompt_attention_mask = outputs["input_ids"], outputs["attention_mask"]

        # responses
        self.tokenizer.padding_side = "right"
        max_response_length = max(max([len(response) for response in inputs["response_ids"]]), self.config.actor_rollout_ref.rollout.response_length)
        outputs = self.tokenizer.pad(
            [{"input_ids": response_ids} for response_ids in inputs["response_ids"]],
            padding="max_length",
            max_length=max_response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        response_ids, response_attention_mask = outputs["input_ids"], outputs["attention_mask"]

        # response_mask
        response_length = response_ids.shape[1]
        loss_masks = [loss_mask + [0]*(response_length - len(loss_mask)) for loss_mask in inputs["loss_masks"]]
        response_mask = torch.tensor(loss_masks, dtype=torch.long)
        assert response_ids.shape == response_mask.shape, (
            f"mismatch in response_ids and response_mask shape: {response_ids.shape} vs {response_mask.shape}"
        )
        response_mask = response_mask * response_attention_mask

        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
            },
            batch_size=len(input_ids),
        )

        return DataProto(batch=batch, non_tensor_batch={"rewards": np.array(inputs["rewards"])}, meta_info={"rollout_metrics": inputs["rollout_metrics"], "timing": {}})

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        skyagent_output = asyncio.run(self.skyagent_generator.run(prompts, val_mode=prompts.meta_info.get("val_mode", False)))
        output = self._postprocess(skyagent_output)
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        return output

    def wake_up(self):
        """Wake up all rollout server instances."""
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    def sleep(self):
        """Sleep all rollout server instances."""
        ray.get([server.sleep.remote() for server in self.async_llm_servers])