from typing import Any, Dict, TypedDict, List, Optional, Type, Callable
from collections import defaultdict
from abc import ABC, abstractmethod
import os
import copy
from omegaconf import OmegaConf, DictConfig
import pandas as pd
from loguru import logger
from skyagent.tasks.base import BaseTask
from transformers import AutoTokenizer
from dataclasses import dataclass

from skyagent.integrations.base import (
    build_backend,
    build_generator_input,
    build_generator_output,
    _import_object,
    AsyncInferBackend,
)
from skyagent.dispatcher.dispatchers import DISPATCHER_REGISTRY, DispatcherType
from skyagent.config.configuration_utils import (
    TASK_CONFIG_REGISTRY,
    get_field_from_config,
    TrajectoryConfig,
)
from skyagent.functional.chat_template import (
    chat_template,
    chat_template_qwen3_thinking,
)
from .mapping import AGENT_TRAJECTORY_REGISTRY


class CompleterOutput:
    pass


@dataclass
class RuntimeConfig:
    runtime_initializer: Optional[Callable]
    instruction_getter: Callable
    config_builder: Optional[Callable]
    completer: Optional[Callable]
    evaluator: Callable

    @classmethod
    def from_dict(cls, cfg: DictConfig):
        def safe_import(cfg, key):
            try:
                val = cfg.get(key, None)
                return _import_object(val) if val else None
            except AttributeError:
                return None

        runtime_initializer = safe_import(cfg, "initializer")
        config_builder = safe_import(cfg, "config_builder")
        instruction_getter = safe_import(
            cfg, "instruction_getter"
        )  # If optional; else raise if missing
        completer = safe_import(cfg, "completer")
        evaluator = safe_import(cfg, "evaluator")
        return cls(
            runtime_initializer=runtime_initializer,
            config_builder=config_builder,
            instruction_getter=instruction_getter,
            completer=completer,
            evaluator=evaluator,
        )


@dataclass
class TrajectoryResult(TypedDict):
    instance_id: str
    trajectory_id: str
    messages: List[Dict[str, str]]
    state: Any
    results: Optional[CompleterOutput]
    error: Optional[str]
    finish: bool
    finish_reason: str
    reward: Optional[bool]
    eval_error: Optional[str]


class BaseTrajectory(ABC):

    def __init__(
        self,
        cfg: TrajectoryConfig,
        data: Dict[str, Any],
        infer_engine: AsyncInferBackend,
        tokenizer: AutoTokenizer,
        task: BaseTask,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.data = data
        self.infer_engine = infer_engine
        self.tokenizer = tokenizer
        self.task = task
        self.agent_cls = _import_object(cfg.agent_cls)

        self.result: TrajectoryResult = None

    @abstractmethod
    async def initialize_trajectory(self):
        pass

    @abstractmethod
    async def generate_trajectory(self):
        pass

    @abstractmethod
    async def evaluate_trajectory(self):
        pass


# TODO(csy): also specify whether loss_mask, attention_mask, etc. are needed -- for training or eval
class AgentRunner:
    def __init__(self, cfg: Dict[str, Any], infer_engine: Any, tokenizer: Any) -> None:
        """
        Initialize the CodeActGenerator with the given configuration.

        Args:
            generation_config: Configuration dictionary containing parameters like max_prompt_length, max_response_length, etc.
        """
        self.cfg = cfg

        # infer engine
        self.infer_engine = build_backend(
            cfg.generator.infer_backend,
            infer_engine=infer_engine,
            cfg=cfg.generator.backend_config,
        )
        self.tokenizer = tokenizer
        self.traj_cls: Type[BaseTrajectory] = _import_object(
            AGENT_TRAJECTORY_REGISTRY.get(cfg.agent_cls)
        )
        self.task: BaseTask = _import_object(cfg.task)()

        # metadata
        self.trajectories: Dict[str, Dict[str, BaseTrajectory]] = {}

        # Will be set in subclasses
        self.agent_config = None

    @classmethod
    def from_task(cls, task: str, infer_engine: Any, tokenizer: Any):
        # Resolve task name or path
        if os.path.exists(task):
            config_path = task
        elif task in TASK_CONFIG_REGISTRY:
            config_path = TASK_CONFIG_REGISTRY[task]
        else:
            raise ValueError(
                f"Unknown task '{task}'. Must be a YAML path or one of: {list(TASK_CONFIG_REGISTRY.keys())}"
            )

        cfg = OmegaConf.load(config_path)

        return cls(cfg, infer_engine, tokenizer)

    def _get_data(self, content) -> Dict[str, Any]:
        """Process input data into trajectory input."""
        data_cfg = self.cfg.get("data", {})
        instance = (
            get_field_from_config(data_cfg.get("instance_key"), content)
            if data_cfg.get("instance_key")
            else None
        )
        instance_id = (
            get_field_from_config(data_cfg.get("instance_id_key"), content)
            if data_cfg.get("instance_id_key")
            else None
        )
        data_source = (
            get_field_from_config(data_cfg.get("data_source_key"), content)
            if data_cfg.get("data_source_key")
            else "default"
        )
        return {
            "instance": instance if instance is not None else content,
            "instance_id": instance_id,
            "data_source": data_source,
        }

    def _initialize_trajectories(self, val_mode: bool = False):
        for batch_id, content in enumerate(self.batch):
            data = self._get_data(content)
            instance_id: str = data["instance_id"] if data["instance_id"] else batch_id
            self.trajectories[instance_id] = {}
            sampling_params = (
                self.cfg.generator.val_config.sampling_params
                if val_mode
                else self.cfg.generator.sampling_params
            )
            num_trajectories = (
                self.cfg.generator.val_config.num_trajectories
                if val_mode
                else self.cfg.generator.num_trajectories
            )

            for traj_id in range(num_trajectories):
                traj_cfg = TrajectoryConfig(
                    instance_id=instance_id,
                    trajectory_id=traj_id,
                    max_prompt_length=self.cfg.generator.max_prompt_length,
                    sampling_params=sampling_params,
                    vision_is_active=self.cfg.generator.vision_is_active,
                    qwen3_enable_thinking=self.cfg.generator.qwen3_enable_thinking,
                    qwen3_acc_thinking=self.cfg.generator.qwen3_acc_thinking,
                    max_iterations=self.cfg.generator.max_iterations,
                    tools=self.cfg.tools,
                    agent_cls=self.cfg.agent_cls,
                )
                traj: BaseTrajectory = self.traj_cls(
                    cfg=traj_cfg,
                    data=data,
                    tokenizer=self.tokenizer,
                    infer_engine=self.infer_engine,
                    task=self.task,
                )
                self.trajectories[instance_id][traj_id] = traj

    def _post_process_results(
        self, return_tensors=False, val_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Post-process the results to convert them into the appropriate output format.

        Returns:
            A dictionary containing the processed results.
        """
        all_results = {}
        matched_results = []
        instance_list = []
        error_list = []
        resolved_list = []
        has_finish_action_list = []
        finish_reason_list = []

        num_trajectories = (
            self.cfg.generator.val_config.num_trajectories
            if val_mode
            else self.cfg.generator.num_trajectories
        )

        for instance_id in self.trajectories:
            for trajectory_id in self.trajectories[instance_id]:
                all_results.setdefault(instance_id, {})[trajectory_id] = (
                    self.trajectories[instance_id][trajectory_id].result
                )

        for batch_idx, content in enumerate(self.batch):
            data = self._get_data(content)
            instance = pd.Series(data["instance"])
            instance_id = data["instance_id"] if data["instance_id"] else batch_idx
            instance["instance_id"] = instance_id  # safe mutation
            trajectories = all_results.get(instance_id, {})
            matched_results.extend(trajectories.values())
            instance_list.extend([instance] * len(trajectories))

        assert len(matched_results) == num_trajectories * len(
            self.batch
        ), f"Expected number of results {num_trajectories * len(self.batch)}, got {len(matched_results)}"

        # Group results by instance_id for message handling
        results_by_instance = {}
        for i, (instance, result) in enumerate(zip(instance_list, matched_results)):
            instance_id = instance["instance_id"]
            results_by_instance.setdefault(instance_id, []).append((i, result))

        global_fallback_set = None
        for results in results_by_instance.values():
            if all(res.get("messages") for _, res in results):
                global_fallback_set = [copy.deepcopy(res) for _, res in results]
                break
        # Handle empty messages by copying from another trajectory of the same instance
        for instance_id, results in results_by_instance.items():
            # Look for a non-empty base result
            fallback = next((res for _, res in results if res.get("messages")), None)
            if not fallback:
                if global_fallback_set:
                    logger.warning(
                        f"[WARN] No local fallback for instance_id {instance_id}, using global fallback set."
                    )

                    for j, (idx, res) in enumerate(results):
                        # Use corresponding global fallback result (same trajectory index)
                        fallback_res = global_fallback_set[j % len(global_fallback_set)]
                        print(
                            f"Empty messages for instance_id {instance_id}, trajectory {idx}. Using global fallback."
                        )
                        for key, value in fallback_res.items():
                            matched_results[idx][key] = copy.deepcopy(value)

                else:
                    logger.error(
                        f"[FATAL] No fallback (local/global) for instance_id {instance_id}. Skipping."
                    )
                    continue
            else:
                for idx, res in results:
                    if not res.get("messages", []):
                        print(
                            f"Empty messages for instance_id {instance_id}, trajectory {idx}. Using local fallback."
                        )
                        for key, value in fallback.items():
                            matched_results[idx][key] = copy.deepcopy(value)

        # Get batch of messages
        all_messages = []
        all_prompts = []
        all_responses = []
        num_turns = []
        for result in matched_results:
            messages = result.get("messages", [])
            all_messages.append(messages)
            # get the response: starting from the first assistant message
            starting_index = 0
            num_turns.append(max(0, (len(messages) - 2) // 2))
            for i, msg in enumerate(messages):
                if msg["role"] == "assistant":
                    starting_index = i
                    break
            if starting_index == 0:
                # If we don't find an assistant, all messages are prompts and there are no responses
                print(
                    f'ERROR: Found no assistant message. len(messages) == {len(messages)} and roles are {[msg["role"] for msg in messages]}'
                )
                starting_index = len(messages)
            prompt = messages[:starting_index]
            all_prompts.append(prompt)
            response = messages[starting_index:]
            all_responses.append(response)

            error_list.append(result.get("error", None))
            resolved_list.append(result.get("reward", False))
            has_finish_action_list.append(result.get("finish", False))
            finish_reason_list.append(result.get("finish_reason", None))

        # Encode messages, get assitant mask and position ids
        prompt_encodings = self.tokenizer.apply_chat_template(
            all_prompts,
            # return_tensors="pt",
            add_generation_prompt=False,
            return_dict=True,
            # padding=True
        )
        prompt_input_ids = prompt_encodings["input_ids"]
        prompt_attention_mask = prompt_encodings["attention_mask"]

        response_encodings = self.tokenizer.apply_chat_template(
            all_responses,
            chat_template=(
                chat_template_qwen3_thinking
                if self.cfg.generator.remove_think_tokens
                else chat_template
            ),
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
            return_dict=True,
        )

        response_ids = response_encodings["input_ids"]
        response_attention_mask = response_encodings["attention_mask"]
        response_assistant_mask = response_encodings["assistant_masks"]

        mask_out_reason = [
            "CONTEXT_WINDOW_EXCEEDED",
            "error_runtime",
            "error_evaluation",
            "max_iterations_reached",
        ]
        loss_mask = [
            [0] * len(mask) if (reason in mask_out_reason) else mask
            for mask, reason in zip(response_assistant_mask, finish_reason_list)
        ]

        rollout_metrics = {}
        rollout_metrics["rollout_metrics/avg_turn"] = sum(num_turns) / len(num_turns)

        total_per_instance = defaultdict(int)
        resolved_per_instance = defaultdict(int)
        for instance, reward in zip(instance_list, resolved_list):
            instance_id = instance["instance_id"]
            total_per_instance[instance_id] += 1
            if reward > 0:
                resolved_per_instance[instance_id] += 1

        # Track how many instances have resolution rate 0% or 100%
        num_resolved_0 = 0
        num_resolved_1 = 0

        # Print ratio and update counts
        for instance in sorted(total_per_instance):
            total = total_per_instance[instance]
            resolved = resolved_per_instance[instance]

            if resolved == 0:
                num_resolved_0 += 1
            elif resolved == total:
                num_resolved_1 += 1

        rollout_metrics["rollout_metrics/num_all_resolved"] = num_resolved_1
        rollout_metrics["rollout_metrics/num_none_resolved"] = num_resolved_0
        rollout_metrics["rollout_metrics/finish_tool_ratio"] = sum(
            1 for reason in finish_reason_list if reason == "FINISH_TOOL"
        ) / len(finish_reason_list)
        rollout_metrics["rollout_metrics/context_exceed_ratio"] = sum(
            1 for reason in finish_reason_list if reason == "CONTEXT_WINDOW_EXCEEDED"
        ) / len(finish_reason_list)
        rollout_metrics["rollout_metrics/max_turn"] = sum(
            1 for reason in finish_reason_list if reason == "max_iterations_reached"
        ) / len(finish_reason_list)
        rollout_metrics["rollout_metrics/stuck_in_a_loop_ratio"] = sum(
            1 for reason in finish_reason_list if reason == "stuck_in_a_loop"
        ) / len(finish_reason_list)
        rollout_metrics["rollout_metrics/error_runtime"] = sum(
            1 for reason in finish_reason_list if reason == "error_runtime"
        ) / len(finish_reason_list)
        rollout_metrics["rollout_metrics/error_evaluation"] = sum(
            1 for reason in finish_reason_list if reason == "error_evaluation"
        ) / len(finish_reason_list)
        rollout_metrics["rollout_metrics/num_mask_out"] = sum(
            1 for mask in loss_mask if all(m == 0 for m in mask)
        )
        rollout_metrics["rollout_metrics/num_mask_non_zero_reward"] = sum(
            1
            for mask, reward in zip(loss_mask, resolved_list)
            if all(m == 0 for m in mask) and reward > 0
        )

        print("rollout metrics:", rollout_metrics)

        print(f"Finish reason: {finish_reason_list}")
        # Create tensor dictionary
        output = {
            "prompt_token_ids": prompt_input_ids,
            "response_ids": response_ids,
            "rewards": resolved_list,
            "loss_masks": loss_mask,
            "rollout_metrics": rollout_metrics,
        }

        return output

    async def run(self, input_batch: Any, val_mode: bool = False) -> Any:
        """
        Generate trajectories for the given prompts using the configured agents.

        Args:
            prompts: A dictionary containing training instances.
            val_mode: Whether we're running validation.

        Returns:
            Results converted to the appropriate output format based on infer backend.
        """
        self.batch = build_generator_input(
            self.cfg.generator.infer_backend, input_batch
        ).input_batch

        if val_mode:
            num_trajectories = self.cfg.generator.val_config.num_trajectories
            sampling_params = self.cfg.generator.val_config.sampling_params
        else:
            sampling_params = self.cfg.generator.sampling_params
            num_trajectories = self.cfg.generator.num_trajectories

        # Initialize agents and other components
        self._initialize_trajectories(val_mode=val_mode)

        generator_dispatcher: DispatcherType | None = DISPATCHER_REGISTRY.get(
            self.cfg.dispatcher.type
        )
        if not generator_dispatcher:
            raise ValueError(f"Unknown generator type: {self.cfg.dispatcher.type}")
        else:
            logger.info(f"Using generator dispatcher: {self.cfg.dispatcher.type}")
            init_fn = "initialize_trajectory"
            run_fn = "generate_trajectory"
            eval_fn = "evaluate_trajectory"
            dispatcher_cfg = {
                "sampling_params": sampling_params,
                "max_parallel_agents": self.cfg.dispatcher.max_parallel_agents,
                "max_eval_parallel_agents": self.cfg.dispatcher.max_eval_parallel_agents,
                "num_instances": len(self.batch),
                "num_trajectories": num_trajectories,
            }
            await generator_dispatcher(
                dispatcher_cfg,
                self.trajectories,
                init_fn=init_fn,
                run_fn=run_fn,
                eval_fn=eval_fn,
            )

        output = self._post_process_results(val_mode=val_mode)

        # reset after run
        self.trajectories = {}

        return build_generator_output(self.cfg.generator.infer_backend, output).result
