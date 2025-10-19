# skyagent/auto.py
from typing import Type
import os
from omegaconf import OmegaConf
from importlib import import_module
from skyagent.agents.mapping import AGENT_GENERATOR_REGISTRY
from skyagent.agents import AgentRunner


def _import_object(path: str):
    module_path, class_name = path.rsplit(".", 1)
    return getattr(import_module(module_path), class_name)


class AutoAgentRunner:
    @classmethod
    def from_task(cls, task_yaml: str, infer_engine, tokenizer) -> AgentRunner:
        # Load config

        if not os.path.exists(task_yaml):
            raise FileNotFoundError(f"Task YAML not found: {task_yaml}")
        cfg = OmegaConf.load(task_yaml)

        runner_path = AGENT_GENERATOR_REGISTRY.get(cfg.agent_cls, None)
        if not runner_path:
            raise ValueError(
                f"`AgentRunner` class for agent {cfg.agent_cls} is not specified. Please ensure that the agent is present in the registry"
            )

        runner_cls: Type[AgentRunner] = _import_object(runner_path)

        return runner_cls.from_task(task_yaml, infer_engine, tokenizer)
