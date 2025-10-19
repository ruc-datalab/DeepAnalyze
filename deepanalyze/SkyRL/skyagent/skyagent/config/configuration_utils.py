from dataclasses import dataclass
from typing import Any, Optional


# TODO(csy): a smarter way?
def get_field_from_config(key_path, context):
    parts = key_path.split(".")
    value = context
    try:
        for part in parts:
            value = value[part]
    except (KeyError, TypeError):
        raise ValueError(f"Path '{key_path}' not found in context.")
    return value


@dataclass
class TrajectoryConfig:
    instance_id: int
    trajectory_id: int
    max_prompt_length: int = 1024
    sampling_params: Optional[Any] = None
    vision_is_active: bool = False
    qwen3_enable_thinking: bool = True
    qwen3_acc_thinking: bool = True
    max_iterations: int = 5
    tools: Optional[list] = None
    agent_cls: str = (
        "skyagent.agents.react.DummyReactAgent"  # Default to DummyReactAgent
    )


# DEPR
@dataclass
class AgentConfig:
    max_iterations: int = 5
    tools: Optional[list] = None


TASK_CONFIG_REGISTRY = {"swe_bench": "swe_bench.yaml"}
