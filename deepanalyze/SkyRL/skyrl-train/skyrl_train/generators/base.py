from typing import List, Dict, Any, TypedDict, Optional, Union
from abc import ABC, abstractmethod
from skyrl_train.inference_engines.base import ConversationType


class GeneratorInput(TypedDict):
    prompts: List[ConversationType]
    env_classes: List[str]
    env_extras: Optional[List[Dict[str, Any]]]
    sampling_params: Optional[Dict[str, Any]]


class GeneratorOutput(TypedDict):
    prompt_token_ids: List[List[int]]
    response_ids: List[List[int]]
    rewards: Union[List[float], List[List[float]]]
    loss_masks: List[List[int]]
    stop_reasons: Optional[List[str]]
    rollout_metrics: Optional[Dict[str, Any]]
    rollout_logprobs: Optional[List[List[float]]]


class GeneratorInterface(ABC):
    @abstractmethod
    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.

        Args:
            input_batch (GeneratorInput): Input batch
        Returns:
            GeneratorOutput: Generated trajectories
        """
        raise NotImplementedError()
