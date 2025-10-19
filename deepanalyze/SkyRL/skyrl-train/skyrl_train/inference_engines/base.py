from abc import ABC, abstractmethod
from typing import List, Dict, TypedDict, Any, Optional, Hashable

MessageType = Dict[str, str]
ConversationType = List[MessageType]


class InferenceEngineInput(TypedDict):
    # Either prompts or prompt_token_ids must be provided, but not both.
    prompts: Optional[List[ConversationType]]
    prompt_token_ids: Optional[List[List[int]]]
    sampling_params: Optional[Dict[str, Any]]
    trajectory_ids: Optional[List[Hashable]]


class InferenceEngineOutput(TypedDict):
    responses: List[str]
    stop_reasons: List[str]
    response_ids: Optional[List[List[int]]]
    response_logprobs: Optional[List[List[float]]]


class NamedWeightsUpdateRequest(TypedDict):
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    extras: Optional[List[Dict[str, Any]]]


class InferenceEngineInterface(ABC):

    @abstractmethod
    async def generate(
        self, input_batch: InferenceEngineInput
    ) -> InferenceEngineOutput:
        raise NotImplementedError()

    @abstractmethod
    async def wake_up(self, *args: Any, **kwargs: Any):
        raise NotImplementedError()

    @abstractmethod
    async def sleep(self, *args: Any, **kwargs: Any):
        raise NotImplementedError()

    @abstractmethod
    async def init_weight_update_communicator(
        self,
        master_addr,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend,
        override_existing: bool = False,
    ):
        raise NotImplementedError()

    @abstractmethod
    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        raise NotImplementedError()

    @abstractmethod
    async def teardown(self):
        raise NotImplementedError()

    @abstractmethod
    async def reset_prefix_cache(self):
        raise NotImplementedError()
