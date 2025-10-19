from skyagent.integrations.base import AsyncInferBackend, GeneratorOutput, GeneratorInput, register_backend, BackendSpec
from typing import Any, List, Dict
import uuid 
from loguru import logger

try: 
    from verl.verl.protocol import DataProto
    from tensordict import TensorDict
except ImportError:
    DataProto = None
    tensordict = None


class VeRLBackend(AsyncInferBackend):
    def __init__(self, infer_engine, cfg: Dict[str, Any] = None):
        self.infer_engine = infer_engine

    async def async_generate_ids(self, input_ids: List[int], sampling_params: Dict[str, Any], request_id):
        response_str, finish_reason = await self.infer_engine.generate(
            request_id=request_id,
            prompt_ids=input_ids,
            sampling_params=sampling_params
        )
        return response_str, finish_reason

    async def async_generate_prompts(self, prompts: Any, sampling_params: Any) -> List[str]:
        raise NotImplementedError

class VeRLGeneratorOutput(GeneratorOutput):
    def __init__(self, result: Dict[str, Any]):
        self.result = result

class VeRLGeneratorInput(GeneratorInput):
    def __init__(self, input_batch: Any):
        self.input_batch: List[Dict[str, Any]] = []
        non_tensor_batch = input_batch.non_tensor_batch
        first_key = next(iter(non_tensor_batch.keys()))
        num_entries = len(non_tensor_batch[first_key])
        for i in range(num_entries):
            data_item: dict = {key: non_tensor_batch[key][i] for key in non_tensor_batch.keys()}
            self.input_batch.append(data_item)
        logger.info(f"input batch of size: {len(self.input_batch)}")
        logger.info(f"keys: {self.input_batch[0].keys()}")
