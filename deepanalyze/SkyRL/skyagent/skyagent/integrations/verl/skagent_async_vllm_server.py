from typing import List, Any, Tuple
from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServerRegular
import logging
import os
import pickle
from typing import Any, Callable, Optional

import ray
import zmq
from vllm import SamplingParams

from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput


@ray.remote(num_cpus=1)
class SkyAgentAsyncvLLMServer(AsyncvLLMServerRegular):
    async def generate(self, prompt_ids: list[int], sampling_params: dict[str, Any], request_id: str) -> Tuple[str, str]:
        max_tokens = self.max_model_len - len(prompt_ids)
        sampling_params.pop("max_tokens", None)
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        prompt = TokensPrompt(prompt_token_ids=prompt_ids)
        generator = self.engine.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id)

        # Get final response
        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        response_str = final_res.outputs[0].text
        stop_reason = final_res.outputs[0].finish_reason

        return response_str, stop_reason 