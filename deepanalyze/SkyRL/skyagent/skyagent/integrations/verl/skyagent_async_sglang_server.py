

from verl.workers.rollout.sglang_rollout.async_sglang_server import AsyncSglangServerRegular
from typing import Any, Dict, List

class SkyAgentAsyncSglangServer(AsyncSglangServerRegular):
    async def generate(self, prompt_ids: list[int], sampling_params: Dict[str, Any], request_id: str) -> List[int]:
        output = await self.master_worker.completion.remote(prompt_ids, sampling_params, request_id)
        return output[0], output[1]