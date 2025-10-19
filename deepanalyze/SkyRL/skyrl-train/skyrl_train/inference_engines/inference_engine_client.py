from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)
import asyncio
from typing import List, Any, Optional


class InferenceEngineClient(InferenceEngineInterface):
    """
    Client to talk to a set of InferenceEngines.

    Note that InferenceEngineClient sub-classes InferenceEngineInterface so it can be used as if talking to a single engine.
    """

    def __init__(self, engines: List[InferenceEngineInterface]):
        self.engines = engines
        print(f"InferenceEngineClient initialized with {len(engines)} engines.")

    async def _run_on_all_engines(self, method_name: str, *args, **kwargs):
        """
        Call a method on all engines concurrently and gather the results.
        """
        assert len(self.engines) > 0, "No engines to call method on"

        awaitables = [
            getattr(engine, method_name)(*args, **kwargs) for engine in self.engines
        ]
        return await asyncio.gather(*awaitables)

    async def generate(
        self, input_batch: InferenceEngineInput
    ) -> InferenceEngineOutput:
        prompts = input_batch.get("prompts")
        prompt_token_ids = input_batch.get("prompt_token_ids")
        trajectory_ids = input_batch.get("trajectory_ids")
        sampling_params = input_batch.get("sampling_params")

        if (prompts is None and prompt_token_ids is None) or (
            prompts is not None and prompt_token_ids is not None
        ):
            raise ValueError(
                "Either `prompts` or `prompt_token_ids` must be provided, but not both."
            )

        # TODO(tgriggs): If there are no traj ids, we'd still like to load balance instead of landing on a single engine.
        if trajectory_ids is not None:
            # Route based on trajectory_ids
            return await self._generate_with_trajectory_routing(
                prompts, prompt_token_ids, trajectory_ids, sampling_params
            )
        else:
            # Split evenly across engines
            return await self._generate_batched(
                prompts, prompt_token_ids, sampling_params
            )

    async def _generate_with_trajectory_routing(
        self, prompts, prompt_token_ids, trajectory_ids, sampling_params
    ):
        """
        Route prompts to engines based on trajectory_ids and return results in the original order of the prompts.
        """
        # Group prompts by engine
        engine_groups: dict[int, dict[str, list]] = {}
        prompts_or_tokens = prompts if prompts is not None else prompt_token_ids
        for i, (prompt_or_token, traj_id) in enumerate(
            zip(prompts_or_tokens, trajectory_ids)
        ):
            engine_idx = abs(hash(str(traj_id))) % len(self.engines)
            group = engine_groups.setdefault(
                engine_idx, {"prompt_or_token": [], "indices": []}
            )
            group["prompt_or_token"].append(prompt_or_token)
            group["indices"].append(i)

        # Build two parallel lists: one of tasks, one of the index‐lists
        tasks: list[asyncio.Task] = []
        indices_list: list[list[int]] = []
        for engine_idx, group in engine_groups.items():
            inp = InferenceEngineInput(
                prompts=group["prompt_or_token"] if prompts is not None else None,
                prompt_token_ids=(
                    group["prompt_or_token"] if prompt_token_ids is not None else None
                ),
                sampling_params=sampling_params,
            )
            coro = self.engines[engine_idx].generate(inp)
            tasks.append(asyncio.create_task(coro))
            indices_list.append(group["indices"])

        results = await asyncio.gather(*tasks)

        # Reconstruct output in original order
        n = len(prompts_or_tokens)
        responses: list[str] = [""] * n
        stop_reasons: list[str] = [""] * n
        response_logprobs: List[Optional[List[float]]] = [None for _ in range(n)]
        response_ids: List[Optional[List[float]]] = [None for _ in range(n)]
        # a bit hacky for now
        add_resp_ids = False
        add_resp_logprobs = False

        for indices, result in zip(indices_list, results):
            for local_idx, original_idx in enumerate(indices):
                responses[original_idx] = result["responses"][local_idx]
                stop_reasons[original_idx] = result["stop_reasons"][local_idx]
                if result.get("response_ids", None):
                    add_resp_ids = True
                    response_ids[original_idx] = result["response_ids"][local_idx]
                if result.get("response_logprobs", None):
                    add_resp_logprobs = True
                    response_logprobs[original_idx] = result["response_logprobs"][
                        local_idx
                    ]

        if any([len(response) == 0 for response in responses]) or (
            add_resp_ids
            and not all([isinstance(sample_ids, list) for sample_ids in response_ids])
        ):
            raise RuntimeError(
                "Did not receive responses / response ids for some prompts. This should never happen. There is likely something wrong with the inference engine"
            )

        return InferenceEngineOutput(
            responses=responses,
            stop_reasons=stop_reasons,
            response_ids=response_ids if add_resp_ids else None,
            response_logprobs=response_logprobs if add_resp_logprobs else None,
        )

    async def _generate_batched(self, prompts, prompt_token_ids, sampling_params):
        """
        Split prompts evenly across engines and return results in the original order of the prompts.
        """
        num_inference_engines = len(self.engines)
        prompts_or_tokens = prompts if prompts is not None else prompt_token_ids
        dp_item_size = (
            len(prompts_or_tokens) + num_inference_engines - 1
        ) // num_inference_engines

        tasks = []
        for dp_rank in range(num_inference_engines):
            start_idx = dp_rank * dp_item_size
            end_idx = (dp_rank + 1) * dp_item_size
            dp_items = prompts_or_tokens[start_idx:end_idx]

            if len(dp_items) <= 0:
                continue

            engine_input = InferenceEngineInput(
                prompts=dp_items if prompts is not None else None,
                prompt_token_ids=dp_items if prompt_token_ids is not None else None,
                sampling_params=sampling_params,
            )
            tasks.append(self.engines[dp_rank].generate(engine_input))

        all_outputs = await asyncio.gather(*tasks)

        # Flatten results
        responses = []
        stop_reasons = []
        response_ids = []
        response_logprobs = []
        for output in all_outputs:
            responses.extend(output["responses"])
            stop_reasons.extend(output["stop_reasons"])
            if output.get("response_ids", None):
                response_ids.extend(output["response_ids"])
            if output.get("response_logprobs", None):
                response_logprobs.extend(output["response_logprobs"])

        return InferenceEngineOutput(
            responses=responses,
            stop_reasons=stop_reasons,
            response_ids=response_ids if len(response_ids) else None,
            response_logprobs=response_logprobs if len(response_logprobs) else None,
        )

    async def wake_up(self, *args: Any, **kwargs: Any):
        return await self._run_on_all_engines("wake_up", *args, **kwargs)

    async def sleep(self, *args: Any, **kwargs: Any):
        return await self._run_on_all_engines("sleep", *args, **kwargs)

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
        tasks = []
        rank_offset_count = rank_offset

        for engine in self.engines:
            assert engine.tp_size is not None, "Engine must have a tp_size"
            tasks.append(
                engine.init_weight_update_communicator(
                    master_addr=master_addr,
                    master_port=master_port,
                    rank_offset=rank_offset_count,
                    world_size=world_size,
                    group_name=group_name,
                    backend=backend,
                    override_existing=override_existing,
                )
            )
            rank_offset_count += engine.tp_size
        await asyncio.gather(*tasks)

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        return await self._run_on_all_engines("update_named_weights", request=request)

    async def reset_prefix_cache(self):
        return await self._run_on_all_engines("reset_prefix_cache")

    async def teardown(self):
        return await self._run_on_all_engines("teardown")
