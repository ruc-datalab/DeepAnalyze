import os
from typing import List, Any, Dict, Optional
from dataclasses import dataclass
import ray
import torch
import asyncio
import vllm
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from torch.distributed import destroy_process_group
from skyrl_train.distributed.utils import init_custom_process_group
from uuid import uuid4
import warnings
from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)
from skyrl_train.utils import str_to_torch_dtype


@dataclass
class Logprob:
    logprob: float
    rank: int
    token_id: str


def setup_envvars_for_vllm(kwargs, bundle_indices):
    noset_visible_devices = kwargs.pop("noset_visible_devices")
    if kwargs.get("distributed_executor_backend") == "ray":
        # a hack to make the script work.
        # stop ray from manipulating *_VISIBLE_DEVICES
        # at the top-level when the distributed_executor_backend is ray.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
    elif noset_visible_devices:
        # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
        # when the distributed_executor_backend is not rayargs and
        # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

    num_gpus = kwargs.pop("num_gpus")
    if bundle_indices is not None:
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
        print(f"creating LLM with bundle_indices={bundle_indices}")


class WorkerWrap:
    def test_rpc(self, *args, **kwargs):
        """Test RPC call to worker"""
        return args, kwargs

    def init_weight_update_communicator(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
        override_existing: bool = False,
    ):
        """Init torch process group for model weights update"""
        assert torch.distributed.is_initialized(), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"

        if getattr(self, "_model_update_group", None):
            if override_existing:
                print("Destroying existing model update group")
                destroy_process_group(self._model_update_group)
                self._model_update_group = None
            else:
                warnings.warn(
                    "Detected an existing weights update group. For overriding, use `generator.override_existing_update_group=True`"
                )

        rank = torch.distributed.get_rank() + rank_offset
        print(
            f"torch.distributed.get_rank(): {torch.distributed.get_rank()}, rank_offset: {rank_offset}, rank: {rank}, world_size: {world_size}, group_name: {group_name}"
        )

        self._model_update_group = init_custom_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        print(
            f"init_weight_update_communicator: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    def update_weights(self, names: List[str], dtypes: List[str], shapes: List[List[int]]):
        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        weight_list = []
        for name, dtype, shape in zip(names, dtypes, shapes):
            dtype = str_to_torch_dtype(dtype)
            assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
            weight = torch.empty(shape, dtype=dtype, device="cuda")
            torch.distributed.broadcast(weight, 0, group=self._model_update_group)
            weight_list.append((name, weight))

        self.model_runner.model.load_weights(weights=weight_list)
        for weight in weight_list:
            del weight

    def update_weights_cuda_ipc(
        self, names: List[str], dtypes: List[str], shapes: List[int], ipc_handles: List[Dict[str, Any]]
    ):

        weight_list = []
        for name, dtype, shape, ipc_handle in zip(names, dtypes, shapes, ipc_handles):

            dtype = str_to_torch_dtype(dtype)
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            physical_gpu_id = str(props.uuid)

            assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"

            handle = ipc_handle[physical_gpu_id]

            device_id = self.device.index
            func, args = handle
            list_args = list(args)
            # the key is to change device id to the current device id
            # in case two processes have different CUDA_VISIBLE_DEVICES
            list_args[6] = device_id
            weight = func(*list_args)
            weight_list.append((name, weight))

        self.model_runner.model.load_weights(weights=weight_list)

        for weight in weight_list:
            del weight

    # TODO (sumanthrh): Add destroy process group RPC as a atexit handler to Trainer code.
    def destroy_weights_update_group(self):
        if not self._model_update_group:
            warnings.warn("No model update group to destroy")
            return
        destroy_process_group(self._model_update_group)


class BaseVLLMInferenceEngine(InferenceEngineInterface):
    """Base class containing shared logic between sync and async VLLM engines."""

    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        setup_envvars_for_vllm(kwargs, bundle_indices)
        vllm_v1_disable_multiproc = kwargs.pop("vllm_v1_disable_multiproc", False)
        if vllm_v1_disable_multiproc or vllm.__version__ == "0.8.2":
            # https://github.com/vllm-project/vllm/blob/effc5d24fae10b29996256eb7a88668ff7941aed/examples/offline_inference/reproduciblity.py#L11
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        # Store common attributes
        self._tp_size = kwargs.get("tensor_parallel_size", 1)
        self.tokenizer = kwargs.pop("tokenizer", None)
        sampling_params_dict = kwargs.pop("sampling_params", None)
        self.sampling_params = (
            SamplingParams(**sampling_params_dict) if sampling_params_dict is not None else SamplingParams()
        )

        # Let subclass create the appropriate engine
        self.llm = self._create_engine(*args, **kwargs)

    def tp_size(self):
        """Return the tensor parallel size."""
        return self._tp_size

    def _create_engine(self, *args, **kwargs):
        """Abstract method for subclasses to implement engine creation."""
        raise NotImplementedError("Subclasses must implement _create_engine")

    def _preprocess_prompts(self, input_batch: InferenceEngineInput):
        """Common prompt preprocessing logic."""
        prompts = input_batch.get("prompts")
        prompt_token_ids = input_batch.get("prompt_token_ids")
        request_sampling_params = input_batch.get("sampling_params")

        if (prompts is None and prompt_token_ids is None) or (prompts is not None and prompt_token_ids is not None):
            raise ValueError("Either `prompts` or `prompt_token_ids` must be provided, but not both.")

        sampling_params = (
            SamplingParams(**request_sampling_params) if request_sampling_params is not None else self.sampling_params
        )

        if prompt_token_ids is None:
            prompt_token_ids = self.tokenizer.apply_chat_template(
                prompts,
                add_generation_prompt=False,
                add_special_tokens=True,
                return_dict=True,
                tokenize=True,
            )["input_ids"]
            # print("AAAAAA")
            # print(self.tokenizer.apply_chat_template(prompts,add_generation_prompt=False,add_special_tokens=True,tokenize=False))
            

        return prompt_token_ids, sampling_params

    def _postprocess_outputs(self, outputs):
        """Common output processing logic."""
        responses: List[str] = []
        stop_reasons: List[str] = []
        response_ids: List[List[int]] = []
        response_logprobs: Optional[List[List[float]]] = []

        for output in outputs:
            # TODO(tgriggs): Support n>1 sampling.
            assert (
                len(output.outputs) == 1
            ), "Each prompt should have only one responses. n>1 sampling is supported by copying prompts."
            resp = output.outputs[0]
            responses.append(resp.text)
            stop_reasons.append(resp.finish_reason)
            response_ids.append(resp.token_ids)
            _logprobs = None
            if resp.logprobs:
                _logprobs = []
                for i, token_logprobs in enumerate(resp.logprobs):
                    token_logprobs: Dict[str, Logprob]
                    token_id = resp.token_ids[i]
                    logprob = token_logprobs[token_id].logprob
                    _logprobs.append(logprob)
                    del token_logprobs
            response_logprobs.append(_logprobs)

        if len(response_logprobs) and response_logprobs[0] is None:
            response_logprobs = None  # hack: assume uniform sampling params

        return InferenceEngineOutput(
            responses=responses,
            stop_reasons=stop_reasons,
            response_ids=response_ids,
            response_logprobs=response_logprobs,
        )

    def _get_engine(self):
        """Get the underlying engine for RPC calls."""
        return self.llm.engine if hasattr(self.llm, "engine") else self.llm

    def reset_prefix_cache(self):
        """Reset the prefix cache. Subclasses override for async version."""
        return self.llm.llm_engine.reset_prefix_cache()


class VLLMInferenceEngine(BaseVLLMInferenceEngine):
    """Synchronous VLLM engine."""

    def _create_engine(self, *args, **kwargs):
        return vllm.LLM(*args, **kwargs)

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        prompt_token_ids, sampling_params = self._preprocess_prompts(input_batch)

        outputs = await asyncio.to_thread(
            self.llm.generate,
            prompts=[TokensPrompt(prompt_token_ids=r) for r in prompt_token_ids],
            sampling_params=sampling_params,
        )
        return self._postprocess_outputs(outputs)

    async def wake_up(self, *args: Any, **kwargs: Any):
        await asyncio.to_thread(self.llm.wake_up, tags=kwargs.get("tags", None))

    async def sleep(self, *args: Any, **kwargs: Any):
        await asyncio.to_thread(self.llm.sleep, level=kwargs.get("level", 1))

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        engine = self._get_engine()
        return await asyncio.to_thread(
            engine.collective_rpc,
            "init_weight_update_communicator",
            args=(master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing),
        )

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        if "names" not in request:
            raise ValueError(f"Expected update weight request with 'names' entry, got keys: {request.keys()}")

        if not len(request["names"]):
            raise ValueError("Update weight request should have atleast one entry in 'names'")

        engine = self._get_engine()
        # Use IPC if handles are provided
        if request.get("extras") and "ipc_handles" in request["extras"][0]:
            return await asyncio.to_thread(
                engine.collective_rpc,
                "update_weights_cuda_ipc",
                args=(
                    request["names"],
                    request["dtypes"],
                    request["shapes"],
                    [extra["ipc_handles"] for extra in request["extras"]],
                ),
            )
        else:
            assert (
                len(request["names"]) == 1
            ), f"Update weights without cuda IPC only supports a single named weight at a time , got request with {len(request['names'])} entries"
            return await asyncio.to_thread(
                engine.collective_rpc, "update_weights", args=(request["names"], request["dtypes"], request["shapes"])
            )

    async def teardown(self):
        await self._destroy_weights_update_group()

    async def reset_prefix_cache(self):
        return await asyncio.to_thread(self.llm.llm_engine.reset_prefix_cache)

    async def _destroy_weights_update_group(self):
        engine = self._get_engine()
        return await asyncio.to_thread(engine.collective_rpc, "destroy_weights_update_group")


class AsyncVLLMInferenceEngine(BaseVLLMInferenceEngine):
    """Asynchronous VLLM engine."""

    def _create_engine(self, *args, **kwargs):
        # TODO (erictang000): potentially enable log requests for a debugging mode
        engine_args = vllm.AsyncEngineArgs(disable_log_requests=True, **kwargs)
        # print("engine_args",engine_args)
        # engine_args.model_impl="transformers"
        return vllm.AsyncLLMEngine.from_engine_args(engine_args)

    async def _collect_outputs(self, prompt_token_ids, request_id: str, sampling_params: SamplingParams):
        """Collect outputs for a single prompt."""
        final_output = None
        async for request_output in self.llm.generate(
            prompt=TokensPrompt(prompt_token_ids=prompt_token_ids),
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            final_output = request_output

        return final_output

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        """Generate responses using vLLM's async engine."""
        # print("1111111")
        # print(input_batch)
        prompt_token_ids, sampling_params = self._preprocess_prompts(input_batch)

        tasks = []
        for prompt in prompt_token_ids:
            # Schedule the collection of outputs for each prompt.
            # Avoid duplicate request_ids
            request_id = str(uuid4().hex)
            # print("=========")
            # print(self.tokenizer.decode(prompt,skip_special_tokens=False))
            # print("++++++++++++")
            task = asyncio.create_task(self._collect_outputs(prompt, request_id, sampling_params))
            tasks.append(task)
        outputs = await asyncio.gather(*tasks)
        
        res=self._postprocess_outputs(outputs)
        # print("EEEEE",res)
        return self._postprocess_outputs(outputs)

    async def wake_up(self, *args: Any, **kwargs: Any):
        await self.llm.wake_up(tags=kwargs.get("tags", None))

    async def sleep(self, *args: Any, **kwargs: Any):
        # TODO(team): remove once vllm fixes this
        # otherwise waking it up will output gibberish: https://github.com/vllm-project/vllm/issues/17103
        await self.reset_prefix_cache()
        await self.llm.sleep(level=kwargs.get("level", 1))

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        engine = self._get_engine()
        return await engine.collective_rpc(
            "init_weight_update_communicator",
            args=(master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing),
        )

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        if "names" not in request:
            raise ValueError(f"Expected update weight request with 'names' entry, got keys: {request.keys()}")

        if not len(request["names"]):
            raise ValueError("Update weight request should have atleast one entry in 'names'")

        engine = self._get_engine()
        # Use IPC if handles are provided

        is_ipc = request.get("extras") and "ipc_handles" in request["extras"][0]

        if is_ipc:
            return await engine.collective_rpc(
                "update_weights_cuda_ipc",
                args=(
                    request["names"],
                    request["dtypes"],
                    request["shapes"],
                    [extra["ipc_handles"] for extra in request["extras"]],
                ),
            )
        else:
            assert (
                len(request["names"]) == 1
            ), f"Update weights without cuda IPC only supports a single named weight at a time , got request with {len(request['names'])} entries"
            return await engine.collective_rpc(
                "update_weights",
                args=(
                    request["names"],
                    request["dtypes"],
                    request["shapes"],
                ),
            )

    async def teardown(self):
        await self._destroy_weights_update_group()

    async def reset_prefix_cache(self):
        engine = self._get_engine()
        await engine.reset_prefix_cache()

    async def _destroy_weights_update_group(self):
        engine = self._get_engine()
        return await engine.collective_rpc("destroy_weights_update_group")




# class AsyncVLLMInferenceEngine(BaseVLLMInferenceEngine):
#     """Asynchronous VLLM engine."""

#     def _create_engine(self, *args, **kwargs):
#         # TODO (erictang000): potentially enable log requests for a debugging mode
#         engine_args = vllm.AsyncEngineArgs(disable_log_requests=True, **kwargs)
#         # print("engine_args",engine_args)
#         # engine_args.model_impl="transformers"
#         return vllm.AsyncLLMEngine.from_engine_args(engine_args)

#     async def _collect_outputs(self, prompt_token_ids, request_id: str, sampling_params: SamplingParams):
#         """Collect outputs for a single prompt."""
#         final_output = None
#         async for request_output in self.llm.generate(
#             prompt=TokensPrompt(prompt_token_ids=prompt_token_ids),
#             sampling_params=sampling_params,
#             request_id=request_id,
#         ):
#             final_output = request_output

#         return final_output

#     async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
#         """Generate responses using vLLM's async engine."""
#         prompt_token_ids, sampling_params = self._preprocess_prompts(input_batch)

#         tasks = []
#         for prompt in prompt_token_ids:
#             # Schedule the collection of outputs for each prompt.
#             # Avoid duplicate request_ids
#             request_id = str(uuid4().hex)
#             task = asyncio.create_task(self._collect_outputs(prompt, request_id, sampling_params))
#             tasks.append(task)
#         outputs = await asyncio.gather(*tasks)

#         return self._postprocess_outputs(outputs)

#     async def wake_up(self, *args: Any, **kwargs: Any):
#         await self.llm.wake_up(tags=kwargs.get("tags", None))

#     async def sleep(self, *args: Any, **kwargs: Any):
#         # TODO(team): remove once vllm fixes this
#         # otherwise waking it up will output gibberish: https://github.com/vllm-project/vllm/issues/17103
#         await self.reset_prefix_cache()
#         await self.llm.sleep(level=kwargs.get("level", 1))

#     async def init_weight_update_communicator(
#         self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
#     ):
#         engine = self._get_engine()
#         return engine.collective_rpc(
#             "init_weight_update_communicator",
#             args=(master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing),
#         )

#     async def update_named_weights(self, request: NamedWeightsUpdateRequest):
#         if "names" not in request:
#             raise ValueError(f"Expected update weight request with 'names' entry, got keys: {request.keys()}")

#         if not len(request["names"]):
#             raise ValueError("Update weight request should have atleast one entry in 'names'")

#         engine = self._get_engine()
#         # Use IPC if handles are provided

#         is_ipc = request.get("extras") and "ipc_handles" in request["extras"][0]

#         if is_ipc:
#             return engine.collective_rpc(
#                 "update_weights_cuda_ipc",
#                 args=(
#                     request["names"],
#                     request["dtypes"],
#                     request["shapes"],
#                     [extra["ipc_handles"] for extra in request["extras"]],
#                 ),
#             )
#         else:
#             assert (
#                 len(request["names"]) == 1
#             ), f"Update weights without cuda IPC only supports a single named weight at a time , got request with {len(request['names'])} entries"
#             return engine.collective_rpc(
#                 "update_weights",
#                 args=(
#                     request["names"],
#                     request["dtypes"],
#                     request["shapes"],
#                 ),
#             )

#     async def teardown(self):
#         await self._destroy_weights_update_group()

#     async def reset_prefix_cache(self):
#         engine = self._get_engine()
#         engine.reset_prefix_cache()

#     async def _destroy_weights_update_group(self):
#         engine = self._get_engine()
#         return engine.collective_rpc("destroy_weights_update_group")

VLLMRayActor = ray.remote(VLLMInferenceEngine)
AsyncVLLMRayActor = ray.remote(AsyncVLLMInferenceEngine)
