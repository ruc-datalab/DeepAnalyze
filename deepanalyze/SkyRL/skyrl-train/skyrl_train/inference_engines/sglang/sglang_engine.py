"""SGLang inference engine implementation."""

import pickle
import base64
import torch
import os
from typing import List, Optional, Tuple
import ray
import multiprocessing as mp

import sglang.srt.entrypoints.engine
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.utils import (
    assert_pkg_version,
    is_cuda,
    maybe_set_triton_cache_manager,
    set_prometheus_multiproc_dir,
    set_ulimit,
    MultiprocessingSerializer,
)
from sglang.srt.managers.tokenizer_manager import (
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromDistributedReqInput,
    InitWeightsUpdateGroupReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)
from skyrl_train.utils import torch_dtype_to_str


# Patch SGLang's _set_envs_and_config to avoid signal handler issues in Ray actors
# Based on VERL's solution: https://github.com/sgl-project/sglang/issues/6723
# https://github.com/volcengine/verl/blob/v0.4.1/verl/workers/rollout/sglang_rollout/sglang_rollout.py#L85
def _patched_set_envs_and_config(server_args):
    """Patched version of SGLang's _set_envs_and_config that removes signal handler registration."""
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(getattr(server_args, "enable_nccl_nvls", False)))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Fix triton bugs
    if server_args.tp_size * server_args.dp_size > 1:
        # FIXME: remove this after https://github.com/triton-lang/triton/pull/4295 is used as a dependency.
        maybe_set_triton_cache_manager()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.2.5",
            "Please uninstall the old version and reinstall the latest version by following the instructions at https://docs.flashinfer.ai/installation.html.",
        )
    if is_cuda():
        assert_pkg_version(
            "sgl-kernel",
            "0.1.1",
            "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
        )

    # Set mp start method
    mp.set_start_method("spawn", force=True)

    # We do NOT register signal handlers here to avoid Ray actor issues
    # Original SGLang code had: signal.signal(signal.SIGCHLD, sigchld_handler)
    # But this fails in Ray actors since signal handlers only work in main thread


# Apply the patch
sglang.srt.entrypoints.engine._set_envs_and_config = _patched_set_envs_and_config


# TODO(charlie): duplicate of setup_envvars_for_vllm, is it needed?
def setup_envvars_for_sglang(kwargs, bundle_indices):
    distributed_executor_backend = kwargs.pop("distributed_executor_backend", None)
    noset_visible_devices = kwargs.pop("noset_visible_devices", None)
    if distributed_executor_backend == "ray":
        # a hack to make the script work.
        # stop ray from manipulating *_VISIBLE_DEVICES
        # at the top-level when the distributed_executor_backend is ray.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        pass
    elif noset_visible_devices:
        # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
        # when the distributed_executor_backend is not rayargs and
        # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])


def update_weights_cuda_ipc(model, named_tensors):
    """
    Custom weight loader for SGLang that handles IPC handles.

    This function is called by SGLang's model runner to load weights.
    It reconstructs tensors from SkyRL's NamedWeightsUpdateRequest that contains IPC handles
    and loads them into the model.
    """
    import torch

    # Extract tensor name and data
    name, tensor = named_tensors[0]
    if name != "ipc_request":
        raise ValueError(f"Expected IPC request tensor name to be 'ipc_request', got: {name}")

    # Convert tensor to bytes, then decode and deserialize
    tensor_bytes = tensor.cpu().numpy().tobytes()
    end_marker = b"__END_OF_REQUEST__"
    end_index = tensor_bytes.find(end_marker)
    if end_index == -1:
        raise ValueError("End marker not found in tensor data")
    request_data = tensor_bytes[:end_index]
    try:
        request_data_decoded = base64.b64decode(request_data)
        request: NamedWeightsUpdateRequest = pickle.loads(request_data_decoded)
    except Exception as e:
        raise ValueError(f"Failed to deserialize request data: {e}")

    weights_to_load = []
    for i in range(len(request["names"])):
        # Extract the request data
        ipc_handles = request["extras"][i]["ipc_handles"]
        dtype = request["dtypes"][i]
        _ = request["shapes"][i]
        weight_name = request["names"][i]

        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        physical_gpu_id = str(props.uuid)

        # Infer model dtype and device index from first parameter
        model_dtype = torch_dtype_to_str(next(model.parameters()).dtype)
        assert dtype == model_dtype, f"mismatch dtype: src {dtype}, dst {model_dtype}"
        device_id = next(model.parameters()).device.index

        handle = ipc_handles[physical_gpu_id]
        func, args = handle
        list_args = list(args)
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
        weight = func(*list_args)
        weights_to_load.append((weight_name, weight))

    model.load_weights(weights_to_load)


CUSTOM_WEIGHT_LOADER_PATH = "skyrl_train.inference_engines.sglang.sglang_engine.update_weights_cuda_ipc"


class SGLangInferenceEngine(InferenceEngineInterface):
    """SGLang inference engine that implements InferenceEngineInterface."""

    def __init__(self, *args, bundle_indices: Optional[List[int]] = None, **kwargs):
        setup_envvars_for_sglang(kwargs, bundle_indices)

        # Store common attributes
        self._tp_size = kwargs.get("tp_size", 1)
        if self._tp_size > 1:
            raise ValueError(
                "As of now, we don't support tensor parallel inference engine with SGLang. "
                "Please set `inference_engine_tensor_parallel_size` to 1."
            )
        self.tokenizer = kwargs.pop("tokenizer", None)

        # Extract sampling params
        sampling_params_dict = kwargs.pop("sampling_params", None)
        self.sampling_params = sampling_params_dict or {}

        # Unused kwargs
        _ = kwargs.pop("num_gpus", 1)

        # Add custom weight loader
        kwargs["custom_weight_loader"] = CUSTOM_WEIGHT_LOADER_PATH

        # Create the SGLang engine (signal handler issue is now fixed by patching)
        self.engine = Engine(**kwargs)
        print(f"Created SGLang engine with kwargs: {kwargs}")

    def tp_size(self):
        """Return the tensor parallel size."""
        return self._tp_size

    def _preprocess_prompts(self, input_batch: InferenceEngineInput):
        """Preprocess prompts for SGLang generation."""
        prompts = input_batch.get("prompts")
        prompt_token_ids = input_batch.get("prompt_token_ids")
        request_sampling_params = input_batch.get("sampling_params")

        if (prompts is None and prompt_token_ids is None) or (prompts is not None and prompt_token_ids is not None):
            raise ValueError("Either `prompts` or `prompt_token_ids` must be provided, but not both.")

        # Use request sampling params if provided, otherwise use defaults
        sampling_params = request_sampling_params if request_sampling_params is not None else self.sampling_params

        if prompt_token_ids is None:
            prompt_token_ids = self.tokenizer.apply_chat_template(
                prompts,
                add_generation_prompt=True,
                add_special_tokens=False,
                return_dict=True,
                tokenize=True,
            )["input_ids"]

        return prompt_token_ids, sampling_params

    def _postprocess_outputs(self, outputs):
        """Process SGLang outputs to match expected format."""
        responses: List[str] = []
        stop_reasons: List[str] = []

        for output in outputs:
            responses.append(output["text"])
            stop_reasons.append(output["meta_info"]["finish_reason"]["type"])

        return InferenceEngineOutput(
            responses=responses,
            # not supported with sglang yet
            response_ids=None,
            stop_reasons=stop_reasons,
        )

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        """Generate responses using SGLang engine."""
        token_ids_prompts, sampling_params = self._preprocess_prompts(input_batch)
        outputs = await self.engine.async_generate(input_ids=token_ids_prompts, sampling_params=sampling_params)
        return self._postprocess_outputs(outputs)

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        """Initialize weight update communicator for SGLang."""
        obj = InitWeightsUpdateGroupReqInput(
            master_address=master_addr,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )

        # NOTE(charlie): Call the async method on tokenizer_manager directly to avoid event loop
        # conflicts. Same underlying implementation: https://github.com/sgl-project/sglang/blob/v0.4.8.post1/python/sglang/srt/model_executor/model_runner.py#L689
        success, message = await self.engine.tokenizer_manager.init_weights_update_group(obj, None)
        return success, message

    async def update_named_weights(self, request: NamedWeightsUpdateRequest) -> Tuple[bool, str]:
        """Update named weights in SGLang engine."""
        if "names" not in request:
            raise ValueError(f"Expected update weight request with 'names' entry, got keys: {request.keys()}")

        extras = request.get("extras")
        if extras is not None and "ipc_handles" in extras[0]:
            # CUDA IPC -- Here we reuse SGLang's update_weights_from_tensor, but actually load the
            # weight from our request data. This will use the update_weights_cuda_ipc defined above.
            # This is a bit hacky, but the only way as of now, since there is no other way to
            # write per-TP worker code besides using `custom_weight_loader`, unlike in vLLM we can
            # use `WorkerWrap`.

            # Serialize the request data
            request_data = pickle.dumps(request)
            request_data_encoded = base64.b64encode(request_data)
            end_marker = b"__END_OF_REQUEST__"
            data_with_marker = request_data_encoded + end_marker

            # Create a tensor large enough to hold the serialized data; round up for alignment
            data_size = len(data_with_marker)
            padded_size = ((data_size + 3) // 4) * 4
            tensor_data = bytearray(data_with_marker)
            tensor_data.extend(b"\x00" * (padded_size - data_size))
            tensor_array = torch.frombuffer(tensor_data, dtype=torch.uint8)

            # Use SGLang's API to update weights with custom loader
            request_tensor = [("ipc_request", tensor_array)]
            obj = UpdateWeightsFromTensorReqInput(
                serialized_named_tensors=[
                    MultiprocessingSerializer.serialize(request_tensor) for _ in range(self._tp_size)
                ],
                load_format=CUSTOM_WEIGHT_LOADER_PATH,
                flush_cache=False,  # TODO(charlie): flush cache on last weight update?
            )

            # Call the underlying async method for the same reason as in `init_weight_update_communicator`
            success, message = await self.engine.tokenizer_manager.update_weights_from_tensor(obj, None)
            return success, message
        else:
            assert (
                len(request["names"]) == 1
            ), f"Update weights without cuda IPC only supports a single named weight at a time , got request with {len(request['names'])} entries"
            # Broadcast
            obj = UpdateWeightsFromDistributedReqInput(
                name=request["names"][0], dtype=request["dtypes"][0], shape=request["shapes"][0]
            )

            # Call the underlying async method for the same reason as in `init_weight_update_communicator`
            success, message = await self.engine.tokenizer_manager.update_weights_from_distributed(obj, None)
            if not success:
                raise RuntimeError(f"Update weight request failed with message: {message}")
            return

    async def wake_up(self, tags: Optional[List[str]] = None):
        """Wake up the engine. For multi-stage waking up, pass in `"weight"` or `"kv_cache"` to tags."""
        obj = ResumeMemoryOccupationReqInput(tags=tags)
        # Call the underlying async method for the same reason as in `init_weight_update_communicator`
        await self.engine.tokenizer_manager.resume_memory_occupation(obj, None)
        print(
            f"From SGLang engine -- Free GPU memory after wake up with tags {tags if tags is not None else 'None'}: "
            + f"{torch.cuda.mem_get_info()[0] / 1024**2:.1f} MB"
        )

    async def sleep(self, tags: Optional[List[str]] = None):
        """Put engine to sleep."""
        obj = ReleaseMemoryOccupationReqInput(tags=tags)
        # Call the underlying async method for the same reason as in `init_weight_update_communicator`
        await self.engine.tokenizer_manager.release_memory_occupation(obj, None)
        print(
            f"From SGLang engine -- Free GPU memory after sleep with tags {tags if tags is not None else 'None'}: "
            + f"{torch.cuda.mem_get_info()[0] / 1024**2:.1f} MB"
        )

    async def teardown(self):
        """Shutdown the SGLang engine."""
        self.engine.shutdown()

    async def reset_prefix_cache(self):
        """Reset prefix cache in SGLang engine."""
        # Call the underlying async method for the same reason as in `init_weight_update_communicator`
        return await self.engine.tokenizer_manager.flush_cache()


SGLangRayActor = ray.remote(SGLangInferenceEngine)
