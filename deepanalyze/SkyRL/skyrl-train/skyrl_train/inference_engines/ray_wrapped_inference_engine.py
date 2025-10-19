import ray
from packaging import version
from ray.actor import ActorHandle
from typing import Dict, Any, Optional, List
from ray.util.placement_group import PlacementGroupSchedulingStrategy, placement_group

from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)


class RayWrappedInferenceEngine(InferenceEngineInterface):
    """
    A thin wrapper around a Ray ActorHandle to another InferenceEngineInterface.
    This class implements the InferenceEngineInterface by delegating calls to the remote actor.
    """

    def __init__(self, inference_engine_actor: ActorHandle):
        self.inference_engine_actor = inference_engine_actor

    @property
    def tp_size(self):
        return ray.get(self.inference_engine_actor.tp_size.remote())

    async def generate(
        self, input_batch: InferenceEngineInput
    ) -> InferenceEngineOutput:
        return await self.inference_engine_actor.generate.remote(
            input_batch=input_batch
        )

    async def wake_up(self, *args: Any, **kwargs: Any):
        return await self.inference_engine_actor.wake_up.remote(*args, **kwargs)

    async def sleep(self, *args: Any, **kwargs: Any):
        return await self.inference_engine_actor.sleep.remote(*args, **kwargs)

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
        return await self.inference_engine_actor.init_weight_update_communicator.remote(
            master_addr,
            master_port,
            rank_offset,
            world_size,
            group_name,
            backend,
            override_existing,
        )

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        return await self.inference_engine_actor.update_named_weights.remote(request)

    async def teardown(self):
        return await self.inference_engine_actor.teardown.remote()

    async def reset_prefix_cache(self):
        return await self.inference_engine_actor.reset_prefix_cache.remote()


def create_ray_wrapped_inference_engines(
    num_inference_engines: int,
    tensor_parallel_size: int,
    model_dtype: str,
    pretrain: str,
    seed: int,
    vllm_v1_disable_multiproc: bool,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    shared_pg=None,
    gpu_memory_utilization=None,
    inference_engine_enable_sleep=False,
    async_engine=False,
    max_num_batched_tokens=8192,
    max_num_seqs=1024,
    sampling_params: Optional[Dict[str, Any]] = None,
    tokenizer=None,
    backend="vllm",
) -> List[InferenceEngineInterface]:
    """
    Create a list of RayWrappedInferenceEngine instances wrapping Ray actor handles to InferenceEngineInterface instances.
    """
    from skyrl_train.utils import (
        ray_noset_visible_devices,
        get_all_env_variables,
        get_ray_pg_ready_with_timeout,
    )

    if backend == "vllm":
        import vllm
        from skyrl_train.inference_engines.vllm.vllm_engine import (
            VLLMRayActor,
            AsyncVLLMRayActor,
        )

        assert version.parse(vllm.__version__) >= version.parse(
            "0.8.3"
        ), "SkyRL-Train only supports vLLM >= 0.8.3"
    elif backend == "sglang":
        # We import SGLang later to avoid importing vllm. See `get_sglang_engine` for more.
        pass
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    inference_engine_actors = []
    noset_visible_devices = ray_noset_visible_devices(
        ray.get(get_all_env_variables.remote())
    )
    # NOTE: we use the ray backend for tensor parallel size > 1 to explicitly manage resource allocation
    # TODO: we should be able to support mp backend by allocating resources at engine level
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    use_hybrid_engine = shared_pg is not None
    num_gpus = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1:
        # every worker will use 0.2 GPU, so that we can schedule
        # 2 instances on the same GPUs.
        num_gpus = 0.2

    if not use_hybrid_engine:
        # Create a big placement group to ensure that all inference engines are packed
        bundles = [
            {"GPU": 1, "CPU": 1}
            for _ in range(num_inference_engines * tensor_parallel_size)
        ]
        shared_pg = placement_group(bundles, strategy="PACK")
        get_ray_pg_ready_with_timeout(shared_pg, timeout=30)

    for i in range(num_inference_engines):
        bundle_indices = None
        if tensor_parallel_size > 1:
            bundle_indices = list(
                range(i * tensor_parallel_size, (i + 1) * tensor_parallel_size)
            )

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=shared_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=i * tensor_parallel_size,
        )

        if backend == "vllm":
            if async_engine:
                actor_class = AsyncVLLMRayActor
            else:
                actor_class = VLLMRayActor

            engine = actor_class.options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                model=pretrain,
                enforce_eager=enforce_eager,
                worker_extension_cls="skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap",
                tensor_parallel_size=tensor_parallel_size,
                seed=seed + i,
                distributed_executor_backend=distributed_executor_backend,
                max_model_len=max_model_len,
                enable_prefix_caching=enable_prefix_caching,
                dtype=model_dtype,
                trust_remote_code=True,
                vllm_v1_disable_multiproc=vllm_v1_disable_multiproc,
                gpu_memory_utilization=gpu_memory_utilization,
                bundle_indices=bundle_indices,
                num_gpus=0.2 if use_hybrid_engine else 1,
                enable_sleep_mode=inference_engine_enable_sleep,
                noset_visible_devices=noset_visible_devices,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                sampling_params=sampling_params,
                tokenizer=tokenizer,
                # only need the logprobs for the chosen token if any
                max_logprobs=1,
            )
        elif backend == "sglang":
            # NOTE: there is no async / sync engine distinction in SGLang

            # NOTE(Charlie): We need `torch.cuda.is_available()` to be True to import SGLang. Otherwise, it requires
            # importing vllm. See https://github.com/sgl-project/sglang/blob/v0.4.8.post1/python/sglang/srt/layers/quantization/utils.py#L11-L17
            # Similar comment: https://github.com/volcengine/verl/blob/9cc307767b0c787e8f5ef581dac929f7bde044ef/verl/workers/fsdp_workers.py#L520-L527
            @ray.remote
            def get_sglang_engine():
                # A workaround to avoid importing vllm is to give this task a GPU.
                import os

                before_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                from skyrl_train.inference_engines.sglang.sglang_engine import (
                    SGLangRayActor,
                )

                os.environ["CUDA_VISIBLE_DEVICES"] = before_cuda_visible_devices

                actor_class = SGLangRayActor
                engine = actor_class.options(
                    num_cpus=num_gpus,
                    num_gpus=num_gpus,
                    scheduling_strategy=scheduling_strategy,
                ).remote(
                    model_path=pretrain,
                    tp_size=tensor_parallel_size,
                    mem_fraction_static=gpu_memory_utilization,
                    random_seed=seed + i,
                    context_length=max_model_len,
                    disable_radix_cache=not enable_prefix_caching,
                    dtype=model_dtype,
                    trust_remote_code=True,
                    max_prefill_tokens=max_num_batched_tokens,
                    max_running_requests=max_num_seqs,
                    # Borrowed from veRL's SGLang rollout
                    mm_attention_backend="fa3",
                    attention_backend="fa3",
                    enable_memory_saver=inference_engine_enable_sleep,
                    # Will be popped before instantiating sgl.Engine
                    distributed_executor_backend=distributed_executor_backend,
                    noset_visible_devices=noset_visible_devices,
                    bundle_indices=bundle_indices,
                    num_gpus=0.2 if use_hybrid_engine else 1,
                    sampling_params=sampling_params,
                    tokenizer=tokenizer,
                )
                return engine

            engine = ray.get(get_sglang_engine.remote())

        inference_engine_actors.append(engine)

    engines = [
        RayWrappedInferenceEngine(actor_handle)
        for actor_handle in inference_engine_actors
    ]

    if inference_engine_enable_sleep:
        sleep_refs = [
            engine.inference_engine_actor.sleep.remote() for engine in engines
        ]
        ray.get(sleep_refs)

    return engines
