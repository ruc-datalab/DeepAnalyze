"""
# Run only vllm tests (requires vllm extra):
uv run --isolated --extra dev --extra vllm pytest tests/gpu/test_engine_generation.py -m "vllm"

# Run only sglang tests (requires sglang extra):
uv run --isolated --extra dev --extra sglang pytest tests/gpu/test_engine_generation.py -m "sglang"
"""

import pytest
import ray
import hydra
from skyrl_train.inference_engines.remote_inference_engine import (
    create_remote_inference_engines,
)
from skyrl_train.inference_engines.ray_wrapped_inference_engine import (
    create_ray_wrapped_inference_engines,
)
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
import asyncio
import subprocess
import os
from tests.gpu.utils import (
    get_available_gpus,
    wait_for_server,
    are_responses_similar,
    get_test_prompts,
)
from transformers import AutoTokenizer
from omegaconf import DictConfig
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import config_dir
from typing import Tuple

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def get_test_actor_config() -> DictConfig:
    """Get base config with test-specific overrides."""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

        cfg.trainer.policy.model.path = MODEL

        return cfg


def init_remote_inference_servers(
    tp_size: int, backend: str
) -> Tuple[InferenceEngineClient, subprocess.Popen]:
    available_gpus = get_available_gpus()
    assert (
        len(available_gpus) >= tp_size
    ), f"Not enough GPUs available. Need {tp_size}, but only {len(available_gpus)} available: {available_gpus}"

    selected_gpus = available_gpus[:tp_size]
    gpu_ids_str = ",".join(map(str, selected_gpus))
    print(f"Using GPUs {gpu_ids_str} for vLLM server (tensor_parallel_size={tp_size})")

    def get_free_port():
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    engine_port = get_free_port()

    # Launch vLLM server using subprocess
    if backend == "vllm":
        remote_server_command = [
            "uv",
            "run",
            "--isolated",
            "--extra",
            "vllm",
            "-m",
            "skyrl_train.inference_engines.vllm.vllm_server",
            "--model",
            MODEL,
            "--enforce-eager",
            "--tensor-parallel-size",
            str(tp_size),
            "--distributed-executor-backend",
            "ray",
            "--dtype",
            "bfloat16",
            "--host",
            "127.0.0.1",
            "--port",
            str(engine_port),
            "--worker-extension-cls",
            "skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap",
        ]
    elif backend == "sglang":
        remote_server_command = [
            "uv",
            "run",
            "--isolated",
            "--extra",
            "sglang",
            "-m",
            "skyrl_train.inference_engines.sglang.sglang_server",
            "--model-path",
            MODEL,
            "--tp-size",
            str(tp_size),
            "--dtype",
            "bfloat16",
            "--host",
            "127.0.0.1",
            "--port",
            str(engine_port),
            "--mm-attention-backend",
            "fa3",
            "--attention-backend",
            "fa3",
        ]
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # Set CUDA_VISIBLE_DEVICES environment variable for the subprocess
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids_str

    # Start the vLLM server process
    server_process = subprocess.Popen(remote_server_command, env=env)

    wait_for_server(url=f"localhost:{engine_port}", health_path="health")
    print(f"Server at localhost:{engine_port} is online")

    engines = create_remote_inference_engines(
        urls=[f"localhost:{engine_port}"],
        model_name=MODEL,
        engine_backend=backend,
        tensor_parallel_size=tp_size,
        sampling_params=get_sampling_params_for_backend(
            backend,
            DictConfig(
                {
                    "temperature": 0.0,
                    "top_p": 1,
                    "top_k": -1,
                    "max_generate_length": 1024,
                    "min_p": 0.0,
                }
            ),
        ),
    )

    return InferenceEngineClient(engines), server_process


def init_ray_inference_engines(backend: str, tp_size: int) -> InferenceEngineClient:
    """Initialize ray-wrapped inference engines for the specified backend"""
    engine = create_ray_wrapped_inference_engines(
        num_inference_engines=1,
        tensor_parallel_size=tp_size,
        model_dtype="bfloat16",
        pretrain=MODEL,
        seed=42,
        vllm_v1_disable_multiproc=True,
        enable_prefix_caching=True,
        enforce_eager=True,
        max_model_len=1536,
        shared_pg=None,
        gpu_memory_utilization=0.8,
        inference_engine_enable_sleep=False,
        async_engine=True,
        max_num_batched_tokens=8192,
        max_num_seqs=1024,
        sampling_params=get_sampling_params_for_backend(
            backend,
            DictConfig(
                {
                    "temperature": 0.0,
                    "top_p": 1,
                    "top_k": -1,
                    "max_generate_length": 1024,
                    "min_p": 0.0,
                }
            ),
        ),
        tokenizer=AutoTokenizer.from_pretrained(MODEL),
        backend=backend,
    )
    client = InferenceEngineClient(engine)
    return client


async def run_batch_generation(client, prompts):
    engine_input = InferenceEngineInput(prompts=prompts)
    engine_output = await client.generate(engine_input)
    return engine_output["responses"], engine_output["stop_reasons"]


async def run_single_generation(client, prompts):
    tasks = []
    for prompt in prompts:
        engine_input = InferenceEngineInput(prompts=[prompt])
        task = client.generate(engine_input)
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    responses = []
    finish_reasons = []
    for result in results:
        responses.extend(result["responses"])
        finish_reasons.extend(result["stop_reasons"])

    return responses, finish_reasons


async def run_batch_generation_with_tokens(client, prompt_token_ids):
    engine_input = InferenceEngineInput(prompt_token_ids=prompt_token_ids)
    engine_output = await client.generate(engine_input)
    return engine_output["responses"], engine_output["stop_reasons"]


async def run_single_generation_with_tokens(client, prompt_token_ids):
    tasks = []
    for tokens in prompt_token_ids:
        engine_input = InferenceEngineInput(prompt_token_ids=[tokens])
        task = client.generate(engine_input)
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    responses = []
    finish_reasons = []
    for result in results:
        responses.extend(result["responses"])
        finish_reasons.extend(result["stop_reasons"])

    return responses, finish_reasons


@pytest.mark.parametrize(
    "backend,tp_size",
    [
        pytest.param("vllm", 2, marks=pytest.mark.vllm),
        # TODO(Charlie): add TP > 1 tests for sglang when we support it
        pytest.param("sglang", 1, marks=pytest.mark.sglang),
    ],
    ids=["vllm", "sglang"],
)
def test_inference_engines_generation(backend: str, tp_size: int):
    """
    Tests generation with both remote and ray-wrapped engines for the specified backend.
    """
    try:
        cfg = get_test_actor_config()
        cfg.generator.backend = backend
        initialize_ray(cfg)

        prompts = get_test_prompts(MODEL)

        try:
            llm_client, remote_server_process = init_remote_inference_servers(
                tp_size, backend
            )

            # Batched generation
            remote_batch_responses, batch_finish_reasons = asyncio.run(
                run_batch_generation(llm_client, prompts)
            )
            assert len(remote_batch_responses) == len(
                prompts
            ), f"Number of responses should match number of prompts, got {len(remote_batch_responses)} responses but {len(prompts)} prompts"
            assert len(batch_finish_reasons) == len(
                prompts
            ), f"Number of finish reasons should match number of prompts, got {len(batch_finish_reasons)} finish reasons but {len(prompts)} prompts"

            # Single generation (ie, submit individual requests)
            remote_single_responses, single_finish_reasons = asyncio.run(
                run_single_generation(llm_client, prompts)
            )
            assert len(remote_single_responses) == len(
                prompts
            ), f"Number of responses should match number of prompts, got {len(remote_single_responses)} responses but {len(prompts)} prompts"
            assert len(single_finish_reasons) == len(
                prompts
            ), f"Number of finish reasons should match number of prompts, got {len(single_finish_reasons)} finish reasons but {len(prompts)} prompts"

            # Ensure batched and single generation outputs are (roughly) the same
            for i in range(len(prompts)):
                if not are_responses_similar(
                    remote_batch_responses[i],
                    remote_single_responses[i],
                    tolerance=0.01,
                ):
                    print(
                        f"Remote batch and single generation responses are not similar, got batch={remote_batch_responses[i]} and single={remote_single_responses[i]}"
                    )

        finally:
            remote_server_process.terminate()
            remote_server_process.wait()

        # Get responses from Ray engine
        llm_client = init_ray_inference_engines(backend, tp_size)

        # Batched generation
        local_batch_responses, batch_finish_reasons = asyncio.run(
            run_batch_generation(llm_client, prompts)
        )
        assert len(local_batch_responses) == len(
            prompts
        ), f"Number of responses should match number of prompts, got {len(local_batch_responses)} responses but {len(prompts)} prompts"
        assert len(batch_finish_reasons) == len(
            prompts
        ), f"Number of finish reasons should match number of prompts, got {len(batch_finish_reasons)} finish reasons but {len(prompts)} prompts"

        # Single generation (ie, submit individual requests)
        local_single_responses, single_finish_reasons = asyncio.run(
            run_single_generation(llm_client, prompts)
        )
        assert len(local_single_responses) == len(
            prompts
        ), f"Number of responses should match number of prompts, got {len(local_single_responses)} responses but {len(prompts)} prompts"
        assert len(single_finish_reasons) == len(
            prompts
        ), f"Number of finish reasons should match number of prompts, got {len(single_finish_reasons)} finish reasons but {len(prompts)} prompts"

        # Ensure batched and single generation outputs are (roughly) the same
        for i in range(len(prompts)):
            if not are_responses_similar(
                local_batch_responses[i], local_single_responses[i], tolerance=0.01
            ):
                print(
                    f"Local batch and single generation responses are not similar, got batch={local_batch_responses[i]} and single={local_single_responses[i]}"
                )

        # Finally, ensure that remote and local outputs are (roughly) the same
        for i in range(len(prompts)):
            if not are_responses_similar(
                remote_batch_responses[i], local_batch_responses[i], tolerance=0.01
            ):
                print(
                    f"Remote and local batch generation responses are not similar, got remote={remote_batch_responses[i]} and local={local_batch_responses[i]}"
                )

    finally:
        ray.shutdown()


@pytest.mark.parametrize(
    "backend,tp_size",
    [
        pytest.param("vllm", 2, marks=pytest.mark.vllm),
        # TODO(Charlie): add TP > 1 tests for sglang when we support it
        pytest.param("sglang", 1, marks=pytest.mark.sglang),
    ],
    ids=["vllm", "sglang"],
)
def test_token_based_generation(backend: str, tp_size: int):
    """Test generation using prompt_token_ids for the specified backend."""

    try:
        cfg = get_test_actor_config()
        cfg.generator.backend = backend
        initialize_ray(cfg)

        prompts = get_test_prompts(MODEL, 3)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        prompt_token_ids = tokenizer.apply_chat_template(
            prompts, add_generation_prompt=True, tokenize=True, return_dict=True
        )["input_ids"]

        llm_client = init_ray_inference_engines(backend, tp_size)

        # Test batch generation with tokens
        token_batch_responses, _ = asyncio.run(
            run_batch_generation_with_tokens(llm_client, prompt_token_ids)
        )
        assert len(token_batch_responses) == len(prompts)

        # Test single generation with tokens
        token_single_responses, _ = asyncio.run(
            run_single_generation_with_tokens(llm_client, prompt_token_ids)
        )
        assert len(token_single_responses) == len(prompts)

        # Compare with prompt-based generation
        prompt_responses, _ = asyncio.run(run_batch_generation(llm_client, prompts))

        # Outputs should be similar since we're using the same inputs
        for i in range(len(prompts)):
            if not are_responses_similar(
                [token_batch_responses[i]], [prompt_responses[i]], tolerance=0.01
            ):
                print(
                    f"Token and prompt responses differ: token={token_batch_responses[i]}, prompt={prompt_responses[i]}"
                )

    finally:
        ray.shutdown()
