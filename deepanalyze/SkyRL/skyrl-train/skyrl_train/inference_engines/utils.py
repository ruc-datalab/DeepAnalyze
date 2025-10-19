from typing import Dict, Any
from omegaconf import DictConfig, ListConfig


def get_vllm_sampling_params(sampling_params: DictConfig) -> Dict[str, Any]:
    vllm_sampling_params = {
        "min_tokens": 1,
        "skip_special_tokens": False,
        "include_stop_str_in_output": True,
        "max_tokens": sampling_params.max_generate_length,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "top_k": sampling_params.top_k,
        "min_p": sampling_params.min_p,
        "logprobs": sampling_params.logprobs,
    }
    exclude_keys = ["max_generate_length"]
    for key, value in sampling_params.items():
        if key not in vllm_sampling_params and key not in exclude_keys:
            # Convert OmegaConf ListConfig to regular list if needed
            if isinstance(value, ListConfig):
                value = list(value)
            vllm_sampling_params[key] = value
    return vllm_sampling_params


def get_sglang_sampling_params(sampling_params: DictConfig) -> Dict[str, Any]:
    # min_tokens, include_stop_str_in_output are not used in sglang
    sglang_sampling_params = {
        "skip_special_tokens": False,
        "max_new_tokens": sampling_params.max_generate_length,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "top_k": sampling_params.top_k,
        "min_p": sampling_params.min_p,
    }
    # logprobs not supported with sglang for now
    exclude_keys = ["max_generate_length", "logprobs"]
    for key, value in sampling_params.items():
        if key not in sglang_sampling_params and key not in exclude_keys:
            # Convert OmegaConf ListConfig to regular list if needed
            if isinstance(value, ListConfig):
                value = list(value)
            sglang_sampling_params[key] = value
    return sglang_sampling_params


def get_sampling_params_for_backend(
    backend: str, sampling_params: DictConfig
) -> Dict[str, Any]:
    if backend == "vllm":
        return get_vllm_sampling_params(sampling_params)
    elif backend == "sglang":
        return get_sglang_sampling_params(sampling_params)
    else:
        raise ValueError(f"Unsupported generation backend: {backend}")
