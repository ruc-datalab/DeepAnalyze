"""
uv run --extra dev --isolated pytest tests/cpu/generators/test_skyrl_gym_generator_chat_templating.py
"""

import pytest
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock
from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl_train.generators.base import GeneratorInput, GeneratorOutput

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from omegaconf import DictConfig
from transformers import AutoTokenizer
from skyrl_gym.envs import register
from skyrl_train.generators.utils import get_custom_chat_template


# Setup for formatting tests
class CPUTestEnv(BaseTextEnv):
    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()
        self.max_turns = 3

    def init(self, prompt):
        return prompt, {}

    def step(self, action: str):
        self.turns += 1
        done = self.turns >= self.max_turns
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": f"{self.turns}"}] if not done else [],
            reward=0,
            done=done,
            metadata={},
        )


def _register_test_env_if_needed():
    """Register the test env only if it's not already registered."""
    try:
        register(
            id="cpu_test_env",
            entry_point="tests.cpu.generators.test_skyrl_gym_generator_chat_templating:CPUTestEnv",
        )
    except Exception:
        # Environment already registered, ignore
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name", ["Qwen/Qwen2.5-0.5B-Instruct", "unsloth/Llama-3.2-1B-Instruct", "Qwen/Qwen3-0.6B"]
)
async def test_skyrl_gym_generator_chat_templating_exact(model_name):
    _register_test_env_if_needed()  # Register only when needed
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mock_llm = MagicMock()

    # Mock the new generate method
    def mock_generate(input_batch):
        num_prompts = len(input_batch["prompts"]) if "prompts" in input_batch else len(input_batch["prompt_token_ids"])
        return {
            "responses": ["b" + tokenizer.eos_token] * num_prompts,
            "stop_reasons": ["stop"] * num_prompts,
            "response_logprobs": None,
        }

    mock_llm.generate = AsyncMock(side_effect=mock_generate)
    # Create a mock generator config
    generator_cfg = DictConfig(
        {
            "sampling_params": {"max_generate_length": 200, "logprobs": None},
            "max_input_length": 200,
            "batched": False,
            "max_turns": 3,
            "zero_reward_on_non_stop": False,
            "apply_overlong_filtering": False,
            "use_conversation_multi_turn": True,
        }
    )
    env_cfg = DictConfig(
        {
            "max_env_workers": 0,
            "env_class": "cpu_test_env",
        }
    )
    generator = SkyRLGymGenerator(
        generator_cfg=generator_cfg,
        skyrl_gym_cfg=env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=tokenizer,
        model_name=model_name,
    )

    prompt = [[{"role": "user", "content": "a"}]]
    extras = [{"answer": "4"}]

    input_batch: GeneratorInput = {
        "prompts": prompt,
        "env_extras": extras,
        "env_classes": [env_cfg.env_class],
    }
    generator_output: GeneratorOutput = await generator.generate(input_batch)

    # assume every actual message is 1 token for loss mask checking
    expected_chat_history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "2"},
        {"role": "assistant", "content": "b"},
    ]

    # check that the full response is exactly string matching with applying the chat template on history
    prompt_str = tokenizer.decode(generator_output["prompt_token_ids"][0])
    resp_str = tokenizer.decode(generator_output["response_ids"][0])
    custom_chat_template = get_custom_chat_template(model_name)
    if custom_chat_template is not None:
        assert prompt_str + resp_str == tokenizer.apply_chat_template(
            expected_chat_history, chat_template=custom_chat_template, tokenize=False
        )
    else:
        assert prompt_str + resp_str == tokenizer.apply_chat_template(expected_chat_history, tokenize=False)

    # check loss mask exact matches
    system_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": ""}] if "Llama" in model_name else [{}], tokenize=True
    )
    empty_user = tokenizer.apply_chat_template([{"role": "user", "content": ""}], tokenize=True)
    empty_user_with_generation_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True
    )
    # TODO (erictang000): consider hard coding the full loss mask for each model to avoid copying logic in code
    generation_prompt_ids = empty_user_with_generation_prompt[len(empty_user) :]
    empty_user = empty_user[len(system_prompt) :]
    expected_assistant_loss_mask = [0] * len(generation_prompt_ids) + [1, 1]  # 1 for single response token, 1 for eos
    if "Qwen" in model_name:
        expected_assistant_loss_mask += [0]  # extra 0 for \n for qwen templates
    expected_user_loss_mask = [0] * len(empty_user) + [0]  # extra 0 for single observation token

    expected_loss_masks = (expected_assistant_loss_mask + expected_user_loss_mask) * 2 + expected_assistant_loss_mask
    assert len(expected_loss_masks) == len(generator_output["loss_masks"][0])
    assert generator_output["loss_masks"][0] == expected_loss_masks
