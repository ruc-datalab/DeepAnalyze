import torch
from typing import List, Tuple, Union, Optional
from collections import defaultdict
import numpy as np
from skyrl_train.generators.base import GeneratorOutput

CUSTOM_CHAT_TEMPLATES = {
    # chat template for qwen3 thinking mode to remove think tokens similar to generation phase
    "qwen3_thinking": (
        "{% for message in messages %}"
        "{% if (message['role'] != 'assistant') %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% elif (message['role'] == 'assistant')%}"
        "{{'<|im_start|>' + message['role'] + '\n'}}"
        "{% generation %}"
        "{% set full_content = message['content'] %}"
        "{% set mycontent = message['content'] %}"
        "{% set is_last_message = loop.last and messages[-1]['role'] == 'assistant' %}"
        "{% if '</think>' in full_content and not is_last_message %}"
        "{% set mycontent = full_content.split('</think>')[-1].lstrip('\n') %}"
        "{% endif %}"
        "{{mycontent + '<|im_end|>'}}"
        "{% endgeneration %}"
        "{{'\n'}}"
        "{% endif %}"
        "{% endfor %}"
    ),
}


def get_custom_chat_template(model_name: str) -> str:
    if "Qwen3" in model_name:
        return CUSTOM_CHAT_TEMPLATES["qwen3_thinking"]
    else:
        return None


def get_generation_prompt_ids(tokenizer) -> List[int]:
    """
    Helper function to get the generation prompt ids for a given tokenizer.
    """
    empty_user = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], tokenize=True
    )
    empty_user_with_generation_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True
    )

    generation_prompt_ids = empty_user_with_generation_prompt[len(empty_user) :]
    return generation_prompt_ids


@torch.no_grad()
def get_metrics_from_generator_output(
    generator_output: GeneratorOutput, uids: List[str]
) -> Tuple[float, Optional[float]]:
    """
    Get `mean_raw_reward` (or avg_score), `pass_at_n` from generator output.
    """
    rewards: Union[List[float], List[List[float]]] = generator_output["rewards"]
    if not len(rewards):
        raise ValueError(f"`rewards` must be a non-empty list, got {rewards}")

    if isinstance(rewards[0], list):
        # We just compute mean over sequence reward.
        # TODO: We should make metrics customizable by the environment
        mean_raw_reward = float(np.mean([sum(seq_rewards) for seq_rewards in rewards]))
        pass_at_n = None  # not computed for token-level rewards since it's ill-defined
    else:
        mean_raw_reward = float(np.mean(rewards))
        # Compute pass@N metrics
        pass_at_n_dict = defaultdict(list)
        for i, reward in enumerate(rewards):
            pass_at_n_dict[uids[i]].append(reward)

        # pass@N metric
        pass_at_n = sum(1 for v in pass_at_n_dict.values() if np.sum(v) > 0) / len(
            pass_at_n_dict
        )

    return mean_raw_reward, pass_at_n


def concatenate_generator_outputs(
    generator_outputs: List[GeneratorOutput],
) -> GeneratorOutput:
    """
    Used in eval to concatenate the generator outputs of multiple batches.

    `rollout_metrics` are not concatenated because they are already aggregated.
    """
    assert len(generator_outputs) > 0
    has_rollout_logprobs = [
        output.get("rollout_logprobs") is not None for output in generator_outputs
    ]
    if any(has_rollout_logprobs) and not all(has_rollout_logprobs):
        raise ValueError(
            "generator outputs are expected to all have null rollout_logprobs or all non-null, but received a mix"
        )
    result: GeneratorOutput = {
        "prompt_token_ids": sum(
            [output["prompt_token_ids"] for output in generator_outputs], []
        ),
        "response_ids": sum(
            [output["response_ids"] for output in generator_outputs], []
        ),
        "rewards": sum([output["rewards"] for output in generator_outputs], []),
        "loss_masks": sum([output["loss_masks"] for output in generator_outputs], []),
        "rollout_logprobs": (
            sum([output["rollout_logprobs"] for output in generator_outputs], [])
            if generator_outputs[0]["rollout_logprobs"]
            else None
        ),
    }
    if "stop_reasons" in generator_outputs[0]:
        result["stop_reasons"] = sum(
            [output["stop_reasons"] for output in generator_outputs], []
        )

    return result


def apply_overlong_filtering(
    loss_masks: List[List[int]],
    response_ids: List[List[int]],
    eos_token_id: int,
) -> List[List[int]]:
    """
    Implements DAPO Overlong Filtering: zero-out every token's mask whenever
    the response does not end with the eos token id (i.e. truncated).

    Returns:
        - The loss masks with tokens zeroed out for truncated responses
    """
    assert len(loss_masks) == len(
        response_ids
    ), "loss_masks and response_ids must have the same length"
    return [
        [0] * len(mask) if not response or response[-1] != eos_token_id else mask
        for mask, response in zip(loss_masks, response_ids)
    ]
