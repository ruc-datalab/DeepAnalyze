from typing import List, Tuple, Optional
import torch
from transformers import AutoTokenizer
from jaxtyping import Float


def _verify_inputs(
    prompts: List[List[int]],
    responses: List[List[int]],
    custom_rewards: Optional[List[torch.Tensor]],
    loss_masks: List[List[int]],
):
    assert (
        len(prompts) == len(responses) and len(prompts) > 0
    ), "prompts and responses must have the same length and length must be greater than 0, got {} and {}".format(
        len(prompts), len(responses)
    )

    if custom_rewards is not None:
        assert len(custom_rewards) == len(
            prompts
        ), "custom_rewards must have the same length as prompts, got {} and {}".format(
            len(custom_rewards), len(prompts)
        )
    assert len(loss_masks) == len(
        prompts
    ), "loss_masks must have the same length as prompt, got {} and {}".format(
        len(loss_masks), len(prompts)
    )


def convert_prompts_responses_to_batch_tensors(
    tokenizer: AutoTokenizer,
    prompts: List[List[int]],
    responses: List[List[int]],
    custom_rewards: List[List[float]],
    loss_masks: List[List[int]],
    logprobs: Optional[List[List[float]]] = None,
) -> Tuple[
    Float[torch.Tensor, "batch seq_len"],
    Float[torch.Tensor, "batch seq_len"],
    Float[torch.Tensor, "batch response_len"],
    Float[torch.Tensor, "batch response_len"],
    Float[torch.Tensor, "batch response_len"],
    Optional[Float[torch.Tensor, "batch response_len"]],
]:
    """
    Convert prompts and responses to batch tensors for training.

    This function concatenates all prompts and responses to the following format:

    | [PAD] [PAD] token token token | token token [PAD] [PAD] |
    | token token token token token | token token [PAD] [PAD] |
    | [PAD] [PAD] [PAD] token token | token token token [PAD] |
    |<---------- prompt ----------->|<-------- answer ------->|

    Assumes that the responses already contain an eos token at index -1.

    Args:
        tokenizer: Model tokenizer
        prompts: List of tokenized prompts
        responses: List of tokenized responses
        custom_rewards: List of custom rewards for each response
        loss_masks: List of loss masks for each response
        logprobs: List of rollout log probs for each response

    Returns:
        sequences: Full trajectories (padded and concatenated prompts and responses). Size: (batch, seq_len).
        attention_mask: Attention mask for the model. Size: (batch, seq_len)
        action_mask: Response mask for the model. Size: (batch, response_len)
        custom_rewards: Custom rewards for each output. Size: (batch, response_len)
        loss_masks: Loss masks for each output. Size: (batch, response_len)
    """
    _verify_inputs(prompts, responses, custom_rewards, loss_masks)

    max_input_len, max_output_len = 0, 0
    prompt_token_lens, response_token_lens = [], []
    inputs_token_ids, outputs_token_ids = [], []
    for prompt, response in zip(prompts, responses):

        inputs_token_ids.append(prompt)
        outputs_token_ids.append(response)

        prompt_token_len = len(prompt)
        response_token_len = len(response)
        prompt_token_lens.append(prompt_token_len)
        response_token_lens.append(response_token_len)

        max_input_len = max(max_input_len, prompt_token_len)
        max_output_len = max(max_output_len, response_token_len)

    pad_token_id = tokenizer.pad_token_id
    sequences = []
    attention_masks = []
    action_masks = []
    for i, prompt in enumerate(prompts):
        # left padding input
        input_len = prompt_token_lens[i]
        input_ids = [pad_token_id] * (max_input_len - input_len) + list(
            inputs_token_ids[i]
        )
        input_attention_mask = [0] * (max_input_len - input_len) + [1] * input_len

        # right padding output
        output_len = response_token_lens[i]
        output_ids = list(outputs_token_ids[i]) + [pad_token_id] * (
            max_output_len - output_len
        )
        output_attention_mask = [1] * output_len + [0] * (max_output_len - output_len)

        # concat input and output
        sequences.append(input_ids + output_ids)
        attention_masks.append(input_attention_mask + output_attention_mask)
        action_masks.append(output_attention_mask)

    sequences = torch.tensor(sequences)
    attention_mask = torch.tensor(attention_masks, dtype=torch.int64)
    action_mask = torch.tensor(action_masks, dtype=torch.int64)

    # initialize ret loss masks to be the same as action mask
    ret_loss_masks = torch.zeros_like(action_mask, dtype=torch.float)
    for i, loss_mask in enumerate(loss_masks):
        ret_loss_masks[i, : len(loss_mask)] = torch.tensor(loss_mask)

    # do the same for custom rewards
    ret_custom_rewards = torch.zeros_like(action_mask, dtype=torch.float)
    for i, custom_reward in enumerate(custom_rewards):
        if isinstance(custom_reward, list):
            custom_reward = torch.tensor(custom_reward)
        ret_custom_rewards[i, : len(custom_reward)] = custom_reward

    logprobs_tensor = None
    if logprobs:
        max_output_len = action_mask.size(1)
        padded_logprobs = [
            sample_logprobs + [0.0] * (max_output_len - len(sample_logprobs))
            for sample_logprobs in logprobs
        ]
        logprobs_tensor = torch.tensor(padded_logprobs, dtype=torch.float)

    return (
        sequences,
        attention_mask,
        action_mask,
        ret_custom_rewards,
        ret_loss_masks,
        logprobs_tensor,
    )
