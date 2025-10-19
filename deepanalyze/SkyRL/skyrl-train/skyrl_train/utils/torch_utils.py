# This code is adapted from VERL
# https://github.com/volcengine/verl/blob/main/verl/utils/torch_functional.py
# The original copyright is reproduced below:
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext
import torch
import torch.nn.functional as F
from jaxtyping import Float, Integer

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False


CHUNK_SIZE = 1024


def chunked_cross_entropy_from_log_probs(
    logprobs: Float[torch.Tensor, "batch_size seqlen vocab_size"],
    requires_grad: bool = False,
) -> Float[torch.Tensor, "batch_size seqlen"]:
    cm = nullcontext() if requires_grad else torch.no_grad()
    with cm:
        # Calculate entropy in chunks to avoid OOM
        chunk_size = CHUNK_SIZE
        num_chunks = (logprobs.size(1) + chunk_size - 1) // chunk_size
        entropy_tensor = torch.zeros(
            (logprobs.shape[0], logprobs.shape[1]),
            dtype=logprobs.dtype,
            device=logprobs.device,
        )

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, logprobs.size(1))
            # (bsz, seq, vocab_size)
            chunk = logprobs[:, start_idx:end_idx]

            # Calculate entropy for this chunk
            chunk_probs = chunk.exp()
            chunk_entropy = -(chunk_probs * chunk).sum(-1)
            entropy_tensor[:, start_idx:end_idx] = chunk_entropy
    return entropy_tensor


# NOTE: we don't actually use jaxtype for runtime type checking since it doesn't play well with torch compile
def chunked_entropy_from_logits(
    logits: Float[torch.Tensor, "batch_size seqlen vocab"], requires_grad: bool = False
) -> Float[torch.Tensor, "batch_size seqlen"]:
    """Chunked entropy calculation from logits.

    Avoids allocating a full log probabilities tensor to save memory. For models like Qwen with large vocab sizes, this can reduce gpu memory significantly (~O(10GB))
    """
    cm = nullcontext() if requires_grad else torch.no_grad()
    with cm:
        # Calculate entropy in chunks to avoid OOM
        chunk_size = CHUNK_SIZE
        num_chunks = (logits.size(1) + chunk_size - 1) // chunk_size
        entropy_tensor = torch.zeros(
            (logits.shape[0], logits.shape[1]), dtype=logits.dtype, device=logits.device
        )

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, logits.size(1))
            # (bsz, seq, vocab_size)
            chunk = logits[:, start_idx:end_idx]
            chunk_logprob = F.log_softmax(chunk, dim=-1)

            # Calculate entropy for this chunk
            chunk_probs = chunk_logprob.exp()
            chunk_entropy = -(chunk_probs * chunk_logprob).sum(-1)
            entropy_tensor[:, start_idx:end_idx] = chunk_entropy
    return entropy_tensor


# Adapt from VERL
def logprobs_from_logits(
    logits: Float[torch.Tensor, "batch_size seqlen vocab_size"],
    labels: Integer[torch.Tensor, "batch_size seqlen"],
    inplace_backward=True,
) -> Float[torch.Tensor, "batch_size seqlen"]:
    """
    Compute per-token log-probabilities for the given labels.

    Uses a Flash-Attention–based cross-entropy (if available) for efficient backward,
    otherwise falls back to a standard log-softmax+gather approach.

    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591

    Args:
        logits (Tensor): Model outputs of shape (..., vocab_size).
        labels (LongTensor): True class indices of shape matching logits[..., :-1].
        inplace_backward (bool): If True and Flash-Attn is available, perform backward in-place.

    Returns:
        Tensor: Log-probabilities of the target labels, shape logits.shape[:-1].
    """
    if FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        logits = logits.reshape(-1, last_dim)
        labels = labels.reshape(-1)
        output = logprobs_from_logits_flash_attn(
            logits, labels, inplace_backward=inplace_backward
        )
        output = output.view(*batch_dim)
    else:
        output = logprobs_from_logits_v2(logits, labels)
    return output


def logprobs_from_logits_flash_attn(logits, labels, inplace_backward=True):
    output = cross_entropy_loss(logits, labels, inplace_backward=inplace_backward)
    assert isinstance(
        output, tuple
    ), "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
    return -output[0]


# Credits: https://www.tylerromero.com/posts/2025-02-selective-log-softmax/
# https://github.com/volcengine/verl/pull/220
def logprobs_from_logits_v2(
    logits: Float[torch.Tensor, "batch_size seqlen vocab_size"],
    labels: Integer[torch.Tensor, "batch_size seqlen"],
) -> Float[torch.Tensor, "batch_size seqlen"]:
    """
    A memory efficient implementation of logprobs_from_logits
    """
    if logits.dtype in [torch.float32, torch.float64]:
        logits_labels = torch.gather(
            logits, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack(
            [torch.logsumexp(logit, dim=-1) for logit in logits]
        )
        logprobs_labels = (
            logits_labels - logsumexp_values
        )  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        logprobs_labels = []
        for row_logits, row_labels in zip(
            logits, labels
        ):  # loop to reduce peak mem consumption
            row_logprobs = F.log_softmax(row_logits, dim=-1)
            row_logprobs_labels = row_logprobs.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            logprobs_labels.append(row_logprobs_labels)
        logprobs_labels = torch.stack(logprobs_labels)
    return logprobs_labels
